"""配置文件驱动的机器人推理控制。

从 YAML 配置一键创建 Robot + Sensors + InferenceClient，
运行 感知→推理→执行 控制循环。每次运行自动将视频和 action/state
数据保存到日志目录（与 Core.logging 统一）。

用法:
    python Example/robot_inference.py Config/rizon4_example.yaml
    python Example/robot_inference.py Config/rizon4_example.yaml --prompt "pick up the red cup"
    python Example/robot_inference.py Config/rizon4_example.yaml --dry-run
    python Example/robot_inference.py Config/rizon4_example.yaml --no-home --show-cameras

控制循环中的按键:
    q — 退出
    r — 重启推理 (reset policy + 清空缓存 + 回 Home)
    Enter — 硬件就绪后开始推理
"""
from __future__ import annotations

import argparse
import csv
import json
import datetime
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import Action, ActionSpace, Observation, load_config, setup_run_logger
from Core.config_schema import AsyncInferenceModeEnum, SystemConfig
from Inference import InferenceClient
from Inference.action_processing import (
    canonicalize_action_chunk,
    format_policy_state_vector,
    max_eef_action_delta,
)
from Inference.action_smoothing import TemporalActionSmoother
from Inference.async_worker import AsyncInferenceWorker, InferenceObservationSnapshot
from Inference.dispatch import ActionDispatcher, build_policy_state_vector
from Inference.tactile_plan_worker import TactilePlanWorker
from Inference.obs_mapping import build_camera_key_map, map_image_keys
from Robot import BaseRobot
from Sensor.manager import SensorManager

log = logging.getLogger(__name__)


# ── 高优先级退出：Ctrl+C 先保存视频/CSV，再触发机器人注销（回零 + 阻尼），硬退出 ──

_running = True
_restart_requested = False
_await_start = False        # 硬件就绪后是否等待按 Enter 开始推理
_reset_key_pressed = False  # stdin/cv2 任一来源的 r 键信号（主循环消费后清零）
_enter_key_pressed = False  # stdin/cv2 任一来源的 Enter 键信号（主循环消费后清零）
_robot_ref: "BaseRobot | None" = None  # 注销目标
_shutdown_refs: dict = {
    "cam_worker": None,   # BackgroundCameraWorker
    "recorder": None,     # VideoRecorder
    "data_logger": None,  # DataLogger
    "action_trace": None,  # ActionTraceLogger
    "go_home_exit_speed_percent": None,
}
_shutdown_triggered = False


def _emergency_exit(sig, frame):
    """SIGINT / SIGTERM 处理器：保存视频/数据 → 机器人注销 → 硬退出。

    disconnect() 是 "死亡代码"——ARX5 / ARX5 双臂的实现是
    reset_to_home()（回到 0 位）→ set_to_damping()（阻尼态）。

    顺序：先停相机后台线程并 release mp4 / flush csv（这些都是毫秒级），
    再执行机器人 disconnect（会阻塞几秒等回零）。若 disconnect 挂死，
    第二次 Ctrl+C 直接硬退出——此时视频/CSV 已经落盘。

    不走 _running=False 软退出路径，那条路径会被 go_home /
    predict_chunk / 插值循环阻塞住。
    """
    global _running, _shutdown_triggered
    _running = False
    if _shutdown_triggered:
        try:
            sys.stderr.write("\n[!!] 二次 Ctrl+C，硬退出\n")
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(130)
    _shutdown_triggered = True
    try:
        sys.stderr.write("\n[!] 收到 Ctrl+C，保存数据并执行机器人注销 ...\n")
        sys.stderr.flush()
    except Exception:
        pass

    # 1) 停止相机后台线程（避免 release 时还在写帧）
    cw = _shutdown_refs.get("cam_worker")
    if cw is not None:
        try:
            cw.stop()
        except Exception as e:
            sys.stderr.write(f"[!] cam_worker.stop 失败: {e!r}\n")

    # 2) 保存视频（VideoWriter.release 写入尾部索引）
    rec = _shutdown_refs.get("recorder")
    if rec is not None:
        try:
            rec.stop()
        except Exception as e:
            sys.stderr.write(f"[!] recorder.stop 失败: {e!r}\n")

    # 3) flush + 关闭 CSV
    dl = _shutdown_refs.get("data_logger")
    if dl is not None:
        try:
            dl.close()
        except Exception as e:
            sys.stderr.write(f"[!] data_logger.close 失败: {e!r}\n")

    at = _shutdown_refs.get("action_trace")
    if at is not None:
        try:
            at.close()
        except Exception as e:
            sys.stderr.write(f"[!] action_trace.close 失败: {e!r}\n")

    # 4) Ctrl+C 后也尽量回 Home，再注销。数据已安全落盘。
    if _robot_ref is not None:
        try:
            sys.stderr.write("[!] Ctrl+C 后自动回 Home ...\n")
            sys.stderr.flush()
            _robot_ref.go_home(velocity=_shutdown_refs.get("go_home_exit_speed_percent"))
            sys.stderr.write("[!] 自动回 Home 完成\n")
            sys.stderr.flush()
        except Exception as e:
            try:
                sys.stderr.write(f"[!] go_home 失败: {e!r}\n")
                sys.stderr.flush()
            except Exception:
                pass
        try:
            _robot_ref.disconnect()
        except Exception as e:
            try:
                sys.stderr.write(f"[!] disconnect 失败: {e!r}\n")
                sys.stderr.flush()
            except Exception:
                pass
    os._exit(130)


signal.signal(signal.SIGINT, _emergency_exit)
signal.signal(signal.SIGTERM, _emergency_exit)


# ── stdin 键盘监听：让 Enter / r / q 在没有 cv2 窗口时也能用 ─────────────

def _start_stdin_listener() -> None:
    """后台线程读 stdin 单字符：Enter 开始推理，'r' 触发 reset，'q' 退出。

    只在 stdin 是 tty 时启动。进入 cbreak 模式以避免回车要求；进程退出
    时由 os._exit 直接结束，不恢复 tty——shell 仍然会 reset 自己的终端。
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios, tty, select
    except Exception:
        return

    def _worker() -> None:
        global _enter_key_pressed, _reset_key_pressed, _running
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while _running:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == "r":
                    _reset_key_pressed = True
                elif ch == "q":
                    _running = False
                elif ch in ("\n", "\r"):
                    _enter_key_pressed = True
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()


def _consume_start_signal(cam_worker) -> str | None:
    """Consume one start-waiting key event.

    Returns "start", "quit", or None. Reset is intentionally ignored while
    waiting for the user to confirm hardware is ready.
    """
    global _enter_key_pressed, _reset_key_pressed, _running

    if cam_worker is not None:
        key = cam_worker.poll_key()
        if key in (10, 13):
            return "start"
        if key == ord("q"):
            _running = False
            return "quit"
        if key == ord("r"):
            log.info("已在等待开始状态，忽略 reset 键")
            return None

    if _enter_key_pressed:
        _enter_key_pressed = False
        return "start"
    if _reset_key_pressed:
        _reset_key_pressed = False
        log.info("已在等待开始状态，忽略 reset 键")
    return None


# ── 视频录制器 ────────────────────────────────────────────────


class VideoRecorder:
    """将多路相机帧水平拼接后通过 ffmpeg subprocess 编码为 H.264 MP4。"""

    def __init__(
        self,
        log_dir: Path,
        fps: float,
        *,
        codec: str = "libx264",
        crf: int = 28,
        preset: str = "veryfast",
    ) -> None:
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._codec = codec
        self._crf = crf
        self._preset = preset
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None  # (w, h)
        self._video_path: Path | None = None
        self._frame_count = 0

    def start(self, episode: int = 1) -> Path:
        """开始新的录制 session，返回视频文件路径。"""
        self.stop()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = self._log_dir / f"episode_{episode:03d}_{ts}.mp4"
        self._frame_count = 0
        log.info("开始录制视频: %s (codec=%s crf=%d preset=%s)",
                 self._video_path, self._codec, self._crf, self._preset)
        return self._video_path

    def write_frame(self, images: dict[str, np.ndarray]) -> None:
        """将多路图像水平拼接后写入一帧。"""
        if not images or self._video_path is None:
            return
        frames = []
        for cam_name in sorted(images.keys()):
            frame = images[cam_name]
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames.append(frame)

        target_h = frames[0].shape[0]
        resized = []
        for f in frames:
            if f.shape[0] != target_h:
                scale = target_h / f.shape[0]
                new_w = int(f.shape[1] * scale)
                f = cv2.resize(f, (new_w, target_h))
            resized.append(f)
        canvas = np.ascontiguousarray(np.hstack(resized))
        h, w = canvas.shape[:2]

        if self._proc is None:
            self._spawn_ffmpeg(w, h)
            if self._proc is None:
                return

        if (w, h) != self._size:
            log.warning("视频帧尺寸由 %s 变为 (%d,%d)，跳过此帧",
                        self._size, w, h)
            return

        try:
            self._proc.stdin.write(canvas.tobytes())
        except (BrokenPipeError, ValueError, OSError) as e:
            log.error("ffmpeg stdin 写入失败 (%r)，停止录制", e)
            self._cleanup_proc(kill=True)
            return
        self._frame_count += 1

    def _spawn_ffmpeg(self, w: int, h: int) -> None:
        """启动 ffmpeg 子进程，rawvideo BGR 帧从 stdin 输入。"""
        # pad 到偶数尺寸（yuv420p 要求宽高都是偶数）
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", f"{self._fps}",
            "-i", "-",
            "-an",
            "-c:v", self._codec,
            "-preset", self._preset,
            "-crf", str(self._crf),
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags", "+faststart",
            str(self._video_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._size = (w, h)
        except FileNotFoundError:
            log.error("未找到 ffmpeg，无法录制视频。请先安装 ffmpeg。")
            self._proc = None
        except Exception as e:
            log.error("启动 ffmpeg 失败: %r", e)
            self._proc = None

    def _cleanup_proc(self, *, kill: bool = False) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        if kill:
            try:
                self._proc.kill()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None
        self._size = None

    def stop(self) -> None:
        """结束当前录制。关闭 stdin 让 ffmpeg flush + 写 moov atom，然后等待退出。"""
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.close()
            except Exception:
                pass
            # 注意：不能用 communicate() — 它会再次 flush 已关闭的 stdin 并抛错。
            try:
                rc = self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log.warning("ffmpeg flush 超时，强制终止")
                self._proc.kill()
                rc = self._proc.wait()
            if rc != 0:
                stderr = b""
                try:
                    if self._proc.stderr is not None:
                        stderr = self._proc.stderr.read() or b""
                except Exception:
                    pass
                msg = stderr.decode(errors="ignore").strip()[:500]
                log.warning("ffmpeg 异常退出 (rc=%d): %s", rc, msg)
            log.info("视频已保存: %s (%d 帧)", self._video_path, self._frame_count)
        self._proc = None
        self._size = None
        self._video_path = None
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self._proc is not None or (
            self._video_path is not None and self._frame_count == 0
        )


# ── 数据记录器 ─────────────────────────────────────────────────


class DataLogger:
    """逐帧将 state + action 写入 CSV，方便事后分析和回放。

    CSV 格式: timestamp_s, step, state_0..N, action_0..M
    """

    def __init__(self, log_dir: Path, episode: int = 1) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = log_dir / f"episode_{episode:03d}_{ts}.csv"
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._header_written = False
        log.info("数据记录: %s", self._path)

    def log_step(self, step: int, state: list[float], action: list[float]) -> None:
        if not self._header_written:
            state_cols = [f"state_{i}" for i in range(len(state))]
            action_cols = [f"action_{i}" for i in range(len(action))]
            self._writer.writerow(["timestamp_s", "step"] + state_cols + action_cols)
            self._header_written = True
        self._writer.writerow(
            [f"{time.perf_counter():.6f}", step]
            + [f"{v:.6f}" for v in state]
            + [f"{v:.6f}" for v in action]
        )
        if step % 30 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        log.info("数据记录已关闭: %s", self._path)


class ActionTraceLogger:
    """Write raw server actions and actual robot publish actions under InferSystem."""

    def __init__(self, root: Path) -> None:
        self._server_path = root / "action_server_log.jsonl"
        self._publish_path = root / "action_publish_log.jsonl"
        self._server_file = open(self._server_path, "w", encoding="utf-8")
        self._publish_file = open(self._publish_path, "w", encoding="utf-8")
        log.info("Server action log: %s", self._server_path)
        log.info("Publish action log: %s", self._publish_path)

    @property
    def server_path(self) -> Path:
        return self._server_path

    @property
    def publish_path(self) -> Path:
        return self._publish_path

    def log_server_chunk(self, *, chunk_id: int, raw_actions) -> None:
        for action_idx, action in enumerate(raw_actions):
            self._write_json(
                self._server_file,
                {
                    "timestamp_s": time.perf_counter(),
                    "chunk_id": chunk_id,
                    "action_idx": action_idx,
                    "action": _to_float_list(action),
                },
            )
        self._server_file.flush()

    def log_publish_step(
        self,
        *,
        step: int,
        chunk_id: int,
        action_idx: int,
        raw_action,
        canonical_action,
        publish_action,
    ) -> None:
        publish = _to_float_list(publish_action)
        canonical = _to_float_list(canonical_action)
        record = {
            "timestamp_s": time.perf_counter(),
            "step": step,
            "chunk_id": chunk_id,
            "action_idx": action_idx,
            "raw_action": _to_float_list(raw_action),
            "canonical_action": canonical,
            "publish_action": publish,
        }
        if len(canonical) == 16:
            record["canonical_left_gripper"] = canonical[7]
            record["canonical_right_gripper"] = canonical[15]
        if len(publish) == 16:
            record["publish_left_gripper_m"] = publish[7]
            record["publish_right_gripper_m"] = publish[15]
        self._write_json(self._publish_file, record)
        self._publish_file.flush()

    def close(self) -> None:
        for f in (self._server_file, self._publish_file):
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    @staticmethod
    def _write_json(file, payload: dict) -> None:
        file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _to_float_list(values) -> list[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [float(v) for v in values]


# ── 后台相机读取 + 录制线程 ──────────────────────────────────────


class BackgroundCameraWorker:
    """在后台线程中读取相机 + 录制视频 + 显示画面，不阻塞控制循环。

    控制循环每帧通过 notify() 触发一次异步采集。
    """

    def __init__(
        self,
        sensors,
        enabled_cameras: list[str] | None,
        recorder: VideoRecorder | None = None,
        show: bool = False,
        read_lock: threading.Lock | None = None,
    ) -> None:
        self._sensors = sensors
        self._enabled_cameras = enabled_cameras
        self._recorder = recorder
        self._show = show
        self._read_lock = read_lock
        self._stop_event = threading.Event()
        self._trigger = threading.Event()
        self._latest_images: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._step_info: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._key_pressed: int = -1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def notify(self, step: int = 0, chunk_id: int = 0, action_idx: int = 0, n_exec: int = 0) -> None:
        """控制循环每帧调用，触发后台采集。"""
        with self._lock:
            self._step_info = (step, chunk_id, action_idx, n_exec)
        self._trigger.set()

    def poll_key(self) -> int:
        """读取后台检测到的按键（非阻塞）。"""
        with self._lock:
            k = self._key_pressed
            self._key_pressed = -1
        return k

    def get_latest_images(self) -> dict[str, np.ndarray]:
        """返回最近一帧相机图像（供推理使用）。"""
        with self._lock:
            return dict(self._latest_images)

    def stop(self) -> None:
        self._stop_event.set()
        self._trigger.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._trigger.wait(timeout=1.0)
            self._trigger.clear()
            if self._stop_event.is_set():
                break
            try:
                if self._read_lock is not None:
                    with self._read_lock:
                        images = self._sensors.read_images(self._enabled_cameras)
                else:
                    images = self._sensors.read_images(self._enabled_cameras)
                with self._lock:
                    self._latest_images = images
                    step, chunk_id, action_idx, n_exec = self._step_info

                if self._recorder is not None:
                    self._recorder.write_frame(images)

                if self._show and images:
                    _show_camera_feeds(images, step, chunk_id, action_idx, n_exec)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        with self._lock:
                            self._key_pressed = key
            except Exception:
                log.warning("后台相机线程异常", exc_info=True)


# ── 按键处理 ──────────────────────────────────────────────────


def _poll_key() -> int:
    """非阻塞读取按键，返回按键值或 -1。"""
    return cv2.waitKey(1) & 0xFF


def _format_vec(values, *, precision: int = 4) -> str:
    """Compact vector formatting for action/state diagnostics."""
    return np.array2string(
        np.asarray(values, dtype=np.float64),
        precision=precision,
        suppress_small=True,
        max_line_width=240,
    )


def _log_action_chunk_debug(
    *,
    chunk_id: int,
    state_vec,
    raw_actions,
    actions: list[list[float]],
    n_skip: int,
    arm_state,
    gripper_width: float,
    gripper_max: float,
) -> None:
    """Print action chunk details before safety checks can reject it."""
    raw_arr = np.asarray(raw_actions, dtype=np.float64)
    can_arr = np.asarray(actions, dtype=np.float64)
    log.info(
        "Action chunk #%d: raw_shape=%s canonical_shape=%s n_skip=%d",
        chunk_id, raw_arr.shape, can_arr.shape, n_skip,
    )
    log.info("  state_sent[%d]: %s", len(state_vec), _format_vec(state_vec))
    if len(raw_actions) > 0:
        raw_first = raw_actions[0]
        raw_exec = raw_actions[min(n_skip, len(raw_actions) - 1)]
        raw_last = raw_actions[-1]
        log.info("  raw[0]:    %s", _format_vec(raw_first))
        log.info("  raw[skip]: %s", _format_vec(raw_exec))
        log.info("  raw[-1]:   %s", _format_vec(raw_last))
    if actions:
        can_first = actions[0]
        can_exec = actions[min(n_skip, len(actions) - 1)]
        can_last = actions[-1]
        log.info("  can[0]:    %s", _format_vec(can_first))
        log.info("  can[skip]: %s", _format_vec(can_exec))
        log.info("  can[-1]:   %s", _format_vec(can_last))
    log.info("  current_eef_pose: %s", _format_vec(arm_state.eef_pose))
    gripper_norm = gripper_width / gripper_max if gripper_max > 0 else 0.0
    log.info(
        "  current_gripper: width=%.5fm max=%.5fm norm=%.4f",
        gripper_width, gripper_max, gripper_norm,
    )


def _build_state_vec_for_config(
    arm_state,
    infer_cfg,
    *,
    gripper_width: float,
    gripper_max: float,
    robot=None,
) -> list[float]:
    state_vec = build_policy_state_vector(
        arm_state,
        action_space=infer_cfg.action_space,
        gripper_width=gripper_width,
        gripper_max_width=gripper_max,
        gripper_mode=infer_cfg.gripper_mode.value,
        gripper_threshold=infer_cfg.gripper_threshold,
        arm_dof=infer_cfg.arm_dof,
        gripper_action_index=infer_cfg.gripper_action_index,
        embedded_gripper_max_widths=_embedded_gripper_max_widths_for_config(
            robot,
            infer_cfg,
        ),
    )
    return format_policy_state_vector(
        state_vec,
        action_space=infer_cfg.action_space,
        policy_format=infer_cfg.policy_format,
        canonical_dim=infer_cfg.canonical_dim,
    )


def _effective_cartesian_action_dim(arm_state, infer_cfg) -> int | None:
    if getattr(infer_cfg.action_space, "value", infer_cfg.action_space) != ActionSpace.CARTESIAN.value:
        return None
    if len(getattr(arm_state, "eef_pose", []) or []) >= 14:
        return 20
    arm_dof = infer_cfg.arm_dof or 0
    if arm_dof >= 14:
        return 20
    return 10


def _embedded_gripper_max_widths_for_config(robot, infer_cfg) -> list[float] | None:
    if robot is None:
        return None
    if infer_cfg.gripper_action_index is not None and infer_cfg.gripper_action_index >= 0:
        return None
    try:
        max_values = list(robot.get_params().joint_position_max)
    except Exception:
        log.debug("读取 robot params 失败，内嵌夹爪 state max_width 使用默认值", exc_info=True)
        return None

    arm_dof = infer_cfg.arm_dof or getattr(robot, "dof", len(max_values))
    if arm_dof >= 14 and len(max_values) > 13:
        return [float(max_values[6]), float(max_values[13])]
    if arm_dof == 7 and len(max_values) > 6:
        return [float(max_values[6])]
    return None


def _read_inference_images(
    sensors,
    enabled_cameras: list[str] | None,
    camera_key_map: dict[str, str],
    sensor_read_lock: threading.Lock,
) -> dict[str, np.ndarray]:
    if sensors is not None:
        with sensor_read_lock:
            images = sensors.read_images(enabled_cameras)
    else:
        images = {
            "mock_cam": np.random.randint(
                0, 255, (480, 640, 3), dtype=np.uint8,
            ),
        }
    return map_image_keys(images, camera_key_map)


def _action_passes_safety(
    *,
    action_vec: list[float],
    arm_state,
    infer_cfg,
    robot,
    chunk_id: int,
) -> bool:
    if infer_cfg.action_space.value == ActionSpace.CARTESIAN.value:
        required_eef_len = 14 if len(action_vec) in (14, 16) else 7
        if len(arm_state.eef_pose) < required_eef_len:
            log.error(
                "安全拒绝：EEF 模式下机器人 eef_pose 长度不足 "
                "(需要 %d，实际 %d)，停止",
                required_eef_len, len(arm_state.eef_pose),
            )
            return False
        pos_delta, rot_delta = max_eef_action_delta(
            arm_state.eef_pose, action_vec,
        )
        if (
            pos_delta > infer_cfg.max_eef_delta_m
            or rot_delta > infer_cfg.max_eef_rotation_delta_rad
        ):
            log.error(
                "安全拒绝：chunk #%d EEF 首帧偏差 %.3fm / %.3frad "
                "超过阈值 %.3fm / %.3frad，停止",
                chunk_id, pos_delta, rot_delta,
                infer_cfg.max_eef_delta_m,
                infer_cfg.max_eef_rotation_delta_rad,
            )
            return False
        log.debug(
            "EEF 安全检查通过：首帧偏差 %.3fm / %.3frad",
            pos_delta, rot_delta,
        )
        return True

    arm_dof_check = infer_cfg.arm_dof or robot.dof
    current_q = np.array(arm_state.joint_positions[:arm_dof_check])
    first_q = np.array(action_vec[:arm_dof_check])
    max_delta = float(np.max(np.abs(first_q - current_q)))
    if max_delta > infer_cfg.max_joint_delta_rad:
        log.error(
            "安全拒绝：chunk #%d 首帧偏差 %.3f rad 超过阈值 %.3f rad，停止",
            chunk_id, max_delta, infer_cfg.max_joint_delta_rad,
        )
        return False
    log.debug("安全检查通过：首帧偏差 %.3f rad", max_delta)
    return True


# ── 主控制循环 ────────────────────────────────────────────────


def run_control_loop(
    config_path: str,
    *,
    prompt: str = "",
    dry_run: bool = False,
    go_home_first: bool = True,
    show_cameras: bool = False,
    log_dir: Path | None = None,
    max_steps: int = 0,
    n_execute: int | None = None,
) -> None:
    """配置驱动的主控制循环: 感知 → 推理 → 执行。

    视频和 action/state CSV 默认保存到 log_dir（由 setup_run_logger 返回）。
    dry_run 模式下跳过视频录制。
    """
    global _running, _restart_requested, _await_start, _reset_key_pressed, _enter_key_pressed

    from Core.config_schema import InferenceConfig

    config = load_config(config_path)
    infer_cfg = config.inference or InferenceConfig()
    _shutdown_refs["go_home_exit_speed_percent"] = infer_cfg.go_home_exit_speed_percent
    fps = infer_cfg.fps
    dt = 1.0 / fps
    # tactile expert 异步推理：prepare/refine（stateful tactile 协议）
    tactile_mode = (
        infer_cfg.async_inference.enabled
        and infer_cfg.async_inference.mode == AsyncInferenceModeEnum.tactile_plan
    )
    if n_execute is None:
        n_execute = infer_cfg.n_execute

    do_record = not dry_run and log_dir is not None

    # ── 1. 创建 Robot ──
    global _robot_ref
    log.info("[1/4] 创建机器人...")
    robot = BaseRobot.from_config(config_path)
    robot.connect()
    _robot_ref = robot  # 注入信号处理器：Ctrl+C 时立即 emergency_stop

    try:
        if robot.is_fault():
            log.info("检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(2.0)

        robot.enable()
        if not robot.wait_until_operational(timeout_s=30.0):
            raise RuntimeError("机器人未能在超时时间内变为 operational")
        log.info("机器人已 operational")

        # 创建夹爪
        gripper = robot.create_gripper()
        if gripper is not None:
            gripper.connect()
            log.info("夹爪已初始化")

        try:
            # Action dispatcher (推理输出 → 机器人 + 夹爪)
            dispatcher = ActionDispatcher.from_config(robot, gripper, config)
            enabled_cameras = infer_cfg.enabled_cameras
            camera_key_map = build_camera_key_map(
                config.cameras,
                enabled_cameras,
                config.tactile,
            )
            sensor_read_lock = threading.Lock()

            # ── 2. 创建 Sensors ──
            log.info("[2/4] 创建传感器...")
            if not dry_run:
                sensors = SensorManager.from_config(
                    config,
                    strict=True,
                    enabled_names=enabled_cameras,
                )
                sensors.open_all()
                # 启动前验证：所有期望图像传感器必须能读出图像，否则中止
                expected_cams = enabled_cameras or list(sensors.sensors.keys())
                with sensor_read_lock:
                    test_images = sensors.read_images(expected_cams)
                missing = [n for n in expected_cams if n not in test_images]
                if missing:
                    raise RuntimeError(
                        f"图像传感器未能读出图像，中止启动: {missing} "
                        f"(已就绪: {sorted(test_images.keys())})"
                    )
                log.info("图像传感器就绪: %s", sorted(test_images.keys()))
            else:
                sensors = None

            try:
                # ── 3. 创建 InferenceClient ──
                log.info("[3/4] 连接推理服务器...")
                client = InferenceClient(
                    infer_cfg.server,
                    recv_timeout_ms=infer_cfg.recv_timeout_ms,
                )
                client.connect()
                action_smoother = TemporalActionSmoother(
                    action_space=infer_cfg.action_space,
                )

                try:
                    # 录制器（视频 + 数据），每个 episode 各一份
                    recorder = VideoRecorder(
                        log_dir, fps,
                        codec=infer_cfg.video_codec,
                        crf=infer_cfg.video_crf,
                        preset=infer_cfg.video_preset,
                    ) if do_record else None
                    data_logger: DataLogger | None = None
                    action_trace = ActionTraceLogger(ROOT)
                    _shutdown_refs["action_trace"] = action_trace
                    async_worker: AsyncInferenceWorker | None = None
                    tactile_worker: TactilePlanWorker | None = None
                    effective_action_dim: int | None = None

                    cam_worker: BackgroundCameraWorker | None = None
                    if sensors is not None and (do_record or show_cameras):
                        cam_worker = BackgroundCameraWorker(
                            sensors, enabled_cameras,
                            recorder=recorder,
                            show=show_cameras,
                            read_lock=sensor_read_lock,
                        )
                    # 注入信号处理器：Ctrl+C 时先 flush 这些再 disconnect
                    _shutdown_refs["recorder"] = recorder
                    _shutdown_refs["cam_worker"] = cam_worker

                    _start_stdin_listener()

                    def _build_async_observation() -> InferenceObservationSnapshot:
                        arm = robot.observe()
                        width = 0.0
                        max_width = 1.0
                        if gripper is not None:
                            gs = gripper.observe()
                            width = gs.width
                            max_width = gs.max_width or 1.0
                        state = _build_state_vec_for_config(
                            arm,
                            infer_cfg,
                            gripper_width=width,
                            gripper_max=max_width,
                            robot=robot,
                        )
                        images_for_infer = _read_inference_images(
                            sensors,
                            enabled_cameras,
                            camera_key_map,
                            sensor_read_lock,
                        )
                        return InferenceObservationSnapshot(
                            images=images_for_infer,
                            state=state,
                            prompt=prompt,
                        )

                    episode = 0
                    _restart_requested = True  # 首次进入走 restart 流程
                    _await_start = True        # 首次硬件就绪后等待人工确认
                    async_obs_period = 0.0
                    next_async_obs_t = 0.0

                    while _running:
                        # ── 重启处理 ──
                        if _restart_requested:
                            _restart_requested = False
                            episode += 1
                            log.info("启动 episode %d ...", episode)

                            if async_worker is not None:
                                async_worker.stop()
                                async_worker = None

                            # 关闭上一个 episode 的记录（视频 + CSV 立即落盘）
                            if recorder is not None:
                                recorder.stop()
                            if data_logger is not None:
                                data_logger.close()
                                data_logger = None
                                _shutdown_refs["data_logger"] = None

                            # 回 Home
                            if go_home_first:
                                log.info("回 Home 位置...")
                                robot.go_home()

                            # 移动到训练起始位置 (inference_home)
                            inf_home = infer_cfg.inference_home
                            if inf_home is not None:
                                log.info("移动到训练起始位置 (%d 维)...", len(inf_home))
                                cur_state = list(robot.observe().joint_positions)
                                target = [float(v) for v in inf_home]
                                # go_home 绕过 dispatcher，必须重置裁切参考点，
                                # 否则会把命令锁在旧 chunk 最后位置附近。
                                dispatcher.reset_velocity_tracking()
                                n_interp = int(2.0 * fps)  # 2 秒
                                next_t = time.perf_counter()
                                for _t in range(n_interp):
                                    alpha = (_t + 1) / n_interp
                                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
                                    interp = [c + alpha * (t - c) for c, t in zip(cur_state, target)]
                                    robot.act(Action(ActionSpace.JOINT_POSITION, interp))
                                    next_t += dt
                                    now = time.perf_counter()
                                    if now < next_t:
                                        time.sleep(next_t - now)
                                log.info("已到达训练起始位置")

                                # 等待到位：确认机器人已稳定在 inference_home
                                # 附近，避免还在惯性滑动就进入推理。
                                settle_tol = 0.02  # rad
                                settle_deadline = time.perf_counter() + 2.0
                                while time.perf_counter() < settle_deadline:
                                    cur = list(robot.observe().joint_positions)
                                    err = max(
                                        abs(c - t) for c, t in zip(cur, target)
                                    )
                                    if err < settle_tol:
                                        break
                                    time.sleep(0.02)
                                else:
                                    log.warning(
                                        "到位等待超时，残差 max=%.3f rad（阈值 %.3f）",
                                        err, settle_tol,
                                    )

                            # 硬件初始化/复位到位后，等待人工确认再启动推理。
                            if _await_start:
                                _enter_key_pressed = False
                                _reset_key_pressed = False
                                log.info("== 硬件已就绪，按 Enter 开始推理（q/Ctrl+C 终止）==")
                                while _running and _await_start:
                                    signal = _consume_start_signal(cam_worker)
                                    if signal == "start":
                                        _await_start = False
                                        break
                                    if signal == "quit":
                                        break
                                    time.sleep(0.05)
                                if not _running:
                                    break

                            # Reset policy。tactile_plan 模式必须经 worker 串行化
                            # 网络访问（REQ socket 非线程安全，且可能有在途请求）。
                            if tactile_mode:
                                if tactile_worker is None:
                                    tactile_worker = TactilePlanWorker(
                                        client,
                                        delay_init=infer_cfg.async_inference.delay_init,
                                        raw_action_callback=(
                                            lambda cid, raw: action_trace.log_server_chunk(
                                                chunk_id=cid,
                                                raw_actions=raw,
                                            )
                                        ),
                                    )
                                    tactile_worker.start()
                                tactile_worker.reset_episode()
                            else:
                                client.reset()
                            action_smoother.clear()
                            dispatcher.reset_velocity_tracking()
                            if infer_cfg.async_inference.enabled:
                                # tactile_plan 默认按控制频率发布观测（每步一帧，
                                # 供 refine 持续修正），chunk 模式保持低频。
                                async_obs_fps = float(
                                    infer_cfg.async_inference.obs_fps
                                    if infer_cfg.async_inference.obs_fps is not None
                                    else (float(fps) if tactile_mode else min(4.0, float(fps)))
                                )
                                async_obs_period = 1.0 / max(async_obs_fps, 1e-6)
                                next_async_obs_t = 0.0
                                effective_action_dim = _effective_cartesian_action_dim(
                                    robot.observe(),
                                    infer_cfg,
                                )
                            if tactile_mode:
                                try:
                                    tactile_worker.update_observation(_build_async_observation())
                                    next_async_obs_t = time.perf_counter() + async_obs_period
                                except Exception:
                                    log.warning("tactile 初始 observation 构建失败，等待控制循环刷新", exc_info=True)
                                log.info(
                                    "异步 tactile plan 推理已启动: obs_fps=%.1f delay_init=%d",
                                    async_obs_fps,
                                    infer_cfg.async_inference.delay_init,
                                )
                            elif infer_cfg.async_inference.enabled:
                                async_worker = AsyncInferenceWorker(
                                    client=client,
                                    smoother=action_smoother,
                                    action_space=infer_cfg.action_space,
                                    policy_format=infer_cfg.policy_format,
                                    canonical_dim=infer_cfg.canonical_dim,
                                    effective_action_dim=effective_action_dim,
                                    overlap_steps=infer_cfg.smooth.overlap_steps,
                                    max_latency_steps=infer_cfg.async_inference.max_latency_steps,
                                    latency_compensation=infer_cfg.latency_compensation,
                                    smooth_enabled=infer_cfg.smooth.enabled,
                                    obs_fps=async_obs_fps,
                                    action_fps=fps,
                                    raw_action_callback=(
                                        lambda cid, raw: action_trace.log_server_chunk(
                                            chunk_id=cid,
                                            raw_actions=raw,
                                        )
                                    ),
                                )
                                try:
                                    async_worker.update_observation(_build_async_observation())
                                    next_async_obs_t = time.perf_counter() + async_obs_period
                                except Exception:
                                    log.warning("异步推理初始 observation 构建失败，等待控制循环刷新", exc_info=True)
                                async_worker.start()
                                log.info(
                                    "异步推理已启动: obs_fps=%.1f latency_compensation=%s smooth=%s max_latency_steps=%d overlap_steps=%d",
                                    async_obs_fps,
                                    infer_cfg.latency_compensation,
                                    infer_cfg.smooth.enabled,
                                    infer_cfg.async_inference.max_latency_steps,
                                    infer_cfg.smooth.overlap_steps,
                                )

                            # 开始新 episode 记录
                            if recorder is not None:
                                recorder.start(episode)
                            if do_record and log_dir is not None:
                                data_logger = DataLogger(log_dir, episode)
                            # 每个 episode 的 data_logger 独立，刷新信号处理器引用
                            _shutdown_refs["data_logger"] = data_logger

                            step = 0
                            chunk_id = 0
                            next_frame_t = time.perf_counter()
                            log.info(
                                "[4/4] 启动控制循环 (%.0f Hz, n_execute=%d)",
                                fps, n_execute,
                            )

                        if 0 < max_steps <= step:
                            log.info("达到最大步数 %d，停止", max_steps)
                            break

                        # ── 感知: 机器人状态（读取失败立即停止）──
                        try:
                            arm_state = robot.observe()
                        except Exception:
                            log.error("机械臂状态读取失败，立即停止控制循环", exc_info=True)
                            _running = False
                            break
                        if not arm_state.joint_positions:
                            log.error("机械臂返回空状态，立即停止控制循环")
                            _running = False
                            break

                        gripper_width = 0.0
                        gripper_max = 1.0
                        if gripper is not None:
                            try:
                                gs = gripper.observe()
                                gripper_width = gs.width
                                gripper_max = gs.max_width or 1.0
                            except Exception:
                                log.error("夹爪状态读取失败，立即停止控制循环", exc_info=True)
                                _running = False
                                break

                        state_vec = _build_state_vec_for_config(
                            arm_state,
                            infer_cfg=infer_cfg,
                            gripper_width=gripper_width,
                            gripper_max=gripper_max,
                            robot=robot,
                        )

                        if infer_cfg.async_inference.enabled:
                            active_worker = tactile_worker if tactile_mode else async_worker
                            if active_worker is not None:
                                now_obs = time.perf_counter()
                                if now_obs >= next_async_obs_t:
                                    images_for_infer = _read_inference_images(
                                        sensors,
                                        enabled_cameras,
                                        camera_key_map,
                                        sensor_read_lock,
                                    )
                                    active_worker.update_observation(
                                        InferenceObservationSnapshot(
                                            images=images_for_infer,
                                            state=state_vec,
                                            prompt=prompt,
                                        )
                                    )
                                    next_async_obs_t = now_obs + async_obs_period
                                chunk_id = active_worker.request_count
                            if tactile_mode:
                                try:
                                    raw_action = tactile_worker.pop_action()
                                except Exception:
                                    log.error("异步 tactile 推理失败，停止控制循环", exc_info=True)
                                    _running = False
                                    break
                                action_vec = None
                                if raw_action is not None:
                                    action_vec = canonicalize_action_chunk(
                                        [raw_action],
                                        infer_cfg.action_space,
                                        policy_format=infer_cfg.policy_format,
                                        canonical_dim=infer_cfg.canonical_dim,
                                        effective_action_dim=effective_action_dim,
                                    )[0]
                            else:
                                action_vec = action_smoother.pop_next()

                            if action_vec is None:
                                if cam_worker is not None:
                                    cam_worker.notify(step, chunk_id, 0, 0)
                                    key = cam_worker.poll_key()
                                    if key == ord("q"):
                                        _running = False
                                    elif key == ord("r"):
                                        _reset_key_pressed = True
                                if _reset_key_pressed:
                                    _reset_key_pressed = False
                                    if active_worker is not None:
                                        active_worker.pause()
                                    log.info("收到 reset（r）: 保存数据 → 回起始位置 → 等待按 Enter 开始")
                                    _restart_requested = True
                                    _await_start = True
                                next_frame_t += dt
                                now = time.perf_counter()
                                if now < next_frame_t:
                                    time.sleep(next_frame_t - now)
                                continue

                            if not _action_passes_safety(
                                action_vec=action_vec,
                                arm_state=arm_state,
                                infer_cfg=infer_cfg,
                                robot=robot,
                                chunk_id=chunk_id,
                            ):
                                _running = False
                                break

                            dispatcher.dispatch(action_vec, state=arm_state)
                            step += 1
                            if tactile_mode and tactile_worker.last_pop_stats is not None:
                                _ps = tactile_worker.last_pop_stats
                                log.info(
                                    "act step=%d off=%d obs_age=%.0fms queue=%.0fms",
                                    step, int(_ps["offset"]),
                                    _ps["obs_age_ms"], _ps["queue_ms"],
                                )
                            action_trace.log_publish_step(
                                step=step,
                                chunk_id=chunk_id,
                                action_idx=0,
                                raw_action=[],
                                canonical_action=getattr(dispatcher, "last_canonical_action_values", action_vec),
                                publish_action=getattr(dispatcher, "last_robot_action_values", action_vec),
                            )

                            if data_logger is not None:
                                data_logger.log_step(step, state_vec, list(action_vec))

                            n_remaining = (
                                tactile_worker.remaining if tactile_mode
                                else action_smoother.remaining
                            )
                            if cam_worker is not None:
                                cam_worker.notify(step, chunk_id, 0, n_remaining + 1)
                                key = cam_worker.poll_key()
                                if key == ord("q"):
                                    _running = False
                                elif key == ord("r"):
                                    _reset_key_pressed = True

                            if _reset_key_pressed:
                                _reset_key_pressed = False
                                if active_worker is not None:
                                    active_worker.pause()
                                log.info("收到 reset（r）: 保存数据 → 回起始位置 → 等待按 Enter 开始")
                                _restart_requested = True
                                _await_start = True

                            next_frame_t += dt
                            now = time.perf_counter()
                            if now < next_frame_t:
                                time.sleep(next_frame_t - now)

                            if step % 30 == 0:
                                actual_dt = (time.perf_counter() - (next_frame_t - dt)) * 1000
                                log.info(
                                    "Step %d: async chunk #%d remaining=%d dt=%.1fms",
                                    step, chunk_id, n_remaining, actual_dt,
                                )
                            continue

                        # ── 感知: 传感器图像 ──
                        images = _read_inference_images(
                            sensors,
                            enabled_cameras,
                            camera_key_map,
                            sensor_read_lock,
                        )

                        # ── 推理 (记录耗时用于跳帧补偿) ──
                        t_req = time.perf_counter()
                        raw_actions = client.predict_chunk(
                            images, state_vec, prompt=prompt,
                        )
                        infer_ms = (time.perf_counter() - t_req) * 1000
                        chunk_id += 1
                        action_trace.log_server_chunk(chunk_id=chunk_id, raw_actions=raw_actions)
                        actions = canonicalize_action_chunk(
                            raw_actions,
                            infer_cfg.action_space,
                            policy_format=infer_cfg.policy_format,
                            canonical_dim=infer_cfg.canonical_dim,
                            effective_action_dim=_effective_cartesian_action_dim(
                                arm_state,
                                infer_cfg,
                            ),
                        )
                        n_total = len(actions)
                        if n_total == 0:
                            log.warning("Chunk #%d 返回空 action，跳过", chunk_id)
                            time.sleep(dt)
                            continue

                        # 可选延迟补偿：跳过推理耗时期间已经过期的 action。
                        if infer_cfg.latency_compensation:
                            n_skip = min(int(infer_ms / 1000.0 * fps), n_total - 1)
                        else:
                            n_skip = 0
                        _log_action_chunk_debug(
                            chunk_id=chunk_id,
                            state_vec=state_vec,
                            raw_actions=raw_actions,
                            actions=actions,
                            n_skip=n_skip,
                            arm_state=arm_state,
                            gripper_width=gripper_width,
                            gripper_max=gripper_max,
                        )
                        if infer_cfg.smooth.enabled:
                            # In synchronous mode we request the next chunk only after executing
                            # n_execute steps. Any leftover tail from the previous chunk is stale
                            # by then and often contains far-future targets; keeping it makes the
                            # next chunk start from the old tail instead of the model's fresh first
                            # action. Async mode updates while the chunk is still being consumed and
                            # uses the normal overlap smoother above.
                            action_smoother.clear()
                            action_smoother.integrate_chunk(
                                actions,
                                latency_steps=n_skip,
                                overlap_steps=infer_cfg.smooth.overlap_steps,
                            )
                            first_action = action_smoother.peek_next()
                            if first_action is None:
                                log.warning("Smooth buffer 为空，跳过 chunk #%d", chunk_id)
                                time.sleep(dt)
                                continue
                            n_exec = min(n_execute, action_smoother.remaining)
                        else:
                            n_exec = min(n_execute, n_total - n_skip)
                            first_action = actions[n_skip]

                        # ── 安全检查：首帧偏差过大则拒绝整个 chunk ──
                        if not _action_passes_safety(
                            action_vec=first_action,
                            arm_state=arm_state,
                            infer_cfg=infer_cfg,
                            robot=robot,
                            chunk_id=chunk_id,
                        ):
                            _running = False
                            break

                        # ── 诊断日志 ──
                        _sv = np.array(state_vec)
                        _a0 = np.array(actions[0])
                        _aL = np.array(actions[-1])
                        _disp = np.abs(_aL - _a0).sum()
                        _cam_info = {k: v.shape for k, v in images.items()}
                        log.info(
                            "Chunk #%d: %d actions, skip %d, 执行 %d, 推理 %.1fms",
                            chunk_id, n_total, n_skip, n_exec, infer_ms,
                        )
                        log.debug("  State[%d]: %s", len(state_vec),
                                  np.array2string(_sv, precision=4, suppress_small=True))
                        log.debug("  Act[0]:   %s", np.array2string(_a0, precision=4, suppress_small=True))
                        log.debug("  Act[-1]:  %s", np.array2string(_aL, precision=4, suppress_small=True))
                        log.debug("  Chunk displacement (L1): %.4f rad", _disp)
                        log.debug("  Cameras: %s", _cam_info)

                        # ── 执行 (精确定时，不被相机/录制阻塞) ──
                        if infer_cfg.smooth.enabled:
                            exec_actions = []
                            for _ in range(n_exec):
                                act = action_smoother.pop_next()
                                if act is None:
                                    break
                                exec_actions.append(act)
                            n_exec = len(exec_actions)
                        else:
                            exec_actions = actions[n_skip:n_skip + n_exec]
                        next_frame_t = time.perf_counter()
                        for i, action_vec in enumerate(exec_actions):
                            if not _running or _restart_requested:
                                break
                            if 0 < max_steps <= step:
                                break

                            # Re-read arm state for every publish. EEF step limiting must be
                            # based on the robot's current pose, matching the old direct-CAN
                            # client; reusing the pre-inference state for an entire chunk makes
                            # every target clamp against a stale pose.
                            try:
                                publish_state = robot.observe()
                            except Exception:
                                log.error("机械臂状态读取失败，立即停止控制循环", exc_info=True)
                                _running = False
                                break

                            dispatcher.dispatch(action_vec, state=publish_state)
                            step += 1
                            raw_idx = min(n_skip + i, len(raw_actions) - 1) if raw_actions else -1
                            action_trace.log_publish_step(
                                step=step,
                                chunk_id=chunk_id,
                                action_idx=raw_idx,
                                raw_action=raw_actions[raw_idx] if raw_idx >= 0 else [],
                                canonical_action=getattr(dispatcher, "last_canonical_action_values", action_vec),
                                publish_action=getattr(dispatcher, "last_robot_action_values", action_vec),
                            )

                            # 记录 state + action
                            if data_logger is not None:
                                data_logger.log_step(step, state_vec, list(action_vec))

                            # 通知后台线程采集相机 (不阻塞)
                            if cam_worker is not None:
                                cam_worker.notify(step, chunk_id, i, n_exec)
                                key = cam_worker.poll_key()
                                if key == ord("q"):
                                    _running = False
                                elif key == ord("r"):
                                    _reset_key_pressed = True

                            # stdin 'r' 或 cv2 'r' 任一触发：保存视频 → 回起始位置 → 等待按 Enter 继续
                            if _reset_key_pressed:
                                _reset_key_pressed = False
                                log.info("收到 reset（r）: 保存数据 → 回起始位置 → 等待按 Enter 开始")
                                _restart_requested = True
                                _await_start = True

                            # 精确帧率控制: 基于绝对时间，不受单帧抖动累积
                            next_frame_t += dt
                            now = time.perf_counter()
                            if now < next_frame_t:
                                time.sleep(next_frame_t - now)

                            if step % 30 == 0:
                                actual_dt = (time.perf_counter() - (next_frame_t - dt)) * 1000
                                log.info(
                                    "Step %d: chunk #%d [%d/%d] dt=%.1fms",
                                    step, chunk_id, i + 1, n_exec, actual_dt,
                                )

                    log.info("控制循环结束: %d 步, %d chunks, %d episodes", step, chunk_id, episode)

                finally:
                    if not dry_run and go_home_first and robot.is_connected():
                        try:
                            log.info("推理结束，自动回 Home 位置...")
                            dispatcher.reset_velocity_tracking()
                            robot.go_home(velocity=infer_cfg.go_home_exit_speed_percent)
                            log.info("自动回 Home 完成")
                        except Exception:
                            log.error("自动回 Home 失败", exc_info=True)
                    if async_worker is not None:
                        async_worker.stop()
                    if tactile_worker is not None:
                        tactile_worker.stop()
                    if cam_worker is not None:
                        cam_worker.stop()
                    if recorder is not None:
                        recorder.stop()
                    if data_logger is not None:
                        data_logger.close()
                    if "action_trace" in locals() and action_trace is not None:
                        action_trace.close()
                        _shutdown_refs["action_trace"] = None
                    client.close()

            finally:
                if sensors is not None:
                    sensors.close_all()
                cv2.destroyAllWindows()

        finally:
            if gripper is not None:
                gripper.disconnect()

    finally:
        robot.disconnect()


def _show_camera_feeds(
    images: dict[str, np.ndarray],
    step: int,
    chunk_id: int,
    action_idx: int,
    n_exec: int,
) -> None:
    """在 OpenCV 窗口中显示相机画面。"""
    if not images:
        return
    vis_frames = []
    for cam_name in sorted(images.keys()):
        frame = images[cam_name].copy()
        cv2.putText(
            frame, cam_name, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.putText(
            frame,
            f"step:{step} chunk:{chunk_id} [{action_idx}/{n_exec}]",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        vis_frames.append(frame)
    if vis_frames:
        canvas = np.hstack(vis_frames)
        cv2.imshow("Robot Cameras", canvas)


# ── 入口 ──


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="配置文件驱动的机器人推理控制",
    )
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--prompt", default="", help="语言指令 (VLA 模型)")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行 (无真实硬件)")
    parser.add_argument("--no-home", action="store_true", help="跳过回 Home")
    parser.add_argument("--show-cameras", action="store_true", help="显示相机画面")
    parser.add_argument("--max-steps", type=int, default=0, help="最大步数 (0=无限)")
    parser.add_argument(
        "--n-execute", type=int, default=None,
        help="每个 chunk 执行的 action 数 (越小越灵敏)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = setup_run_logger(__file__, args.config)
    config_for_prompt = SystemConfig.from_yaml(args.config)
    prompt = args.prompt or ((config_for_prompt.inference.prompt if config_for_prompt.inference else "") or "")
    run_control_loop(
        config_path=args.config,
        prompt=prompt,
        dry_run=args.dry_run,
        go_home_first=not args.no_home,
        show_cameras=args.show_cameras,
        log_dir=log_dir,
        max_steps=args.max_steps,
        n_execute=args.n_execute,
    )


if __name__ == "__main__":
    main()
