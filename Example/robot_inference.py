"""配置文件驱动的机器人推理控制。

从 YAML 配置一键创建 Robot + Sensors + InferenceClient，
运行 感知→推理→执行 控制循环。

用法:
    python Example/robot_inference.py Config/rizon4_example.yaml
    python Example/robot_inference.py Config/rizon4_example.yaml --prompt "pick up the red cup"
    python Example/robot_inference.py Config/rizon4_example.yaml --dry-run
    python Example/robot_inference.py Config/rizon4_example.yaml --no-home --show-cameras
    python Example/robot_inference.py Config/rizon4_example.yaml --save-video
    python Example/robot_inference.py Config/rizon4_example.yaml --save-video --video-dir ./videos

控制循环中的按键:
    q — 退出
    r — 重启推理 (reset policy + 清空缓存 + 回 Home)
"""
from __future__ import annotations

import argparse
import datetime
import logging
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import Observation, load_yaml
from Inference import InferenceClient
from Inference.dispatch import ActionDispatcher, build_state_vector
from Robot import BaseRobot
from Sensor.manager import SensorManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── 优雅退出 ──

_running = True
_restart_requested = False


def _signal_handler(sig, frame):
    global _running
    log.info("收到退出信号，正在停止...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── 视频录制器 ────────────────────────────────────────────────


class VideoRecorder:
    """将多路相机帧水平拼接后写入视频文件。"""

    def __init__(self, video_dir: str, fps: float) -> None:
        self._video_dir = Path(video_dir)
        self._video_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._frame_count = 0

    def start(self) -> Path:
        """开始新的录制 session，返回视频文件路径。"""
        self.stop()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_path = self._video_dir / f"recording_{ts}.mp4"
        self._frame_count = 0
        log.info("开始录制视频: %s", self._video_path)
        return self._video_path

    def write_frame(self, images: dict[str, np.ndarray]) -> None:
        """将多路图像水平拼接后写入一帧。"""
        if not images:
            return
        frames = []
        for cam_name in sorted(images.keys()):
            frame = images[cam_name]
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frames.append(frame)

        # 统一高度后水平拼接
        target_h = frames[0].shape[0]
        resized = []
        for f in frames:
            if f.shape[0] != target_h:
                scale = target_h / f.shape[0]
                new_w = int(f.shape[1] * scale)
                f = cv2.resize(f, (new_w, target_h))
            resized.append(f)
        canvas = np.hstack(resized)

        # 延迟初始化 writer (需要知道帧尺寸)
        if self._writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._video_path), fourcc, self._fps, (w, h),
            )
            if not self._writer.isOpened():
                log.error("无法创建视频文件: %s", self._video_path)
                self._writer = None
                return

        self._writer.write(canvas)
        self._frame_count += 1

    def stop(self) -> None:
        """结束当前录制。"""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info(
                "视频已保存: %s (%d 帧)", self._video_path, self._frame_count,
            )

    @property
    def is_recording(self) -> bool:
        return self._writer is not None or self._video_path is not None and self._frame_count == 0


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
    ) -> None:
        self._sensors = sensors
        self._enabled_cameras = enabled_cameras
        self._recorder = recorder
        self._show = show
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
                pass


# ── 按键处理 ──────────────────────────────────────────────────


def _poll_key() -> int:
    """非阻塞读取按键，返回按键值或 -1。"""
    return cv2.waitKey(1) & 0xFF


# ── 主控制循环 ────────────────────────────────────────────────


def run_control_loop(
    config_path: str,
    *,
    prompt: str = "",
    dry_run: bool = False,
    go_home_first: bool = True,
    show_cameras: bool = False,
    save_video: bool = False,
    video_dir: str = "./videos",
    max_steps: int = 0,
    n_execute: int | None = None,
) -> None:
    """配置驱动的主控制循环: 感知 → 推理 → 执行。"""
    global _running, _restart_requested

    config = load_yaml(config_path)
    infer_cfg = config.get("inference", {})
    fps = infer_cfg.get("fps", 30)
    dt = 1.0 / fps
    if n_execute is None:
        n_execute = infer_cfg.get("n_execute", 100)

    # ── 1. 创建 Robot ──
    log.info("[1/4] 创建机器人...")
    robot = BaseRobot.from_config(config_path)
    robot.connect()

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

            # ── 2. 创建 Sensors ──
            log.info("[2/4] 创建传感器...")
            if not dry_run:
                sensors = SensorManager.from_config(config)
                sensors.open_all()
            else:
                sensors = None

            try:
                # ── 3. 创建 InferenceClient ──
                log.info("[3/4] 连接推理服务器...")
                server_addr = infer_cfg.get("server", "localhost:5555")
                client = InferenceClient(
                    server_addr,
                    jpeg_quality=infer_cfg.get("jpeg_quality", 90),
                    recv_timeout_ms=infer_cfg.get("recv_timeout_ms", 30000),
                )
                client.connect()

                try:
                    # 确定推理使用的相机子集
                    enabled_cameras = infer_cfg.get("enabled_cameras")

                    # 视频录制器
                    recorder = VideoRecorder(video_dir, fps) if save_video else None

                    # 后台相机 worker (录制+显示不阻塞控制帧率)
                    cam_worker: BackgroundCameraWorker | None = None
                    if sensors is not None and (save_video or show_cameras):
                        cam_worker = BackgroundCameraWorker(
                            sensors, enabled_cameras,
                            recorder=recorder,
                            show=show_cameras,
                        )

                    # 初始启动
                    _restart_requested = True  # 首次进入也走 restart 流程

                    while _running:
                        # ── 重启处理 ──
                        if _restart_requested:
                            _restart_requested = False
                            log.info("正在重启推理...")

                            # 停止当前录制
                            if recorder is not None:
                                recorder.stop()

                            # 回 Home
                            if go_home_first:
                                log.info("回 Home 位置...")
                                robot.go_home()

                            # 移动到训练起始位置 (inference_home)
                            inf_home = infer_cfg.get("inference_home")
                            if inf_home is not None:
                                log.info("移动到训练起始位置 (%d 维)...", len(inf_home))
                                # smoothstep 插值到 inference_home (起止速度为零，避免阶跃冲击)
                                cur_state = list(robot.observe().joint_positions)
                                target = [float(v) for v in inf_home]
                                n_interp = int(2.0 * fps)  # 2 秒
                                next_t = time.perf_counter()
                                for _t in range(n_interp):
                                    alpha = (_t + 1) / n_interp
                                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
                                    interp = [c + alpha * (t - c) for c, t in zip(cur_state, target)]
                                    dispatcher.dispatch(interp)
                                    next_t += dt
                                    now = time.perf_counter()
                                    if now < next_t:
                                        time.sleep(next_t - now)
                                log.info("已到达训练起始位置")

                            # Reset policy + 清空缓存
                            client.reset()

                            # 开始新录制
                            if recorder is not None:
                                recorder.start()

                            step = 0
                            chunk_id = 0
                            log.info(
                                "[4/4] 启动控制循环 (%.0f Hz, n_execute=%d)",
                                fps, n_execute,
                            )

                        if 0 < max_steps <= step:
                            log.info("达到最大步数 %d，停止", max_steps)
                            break

                        # ── 感知: 机器人状态 ──
                        arm_state = robot.observe()
                        gripper_width = 0.0
                        gripper_max = 1.0
                        if gripper is not None:
                            try:
                                gs = gripper.observe()
                                gripper_width = gs.width
                                gripper_max = gs.max_width or 1.0
                            except Exception:
                                pass

                        if gripper is not None:
                            state_vec = build_state_vector(
                                arm_state, gripper_width, gripper_max,
                            )
                        else:
                            state_vec = list(arm_state.joint_positions)

                        # ── 感知: 传感器图像 ──
                        if sensors is not None:
                            images = sensors.read_images(enabled_cameras)
                        else:
                            images = {
                                "mock_cam": np.random.randint(
                                    0, 255, (480, 640, 3), dtype=np.uint8,
                                ),
                            }

                        # ── 推理 (记录耗时用于跳帧补偿) ──
                        t_req = time.perf_counter()
                        actions = client.predict_chunk(
                            images, state_vec, prompt=prompt,
                        )
                        infer_ms = (time.perf_counter() - t_req) * 1000
                        chunk_id += 1
                        n_total = len(actions)

                        # 跳过推理耗时期间本该已执行的 action，补偿控制空窗
                        n_skip = min(int(infer_ms / 1000.0 * fps), n_total - 1)
                        n_exec = min(n_execute, n_total - n_skip)

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
                        log.info("  State[14]: %s", np.array2string(_sv, precision=4, suppress_small=True))
                        log.info("  Act[0]:    %s", np.array2string(_a0, precision=4, suppress_small=True))
                        log.info("  Act[-1]:   %s", np.array2string(_aL, precision=4, suppress_small=True))
                        log.info("  Chunk displacement (L1): %.4f rad", _disp)
                        log.info("  Cameras: %s", _cam_info)

                        # ── 执行 (精确定时，不被相机/录制阻塞) ──
                        exec_actions = actions[n_skip:n_skip + n_exec]
                        next_frame_t = time.perf_counter()
                        for i, action_vec in enumerate(exec_actions):
                            if not _running or _restart_requested:
                                break
                            if 0 < max_steps <= step:
                                break

                            dispatcher.dispatch(action_vec)
                            step += 1

                            # 通知后台线程采集相机 (不阻塞)
                            if cam_worker is not None:
                                cam_worker.notify(step, chunk_id, i, n_exec)
                                key = cam_worker.poll_key()
                                if key == ord("q"):
                                    _running = False
                                elif key == ord("r"):
                                    log.info("收到重启请求 (按键 r)")
                                    _restart_requested = True

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

                    log.info("控制循环结束: %d 步, %d chunks", step, chunk_id)

                finally:
                    if cam_worker is not None:
                        cam_worker.stop()
                    if recorder is not None:
                        recorder.stop()
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
    parser.add_argument("--save-video", action="store_true", help="保存视频 (多路相机水平拼接)")
    parser.add_argument("--video-dir", default="./videos", help="视频保存目录 (默认: ./videos)")
    parser.add_argument("--max-steps", type=int, default=0, help="最大步数 (0=无限)")
    parser.add_argument(
        "--n-execute", type=int, default=None,
        help="每个 chunk 执行的 action 数 (越小越灵敏)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_control_loop(
        config_path=args.config,
        prompt=args.prompt,
        dry_run=args.dry_run,
        go_home_first=not args.no_home,
        show_cameras=args.show_cameras,
        save_video=args.save_video,
        video_dir=args.video_dir,
        max_steps=args.max_steps,
        n_execute=args.n_execute,
    )


if __name__ == "__main__":
    main()
