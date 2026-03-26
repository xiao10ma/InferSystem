"""机器人端远程推理客户端 (运行在 feixi_infer 上)

读取 RealSense 相机图像和 Flexiv 机器人状态，
通过 ZMQ 发送给推理服务器，接收动作并执行。

数据格式:
  observation.state: [joint_1..joint_7, gripper] (8-dim, rad)
  action:            [joint_1..joint_7, gripper] (8-dim, rad)
  images: main_cam (480x640 RGB), wrist_left (480x640 RGB)

用法:
    python robot_client.py <robot_sn> --server <4090_ip>:5555 [--fps 30] [--dry-run]

示例:
    python robot_client.py Rizon4-063609 --server 192.168.50.225:5555 --fps 30
    python robot_client.py Rizon4-063609 --server 192.168.50.225:5555 --dry-run  # 仅打印，不控制
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Graceful shutdown
_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Received signal, stopping...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class CameraManager:
    """管理混合类型相机 (RealSense + OpenCV)。"""

    # 相机硬件配置: 与采集系统一致
    CAMERA_DEFS = {
        "main_cam": {"type": "realsense", "serial_number": "409122273675"},
        "wrist_left": {"type": "opencv", "index_or_path": "/dev/v4l/by-path/pci-0000:08:00.3-usb-0:1:1.0-video-index0"},
        "wrist_right": {"type": "opencv", "index_or_path": "/dev/v4l/by-path/pci-0000:08:00.3-usb-0:2:1.0-video-index0"},
        "third_view_left": {"type": "realsense", "serial_number": "349622074137"},
    }

    def __init__(self, cam_names: list[str] | None = None):
        """
        Args:
            cam_names: 要启用的相机名称列表。默认 None 表示只启用推理所需的
                       ["main_cam", "wrist_left"]。
        """
        import pyrealsense2 as rs

        if cam_names is None:
            cam_names = ["main_cam", "wrist_left"]

        self._rs_pipelines: dict[str, rs.pipeline] = {}
        self._cv_captures: dict[str, cv2.VideoCapture] = {}
        self._cam_names = cam_names

        for name in cam_names:
            cfg = self.CAMERA_DEFS.get(name)
            if cfg is None:
                log.warning(f"Unknown camera '{name}', skipping.")
                continue

            if cfg["type"] == "realsense":
                pipeline = rs.pipeline()
                rs_config = rs.config()
                rs_config.enable_device(cfg["serial_number"])
                rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(rs_config)
                self._rs_pipelines[name] = pipeline
                log.info(f"RealSense '{name}' (SN={cfg['serial_number']}) started.")

            elif cfg["type"] == "opencv":
                cap = cv2.VideoCapture(cfg["index_or_path"])
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if not cap.isOpened():
                    log.error(f"OpenCV camera '{name}' ({cfg['index_or_path']}) failed to open!")
                    continue
                self._cv_captures[name] = cap
                log.info(f"OpenCV '{name}' ({cfg['index_or_path']}) started.")

    def read(self) -> dict[str, np.ndarray]:
        """读取所有相机当前帧，返回 {cam_name: BGR numpy array (H,W,3)}。"""
        frames = {}

        # RealSense 相机
        for name, pipeline in self._rs_pipelines.items():
            rs_frames = pipeline.wait_for_frames()
            color_frame = rs_frames.get_color_frame()
            if color_frame:
                frames[name] = np.asanyarray(color_frame.get_data())
            else:
                log.warning(f"RealSense '{name}' returned empty frame.")

        # OpenCV 相机
        for name, cap in self._cv_captures.items():
            ret, frame = cap.read()
            if ret:
                frames[name] = frame
            else:
                log.warning(f"OpenCV '{name}' returned empty frame.")

        return frames

    def close(self):
        for name, pipeline in self._rs_pipelines.items():
            pipeline.stop()
            log.info(f"RealSense '{name}' stopped.")
        for name, cap in self._cv_captures.items():
            cap.release()
            log.info(f"OpenCV '{name}' stopped.")


class MockCameraManager:
    """用于 dry-run 模式的虚拟相机。"""

    def read(self) -> dict[str, np.ndarray]:
        return {
            "main_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "wrist_left": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class FlexivRobotClient:
    """封装 Flexiv 机器人的状态读取和控制。"""

    def __init__(self, robot_sn: str, gripper_name: str = "Flexiv-GN01"):
        import flexivrdk

        self.robot = flexivrdk.Robot(robot_sn)

        # 清除故障 & 使能
        if self.robot.fault():
            log.info("Clearing robot fault...")
            self.robot.ClearFault()
            time.sleep(2.0)

        self.robot.Enable()

        # 等待 operational
        timeout = 30
        t0 = time.time()
        while not self.robot.operational():
            if time.time() - t0 > timeout:
                raise RuntimeError("Robot not operational within timeout")
            time.sleep(0.1)
        log.info(f"Flexiv robot {robot_sn} connected and operational.")

        # 初始化夹爪 (与 replay_parquet.py 一致: Enable + Tool.Switch + Init)
        self._gripper_enabled = False
        self._gripper_max_width = 0.09  # 默认值，Init 后从 params() 更新
        self._gripper_speed = 0.2       # m/s (GN01 max: 0.2)
        self._gripper_force = 15.0      # N
        try:
            self.gripper = flexivrdk.Gripper(self.robot)
            self.gripper.Enable(gripper_name)
            tool = flexivrdk.Tool(self.robot)
            tool.Switch(gripper_name)
            self.gripper.Init()
            time.sleep(3.0)
            gripper_params = self.gripper.params()
            self._gripper_max_width = gripper_params.max_width
            self._gripper_enabled = True
            log.info(f"Gripper '{gripper_name}' ready (max_width={self._gripper_max_width:.4f}m)")
            # 初始打开夹爪
            self.gripper.Move(self._gripper_max_width, self._gripper_speed, self._gripper_force)
            log.info("Opening gripper...")
            time.sleep(1.0)
            log.info(f"Gripper width: {self.gripper.states().width:.4f}m")
        except Exception as e:
            log.warning(f"Gripper init failed (non-fatal): {e}")
            self.gripper = None

        self._last_gripper_open = True  # 初始全开

        # 切换到 NRT_JOINT_POSITION 模式
        self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)
        log.info("Robot mode: NRT_JOINT_POSITION")

    def get_state(self) -> np.ndarray:
        """获取当前状态 [joint_1..joint_7, gripper_width] (8-dim)。"""
        rs = self.robot.states()
        q = list(rs.q)  # 7 个关节角度 (rad)
        # gripper 宽度归一化到 [0, 1]
        gripper_pos = 0.0
        if self._gripper_enabled and self.gripper is not None:
            try:
                gripper_pos = self.gripper.states().width / self._gripper_max_width
            except Exception:
                pass
        return np.array(q + [gripper_pos], dtype=np.float32)

    def apply_action(self, action: list[float]):
        """执行动作 [joint_1..joint_7, gripper]。"""
        joint_pos = action[:7]
        gripper_val = action[7]

        # 发送关节位置
        # API: SendJointPosition(target_pos, target_vel, max_vel, max_acc)
        zero_vel = [0.0] * 7
        max_vel = [2.0] * 7   # rad/s
        max_acc = [3.0] * 7   # rad/s^2
        self.robot.SendJointPosition(joint_pos, zero_vel, max_vel, max_acc)

        # 夹爪二值控制: 阈值 0.6，只有全开/全关
        if self._gripper_enabled and self.gripper is not None:
            try:
                gripper_open = gripper_val >= 0.6
                if gripper_open != self._last_gripper_open:
                    actual_width = self.gripper.states().width
                    if gripper_open:
                        self.gripper.Move(self._gripper_max_width, self._gripper_speed, self._gripper_force)
                    else:
                        self.gripper.Grasp(self._gripper_force)
                    self._last_gripper_open = gripper_open
                    log.info(
                        f"[Gripper] cmd={gripper_val:.3f} -> {'OPEN' if gripper_open else 'CLOSE'} "
                        f"actual={actual_width*1000:.1f}mm"
                    )
            except Exception as e:
                log.warning(f"Gripper move failed: {e}")

    def go_home(self, timeout_s: float = 30.0):
        """回到起始位置 (使用 MoveJ Primitive)。"""
        import flexivrdk
        self.robot.SwitchMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        # 与采集配置一致的起始位置 (degrees)
        start_pos_deg = [0.0, -20.0, 0.0, 90.0, 0.0, 20.0, 0.0]
        start_jpos = flexivrdk.JPos(start_pos_deg)
        self.robot.ExecutePrimitive("MoveJ", {"target": start_jpos, "jntVelScale": 30})
        # 等待 reachedTarget
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ps = self.robot.primitive_states()
            if ps.get("reachedTarget", 0) == 1:
                break
            time.sleep(0.1)
        log.info("Start position reached.")
        self.robot.SwitchMode(flexivrdk.Mode.NRT_JOINT_POSITION)

    def close(self):
        self.robot.Stop()


class MockRobotClient:
    """用于 dry-run 模式的虚拟机器人。"""

    def get_state(self) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    def apply_action(self, action: list[float], max_vel=None):
        pass

    def go_home(self):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# ZMQ Client
# ---------------------------------------------------------------------------

class InferenceClient:
    """ZMQ 推理客户端，与远程推理服务器通信。"""

    def __init__(self, server_addr: str, jpeg_quality: int = 90):
        self.jpeg_quality = jpeg_quality
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30s timeout (首次推理较慢)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.connect(f"tcp://{server_addr}")
        log.info(f"Connected to inference server at {server_addr}")

    def reset_policy(self):
        """通知服务器 reset policy 的 action queue。"""
        data = msgpack.packb({"cmd": "reset"})
        self.socket.send(data)
        resp = msgpack.unpackb(self.socket.recv(), raw=False)
        log.info(f"Policy reset: {resp}")

    def predict_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
    ) -> tuple[list[list[float]], float]:
        """
        发送观测，获取完整 action chunk。

        Args:
            images: {cam_name: BGR numpy (H,W,3)}
            state: (8,) float32 array

        Returns:
            (actions_list: n_action_steps x 8, infer_time_ms)
        """
        # JPEG 压缩图像
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        payload = {"state": state.tolist()}
        for cam_name, img_bgr in images.items():
            ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
            if not ok:
                raise RuntimeError(f"Failed to JPEG-encode {cam_name}")
            payload[cam_name] = buf.tobytes()

        # 发送并接收
        self.socket.send(msgpack.packb(payload))
        resp = msgpack.unpackb(self.socket.recv(), raw=False)

        return resp["actions"], resp.get("infer_time_ms", 0.0)

    def close(self):
        self.socket.close()
        self.context.term()


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run_control_loop(
    robot_sn: str,
    server_addr: str,
    fps: float = 30.0,
    dry_run: bool = False,
    max_steps: int = 0,
    go_home_first: bool = True,
    show_cameras: bool = False,
    n_execute: int = 100,
):
    """主控制循环：采集 -> 推理 -> 执行。"""
    global _running

    dt = 1.0 / fps

    # 初始化
    if dry_run:
        robot = MockRobotClient()
        cameras = MockCameraManager()
    else:
        robot = FlexivRobotClient(robot_sn)
        cameras = CameraManager()  # 默认启用 main_cam + wrist_left

    infer_client = InferenceClient(server_addr)

    # 回 Home
    if go_home_first and not dry_run:
        log.info("Going home first...")
        robot.go_home()

    # Reset policy action queue
    infer_client.reset_policy()

    log.info(f"Starting control loop at {fps} Hz (dt={dt*1000:.1f}ms), max_steps={max_steps or 'inf'}")

    step = 0
    chunk_id = 0
    try:
        while _running:
            if max_steps > 0 and step >= max_steps:
                log.info(f"Reached max_steps={max_steps}, stopping.")
                break

            # ── 采集观测 & 请求推理 (每个 chunk 只做一次) ──
            images = cameras.read()
            state = robot.get_state()

            t_req = time.perf_counter()
            actions, infer_ms = infer_client.predict_chunk(images, state)
            t_req_done = time.perf_counter()
            chunk_id += 1
            n_total = len(actions)
            n_exec = min(n_execute, n_total)

            log.info(
                f"Chunk #{chunk_id}: got {n_total} actions, executing {n_exec}, "
                f"infer={infer_ms:.1f}ms, "
                f"network={((t_req_done - t_req) * 1000 - infer_ms):.1f}ms"
            )

            # ── 逐个执行 chunk 内前 n_exec 个 action ──
            for i, action in enumerate(actions[:n_exec]):
                if not _running:
                    break
                if max_steps > 0 and step >= max_steps:
                    break

                t_start = time.perf_counter()

                # 执行动作
                if not dry_run:
                    robot.apply_action(action)

                # 可视化相机 (执行期间持续刷新画面)
                if show_cameras:
                    # 每隔几步刷新一次相机画面，避免影响控制频率
                    if i % 5 == 0:
                        images = cameras.read()
                    if images:
                        vis_frames = []
                        for cam_name in sorted(images.keys()):
                            frame = images[cam_name].copy()
                            cv2.putText(frame, cam_name, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(frame, f"step:{step} chunk:{chunk_id} [{i}/{n_exec}]",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            vis_frames.append(frame)
                        canvas = np.hstack(vis_frames)
                        cv2.imshow("Robot Cameras", canvas)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            log.info("Quit requested via keyboard.")
                            _running = False

                step += 1

                # 频率控制
                t_elapsed = time.perf_counter() - t_start
                sleep_time = dt - t_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step % 30 == 0:
                    actual_dt = time.perf_counter() - t_start
                    log.info(
                        f"Step {step}: chunk #{chunk_id} [{i+1}/{n_exec}], "
                        f"loop={actual_dt*1000:.1f}ms, "
                        f"action={action[:3]}"
                    )

    except Exception as e:
        log.error(f"Error in control loop: {e}", exc_info=True)
    finally:
        log.info(f"Control loop ended after {step} steps ({chunk_id} chunks).")
        if show_cameras:
            cv2.destroyAllWindows()
        cameras.close()
        infer_client.close()
        if not dry_run:
            robot.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robot Inference Client (ZMQ)")
    parser.add_argument("robot_sn", type=str, help="Flexiv robot serial number")
    parser.add_argument(
        "--server", type=str, default="192.168.50.225:5555",
        help="Inference server address (ip:port)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Control frequency")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no real robot/camera)")
    parser.add_argument("--max-steps", type=int, default=0, help="Max steps (0=infinite)")
    parser.add_argument("--no-home", action="store_true", help="Skip go-home on start")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG compression quality")
    parser.add_argument("--show-cameras", action="store_true", help="Show camera feeds in OpenCV window")
    parser.add_argument(
        "--n-execute", type=int, default=100,
        help="每个 chunk 执行的 action 数量 (1~100)。越小越灵敏，越大越高效。",
    )
    args = parser.parse_args()

    run_control_loop(
        robot_sn=args.robot_sn,
        server_addr=args.server,
        fps=args.fps,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        go_home_first=not args.no_home,
        show_cameras=args.show_cameras,
        n_execute=args.n_execute,
    )


if __name__ == "__main__":
    main()
