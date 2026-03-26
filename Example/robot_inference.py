"""配置文件驱动的机器人推理控制。

从 YAML 配置一键创建 Robot + Sensors + InferenceClient，
运行 感知→推理→执行 控制循环。

用法:
    python Example/robot_inference.py Config/rizon4_example.yaml
    python Example/robot_inference.py Config/rizon4_example.yaml --prompt "pick up the red cup"
    python Example/robot_inference.py Config/rizon4_example.yaml --dry-run
    python Example/robot_inference.py Config/rizon4_example.yaml --no-home --show-cameras
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
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


def _signal_handler(sig, frame):
    global _running
    log.info("收到退出信号，正在停止...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── 主控制循环 ────────────────────────────────────────────────


def run_control_loop(
    config_path: str,
    *,
    prompt: str = "",
    dry_run: bool = False,
    go_home_first: bool = True,
    show_cameras: bool = False,
    max_steps: int = 0,
    n_execute: int | None = None,
) -> None:
    """配置驱动的主控制循环: 感知 → 推理 → 执行。"""
    global _running

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

                    # 回 Home
                    if go_home_first:
                        log.info("回 Home 位置...")
                        robot.go_home()

                    # Reset policy
                    client.reset()

                    # ── 4. 控制循环 ──
                    log.info(
                        "[4/4] 启动控制循环 (%.0f Hz, n_execute=%d)",
                        fps, n_execute,
                    )
                    step = 0
                    chunk_id = 0

                    while _running:
                        if 0 < max_steps <= step:
                            log.info("达到最大步数 %d，停止", max_steps)
                            break

                        # 感知: 机器人状态
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

                        state_vec = build_state_vector(
                            arm_state, gripper_width, gripper_max,
                        )

                        # 感知: 传感器图像
                        if sensors is not None:
                            images = sensors.read_images(enabled_cameras)
                        else:
                            images = {
                                "mock_cam": np.random.randint(
                                    0, 255, (480, 640, 3), dtype=np.uint8,
                                ),
                            }

                        # 推理
                        t_req = time.perf_counter()
                        actions = client.predict_chunk(
                            images, state_vec, prompt=prompt,
                        )
                        infer_ms = (time.perf_counter() - t_req) * 1000
                        chunk_id += 1
                        n_total = len(actions)
                        n_exec = min(n_execute, n_total)

                        log.info(
                            "Chunk #%d: %d actions, 执行 %d, 推理 %.1fms",
                            chunk_id, n_total, n_exec, infer_ms,
                        )

                        # 执行
                        for i, action_vec in enumerate(actions[:n_exec]):
                            if not _running:
                                break
                            if 0 < max_steps <= step:
                                break

                            t_start = time.perf_counter()
                            dispatcher.dispatch(action_vec)
                            step += 1

                            # 可视化
                            if show_cameras and sensors is not None and i % 5 == 0:
                                _show_camera_feeds(
                                    sensors.read_images(enabled_cameras),
                                    step, chunk_id, i, n_exec,
                                )

                            # 频率控制
                            elapsed = time.perf_counter() - t_start
                            if elapsed < dt:
                                time.sleep(dt - elapsed)

                            if step % 30 == 0:
                                log.info(
                                    "Step %d: chunk #%d [%d/%d]",
                                    step, chunk_id, i + 1, n_exec,
                                )

                    log.info("控制循环结束: %d 步, %d chunks", step, chunk_id)

                finally:
                    client.close()

            finally:
                if sensors is not None:
                    sensors.close_all()
                if show_cameras:
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
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            global _running
            _running = False


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
    run_control_loop(
        config_path=args.config,
        prompt=args.prompt,
        dry_run=args.dry_run,
        go_home_first=not args.no_home,
        show_cameras=args.show_cameras,
        max_steps=args.max_steps,
        n_execute=args.n_execute,
    )


if __name__ == "__main__":
    main()
