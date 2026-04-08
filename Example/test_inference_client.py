"""推理客户端测试：连接远程 server，运行 observe → predict → dispatch 回环。

在机器人侧运行，从 YAML 配置创建 Robot + Sensors，
将真实观测发送到远程 server，接收 action chunk 后分发执行。

在机器人机器上运行:
    python Example/test_inference_client.py Config/arx5_example.yaml --server 192.168.50.225:5555
    python Example/test_inference_client.py Config/arx5_bimanual_example.yaml --server 192.168.50.225:5555 --steps 60
    python Example/test_inference_client.py Config/rizon4_example.yaml --dry-run --server 127.0.0.1:5555
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import load_yaml
from Inference import InferenceClient
from Inference.dispatch import ActionDispatcher, build_state_vector
from Robot import BaseRobot
from Sensor.manager import SensorManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="推理客户端测试 (连接远程 server)")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--server", default=None, help="服务器地址 (覆盖 YAML 中的 inference.server)")
    parser.add_argument("--steps", type=int, default=30, help="控制循环步数")
    parser.add_argument("--dry-run", action="store_true", help="无真实硬件 (mock 观测)")
    parser.add_argument("--prompt", default="test prompt", help="语言指令")
    parser.add_argument("--timeout", type=float, default=20.0, help="等待 operational 超时秒数")
    args = parser.parse_args()

    config = load_yaml(args.config)
    infer_cfg = config.get("inference", {})
    fps = infer_cfg.get("fps", 30)
    n_execute = infer_cfg.get("n_execute", 10)
    enabled_cameras = infer_cfg.get("enabled_cameras")
    server_addr = args.server or infer_cfg.get("server", "127.0.0.1:5555")

    dt = 1.0 / fps
    robot = None
    gripper = None
    sensors = None

    # ── 1. Robot + Sensors ──
    if args.dry_run:
        robot_cfg = config.get("robot", {})
        if "bimanual" in robot_cfg.get("type", ""):
            mock_dof = 14
        else:
            ctrl = robot_cfg.get("control", {})
            mock_dof = int(robot_cfg.get("dof", 6))
            if ctrl.get("enable_gripper", False):
                mock_dof += 1
        mock_state = [0.0] * mock_dof
        log.info("[1/3] Dry-run 模式 (mock_dof=%d)", mock_dof)
    else:
        log.info("[1/3] 创建机器人...")
        robot = BaseRobot.from_config(args.config)
        robot.connect()
        if robot.is_fault():
            robot.clear_fault()
            time.sleep(1.5)
        robot.enable()
        if not robot.wait_until_operational(timeout_s=args.timeout):
            raise RuntimeError("机器人未在超时内进入 operational")
        gripper = robot.create_gripper()
        if gripper is not None:
            gripper.connect()
        sensors = SensorManager.from_config(config)
        sensors.open_all()
        log.info("机器人 '%s' 已就绪 (dof=%d)", robot.name, robot.dof)

    # ── 2. Client ──
    log.info("[2/3] 连接推理服务器: %s", server_addr)
    client = InferenceClient(
        server_addr,
        jpeg_quality=infer_cfg.get("jpeg_quality", 90),
        recv_timeout_ms=infer_cfg.get("recv_timeout_ms", 30000),
    )
    client.connect()

    dispatcher = ActionDispatcher.from_config(robot, gripper, config) if robot else None

    # ── 3. 控制循环 ──
    log.info("[3/3] 开始回环测试 (%d steps, %.0f Hz)", args.steps, fps)
    latencies = []

    try:
        step = 0
        while step < args.steps:
            t0 = time.perf_counter()

            # 感知
            if robot is not None:
                arm_state = robot.observe()
                g_width, g_max = 0.0, 1.0
                if gripper is not None:
                    try:
                        gs = gripper.observe()
                        g_width, g_max = gs.width, gs.max_width or 1.0
                    except Exception:
                        pass
                if gripper is not None:
                    state_vec = build_state_vector(arm_state, g_width, g_max)
                else:
                    state_vec = list(arm_state.joint_positions)
                images = sensors.read_images(enabled_cameras) if sensors else {}
            else:
                state_vec = mock_state
                images = {"mock_cam": np.zeros((480, 640, 3), dtype=np.uint8)}

            # 推理
            t_infer = time.perf_counter()
            actions = client.predict_chunk(images, state_vec, prompt=args.prompt)
            infer_ms = (time.perf_counter() - t_infer) * 1000
            latencies.append(infer_ms)

            # 执行
            n_exec = min(n_execute, len(actions))
            for action_vec in actions[:n_exec]:
                if dispatcher is not None:
                    dispatcher.dispatch(action_vec)
                step += 1
                if step >= args.steps:
                    break
                remain = dt - (time.perf_counter() - t0 - (step - 1) * dt)
                if remain > 0:
                    time.sleep(remain)

            loop_ms = (time.perf_counter() - t0) * 1000
            if step % 10 == 0 or step >= args.steps:
                log.info(
                    "step=%d/%d  chunk=%d actions  infer=%.1fms  loop=%.1fms",
                    step, args.steps, len(actions), infer_ms, loop_ms,
                )

    finally:
        client.close()
        if sensors is not None:
            sensors.close_all()
        if gripper is not None:
            gripper.disconnect()
        if robot is not None:
            robot.disconnect()

    # 统计
    if latencies:
        arr = np.array(latencies)
        log.info(
            "推理延迟: mean=%.1fms  std=%.1fms  min=%.1fms  max=%.1fms  (%d chunks)",
            arr.mean(), arr.std(), arr.min(), arr.max(), len(arr),
        )
    log.info("回环测试完成 (%d steps)", step)


if __name__ == "__main__":
    main()
