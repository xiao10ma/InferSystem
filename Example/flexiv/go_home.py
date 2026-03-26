"""控制机器人回到原点 (Home)。

用法:
    python Example/flexiv/go_home.py Config/rizon4_example.yaml
    python Example/flexiv/go_home.py Config/rizon4_example.yaml --velocity 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Robot import BaseRobot


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="控制机器人回到原点")
    parser.add_argument("config", help="YAML 配置文件路径")
    parser.add_argument("--velocity", type=int, default=50, help="速度百分比 (1-100)")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
    args = parser.parse_args()

    log("[1] 从配置创建机器人...")
    with BaseRobot.from_config(args.config) as robot:
        if robot.is_fault():
            log("[2] 检测到故障，正在清除...")
            robot.clear_fault()
            time.sleep(2.0)

        if not robot.is_operational():
            log("[2] 使能机器人...")
            robot.enable()
            if not robot.wait_until_operational(timeout_s=args.timeout):
                raise SystemExit("机器人未能在超时时间内变为 operational")
        log(f"[2] 机器人 '{robot.name}' 已 operational")

        log(f"[3] 执行 Home (速度 {args.velocity}%) ...")
        reached = robot.go_home(velocity=args.velocity / 100.0)

        state = robot.observe()
        log(f"[4] 回原点{'完成' if reached else '超时'}!")
        log(f"    TCP: [{', '.join(f'{v:.4f}' for v in state.eef_pose)}]")
        log(f"    关节: [{', '.join(f'{v:.4f}' for v in state.joint_positions)}]")


if __name__ == "__main__":
    main()
