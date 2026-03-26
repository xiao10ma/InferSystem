"""控制飞夕机器人回到原点 (Home)。

用法:
    python Example/flexiv/go_home.py <robot_sn>
    python Example/flexiv/go_home.py Rizon4-123456 --timeout 30 --velocity 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Robot import FlexivRobot


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="控制飞夕机器人回到原点")
    parser.add_argument("robot_sn", help="机器人序列号，例如 Rizon4-123456")
    parser.add_argument("--nic", action="append", default=[], help="网络接口白名单")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
    parser.add_argument("--velocity", type=int, default=50, help="速度百分比 (1-100)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log(f"[1] 连接机器人 {args.robot_sn} ...")
    with FlexivRobot(
        args.robot_sn,
        network_interface_whitelist=args.nic,
    ) as robot:
        # 清除故障 & 使能
        if robot.is_fault():
            log("[2] 检测到故障，正在清除 ...")
            robot.clear_fault()
            time.sleep(2.0)

        if not robot.is_operational():
            log("[2] 使能机器人 ...")
            robot.enable()
            if not robot.wait_until_operational(timeout_s=args.timeout):
                raise SystemExit("机器人未能在超时时间内变为 operational")
        log("[2] 机器人已 operational")

        # 执行 Home (velocity 参数: 百分比转 [0,1] 比例)
        log(f"[3] 执行 Home (速度 {args.velocity}%) ...")
        reached = robot.go_home(velocity=args.velocity / 100.0)

        # 打印最终位姿
        state = robot.observe()
        log(f"[4] 回原点{'完成' if reached else '超时'}!")
        log(f"    TCP: [{', '.join(f'{v:.4f}' for v in state.eef_pose)}]")
        log(f"    关节: [{', '.join(f'{v:.4f}' for v in state.joint_positions)}]")


if __name__ == "__main__":
    main()
