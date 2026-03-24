"""控制飞夕机器人回到原点 (Home)。

用法:
    python flexiv_go_home.py <robot_sn>
    python flexiv_go_home.py Rizon4-123456 --timeout 30 --velocity 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    robot = FlexivRobot(
        args.robot_sn,
        network_interface_whitelist=args.nic,
    )

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

    # 切换到 primitive 模式
    log("[3] 切换到 NRT_PRIMITIVE_EXECUTION 模式 ...")
    robot.switch_mode("primitive_execution")

    # 设置速度（必须在切换模式之后）
    robot.set_velocity_scale(args.velocity)

    log(f"[4] 执行 Home (速度 {args.velocity}%) ...")
    robot.execute_primitive("Home", input_params={})

    # 等待运动完成
    log("[5] 等待回原点完成 ...")
    if not robot.wait_until_done(timeout_s=60.0):
        log("警告: 等待超时 (60s)，机器人可能仍在运动")

    # 打印最终位姿
    state = robot.get_state()
    tcp = state.metadata["tcp_pose"]
    q = state.metadata["q"]
    log("[6] 回原点完成!")
    log(f"    TCP: [{', '.join(f'{v:.4f}' for v in tcp)}]")
    log(f"    关节: [{', '.join(f'{v:.4f}' for v in q)}]")


if __name__ == "__main__":
    main()
