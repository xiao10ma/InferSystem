"""控制飞夕机器人夹爪开合。

用法:
    python flexiv_gripper.py <robot_sn> open
    python flexiv_gripper.py <robot_sn> close
    python flexiv_gripper.py <robot_sn> open close open
    python flexiv_gripper.py <robot_sn> move --width 0.05
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flexivrdk
from Robot import FlexivRobot


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="控制飞夕机器人夹爪")
    parser.add_argument("robot_sn", help="机器人序列号")
    parser.add_argument(
        "actions", nargs="+", choices=["open", "close", "move"],
        help="夹爪动作：open / close / move",
    )
    parser.add_argument("--nic", action="append", default=[], help="网络接口白名单")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待 operational 超时秒数")
    parser.add_argument("--width", type=float, default=0.09, help="open/move 目标宽度 (米)")
    parser.add_argument("--velocity", type=float, default=0.1, help="夹爪速度 (米/秒)")
    parser.add_argument("--force", type=float, default=30.0, help="夹持力 (N)")
    parser.add_argument("--interval", type=float, default=2.0, help="动作间等待 (秒)")
    return parser.parse_args()


def wait_gripper_done(gripper: flexivrdk.Gripper, timeout_s: float = 10.0) -> None:
    time.sleep(0.3)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        states = gripper.states()
        if not states.is_moving:
            return
        time.sleep(0.05)
    log("  警告: 夹爪运动超时")


def print_gripper_state(gripper: flexivrdk.Gripper) -> None:
    states = gripper.states()
    log(f"  宽度: {states.width:.4f} m, 力: {states.force:.2f} N")


def main() -> None:
    args = parse_args()

    log(f"[1] 连接机器人 {args.robot_sn} ...")
    robot = FlexivRobot(args.robot_sn, network_interface_whitelist=args.nic)

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

    log("[3] 初始化夹爪 ...")
    gripper = flexivrdk.Gripper(robot.native_handle)
    gripper.Init()
    time.sleep(2.0)
    print_gripper_state(gripper)

    for i, action in enumerate(args.actions):
        if i > 0:
            time.sleep(args.interval)

        if action == "open":
            log(f"[{4 + i}] 打开夹爪 (宽度={args.width}m) ...")
            gripper.Move(args.width, args.velocity, args.force)
            wait_gripper_done(gripper)
            print_gripper_state(gripper)

        elif action == "close":
            log(f"[{4 + i}] 关闭夹爪 (力={args.force}N) ...")
            gripper.Grasp(args.force)
            wait_gripper_done(gripper)
            print_gripper_state(gripper)

        elif action == "move":
            log(f"[{4 + i}] 移动夹爪到 {args.width}m ...")
            gripper.Move(args.width, args.velocity, args.force)
            wait_gripper_done(gripper)
            print_gripper_state(gripper)

    log("完成!")


if __name__ == "__main__":
    main()
