"""飞夕机器人连接探针 — 连接并轮询状态快照。

用法:
    python Example/flexiv/probe.py <robot_sn>
    python Example/flexiv/probe.py Rizon4-123456 --polls 10 --interval 0.2 --enable
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Robot import FlexivRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flexiv 机器人连接探针 — 轮询状态快照",
    )
    parser.add_argument("robot_sn", help="机器人序列号，例如 Rizon4-123456")
    parser.add_argument("--nic", action="append", default=[], help="网络接口白名单")
    parser.add_argument("--polls", type=int, default=5, help="轮询次数")
    parser.add_argument("--interval", type=float, default=0.5, help="轮询间隔 (秒)")
    parser.add_argument("--timeout", type=float, default=10.0, help="等待 operational 超时 (秒)")
    parser.add_argument("--enable", action="store_true", help="连接后自动使能")
    parser.add_argument("--clear-fault", action="store_true", help="连接后自动清除故障")
    parser.add_argument("--lite", action="store_true", help="使用 RDK lite 模式")
    parser.add_argument("--quiet-rdk", action="store_true", help="禁用 RDK 初始化日志")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        with FlexivRobot(
            args.robot_sn,
            network_interface_whitelist=args.nic,
            verbose=not args.quiet_rdk,
            lite=args.lite,
        ) as robot:
            if args.clear_fault and robot.is_fault():
                robot.clear_fault()
                time.sleep(1.0)

            if args.enable:
                robot.enable()
                ok = robot.wait_until_operational(timeout_s=args.timeout)
                print(json.dumps({"enabled": ok}, ensure_ascii=True))

            for poll_index in range(args.polls):
                state = robot.observe()
                payload = {
                    "poll": poll_index,
                    "timestamp": state.timestamp,
                    "connected": robot.is_connected(),
                    "operational": robot.is_operational(),
                    "fault": robot.is_fault(),
                    "joint_positions": state.joint_positions,
                    "eef_pose": state.eef_pose,
                }
                print(json.dumps(payload, ensure_ascii=True))
                if poll_index + 1 < args.polls:
                    time.sleep(args.interval)

    except Exception as exc:
        raise SystemExit(f"Flexiv probe failed: {exc}") from exc


if __name__ == "__main__":
    main()
