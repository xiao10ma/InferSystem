from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Core import InferenceCommand
from Robot import FlexivRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Flexiv RDK connectivity probe for InferSystem."
    )
    parser.add_argument("robot_sn", help="Robot serial number, e.g. Rizon4-123456")
    parser.add_argument(
        "--nic",
        action="append",
        default=[],
        help="Optional network interface whitelist entry. Repeat to provide more than one.",
    )
    parser.add_argument(
        "--polls",
        type=int,
        default=5,
        help="Number of state snapshots to print.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between state snapshots.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the robot to become operational after enable.",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Call Robot.Enable() before polling state.",
    )
    parser.add_argument(
        "--clear-fault",
        action="store_true",
        help="Call Robot.ClearFault() before enabling or polling.",
    )
    parser.add_argument(
        "--mode",
        help="Optional mode name such as NRT_PRIMITIVE_EXECUTION.",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Create the RDK robot handle in lite mode.",
    )
    parser.add_argument(
        "--quiet-rdk",
        action="store_true",
        help="Disable verbose Flexiv RDK output during initialization.",
    )
    return parser.parse_args()


def snapshot_payload(robot: FlexivRobot) -> dict[str, object]:
    state = robot.get_state()
    metadata = state.metadata
    return {
        "robot_name": state.robot_name,
        "robot_type": state.robot_type,
        "pose": {
            "x": state.pose.x,
            "y": state.pose.y,
            "z": state.pose.z,
        },
        "connected": metadata["connected"],
        "operational": metadata["operational"],
        "fault": metadata["fault"],
        "mode": metadata["mode"],
        "operational_status": metadata["operational_status"],
        "software_version": metadata["software_version"],
        "q": metadata["q"],
        "tcp_pose": metadata["tcp_pose"],
    }


def main() -> None:
    args = parse_args()
    try:
        robot = FlexivRobot(
            args.robot_sn,
            network_interface_whitelist=args.nic,
            verbose=not args.quiet_rdk,
            lite=args.lite,
        )

        if args.clear_fault:
            robot.apply_command(InferenceCommand(action="clear_fault"))
            time.sleep(1.0)

        if args.enable:
            robot.apply_command(InferenceCommand(action="enable"))
            is_operational = robot.wait_until_operational(timeout_s=args.timeout)
            print(json.dumps({"enabled": is_operational}, ensure_ascii=True))

        if args.mode:
            robot.apply_command(
                InferenceCommand(
                    action="switch_mode",
                    parameters={"mode": args.mode},
                )
            )

        for poll_index in range(args.polls):
            print(
                json.dumps(
                    {"poll": poll_index, "state": snapshot_payload(robot)},
                    ensure_ascii=True,
                )
            )
            if poll_index + 1 < args.polls:
                time.sleep(args.interval)
    except Exception as exc:
        raise SystemExit(f"Flexiv probe failed: {exc}") from exc


if __name__ == "__main__":
    main()
