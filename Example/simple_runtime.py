from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference.rule_engine import RuleBasedInferenceEngine
from Robot.mock_mobile import MockMobileRobot
from SDK.runtime import RobotRuntime
from Sensor.rgb_camera.mock_camera import MockRGBCamera


def main() -> None:
    robot = MockMobileRobot(name="ground_bot")
    sensors = [
        MockRGBCamera(
            robot_name=robot.name,
            distance_sequence=(2.0, 1.6, 0.4, 2.5),
        )
    ]
    engine = RuleBasedInferenceEngine(obstacle_threshold=1.0, step_size=0.4)
    runtime = RobotRuntime(robot=robot, sensors=sensors, engine=engine)

    for step_index in range(4):
        result = runtime.step()
        print(
            f"step={step_index} "
            f"command={result.command.action} "
            f"x={result.state_after.pose.x:.2f} "
            f"battery={result.state_after.battery_level:.2f}"
        )


if __name__ == "__main__":
    main()
