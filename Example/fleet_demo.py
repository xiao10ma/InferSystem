from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference.rule_engine import RuleBasedInferenceEngine
from Robot.mock_mobile import MockMobileRobot
from SDK.fleet import FleetManager
from SDK.runtime import RobotRuntime
from Sensor.rgb_camera.mock_camera import MockRGBCamera


def build_runtime(name: str, distances: tuple[float, ...]) -> RobotRuntime:
    robot = MockMobileRobot(name=name)
    sensors = [MockRGBCamera(robot_name=name, distance_sequence=distances)]
    engine = RuleBasedInferenceEngine(obstacle_threshold=0.8, step_size=0.3)
    return RobotRuntime(robot=robot, sensors=sensors, engine=engine)


def main() -> None:
    fleet = FleetManager()
    fleet.register(build_runtime("carrier_bot", (2.5, 2.2, 1.8)))
    fleet.register(build_runtime("patrol_bot", (0.6, 0.5, 1.4)))

    for cycle in range(3):
        results = fleet.step_all()
        print(f"cycle={cycle}")
        for robot_name, result in results.items():
            print(
                f"  {robot_name}: command={result.command.action} "
                f"x={result.state_after.pose.x:.2f}"
            )


if __name__ == "__main__":
    main()
