from __future__ import annotations

from SDK.runtime import RobotRuntime, RuntimeStep


class FleetManager:
    def __init__(self) -> None:
        self._runtimes: dict[str, RobotRuntime] = {}

    def register(self, runtime: RobotRuntime) -> None:
        robot_name = runtime.robot.name
        if robot_name in self._runtimes:
            raise ValueError(f"Robot runtime already registered: {robot_name}")
        self._runtimes[robot_name] = runtime

    def step_all(self) -> dict[str, RuntimeStep]:
        return {
            robot_name: runtime.step()
            for robot_name, runtime in self._runtimes.items()
        }

