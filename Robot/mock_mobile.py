from __future__ import annotations

from Core import InferenceCommand, Pose, RobotState
from Robot.base import BaseRobot


class MockMobileRobot(BaseRobot):
    def __init__(self, name: str, robot_type: str = "mobile") -> None:
        super().__init__(name=name, robot_type=robot_type)
        self._state = RobotState(
            robot_name=name,
            robot_type=robot_type,
            pose=Pose(),
            battery_level=1.0,
            metadata={"last_action": "idle"},
        )

    def get_state(self) -> RobotState:
        return self._state

    def apply_command(self, command: InferenceCommand) -> None:
        action = command.action
        self._state.metadata["last_action"] = action

        if action == "move":
            self._state.pose.x += float(command.parameters.get("dx", 0.0))
            self._state.pose.y += float(command.parameters.get("dy", 0.0))
        elif action == "rotate":
            self._state.pose.yaw += float(command.parameters.get("yaw", 0.0))
        elif action == "stop":
            self._state.metadata["stop_reason"] = command.parameters.get("reason", "unknown")

        self._state.battery_level = max(self._state.battery_level - 0.01, 0.0)

