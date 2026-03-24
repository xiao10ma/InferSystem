from __future__ import annotations

from abc import ABC, abstractmethod

from Core import InferenceCommand, RobotState


class BaseRobot(ABC):
    def __init__(self, name: str, robot_type: str) -> None:
        self.name = name
        self.robot_type = robot_type

    @abstractmethod
    def get_state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def apply_command(self, command: InferenceCommand) -> None:
        raise NotImplementedError

