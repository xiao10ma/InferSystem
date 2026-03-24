from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from Core import InferenceCommand, RobotState, SensorFrame
from Inference.base import BaseInferenceEngine
from Robot.base import BaseRobot
from Sensor.base import BaseSensor


@dataclass(slots=True)
class RuntimeStep:
    robot_name: str
    frames: list[SensorFrame]
    state_before: RobotState
    command: InferenceCommand
    state_after: RobotState


class RobotRuntime:
    def __init__(
        self,
        robot: BaseRobot,
        sensors: Sequence[BaseSensor],
        engine: BaseInferenceEngine,
    ) -> None:
        self.robot = robot
        self.sensors = list(sensors)
        self.engine = engine

    def step(self) -> RuntimeStep:
        frames = [sensor.read() for sensor in self.sensors]
        state_before = deepcopy(self.robot.get_state())
        command = self.engine.infer(frames, state_before)
        self.robot.apply_command(command)
        state_after = deepcopy(self.robot.get_state())
        return RuntimeStep(
            robot_name=self.robot.name,
            frames=frames,
            state_before=state_before,
            command=command,
            state_after=state_after,
        )

