from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from Core import InferenceCommand, RobotState, SensorFrame


class BaseInferenceEngine(ABC):
    @abstractmethod
    def infer(
        self,
        sensor_frames: Iterable[SensorFrame],
        robot_state: RobotState,
    ) -> InferenceCommand:
        raise NotImplementedError

