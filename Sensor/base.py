from __future__ import annotations

from abc import ABC, abstractmethod

from Core import SensorFrame


class BaseSensor(ABC):
    def __init__(self, name: str, sensor_type: str, robot_name: str | None = None) -> None:
        self.name = name
        self.sensor_type = sensor_type
        self.robot_name = robot_name

    @abstractmethod
    def read(self) -> SensorFrame:
        raise NotImplementedError

