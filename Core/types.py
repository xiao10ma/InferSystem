from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


@dataclass(slots=True)
class SensorFrame:
    sensor_name: str
    sensor_type: str
    timestamp: datetime = field(default_factory=utc_now)
    robot_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RobotState:
    robot_name: str
    robot_type: str
    pose: Pose = field(default_factory=Pose)
    battery_level: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceCommand:
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    target_robot: str | None = None
    priority: int = 0

