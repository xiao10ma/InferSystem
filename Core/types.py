from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── 基础数据类型 ──────────────────────────────────────────────

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


# ── 机器人参数 ────────────────────────────────────────────────

@dataclass(slots=True)
class GripperParams:
    """夹爪硬件参数。"""
    max_width: float            # 最大开口宽度 (m)
    min_width: float = 0.0      # 最小开口宽度 (m)
    max_velocity: float = 0.0   # 最大运动速度 (m/s)
    max_force: float = 0.0      # 最大夹持力 (N)


@dataclass(slots=True)
class RobotParams:
    """机器人硬件参数 — 描述机器人的物理能力和极限。"""
    dof: int
    joint_position_min: list[float]         # 各关节最小位置 (rad)
    joint_position_max: list[float]         # 各关节最大位置 (rad)
    joint_velocity_max: list[float]         # 各关节最大速度 (rad/s)
    joint_acceleration_max: list[float]     # 各关节最大加速度 (rad/s²)
    joint_torque_max: list[float]           # 各关节最大力矩 (Nm)
    gripper: GripperParams | None = None    # 夹爪参数，无夹爪时为 None
    home_position: list[float] = field(default_factory=list)  # Home 位置 (rad)
    control_frequency_hz: float = 1000.0    # 实时控制频率 (Hz)
