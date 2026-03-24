from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from Core import InferenceCommand, RobotState


@dataclass(slots=True)
class GripperState:
    """夹爪状态。"""
    width: float = 0.0        # 当前开口宽度 (米)
    max_width: float = 0.0    # 最大开口宽度 (米)
    force: float = 0.0        # 当前夹持力 (N)
    is_moving: bool = False   # 是否正在运动


class BaseRobot(ABC):
    """机械臂基类。

    定义三种运动控制接口:
      1. 关节位置控制  move_joint()
      2. 关节速度控制  move_joint_velocity()
      3. 末端位姿控制  move_eef()

    定义两种夹爪控制接口:
      1. 二值控制 (开/合)  gripper_set()
      2. 宽度控制          gripper_move()
    """

    def __init__(self, name: str, robot_type: str, *, dof: int = 7) -> None:
        self.name = name
        self.robot_type = robot_type
        self.dof = dof

    # ── 生命周期 ──────────────────────────────────────────────

    def connect(self) -> None:
        """连接机器人，子类可覆盖。"""

    def disconnect(self) -> None:
        """断开连接，子类可覆盖。"""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # ── 状态读取 ──────────────────────────────────────────────

    @abstractmethod
    def get_state(self) -> RobotState:
        """读取完整机器人状态。"""
        raise NotImplementedError

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """返回当前各关节位置 (rad)。"""
        raise NotImplementedError

    @abstractmethod
    def get_joint_velocities(self) -> list[float]:
        """返回当前各关节速度 (rad/s)。"""
        raise NotImplementedError

    @abstractmethod
    def get_eef_pose(self) -> list[float]:
        """返回末端位姿 [x, y, z, qw, qx, qy, qz] 或子类约定的格式。"""
        raise NotImplementedError

    @abstractmethod
    def get_gripper_state(self) -> GripperState:
        """返回夹爪状态。"""
        raise NotImplementedError

    # ── 关节位置控制 ──────────────────────────────────────────

    @abstractmethod
    def move_joint(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        blocking: bool = True,
    ) -> None:
        """移动到目标关节位置 (rad)。

        Args:
            positions: 目标关节角度，长度 = dof。
            velocity: 速度缩放 (0~1)，None 表示使用默认值。
            acceleration: 加速度缩放 (0~1)，None 表示使用默认值。
            blocking: 是否阻塞直到到达目标。
        """
        raise NotImplementedError

    # ── 关节速度控制 ──────────────────────────────────────────

    @abstractmethod
    def move_joint_velocity(
        self,
        velocities: Sequence[float],
    ) -> None:
        """发送关节速度指令 (rad/s)。

        需要在实时控制循环中持续调用。
        停止运动请发送全零速度。

        Args:
            velocities: 各关节目标速度，长度 = dof。
        """
        raise NotImplementedError

    # ── 末端位姿控制 ──────────────────────────────────────────

    @abstractmethod
    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,
        blocking: bool = True,
    ) -> None:
        """移动末端到目标位姿。

        Args:
            position: 目标位置 [x, y, z] (米)。
            orientation: 目标姿态，格式由子类定义 (四元数/欧拉角/旋转向量)。
                         None 表示保持当前姿态。
            velocity: 速度缩放 (0~1)，None 表示使用默认值。
            blocking: 是否阻塞直到到达目标。
        """
        raise NotImplementedError

    # ── 夹爪控制: 二值 ────────────────────────────────────────

    @abstractmethod
    def gripper_set(self, open: bool) -> None:
        """二值夹爪控制。

        Args:
            open: True=打开, False=关闭。
        """
        raise NotImplementedError

    # ── 夹爪控制: 宽度 ────────────────────────────────────────

    @abstractmethod
    def gripper_move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
    ) -> None:
        """移动夹爪到指定宽度。

        Args:
            width: 目标宽度 (米)。
            velocity: 运动速度 (米/秒)，None 使用默认值。
            force: 最大夹持力 (N)，None 使用默认值。
        """
        raise NotImplementedError

    # ── 等待 ──────────────────────────────────────────────────

    @abstractmethod
    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
        """等待当前运动完成。返回是否在超时前完成。"""
        raise NotImplementedError

    # ── 便捷方法 ──────────────────────────────────────────────

    def read_state(self) -> RobotState:
        return self.get_state()

    def read_data(self) -> dict[str, Any]:
        """返回可序列化的状态字典，子类可覆盖以添加更多字段。"""
        state = self.get_state()
        return {
            "robot_name": state.robot_name,
            "robot_type": state.robot_type,
            "pose": {
                "x": state.pose.x,
                "y": state.pose.y,
                "z": state.pose.z,
                "yaw": state.pose.yaw,
            },
            "battery_level": state.battery_level,
            "metadata": deepcopy(state.metadata),
        }
