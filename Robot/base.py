from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Core import RobotParams


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(slots=True)
class GripperState:
    """夹爪当前状态。"""
    width: float = 0.0
    max_width: float = 0.0
    force: float = 0.0
    is_moving: bool = False


class BaseRobot(ABC):
    """机械臂基类。

    职责:
      1. 初始化机器人
      2. 获取所有能获取的数据
      3. 执行所有能执行的控制

    控制接口分两组，每组四个空间:

      阻塞控制 (move_*) — 发送目标，等待到达后返回:
        move_joint_position    关节位置
        move_joint_velocity    关节速度 (运动指定时长)
        move_joint_torque      关节力矩 (施加指定时长)
        move_eef               笛卡尔末端位姿

      流式控制 (send_*) — 非阻塞，需在控制循环中持续调用:
        send_joint_position    关节位置
        send_joint_velocity    关节速度
        send_joint_torque      关节力矩
        send_eef               笛卡尔末端位姿 + 力

    配置文件:
      通过 from_config(path) 加载 YAML，一个文件描述整套系统
      (机器人 + 相机 + 触觉传感器 + 推理参数)。
      机器人只消费 robot: 段，其余段通过 self.config 暴露给上层。
    """

    def __init__(self, name: str, robot_type: str, *, dof: int = 7) -> None:
        self.name = name
        self.robot_type = robot_type
        self.dof = dof
        self.config: dict[str, Any] = {}  # 完整配置，供上层读取

    # ── 从配置文件创建 ────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path) -> BaseRobot:
        """从 YAML 配置文件创建机器人实例。

        YAML 结构:
          robot:     机器人配置 (type, serial_number, gripper, control, ...)
          cameras:   相机配置
          tactile:   触觉传感器配置
          inference: 推理参数

        机器人只消费 robot: 段，完整配置存入 self.config。
        """
        config = _load_yaml(config_path)
        robot_cfg = config.get("robot", {})
        robot_type = robot_cfg.get("type", "")

        # 根据 type 分发到具体子类
        registry = BaseRobot._get_registry()
        factory = registry.get(robot_type)
        if factory is None:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown robot type '{robot_type}'. Available: {available}"
            )

        robot = factory(robot_cfg)
        robot.config = config
        return robot

    @staticmethod
    def _get_registry() -> dict[str, Any]:
        """返回 robot_type -> factory 映射。延迟导入避免循环依赖。"""
        from Robot.flexiv import FlexivRobot
        return {
            "flexiv": FlexivRobot._from_config_dict,
        }

    # ================================================================
    #  1. 初始化 / 生命周期
    # ================================================================

    def connect(self) -> None:
        """连接机器人，子类可覆盖。"""

    def disconnect(self) -> None:
        """断开连接，子类可覆盖。"""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    @abstractmethod
    def enable(self) -> None:
        """使能机器人。"""

    @abstractmethod
    def stop(self) -> None:
        """停止所有运动。"""

    @abstractmethod
    def clear_fault(self) -> None:
        """清除故障。"""

    @abstractmethod
    def switch_mode(self, mode: str) -> None:
        """切换底层控制模式。"""

    @abstractmethod
    def get_params(self) -> RobotParams:
        """返回机器人硬件参数。"""

    # ================================================================
    #  2. 数据读取
    # ================================================================

    # ── 关节空间 ──

    @abstractmethod
    def get_joint_positions(self) -> list[float]:
        """当前各关节位置 (rad)。"""

    @abstractmethod
    def get_joint_velocities(self) -> list[float]:
        """当前各关节速度 (rad/s)。"""

    @abstractmethod
    def get_joint_torques(self) -> list[float]:
        """当前各关节实际力矩 (Nm)。"""

    @abstractmethod
    def get_joint_external_torques(self) -> list[float]:
        """当前各关节外部力矩 (Nm)。"""

    @abstractmethod
    def get_joint_positions_desired(self) -> list[float]:
        """电机侧 (期望) 关节位置 (rad)。"""

    # ── 笛卡尔空间 ──

    @abstractmethod
    def get_eef_pose(self) -> list[float]:
        """末端位姿，格式由子类定义。"""

    @abstractmethod
    def get_eef_velocity(self) -> list[float]:
        """末端速度。"""

    @abstractmethod
    def get_external_wrench_in_tcp(self) -> list[float]:
        """TCP 坐标系下的外力/力矩 [fx,fy,fz,tx,ty,tz]。"""

    @abstractmethod
    def get_external_wrench_in_world(self) -> list[float]:
        """世界坐标系下的外力/力矩 [fx,fy,fz,tx,ty,tz]。"""

    # ── 夹爪 ──

    @abstractmethod
    def get_gripper_state(self) -> GripperState:
        """返回夹爪当前状态。"""

    # ── 状态查询 ──

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def is_operational(self) -> bool: ...

    @abstractmethod
    def is_busy(self) -> bool: ...

    @abstractmethod
    def is_fault(self) -> bool: ...

    # ================================================================
    #  3a. 阻塞控制 (move_*)
    # ================================================================

    @abstractmethod
    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        """移动到目标关节位置，阻塞直到到达。"""

    @abstractmethod
    def move_joint_velocity(
        self,
        velocities: Sequence[float],
        duration: float,
    ) -> None:
        """以指定关节速度运动一段时间，结束后自动停止。"""

    @abstractmethod
    def move_joint_torque(
        self,
        torques: Sequence[float],
        duration: float,
    ) -> None:
        """施加指定关节力矩一段时间，结束后自动停止。"""

    @abstractmethod
    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,
    ) -> None:
        """移动末端到目标位姿，阻塞直到到达。"""

    # ================================================================
    #  3b. 流式控制 (send_*)
    # ================================================================

    @abstractmethod
    def send_joint_position(
        self,
        positions: Sequence[float],
        velocities: Sequence[float] | None = None,
        max_vel: Sequence[float] | None = None,
        max_acc: Sequence[float] | None = None,
    ) -> None:
        """发送关节位置指令（非阻塞）。"""

    @abstractmethod
    def send_joint_velocity(
        self,
        velocities: Sequence[float],
    ) -> None:
        """发送关节速度指令（非阻塞），需持续调用。"""

    @abstractmethod
    def send_joint_torque(
        self,
        torques: Sequence[float],
    ) -> None:
        """发送关节力矩指令（非阻塞），需持续调用。"""

    @abstractmethod
    def send_eef(
        self,
        pose: Sequence[float],
        wrench: Sequence[float] | None = None,
    ) -> None:
        """发送末端位姿 + 力指令（非阻塞），需持续调用。"""

    # ================================================================
    #  3c. 夹爪控制
    # ================================================================

    @abstractmethod
    def gripper_set(self, open: bool) -> None:
        """二值夹爪控制。True=打开, False=关闭。"""

    @abstractmethod
    def gripper_move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
    ) -> None:
        """移动夹爪到指定宽度 (m)。"""

    # ================================================================
    #  辅助
    # ================================================================

    @abstractmethod
    def go_home(self, *, velocity: float | None = None) -> None:
        """回到 Home 位置（阻塞）。"""

    @abstractmethod
    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
        """等待当前运动完成。"""

    @abstractmethod
    def wait_until_operational(self, timeout_s: float = 10.0) -> bool:
        """等待机器人进入 operational 状态。"""
