from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

from Core import GripperParams, RobotParams
from Robot.base import BaseRobot, GripperState


class FlexivRobot(BaseRobot):
    """飞夕机械臂驱动 (Rizon 系列)。

    阻塞控制 (move_*):
      - move_joint_position   → MoveJ primitive
      - move_joint_velocity   → RT 关节位置循环 (速度积分)
      - move_joint_torque     → RT 关节力矩循环
      - move_eef              → MoveL primitive

    流式控制 (send_*):
      - send_joint_position   → SendJointPosition (NRT)
      - send_joint_velocity   → SendJointPosition (RT, 速度积分)
      - send_joint_torque     → SendJointTorque (RT)
      - send_eef              → SendCartesianMotionForce (RT)
    """

    # ── Rizon4 默认硬件参数 ───────────────────────────────────

    _DEFAULT_PARAMS = RobotParams(
        dof=7,
        joint_position_min=[-2.7925, -2.2689, -2.9671, -1.8675, -2.9671, -1.3963, -2.9671],
        joint_position_max=[+2.7925, +2.2689, +2.9671, +2.6878, +2.9671, +4.5379, +2.9671],
        joint_velocity_max=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        joint_acceleration_max=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        joint_torque_max=[100.0, 100.0, 50.0, 50.0, 20.0, 20.0, 20.0],
        gripper=GripperParams(
            max_width=0.1,
            min_width=0.0,
            max_velocity=0.2,
            max_force=30.0,
        ),
        home_position=[0.0, -0.3491, 0.0, 1.5708, 0.0, 0.3491, 0.0],
        control_frequency_hz=1000.0,
    )

    # ── RDK 模式映射 ─────────────────────────────────────────

    _MODE_MAP: dict[str, str] = {
        "primitive_execution": "NRT_PRIMITIVE_EXECUTION",
        "plan_execution": "NRT_PLAN_EXECUTION",
        "joint_position": "NRT_JOINT_POSITION",
        "joint_impedance": "NRT_JOINT_IMPEDANCE",
        "cartesian_motion_force": "NRT_CARTESIAN_MOTION_FORCE",
        "rt_joint_position": "RT_JOINT_POSITION",
        "rt_joint_torque": "RT_JOINT_TORQUE",
        "rt_cartesian_motion_force": "RT_CARTESIAN_MOTION_FORCE",
    }

    # ================================================================
    #  1. 初始化
    # ================================================================

    def __init__(
        self,
        serial_number: str,
        *,
        name: str | None = None,
        network_interface_whitelist: Sequence[str] | None = None,
        verbose: bool = True,
        lite: bool = False,
        robot_handle: Any | None = None,
        rdk_module: Any | None = None,
        params: RobotParams | None = None,
        gripper_config: dict[str, Any] | None = None,
        control_config: dict[str, Any] | None = None,
    ) -> None:
        if rdk_module is None:
            import flexivrdk as rdk_module  # type: ignore[no-redef]

        self.serial_number = serial_number
        self._rdk = rdk_module
        self._robot = (
            robot_handle
            if robot_handle is not None
            else self._rdk.Robot(
                serial_number,
                list(network_interface_whitelist or []),
                verbose,
                lite,
            )
        )
        self._gripper: Any | None = None
        self._gripper_inited: bool = False

        # 参数: 外部传入 > 默认值
        self._params = params if params is not None else self._DEFAULT_PARAMS

        # 夹爪配置 (来自 YAML)
        self._gripper_config = gripper_config or {}
        self._gripper_name: str = self._gripper_config.get("name", "")
        self._default_gripper_velocity: float = self._gripper_config.get(
            "default_velocity", 0.1,
        )
        self._default_gripper_force: float = self._gripper_config.get(
            "default_force", 30.0,
        )

        # 控制配置 (来自 YAML)
        self._control_config = control_config or {}

        # 查询 DOF
        info = self._robot.info()
        dof = getattr(info, "DoF", 7) or 7

        super().__init__(
            name=name or serial_number,
            robot_type="flexiv",
            dof=dof,
        )

    # ── 从 config dict 创建 (供 BaseRobot.from_config 调用) ──

    @classmethod
    def _from_config_dict(cls, robot_cfg: dict[str, Any]) -> FlexivRobot:
        """从 YAML robot: 段创建 FlexivRobot。"""
        gripper_cfg = robot_cfg.get("gripper", {})
        control_cfg = robot_cfg.get("control", {})

        # 从 YAML 构建 RobotParams (如果有 joint_limits 段则覆盖默认)
        params = cls._build_params(robot_cfg)

        return cls(
            serial_number=robot_cfg["serial_number"],
            name=robot_cfg.get("name"),
            network_interface_whitelist=robot_cfg.get("network_interface_whitelist"),
            verbose=robot_cfg.get("verbose", True),
            lite=robot_cfg.get("lite", False),
            params=params,
            gripper_config=gripper_cfg,
            control_config=control_cfg,
        )

    @classmethod
    def _build_params(cls, robot_cfg: dict[str, Any]) -> RobotParams:
        """从 YAML 构建 RobotParams，缺省字段使用默认值。"""
        default = cls._DEFAULT_PARAMS
        gripper_cfg = robot_cfg.get("gripper", {})
        control_cfg = robot_cfg.get("control", {})
        limits_cfg = robot_cfg.get("joint_limits", {})

        # 夹爪参数
        gripper = GripperParams(
            max_width=gripper_cfg.get("max_width", default.gripper.max_width),
            min_width=gripper_cfg.get("min_width", default.gripper.min_width),
            max_velocity=gripper_cfg.get("max_velocity", default.gripper.max_velocity),
            max_force=gripper_cfg.get("max_force", default.gripper.max_force),
        ) if default.gripper else None

        # Home 位置: YAML 用度，内部存弧度
        home_deg = control_cfg.get("home_position_deg")
        if home_deg is not None:
            home_rad = [math.radians(d) for d in home_deg]
        else:
            home_rad = list(default.home_position)

        # 关节极限: YAML 可选覆盖
        pos_min_deg = limits_cfg.get("position_min_deg")
        pos_max_deg = limits_cfg.get("position_max_deg")

        return RobotParams(
            dof=default.dof,
            joint_position_min=(
                [math.radians(d) for d in pos_min_deg]
                if pos_min_deg else list(default.joint_position_min)
            ),
            joint_position_max=(
                [math.radians(d) for d in pos_max_deg]
                if pos_max_deg else list(default.joint_position_max)
            ),
            joint_velocity_max=limits_cfg.get(
                "velocity_max", list(default.joint_velocity_max),
            ),
            joint_acceleration_max=limits_cfg.get(
                "acceleration_max", list(default.joint_acceleration_max),
            ),
            joint_torque_max=limits_cfg.get(
                "torque_max", list(default.joint_torque_max),
            ),
            gripper=gripper,
            home_position=home_rad,
            control_frequency_hz=control_cfg.get(
                "frequency_hz", default.control_frequency_hz,
            ),
        )

    # ── 底层句柄 ──────────────────────────────────────────────

    @property
    def native_handle(self) -> Any:
        return self._robot

    @property
    def gripper_handle(self) -> Any:
        if not self._gripper_inited:
            self._gripper = self._rdk.Gripper(self._robot)
            if self._gripper_name:
                self._gripper.Enable(self._gripper_name)
                tool = self._rdk.Tool(self._robot)
                tool.Switch(self._gripper_name)
            self._gripper.Init()
            time.sleep(2.0)
            self._gripper_inited = True
        return self._gripper

    # ── 生命周期 ──────────────────────────────────────────────

    def enable(self) -> None:
        self._robot.Enable()

    def stop(self) -> None:
        self._robot.Stop()

    def clear_fault(self) -> None:
        self._robot.ClearFault()

    def switch_mode(self, mode_name: str) -> None:
        rdk_name = self._MODE_MAP.get(mode_name, mode_name)
        try:
            rdk_mode = getattr(self._rdk.Mode, rdk_name)
        except AttributeError as exc:
            available = ", ".join(sorted(self._rdk.Mode.__members__.keys()))
            raise ValueError(
                f"Unknown mode '{mode_name}'. Available: {available}"
            ) from exc
        self._robot.SwitchMode(rdk_mode)

    def get_params(self) -> RobotParams:
        return self._params

    # ================================================================
    #  2. 数据读取
    # ================================================================

    def get_joint_positions(self) -> list[float]:
        return _to_list(self._robot.states().q)

    def get_joint_velocities(self) -> list[float]:
        return _to_list(self._robot.states().dq)

    def get_joint_torques(self) -> list[float]:
        return _to_list(self._robot.states().tau)

    def get_joint_external_torques(self) -> list[float]:
        return _to_list(self._robot.states().tau_ext)

    def get_joint_positions_desired(self) -> list[float]:
        return _to_list(self._robot.states().theta)

    def get_eef_pose(self) -> list[float]:
        """TCP 位姿 [x, y, z, qw, qx, qy, qz]。"""
        return _to_list(self._robot.states().tcp_pose)

    def get_eef_velocity(self) -> list[float]:
        return _to_list(self._robot.states().tcp_vel)

    def get_external_wrench_in_tcp(self) -> list[float]:
        return _to_list(self._robot.states().ext_wrench_in_tcp)

    def get_external_wrench_in_world(self) -> list[float]:
        return _to_list(self._robot.states().ext_wrench_in_world)

    def get_gripper_state(self) -> GripperState:
        gs = self.gripper_handle.states()
        return GripperState(
            width=gs.width,
            max_width=self._params.gripper.max_width if self._params.gripper else 0.0,
            force=gs.force,
            is_moving=gs.is_moving,
        )

    def is_connected(self) -> bool:
        return bool(self._robot.connected())

    def is_operational(self) -> bool:
        return bool(self._robot.operational())

    def is_busy(self) -> bool:
        return bool(self._robot.busy())

    def is_fault(self) -> bool:
        return bool(self._robot.fault())

    # ================================================================
    #  3a. 阻塞控制 (move_*)
    # ================================================================

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
    ) -> None:
        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")
        if velocity is not None:
            self._robot.SetVelocityScale(_to_scale(velocity))
        jpos = self._rdk.JPos(list(positions))
        self._robot.ExecutePrimitive("MoveJ", {"target": jpos}, True)
        self.wait_until_done()

    def move_joint_velocity(
        self,
        velocities: Sequence[float],
        duration: float,
    ) -> None:
        self._ensure_mode("RT_JOINT_POSITION")
        dt = 1.0 / self._params.control_frequency_hz
        steps = int(duration / dt)
        vel = [float(v) for v in velocities]
        for _ in range(steps):
            q = _to_list(self._robot.states().q)
            target = [q[i] + vel[i] * dt for i in range(self.dof)]
            self._robot.SendJointPosition(
                target, vel,
                [0.0] * self.dof, [0.0] * self.dof,
            )
            time.sleep(dt)
        # 停止
        q = _to_list(self._robot.states().q)
        zeros = [0.0] * self.dof
        self._robot.SendJointPosition(q, zeros, zeros, zeros)

    def move_joint_torque(
        self,
        torques: Sequence[float],
        duration: float,
    ) -> None:
        self._ensure_mode("RT_JOINT_TORQUE")
        dt = 1.0 / self._params.control_frequency_hz
        steps = int(duration / dt)
        tau = [float(t) for t in torques]
        for _ in range(steps):
            self._robot.SendJointTorque(tau, False, [0.0] * self.dof)
            time.sleep(dt)
        self._robot.SendJointTorque([0.0] * self.dof, False, [0.0] * self.dof)

    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,
    ) -> None:
        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")
        if velocity is not None:
            self._robot.SetVelocityScale(_to_scale(velocity))
        if orientation is None:
            tcp = _to_list(self._robot.states().tcp_pose)
            orientation = _quat_to_rotvec(tcp[3], tcp[4], tcp[5], tcp[6])
        coord = self._rdk.Coord(
            list(position), list(orientation), ["world", "world"],
        )
        self._robot.ExecutePrimitive("MoveL", {"target": coord}, True)
        self.wait_until_done()

    # ================================================================
    #  3b. 流式控制 (send_*)
    # ================================================================

    def send_joint_position(
        self,
        positions: Sequence[float],
        velocities: Sequence[float] | None = None,
        max_vel: Sequence[float] | None = None,
        max_acc: Sequence[float] | None = None,
    ) -> None:
        p = self._params
        vel = list(velocities) if velocities is not None else [0.0] * self.dof
        mv = list(max_vel) if max_vel is not None else list(p.joint_velocity_max)
        ma = list(max_acc) if max_acc is not None else list(p.joint_acceleration_max)
        self._robot.SendJointPosition(list(positions), vel, mv, ma)

    def send_joint_velocity(
        self,
        velocities: Sequence[float],
    ) -> None:
        dt = 1.0 / self._params.control_frequency_hz
        vel = [float(v) for v in velocities]
        q = _to_list(self._robot.states().q)
        target = [q[i] + vel[i] * dt for i in range(self.dof)]
        self._robot.SendJointPosition(
            target, vel,
            [0.0] * self.dof, [0.0] * self.dof,
        )

    def send_joint_torque(
        self,
        torques: Sequence[float],
    ) -> None:
        self._robot.SendJointTorque(
            [float(t) for t in torques], False, [0.0] * self.dof,
        )

    def send_eef(
        self,
        pose: Sequence[float],
        wrench: Sequence[float] | None = None,
    ) -> None:
        w = list(wrench) if wrench is not None else [0.0] * 6
        self._robot.SendCartesianMotionForce(list(pose), w)

    # ================================================================
    #  3c. 夹爪控制
    # ================================================================

    def gripper_set(self, open: bool) -> None:
        gp = self._params.gripper
        if open:
            self.gripper_handle.Move(
                gp.max_width if gp else 0.09,
                self._default_gripper_velocity,
                self._default_gripper_force,
            )
        else:
            self.gripper_handle.Grasp(self._default_gripper_force)
        self._wait_gripper()

    def gripper_move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
    ) -> None:
        self.gripper_handle.Move(
            width,
            velocity if velocity is not None else self._default_gripper_velocity,
            force if force is not None else self._default_gripper_force,
        )
        self._wait_gripper()

    # ================================================================
    #  辅助
    # ================================================================

    def go_home(self, *, velocity: float | None = None) -> None:
        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")
        vel_scale = self._control_config.get("home_velocity_scale", 50)
        if velocity is not None:
            vel_scale = _to_scale(velocity)
        self._robot.SetVelocityScale(vel_scale)
        self._robot.ExecutePrimitive("Home", {}, True)
        self.wait_until_done(timeout_s=60.0)

    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ps = self._robot.primitive_states()
            if ps.get("reachedTarget", 0) == 1:
                return True
            time.sleep(0.1)
        return self._robot.primitive_states().get("reachedTarget", 0) == 1

    def wait_until_operational(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._robot.operational():
                return True
            time.sleep(0.2)
        return self._robot.operational()

    # ── 内部工具 ──────────────────────────────────────────────

    def _ensure_mode(self, target_mode: str) -> None:
        current = _enum_name(self._robot.mode())
        if current != target_mode:
            self.switch_mode(target_mode)

    def _wait_gripper(self, timeout_s: float = 10.0) -> None:
        time.sleep(0.3)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.gripper_handle.states().is_moving:
                return
            time.sleep(0.05)


# ── 模块级工具函数 ────────────────────────────────────────────

def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def _to_scale(value: float) -> int:
    return max(1, min(100, int(value * 100)))


def _quat_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> list[float]:
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, qw)
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]
