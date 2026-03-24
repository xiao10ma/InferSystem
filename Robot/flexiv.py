from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from Core import InferenceCommand, Pose, RobotState
from Robot.base import BaseRobot, GripperState


class FlexivRobot(BaseRobot):
    """飞夕机械臂驱动。

    支持:
      - 关节位置控制  (MoveJ primitive / NRT)
      - 关节速度控制  (SendJointPosition / RT)
      - 末端位姿控制  (MoveL primitive / NRT)
      - 夹爪二值控制  (Grasp / Move)
      - 夹爪宽度控制  (Move)
    """

    # ── RDK 模式映射 ──────────────────────────────────────────
    _MODE_MAP: dict[str, str] = {
        "primitive_execution": "NRT_PRIMITIVE_EXECUTION",
        "plan_execution": "NRT_PLAN_EXECUTION",
        "joint_position": "NRT_JOINT_POSITION",
        "joint_impedance": "NRT_JOINT_IMPEDANCE",
        "cartesian_motion_force": "NRT_CARTESIAN_MOTION_FORCE",
    }

    # 夹爪默认参数
    _DEFAULT_GRIPPER_VELOCITY: float = 0.1   # m/s
    _DEFAULT_GRIPPER_FORCE: float = 30.0     # N
    _DEFAULT_GRIPPER_OPEN_WIDTH: float = 0.09  # m

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

        # 查询 DOF
        info = self._robot.info()
        dof = getattr(info, "DoF", 7) or 7

        super().__init__(
            name=name or serial_number,
            robot_type="flexiv",
            dof=dof,
        )

    # ── 底层句柄 ──────────────────────────────────────────────

    @property
    def native_handle(self) -> Any:
        return self._robot

    @property
    def gripper_handle(self) -> Any:
        """获取夹爪句柄，首次访问时自动创建并初始化。"""
        if self._gripper is None:
            self._gripper = self._rdk.Gripper(self._robot)
            self._gripper.Init()
            time.sleep(2.0)
        return self._gripper

    # ── RDK 状态查询 ──────────────────────────────────────────

    def is_connected(self) -> bool:
        return bool(self._robot.connected())

    def is_operational(self) -> bool:
        return bool(self._robot.operational())

    def is_busy(self) -> bool:
        return bool(self._robot.busy())

    def is_fault(self) -> bool:
        return bool(self._robot.fault())

    def is_stopped(self) -> bool:
        return bool(self._robot.stopped())

    # ── 状态读取 ──────────────────────────────────────────────

    def get_state(self) -> RobotState:
        snap = self._snapshot()
        tcp = snap["tcp_pose"]
        return RobotState(
            robot_name=self.name,
            robot_type=self.robot_type,
            pose=Pose(
                x=_at(tcp, 0),
                y=_at(tcp, 1),
                z=_at(tcp, 2),
            ),
            battery_level=1.0,
            metadata=snap,
        )

    def get_joint_positions(self) -> list[float]:
        return _to_list(self._robot.states().q)

    def get_joint_velocities(self) -> list[float]:
        return _to_list(self._robot.states().dq)

    def get_eef_pose(self) -> list[float]:
        """返回 TCP 位姿 [x, y, z, qw, qx, qy, qz]。"""
        return _to_list(self._robot.states().tcp_pose)

    def get_gripper_state(self) -> GripperState:
        gs = self.gripper_handle.states()
        return GripperState(
            width=gs.width,
            max_width=self._DEFAULT_GRIPPER_OPEN_WIDTH,
            force=gs.force,
            is_moving=gs.is_moving,
        )

    def read_data(self) -> dict[str, Any]:
        snap = self._snapshot()
        return {
            "robot_name": self.name,
            "robot_type": self.robot_type,
            "pose": {
                "x": _at(snap["tcp_pose"], 0),
                "y": _at(snap["tcp_pose"], 1),
                "z": _at(snap["tcp_pose"], 2),
                "yaw": 0.0,
            },
            "status": {
                "connected": snap["connected"],
                "operational": snap["operational"],
                "busy": snap["busy"],
                "fault": snap["fault"],
                "mode": snap["mode"],
                "operational_status": snap["operational_status"],
            },
            "info": {
                "serial_number": snap["serial_number"],
                "model_name": snap["model_name"],
                "software_version": snap["software_version"],
                "license_type": snap["license_type"],
                "dof": snap["dof"],
            },
            "states": {
                "q": snap["q"],
                "dq": snap["dq"],
                "theta": snap["theta"],
                "tau": snap["tau"],
                "tau_ext": snap["tau_ext"],
                "tcp_pose": snap["tcp_pose"],
                "tcp_velocity": snap["tcp_velocity"],
                "ext_wrench_in_tcp": snap["ext_wrench_in_tcp"],
                "ext_wrench_in_world": snap["ext_wrench_in_world"],
                "timestamp": snap["timestamp"],
            },
        }

    def _snapshot(self) -> dict[str, Any]:
        robot = self._robot
        states = robot.states()
        info = robot.info()
        return {
            "serial_number": getattr(info, "serial_num", self.serial_number),
            "model_name": getattr(info, "model_name", "unknown"),
            "software_version": getattr(info, "software_ver", "unknown"),
            "license_type": getattr(info, "license_type", "unknown"),
            "connected": robot.connected(),
            "operational": robot.operational(),
            "busy": robot.busy(),
            "fault": robot.fault(),
            "mode": _enum_name(robot.mode()),
            "operational_status": _enum_name(robot.operational_status()),
            "dof": getattr(info, "DoF", None),
            "q": _to_list(getattr(states, "q", [])),
            "dq": _to_list(getattr(states, "dq", [])),
            "theta": _to_list(getattr(states, "theta", [])),
            "tau": _to_list(getattr(states, "tau", [])),
            "tau_ext": _to_list(getattr(states, "tau_ext", [])),
            "tcp_pose": _to_list(getattr(states, "tcp_pose", [])),
            "tcp_velocity": _to_list(getattr(states, "tcp_vel", [])),
            "ext_wrench_in_tcp": _to_list(getattr(states, "ext_wrench_in_tcp", [])),
            "ext_wrench_in_world": _to_list(getattr(states, "ext_wrench_in_world", [])),
            "timestamp": getattr(states, "timestamp", None),
        }

    # ── 关节位置控制 (MoveJ / NRT) ────────────────────────────

    def move_joint(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        acceleration: float | None = None,
        blocking: bool = True,
    ) -> None:
        """MoveJ 到目标关节位置。

        Args:
            positions: 目标关节角度 (rad)，长度 = dof。
            velocity: 速度缩放 (0~1)，映射到 SetVelocityScale(1~100)。
            acceleration: 加速度缩放，目前未使用（MoveJ primitive 不支持独立加速度参数）。
            blocking: True 则阻塞直到 reachedTarget。
        """
        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")

        if velocity is not None:
            self._robot.SetVelocityScale(max(1, min(100, int(velocity * 100))))

        jpos = self._rdk.JPos(list(positions))
        self._robot.ExecutePrimitive("MoveJ", {"target": jpos}, True)

        if blocking:
            self.wait_until_done()

    # ── 关节速度控制 (RT) ─────────────────────────────────────

    def move_joint_velocity(
        self,
        velocities: Sequence[float],
    ) -> None:
        """发送关节速度指令。

        通过 SendJointPosition 实现：在当前位置上按速度积分一步。
        需要在外部循环中以固定频率 (推荐 1kHz) 持续调用。
        停止运动请发送全零速度。

        注意: 首次调用前需确保已通过 switch_mode('RT_JOINT_POSITION')
        切换到实时关节位置模式。
        """
        vel = [float(v) for v in velocities]
        # 当前位置 + 速度 * dt 作为目标（dt 由 RDK 内部控制）
        q = _to_list(self._robot.states().q)
        dt = 0.001  # RDK RT 周期 1ms
        target = [q[i] + vel[i] * dt for i in range(len(q))]
        self._robot.SendJointPosition(
            target,
            vel,
            [0.0] * len(target),   # acceleration
            [0.0] * len(target),   # feedforward torque
        )

    # ── 末端位姿控制 (MoveL / NRT) ───────────────────────────

    def move_eef(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
        *,
        velocity: float | None = None,
        blocking: bool = True,
    ) -> None:
        """MoveL 直线运动到目标末端位姿。

        Args:
            position: 目标位置 [x, y, z] (米)。
            orientation: 目标姿态 [rx, ry, rz] (rad, 旋转向量)。
                         None 则保持当前姿态。
            velocity: 速度缩放 (0~1)，映射到 SetVelocityScale(1~100)。
            blocking: True 则阻塞直到 reachedTarget。
        """
        self._ensure_mode("NRT_PRIMITIVE_EXECUTION")

        if velocity is not None:
            self._robot.SetVelocityScale(max(1, min(100, int(velocity * 100))))

        # 当前姿态作为默认
        if orientation is None:
            tcp = _to_list(self._robot.states().tcp_pose)
            # tcp_pose 格式: [x, y, z, qw, qx, qy, qz]
            # Coord 需要旋转向量 [rx, ry, rz]，从四元数转换
            orientation = _quat_to_rotvec(tcp[3], tcp[4], tcp[5], tcp[6])

        coord = self._rdk.Coord(
            list(position),         # [x, y, z]
            list(orientation),      # [rx, ry, rz]
            ["world", "world"],     # ref_frame
        )
        self._robot.ExecutePrimitive("MoveL", {"target": coord}, True)

        if blocking:
            self.wait_until_done()

    # ── 夹爪控制: 二值 ────────────────────────────────────────

    def gripper_set(self, open: bool) -> None:
        """二值夹爪控制: True=打开, False=关闭 (夹紧)。"""
        if open:
            self.gripper_handle.Move(
                self._DEFAULT_GRIPPER_OPEN_WIDTH,
                self._DEFAULT_GRIPPER_VELOCITY,
                self._DEFAULT_GRIPPER_FORCE,
            )
        else:
            self.gripper_handle.Grasp(self._DEFAULT_GRIPPER_FORCE)
        self._wait_gripper()

    # ── 夹爪控制: 宽度 ────────────────────────────────────────

    def gripper_move(
        self,
        width: float,
        *,
        velocity: float | None = None,
        force: float | None = None,
    ) -> None:
        """移动夹爪到指定宽度 (米)。"""
        self.gripper_handle.Move(
            width,
            velocity if velocity is not None else self._DEFAULT_GRIPPER_VELOCITY,
            force if force is not None else self._DEFAULT_GRIPPER_FORCE,
        )
        self._wait_gripper()

    def _wait_gripper(self, timeout_s: float = 10.0) -> None:
        time.sleep(0.3)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.gripper_handle.states().is_moving:
                return
            time.sleep(0.05)

    # ── 等待运动完成 ──────────────────────────────────────────

    def wait_until_done(self, timeout_s: float = 30.0) -> bool:
        """等待当前 primitive 执行完成 (reachedTarget)。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ps = self._robot.primitive_states()
            if ps.get("reachedTarget", 0) == 1:
                return True
            time.sleep(0.1)
        return self._robot.primitive_states().get("reachedTarget", 0) == 1

    # ── 生命周期 ──────────────────────────────────────────────

    def wait_until_operational(
        self, timeout_s: float = 10.0, interval_s: float = 0.2
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._robot.operational():
                return True
            time.sleep(interval_s)
        return self._robot.operational()

    def stop(self) -> None:
        self._robot.Stop()

    def clear_fault(self) -> None:
        self._robot.ClearFault()

    def enable(self) -> None:
        self._robot.Enable()

    def run_auto_recovery(self) -> None:
        self._robot.RunAutoRecovery()

    # ── 底层模式控制 ──────────────────────────────────────────

    def switch_mode(self, mode_name: str) -> None:
        """切换 RDK 控制模式。支持友好名称或原始模式名。"""
        rdk_name = self._MODE_MAP.get(mode_name, mode_name)
        try:
            rdk_mode = getattr(self._rdk.Mode, rdk_name)
        except AttributeError as exc:
            available = ", ".join(sorted(self._rdk.Mode.__members__.keys()))
            raise ValueError(
                f"Unknown mode '{mode_name}'. Available: {available}"
            ) from exc
        self._robot.SwitchMode(rdk_mode)

    def set_velocity_scale(self, value: int) -> None:
        self._robot.SetVelocityScale(max(1, min(100, int(value))))

    def _ensure_mode(self, target_mode: str) -> None:
        """确保当前处于目标模式，否则自动切换。"""
        current = _enum_name(self._robot.mode())
        if current != target_mode:
            self.switch_mode(target_mode)

    # ── 底层 primitive / plan 接口 ────────────────────────────

    def execute_primitive(
        self,
        primitive_name: str,
        *,
        input_params: dict[str, Any] | None = None,
        block_until_started: bool = True,
    ) -> None:
        self._robot.ExecutePrimitive(
            primitive_name,
            dict(input_params or {}),
            block_until_started,
        )

    def execute_plan(
        self,
        plan: Any,
        *,
        continue_exec: bool = False,
        block_until_started: bool = True,
    ) -> None:
        self._robot.ExecutePlan(plan, continue_exec, block_until_started)

    # ── apply_command (兼容 SDK/runtime) ──────────────────────

    def apply_command(self, command: InferenceCommand) -> None:
        action = command.action.lower()
        params = dict(command.parameters)

        if action in ("noop", "idle"):
            return

        dispatch: dict[str, Any] = {
            "stop": lambda: self.stop(),
            "clear_fault": lambda: self.clear_fault(),
            "enable": lambda: self.enable(),
            "move_joint": lambda: self.move_joint(
                params["positions"],
                velocity=params.get("velocity"),
                acceleration=params.get("acceleration"),
            ),
            "move_eef": lambda: self.move_eef(
                params["position"],
                orientation=params.get("orientation"),
                velocity=params.get("velocity"),
            ),
            "gripper_set": lambda: self.gripper_set(params["open"]),
            "gripper_move": lambda: self.gripper_move(
                params["width"],
                velocity=params.get("velocity"),
                force=params.get("force"),
            ),
            "switch_mode": lambda: self.switch_mode(str(params["mode"])),
            "set_velocity_scale": lambda: self.set_velocity_scale(int(params["value"])),
            "execute_plan": lambda: self.execute_plan(
                params["plan"],
                continue_exec=params.get("continue_exec", False),
            ),
            "execute_primitive": lambda: self.execute_primitive(
                str(params["name"]),
                input_params=params.get("input_params"),
            ),
        }

        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"Unsupported command: {command.action}")
        handler()


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


def _at(values: Sequence[Any], index: int) -> float:
    return float(values[index]) if len(values) > index else 0.0


def _quat_to_rotvec(qw: float, qx: float, qy: float, qz: float) -> list[float]:
    """四元数 (w, x, y, z) -> 旋转向量 [rx, ry, rz]。"""
    import math
    # 确保 qw >= 0 (取正半球)
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    sin_half = math.sqrt(qx * qx + qy * qy + qz * qz)
    if sin_half < 1e-10:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(sin_half, qw)
    scale = angle / sin_half
    return [qx * scale, qy * scale, qz * scale]
