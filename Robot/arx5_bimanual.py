from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from Core import Action, ActionSpace, ArmState, RobotParams
from Robot.arx5 import (
    GRIPPER_CLOSED_SLACK_DEFAULT_M,
    _pose7_to_pose6d,
    apply_robot_config_overrides,
    infer_gripper_sdk_sign,
)
from Robot.base import BaseRobot

logger = logging.getLogger(__name__)

_INIT_TIMEOUT_S = 30.0
_RESET_TIMEOUT_S = 30.0
_GET_STATE_TIMEOUT_S = 0.5


@BaseRobot.register("arx5_bimanual")
class Arx5BimanualRobot(BaseRobot):
    """ARX5 双臂驱动，统一暴露 14 维 joint_position:
    [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper]
    """

    def __init__(
        self,
        *,
        left_model: str,
        left_interface: str,
        right_model: str,
        right_interface: str,
        name: str | None = None,
        use_background_send_recv: bool = True,
        log_level: str = "INFO",
        params: RobotParams,
        ctrl_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name or "arx5_bimanual", robot_type="arx5_bimanual", dof=params.dof)
        self._left_model = left_model
        self._left_interface = left_interface
        self._right_model = right_model
        self._right_interface = right_interface
        self._use_background_send_recv = use_background_send_recv
        self._log_level_name = log_level.upper()
        self._params = params
        self._ctrl_cfg = ctrl_cfg or {}

        self._enable_gripper = bool(self._ctrl_cfg.get("enable_gripper", True))
        self._flip_gripper_sign = bool(self._ctrl_cfg.get("flip_gripper_sign", False))
        # (left_kp, right_kp) / (left_kd, right_kd)，右臂可单独覆盖
        self._gripper_kp = (
            float(self._ctrl_cfg.get("gripper_kp", 4.0)),
            float(self._ctrl_cfg.get("right_gripper_kp", self._ctrl_cfg.get("gripper_kp", 4.0))),
        )
        self._gripper_kd = (
            float(self._ctrl_cfg.get("gripper_kd", 0.24)),
            float(self._ctrl_cfg.get("right_gripper_kd", self._ctrl_cfg.get("gripper_kd", 0.24))),
        )

        self._arx5: Any | None = None
        self._ctrls: tuple[Any, Any] | None = None  # (left_ctrl, right_ctrl)
        self._solvers: tuple[Any, Any] | None = None  # (left_solver, right_solver) 用于 CARTESIAN IK
        self._connected = False
        # connect 后按各臂 gripper_open_readout / YAML 覆盖刷新，与 Arx5Robot 一致
        self._gripper_sdk_sign: tuple[int, int] = (1, 1)

    # ── 工厂 ──────────────────────────────────────────────────

    @classmethod
    def _from_config_dict(cls, robot_cfg: dict[str, Any]) -> Arx5BimanualRobot:
        """从 YAML robot: 段创建实例，读取 left_arm / right_arm / control 配置。"""
        left_cfg = robot_cfg.get("left_arm", {})
        right_cfg = robot_cfg.get("right_arm", {})
        control_cfg = robot_cfg.get("control", {})
        merged = dict(control_cfg)
        if "joint_limits" in robot_cfg and "joint_limits" not in merged:
            merged["joint_limits"] = robot_cfg["joint_limits"]
        return cls(
            left_model=left_cfg.get("model", "X5"),
            left_interface=left_cfg.get("interface_name", left_cfg.get("interface", "can0")),
            right_model=right_cfg.get("model", "X5"),
            right_interface=right_cfg.get("interface_name", right_cfg.get("interface", "can1")),
            name=robot_cfg.get("name"),
            use_background_send_recv=bool(control_cfg.get("background_send_recv", True)),
            log_level=control_cfg.get("log_level", "INFO"),
            params=cls._build_params(robot_cfg),
            ctrl_cfg=merged,
        )

    @classmethod
    def _build_params(cls, robot_cfg: dict[str, Any]) -> RobotParams:
        """从 YAML 构建 14 维 RobotParams（connect 后会被 SDK 实际值覆盖）。

        布局: [left_6joints, left_gripper, right_6joints, right_gripper]
        两臂共用同一组关节限位，gripper 范围从 control 段读取。
        """
        ctrl = robot_cfg.get("control", {})
        lim = robot_cfg.get("joint_limits", {})
        min_deg = lim.get("position_min_deg", [-180.0] * 6)
        max_deg = lim.get("position_max_deg", [180.0] * 6)
        vel = [float(v) for v in lim.get("velocity_max", [2.0] * 6)]
        acc = [float(v) for v in lim.get("acceleration_max", [3.0] * 6)]
        tau = [float(v) for v in lim.get("torque_max", [30.0] * 6)]
        jmin = [math.radians(float(v)) for v in min_deg]
        jmax = [math.radians(float(v)) for v in max_deg]
        slack = float(ctrl.get("gripper_closed_slack", GRIPPER_CLOSED_SLACK_DEFAULT_M))
        lg_min = min(float(ctrl.get("left_gripper_min", -slack)), -slack)
        lg_max = float(ctrl.get("left_gripper_max", 0.2))
        rg_min = min(float(ctrl.get("right_gripper_min", -slack)), -slack)
        rg_max = float(ctrl.get("right_gripper_max", 0.2))
        return RobotParams(
            dof=14,
            joint_position_min=jmin + [lg_min] + jmin + [rg_min],
            joint_position_max=jmax + [lg_max] + jmax + [rg_max],
            joint_velocity_max=vel + [1.0] + vel + [1.0],
            joint_acceleration_max=acc + [1.0] + acc + [1.0],
            joint_torque_max=tau + [1.0] + tau + [1.0],
            gripper=None,
            home_position=[math.radians(float(v)) for v in ctrl.get("home_position_deg_14", [0.0] * 14)],
            control_frequency_hz=float(ctrl.get("frequency_hz", 500.0)),
        )

    # ── 生命周期 ──────────────────────────────────────────────

    def connect(self) -> None:
        """延迟导入 arx5_interface，依次创建左右臂控制器并从 SDK 同步参数。"""
        if self._connected:
            return
        if self._left_interface == self._right_interface:
            raise ValueError("left/right arm interface must be different")

        import arx5_interface as arx5

        self._arx5 = arx5
        left_ctrl = self._init_one(self._left_model, self._left_interface, arm_side="left")
        right_ctrl = self._init_one(self._right_model, self._right_interface, arm_side="right")
        self._ctrls = (left_ctrl, right_ctrl)
        self._solvers = self._create_solvers()
        self._set_log_level(self._log_level_name)
        self._sync_params_from_sdk()
        self._apply_gripper_gains()
        self._connected = True
        import atexit
        atexit.register(self.disconnect)
        logger.info(
            "已连接 ARX5 双臂: left=%s@%s right=%s@%s",
            self._left_model, self._left_interface,
            self._right_model, self._right_interface,
        )

    def disconnect(self) -> None:
        """安全断开：先尝试双臂回零（防止松弛坠落），再进入被动态。"""
        if not self._connected:
            return
        self._connected = False
        if self._ctrls is not None:
            # 先尝试回零
            for i, tag in enumerate(("left", "right")):
                try:
                    logger.info("ARX5 %s arm 断开前回零 ...", tag)
                    self._ctrls[i].reset_to_home()
                except Exception:
                    logger.warning("ARX5 %s arm 断开前回零失败", tag, exc_info=True)
            # 再进入阻尼态
            for i, tag in enumerate(("left", "right")):
                try:
                    self._ctrls[i].set_to_damping()
                except Exception:
                    logger.warning("ARX5 %s arm set_to_damping 失败", tag, exc_info=True)
        self._ctrls = None
        self._solvers = None
        self._arx5 = None
        logger.info("已断开 ARX5 双臂: %s", self.name)

    def enable(self) -> None:
        """ARX5 SDK 无显式 enable 接口，连接后即就绪。"""
        self._require_connected()

    def stop(self) -> None:
        """双臂同时进入阻尼态。"""
        self._require_connected()
        for ctrl in self._ctrls:
            ctrl.set_to_damping()

    def emergency_stop(self) -> None:
        """紧急停止：同 stop()，SDK 无更高优先级的停止方式。"""
        self.stop()
        logger.warning("ARX5 双臂紧急停止已执行 (set_to_damping)")

    def clear_fault(self) -> None:
        """SDK 无故障清除接口，尝试 go_home 恢复。"""
        self._require_connected()
        self.go_home()

    def get_params(self) -> RobotParams:
        return self._params

    def is_connected(self) -> bool:
        return self._connected and self._ctrls is not None

    def is_operational(self) -> bool:
        return self.is_connected()

    def is_busy(self) -> bool:
        return False

    def is_fault(self) -> bool:
        return False

    # ── 原语 ──────────────────────────────────────────────────

    def observe(self) -> ArmState:
        """并行读取左右臂关节状态，合并为 14 维向量。

        注意：两臂读取存在微小时间差（约 <2ms），非严格原子快照。
        """
        self._require_connected()
        left_js, right_js = _run_dual(
            lambda: self._ctrls[0].get_joint_state(),
            lambda: self._ctrls[1].get_joint_state(),
            timeout=_GET_STATE_TIMEOUT_S,
            err_msg="get_joint_state timeout: ARX5 arm communication blocked",
        )
        q14 = _merge_arm_field(left_js, right_js, "pos", "gripper_pos")
        dq14 = _merge_arm_field(left_js, right_js, "vel", "gripper_vel")
        tau14 = _merge_arm_field(left_js, right_js, "torque", "gripper_torque")
        ls, rs = self._gripper_sdk_sign
        q14[6] *= ls
        q14[13] *= rs
        dq14[6] *= ls
        dq14[13] *= rs
        tau14[6] *= ls
        tau14[13] *= rs
        return ArmState(
            timestamp=time.perf_counter(),
            joint_positions=q14,
            joint_velocities=dq14,
            joint_torques=tau14,
            joint_external_torques=[],
            joint_positions_desired=[],
            eef_pose=[],
            eef_velocity=[],
            wrench_in_tcp=[],
            wrench_in_world=[],
        )

    def act(self, action: Action, *, state: ArmState | None = None) -> None:
        """根据 ActionSpace 分发：JOINT_POSITION 直接下发，CARTESIAN 通过 IK 解算后下发。"""
        self._require_connected()
        try:
            if action.space == ActionSpace.JOINT_POSITION:
                self._act_joint_position(action.values)
            elif action.space == ActionSpace.CARTESIAN:
                self._act_cartesian(action.values, state=state)
            else:
                logger.warning("ARX5 双臂不支持动作空间 %s，已跳过", action.space.value)
        except Exception:
            logger.error("ARX5 双臂 act() 异常，已跳过本步", exc_info=True)

    def move_joint_position(
        self,
        positions: Sequence[float],
        *,
        velocity: float | None = None,
        tolerance: float = 0.01,
        timeout_s: float = 30.0,
    ) -> bool:
        """smoothstep 插值规划移动到目标 14 维关节位置（含 gripper）。

        以 YAML control.frequency_hz 为插值步进频率，使用 smoothstep (3t²−2t³) 插值，
        起止速度为零，避免阶跃冲击。距离很近时直接下发目标，由 PD 控制收敛。
        duration 由纯关节（排除 gripper index 6/13）的最大位移 / velocity 决定。
        """
        self._require_connected()
        self._validate_length("positions", positions)
        target = np.asarray(positions, dtype=np.float64)
        self._clip_14d(target)

        state = self.observe()
        start = np.asarray(state.joint_positions, dtype=np.float64)

        joint_indices = [i for i in range(14) if i not in (6, 13)]
        max_disp = max(abs(float(target[i] - start[i])) for i in joint_indices)
        if max_disp < tolerance:
            return True

        dt = 1.0 / self._params.control_frequency_hz

        # 距离足够小，直接下发目标让 PD 收敛
        direct_threshold = 0.02  # rad
        if max_disp < direct_threshold:
            hold_steps = max(int(round(0.3 / dt)), 1)
            next_time = time.perf_counter()
            for _ in range(hold_steps):
                self.act(Action(ActionSpace.JOINT_POSITION, target.tolist()))
                next_time += dt
                self._rt_sleep_until(next_time)
            final = self.observe()
            return all(abs(final.joint_positions[i] - positions[i]) < tolerance for i in joint_indices)

        vel = velocity if velocity is not None else 1.0
        duration = max_disp / vel
        steps = max(int(round(duration / dt)), 1)

        diff = target - start
        next_time = time.perf_counter()
        for i in range(1, steps + 1):
            t = i / steps
            alpha = t * t * (3.0 - 2.0 * t)  # smoothstep
            cmd = start + diff * alpha
            self.act(Action(ActionSpace.JOINT_POSITION, cmd.tolist()))
            next_time += dt
            self._rt_sleep_until(next_time)

        final = self.observe()
        return all(abs(final.joint_positions[i] - positions[i]) < tolerance for i in joint_indices)

    def go_home(self, *, velocity: float | None = None, timeout_s: float = 60.0) -> bool:
        """双臂并行 reset_to_home，阻塞直到完成或超时。"""
        self._require_connected()
        _run_dual(
            lambda: self._ctrls[0].reset_to_home(),
            lambda: self._ctrls[1].reset_to_home(),
            timeout=min(_RESET_TIMEOUT_S, timeout_s),
            err_msg="ARX5 bimanual reset_to_home timeout",
        )
        return True

    # ── 动作实现 ──────────────────────────────────────────────

    def _act_joint_position(self, values: list[float]) -> None:
        """下发 14 维关节位置指令。

        接受 12 维（纯关节，gripper 保持当前值）或 14 维（含 gripper）。
        流程: 维度补全 → 限位校验 → gripper flip/clamp → 分发左右臂。
        """
        v = np.asarray(values, dtype=np.float64)
        if v.shape[0] not in (12, 14):
            logger.warning("ARX5 双臂 joint_position 期望 12 或 14 维，实际 %d，已跳过", v.shape[0])
            return

        # 12 维补全：读取当前 gripper 值填入 index 6 和 13
        if v.shape[0] == 12:
            q_now = np.asarray(self.observe().joint_positions, dtype=np.float64)
            q_now[:6] = v[:6]
            q_now[7:13] = v[6:12]
            v = q_now

        self._clip_14d(v)

        # gripper: 上层为规范开合量（张开为正，与 gripper_width 同向）；下发 SDK 需乘各臂 sign
        ls, rs = self._gripper_sdk_sign
        left_g, right_g = float(v[6]), float(v[13])
        if not self._enable_gripper:
            q_now = self.observe().joint_positions
            left_g, right_g = float(q_now[6]), float(q_now[13])
        elif self._flip_gripper_sign:
            left_g, right_g = -left_g, -right_g
        left_g = float(np.clip(left_g, self._params.joint_position_min[6], self._params.joint_position_max[6]))
        right_g = float(np.clip(right_g, self._params.joint_position_min[13], self._params.joint_position_max[13]))
        left_sdk = ls * left_g
        right_sdk = rs * right_g

        left_cmd = self._arx5.JointState(6)
        left_cmd.pos()[:] = v[:6]
        left_cmd.gripper_pos = left_sdk
        self._ctrls[0].set_joint_cmd(left_cmd)

        right_cmd = self._arx5.JointState(6)
        right_cmd.pos()[:] = v[7:13]
        right_cmd.gripper_pos = right_sdk
        self._ctrls[1].set_joint_cmd(right_cmd)

        if not self._use_background_send_recv:
            self._ctrls[0].send_recv_once()
            self._ctrls[1].send_recv_once()

    def _act_cartesian(self, values: Sequence[float], *, state: ArmState | None = None) -> None:
        """笛卡尔空间控制：对左右臂分别 IK 解算后下发关节位置。

        接受 16 维 [left_x,y,z,qw,qx,qy,qz, left_gripper, right_x,y,z,qw,qx,qy,qz, right_gripper]
        或 14 维 [left_pose_7d, right_pose_7d]（gripper 保持当前值）。
        任一臂 IK 解算失败时记录警告并保持当前关节位置，不中断控制循环。
        """
        n = len(values)
        if n not in (14, 16):
            logger.warning("CARTESIAN action 期望 14 或 16 维，实际 %d，已跳过", n)
            return

        obs = state if state is not None else self.observe()
        q14 = obs.joint_positions

        if n == 16:
            left_pose, left_grip = values[:7], float(values[7])
            right_pose, right_grip = values[8:15], float(values[15])
        else:
            left_pose, left_grip = values[:7], float(q14[6])
            right_pose, right_grip = values[7:14], float(q14[13])

        left_6d = _pose7_to_pose6d(left_pose)
        right_6d = _pose7_to_pose6d(right_pose)

        left_q = np.asarray(q14[:6], dtype=np.float64)
        right_q = np.asarray(q14[7:13], dtype=np.float64)

        l_status, l_sol = self._solvers[0].multi_trial_ik(left_6d, left_q, 5)
        r_status, r_sol = self._solvers[1].multi_trial_ik(right_6d, right_q, 5)

        if l_status != 0:
            name = self._solvers[0].get_ik_status_name(l_status)
            logger.warning("左臂 IK 解算失败: %s (status=%d)，保持当前关节位置", name, l_status)
            return
        if r_status != 0:
            name = self._solvers[1].get_ik_status_name(r_status)
            logger.warning("右臂 IK 解算失败: %s (status=%d)，保持当前关节位置", name, r_status)
            return

        target_14 = l_sol.tolist() + [left_grip] + r_sol.tolist() + [right_grip]
        self._act_joint_position(target_14)

    # ── 内部工具 ──────────────────────────────────────────────

    def _create_solvers(self) -> tuple[Any, Any]:
        """从左右臂 RobotConfig 创建 Arx5Solver（用于 CARTESIAN IK 解算）。"""
        solvers = []
        for ctrl in self._ctrls:
            rc = ctrl.get_robot_config()
            solvers.append(self._arx5.Arx5Solver(
                rc.urdf_path, rc.joint_dof,
                rc.joint_pos_min, rc.joint_pos_max,
                rc.base_link_name, rc.eef_link_name,
                rc.gravity_vector,
            ))
        return tuple(solvers)

    def _init_one(self, model: str, interface_name: str, *, arm_side: str | None = None) -> Any:
        """创建单臂 Arx5JointController（带超时保护，防止 CAN 通信阻塞卡死）。"""
        robot_cfg = self._arx5.RobotConfigFactory.get_instance().get_config(model)
        ctrl_cfg = self._arx5.ControllerConfigFactory.get_instance().get_config("joint_controller", robot_cfg.joint_dof)
        ctrl_cfg.background_send_recv = bool(self._use_background_send_recv)
        if "controller_dt" in self._ctrl_cfg:
            ctrl_cfg.controller_dt = float(self._ctrl_cfg["controller_dt"])
        if "over_current_cnt_max" in self._ctrl_cfg:
            ctrl_cfg.over_current_cnt_max = int(self._ctrl_cfg["over_current_cnt_max"])
        elif int(ctrl_cfg.over_current_cnt_max) < 1000:
            # 提高默认阈值，降低因瞬态过流导致的误触发通信阻塞
            ctrl_cfg.over_current_cnt_max = 1000

        apply_robot_config_overrides(robot_cfg, self._ctrl_cfg, arm_side=arm_side)

        result: list[Any] = [None]
        error: list[BaseException | None] = [None]

        def _create() -> None:
            try:
                result[0] = self._arx5.Arx5JointController(robot_cfg, ctrl_cfg, interface_name)
            except BaseException as exc:
                error[0] = exc

        t = threading.Thread(target=_create, daemon=True)
        t.start()
        t.join(timeout=_INIT_TIMEOUT_S)
        if t.is_alive():
            raise RuntimeError(f"Arx5JointController init timed out after {_INIT_TIMEOUT_S}s on {interface_name}")
        if error[0] is not None:
            raise error[0]
        return result[0]

    def _sync_params_from_sdk(self) -> None:
        """用 SDK 返回的实际硬件参数覆盖 YAML 初始值。

        分别读取左右臂的关节限位和 gripper_width，组合为 14 维 RobotParams。
        gripper 规范范围默认约 [-slack, gripper_width]（slack 见 gripper_closed_slack），可通过 left/right_gripper_min 覆盖。
        """
        left_rc = self._ctrls[0].get_robot_config()
        right_rc = self._ctrls[1].get_robot_config()
        left_cc = self._ctrls[0].get_controller_config()
        right_cc = self._ctrls[1].get_controller_config()

        ls = infer_gripper_sdk_sign(left_rc, self._ctrl_cfg, arm="left")
        rs = infer_gripper_sdk_sign(right_rc, self._ctrl_cfg, arm="right")
        self._gripper_sdk_sign = (ls, rs)
        logger.info(
            "ARX5 双臂 gripper readout_sign: left=%d right=%d (默认 1；仅当显式 YAML 时才翻转；open_readout 由 SDK 换算)",
            ls,
            rs,
        )

        def _arm_limits(rc: Any) -> tuple[list[float], list[float], list[float], list[float], float]:
            return (
                np.asarray(rc.joint_pos_min, dtype=float).tolist(),
                np.asarray(rc.joint_pos_max, dtype=float).tolist(),
                np.asarray(rc.joint_vel_max, dtype=float).tolist(),
                np.asarray(rc.joint_torque_max, dtype=float).tolist(),
                abs(float(rc.gripper_width)),
            )

        l_min, l_max, l_vel, l_tau, l_gw = _arm_limits(left_rc)
        r_min, r_max, r_vel, r_tau, r_gw = _arm_limits(right_rc)
        slack = float(self._ctrl_cfg.get("gripper_closed_slack", GRIPPER_CLOSED_SLACK_DEFAULT_M))
        lg_min = min(float(self._ctrl_cfg.get("left_gripper_min", -slack)), -slack)
        lg_max = float(self._ctrl_cfg.get("left_gripper_max", l_gw))
        rg_min = min(float(self._ctrl_cfg.get("right_gripper_min", -slack)), -slack)
        rg_max = float(self._ctrl_cfg.get("right_gripper_max", r_gw))
        dt = max(float(left_cc.controller_dt), float(right_cc.controller_dt))

        self._params = RobotParams(
            dof=14,
            joint_position_min=l_min + [lg_min] + r_min + [rg_min],
            joint_position_max=l_max + [lg_max] + r_max + [rg_max],
            joint_velocity_max=l_vel + [1.0] + r_vel + [1.0],
            joint_acceleration_max=[3.0] * 14,
            joint_torque_max=l_tau + [1.0] + r_tau + [1.0],
            gripper=None,
            home_position=self._params.home_position or [0.0] * 14,
            control_frequency_hz=1.0 / dt,
        )
        self.dof = 14

    def _apply_gripper_gains(self) -> None:
        """为左右臂设置 gripper PD 增益：禁用时清零，启用时补齐 SDK 默认值为 0 的情况。"""
        for i, (kp, kd) in enumerate(zip(self._gripper_kp, self._gripper_kd)):
            gain = self._ctrls[i].get_gain()
            if not self._enable_gripper:
                gain.gripper_kp = 0.0
                gain.gripper_kd = 0.0
            else:
                if gain.gripper_kp <= 1e-6:
                    gain.gripper_kp = kp
                if gain.gripper_kd <= 1e-6:
                    gain.gripper_kd = kd
            self._ctrls[i].set_gain(gain)

    def _set_log_level(self, level_name: str) -> None:
        if self._arx5 is None or self._ctrls is None:
            return
        level = getattr(self._arx5.LogLevel, level_name, self._arx5.LogLevel.INFO)
        for ctrl in self._ctrls:
            ctrl.set_log_level(level)

    def _clip_14d(self, v: np.ndarray) -> None:
        """将 14 维位置裁切到限位范围内（推理输出可能略越界）。"""
        p = self._params
        lo = np.asarray(p.joint_position_min, dtype=np.float64)
        hi = np.asarray(p.joint_position_max, dtype=np.float64)
        mask = (v < lo) | (v > hi)
        if mask.any():
            for i in np.where(mask)[0]:
                side = "left" if int(i) < 7 else "right"
                kind = "gripper" if int(i) in (6, 13) else f"joint_{int(i) % 7}"
                logger.warning(
                    "ARX5 双臂 %s %s 位置 %.4f 超限 [%.4f, %.4f]，已裁切",
                    side, kind, float(v[i]), float(lo[i]), float(hi[i]),
                )
            np.clip(v, lo, hi, out=v)

    def _require_connected(self) -> None:
        if not self.is_connected():
            raise RuntimeError(f"机器人 '{self.name}' 尚未连接，请先调用 connect() 或使用 with 语句")


# ── 模块级工具 ────────────────────────────────────────────────


def _run_dual(fn_left, fn_right, *, timeout: float, err_msg: str) -> tuple[Any, Any]:
    """并行执行左右臂操作，带超时和异常传播。任一臂超时或异常则 raise。"""
    results: list[Any] = [None, None]
    errors: list[BaseException | None] = [None, None]

    def _wrap(idx, fn):
        try:
            results[idx] = fn()
        except BaseException as exc:
            errors[idx] = exc

    tl = threading.Thread(target=_wrap, args=(0, fn_left), daemon=True)
    tr = threading.Thread(target=_wrap, args=(1, fn_right), daemon=True)
    tl.start()
    tr.start()
    tl.join(timeout=timeout)
    tr.join(timeout=timeout)
    if tl.is_alive() or tr.is_alive():
        raise RuntimeError(err_msg)
    for e in errors:
        if e is not None:
            raise e
    return results[0], results[1]


def _merge_arm_field(left_js: Any, right_js: Any, vec_attr: str, gripper_attr: str) -> list[float]:
    """合并左右臂 JointState 的某个字段为 14 维列表。

    vec_attr: JointState 上的方法名（如 "pos"），返回 6 维 ndarray。
    gripper_attr: JointState 上的属性名（如 "gripper_pos"），返回标量。
    """
    left_vec = np.asarray(getattr(left_js, vec_attr)(), dtype=float).tolist()
    right_vec = np.asarray(getattr(right_js, vec_attr)(), dtype=float).tolist()
    return left_vec + [float(getattr(left_js, gripper_attr))] + right_vec + [float(getattr(right_js, gripper_attr))]
