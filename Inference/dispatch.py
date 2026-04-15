"""推理输出 → 机器人控制的桥接层。

推理模型输出原始向量 (如 [j1..j7, gripper])，
ActionDispatcher 按配置拆分并分发给机器人手臂和夹爪。

用法::

    from Inference.dispatch import ActionDispatcher, build_state_vector

    dispatcher = ActionDispatcher.from_config(robot, gripper, config)

    # 控制循环
    for action_vec in actions:
        dispatcher.dispatch(action_vec)

    # 构建推理输入状态
    state = build_state_vector(arm_state, gripper_width, gripper_max_width)
"""
from __future__ import annotations

import logging
from typing import Any

from Core import Action, ActionSpace, ArmState
from Robot.base import BaseRobot
from Robot.gripper import BaseGripper

logger = logging.getLogger(__name__)


class ActionDispatcher:
    """将推理输出的原始向量分发给机器人和夹爪。

    推理模型输出的 action 通常是一个扁平向量，如:
      [j1, j2, j3, j4, j5, j6, j7, gripper]  (8 维)

    ActionDispatcher 根据配置把它拆分并分发:
      - 前 arm_dim 个维度 → robot.act(Action)
      - gripper_index 位置的维度 → gripper 控制

    配置 (YAML inference 段)::

        inference:
          arm_dim: 7             # 手臂动作维度 (默认 = robot.dof)
          gripper_index: 7       # 夹爪维度索引 (-1 表示无夹爪)
          gripper_threshold: 0.5 # 二值夹爪阈值
          gripper_mode: binary   # 夹爪控制模式:
                                 #   binary      — 阈值后调 set() (默认)
                                 #   binary_move — 阈值后调 move(0) / move(max_width)
                                 #                 适合位控夹爪抓力不足的场景:
                                 #                 action 仍为位置值，但执行时
                                 #                 强制打到两端极限，电机持续施力
          action_space: joint_position  # 动作空间 (默认)
    """

    _SPACE_MAP: dict[str, ActionSpace] = {
        "joint_position": ActionSpace.JOINT_POSITION,
        "joint_velocity": ActionSpace.JOINT_VELOCITY,
        "joint_torque": ActionSpace.JOINT_TORQUE,
        "cartesian": ActionSpace.CARTESIAN,
    }

    def __init__(
        self,
        robot: BaseRobot,
        gripper: BaseGripper | None = None,
        *,
        arm_dim: int | None = None,
        gripper_index: int = -1,
        gripper_threshold: float = 0.5,
        gripper_mode: str = "binary",
        action_space: ActionSpace = ActionSpace.JOINT_POSITION,
    ) -> None:
        self._robot = robot
        self._gripper = gripper
        self._arm_dim = arm_dim if arm_dim is not None else robot.dof
        self._gripper_index = gripper_index
        self._gripper_threshold = gripper_threshold
        self._gripper_mode = gripper_mode
        self._action_space = action_space
        self._last_gripper_open: bool | None = None
        self._gripper_max_width: float | None = None  # binary_move 模式下缓存

    @classmethod
    def from_config(
        cls,
        robot: BaseRobot,
        gripper: BaseGripper | None,
        config: dict[str, Any],
    ) -> ActionDispatcher:
        """从 YAML 配置创建 dispatcher。

        读取 config["inference"] 段中的:
          arm_dim, gripper_index, gripper_threshold, gripper_mode, action_space
        """
        infer_cfg = config.get("inference", {})

        space_name = infer_cfg.get("action_space", "joint_position")
        action_space = cls._SPACE_MAP.get(space_name, ActionSpace.JOINT_POSITION)

        return cls(
            robot=robot,
            gripper=gripper,
            arm_dim=infer_cfg.get("arm_dim", robot.dof),
            gripper_index=infer_cfg.get(
                "gripper_index",
                robot.dof if gripper is not None else -1,
            ),
            gripper_threshold=infer_cfg.get("gripper_threshold", 0.5),
            gripper_mode=infer_cfg.get("gripper_mode", "binary"),
            action_space=action_space,
        )

    def dispatch(self, action_values: list[float]) -> None:
        """将推理输出分发给机器人和夹爪。

        Args:
            action_values: 推理模型输出的原始向量
        """
        # 手臂控制
        arm_values = action_values[:self._arm_dim]
        action = Action(self._action_space, arm_values)
        self._robot.act(action)

        # 夹爪控制
        if (
            self._gripper is not None
            and 0 <= self._gripper_index < len(action_values)
        ):
            gripper_val = action_values[self._gripper_index]
            gripper_open = gripper_val >= self._gripper_threshold
            if gripper_open != self._last_gripper_open:
                self._dispatch_gripper(gripper_open)
                self._last_gripper_open = gripper_open
                logger.info(
                    "[夹爪] val=%.3f → %s (%s)",
                    gripper_val, "打开" if gripper_open else "关闭",
                    self._gripper_mode,
                )

    def _dispatch_gripper(self, open: bool) -> None:
        """根据 gripper_mode 执行夹爪动作。

        binary      — 调 set()，由夹爪实现决定具体行为
        binary_move — 调 move(0) / move(max_width)，强制走到两端极限
                      适合位控夹爪: 电机持续施力直到到达目标位置，抓力更大
        """
        if self._gripper_mode == "binary_move":
            if open:
                max_width = self._get_gripper_max_width()
                self._gripper.move(max_width)
            else:
                self._gripper.move(-0.01)
        else:
            # 默认 binary 模式
            self._gripper.set(open)

    def _get_gripper_max_width(self) -> float:
        """获取夹爪最大宽度，首次调用时从硬件读取并缓存。"""
        if self._gripper_max_width is None:
            try:
                state = self._gripper.observe()
                if state.max_width > 0:
                    self._gripper_max_width = state.max_width
                    logger.info("[夹爪] max_width 缓存: %.4f m", self._gripper_max_width)
            except Exception:
                pass
        return self._gripper_max_width or 0.08  # 读取失败时回退到 8cm


def build_state_vector(
    arm_state: ArmState,
    gripper_width: float = 0.0,
    gripper_max_width: float = 1.0,
) -> list[float]:
    """构建推理需要的状态向量。

    默认格式: [q1..qN, gripper_normalized]
    gripper_normalized = width / max_width ∈ [0, 1]

    Args:
        arm_state: 机器人原子状态快照
        gripper_width: 当前夹爪宽度 (m)
        gripper_max_width: 夹爪最大宽度 (m)，用于归一化

    Returns:
        状态向量 (list[float])
    """
    state = list(arm_state.joint_positions)
    if gripper_max_width > 0:
        state.append(gripper_width / gripper_max_width)
    return state
