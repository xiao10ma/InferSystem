import math

import numpy as np
import pytest
from pydantic import ValidationError

from Core import ActionSpace, ArmState, RobotParams
from Core.config_schema import InferenceConfig
from Inference.action_processing import (
    blend_action_values,
    canonicalize_action_values,
    format_policy_state_vector,
    eef_pose_delta,
    rotation6d_to_quaternion,
)
from Inference.action_smoothing import TemporalActionSmoother
from Inference.dispatch import (
    ActionDispatcher,
    build_policy_state_vector,
    quaternion_to_rotation6d,
)


def test_inference_config_accepts_existing_aliases_and_smooth_config():
    cfg = InferenceConfig.model_validate(
        {
            "action_space": "cartesian",
            "arm_dim": 14,
            "gripper_index": -1,
            "latency_compensation": False,
            "smooth": {
                "enabled": True,
                "overlap_steps": 5,
            },
        }
    )

    assert cfg.arm_dof == 14
    assert cfg.gripper_action_index == -1
    assert cfg.latency_compensation is False
    assert cfg.smooth.enabled is True
    assert cfg.smooth.overlap_steps == 5


def test_inference_config_accepts_async_config():
    cfg = InferenceConfig.model_validate(
        {
            "async": {
                "enabled": True,
                "obs_fps": 12.5,
                "max_latency_steps": 3,
            },
        }
    )

    assert cfg.async_inference.enabled is True
    assert cfg.async_inference.obs_fps == 12.5
    assert cfg.async_inference.max_latency_steps == 3


def test_inference_config_rejects_removed_binary_move_gripper_mode():
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate({"gripper_mode": "binary_move"})


def test_inference_config_allows_canonical_only_for_cartesian():
    cfg = InferenceConfig.model_validate(
        {
            "action_space": "cartesian",
            "policy_format": "canonical",
            "canonical_dim": 32,
        }
    )

    assert cfg.policy_format.value == "canonical"
    assert cfg.canonical_dim == 32

    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(
            {
                "action_space": "joint_position",
                "policy_format": "canonical",
            }
        )


def test_inference_config_accepts_legacy_smooth_overlap_alias():
    cfg = InferenceConfig.model_validate(
        {
            "smooth": {
                "enabled": True,
                "min_smooth_steps": 6,
            },
        }
    )

    assert cfg.smooth.overlap_steps == 6


def test_rotation6d_column_major_identity_to_quaternion():
    quat = rotation6d_to_quaternion([1, 0, 0, 0, 1, 0])

    assert quat == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_quaternion_to_rotation6d_identity_column_major():
    rot6d = quaternion_to_rotation6d([1.0, 0.0, 0.0, 0.0])

    assert rot6d == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_cartesian_policy_state_single_arm_is_eef_rot6d_with_gripper():
    state = ArmState(
        timestamp=0.0,
        joint_positions=[0.0] * 7,
        eef_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
    )

    vec = build_policy_state_vector(
        state,
        action_space=ActionSpace.CARTESIAN,
        gripper_width=0.02,
        gripper_max_width=0.10,
        gripper_mode="width",
    )

    assert len(vec) == 10
    assert vec == pytest.approx([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.02])


def test_canonical_policy_state_pads_cartesian_state_to_fixed_dim():
    normal_state = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.02]

    vec = format_policy_state_vector(
        normal_state,
        action_space=ActionSpace.CARTESIAN,
        policy_format="canonical",
        canonical_dim=32,
    )

    assert len(vec) == 32
    assert vec[:10] == pytest.approx(normal_state)
    assert vec[10:] == pytest.approx([0.0] * 22)


def test_canonical_policy_state_rejects_joint_position():
    with pytest.raises(ValueError, match="canonical policy_format only supports cartesian"):
        format_policy_state_vector(
            [0.0] * 7,
            action_space=ActionSpace.JOINT_POSITION,
            policy_format="canonical",
            canonical_dim=32,
        )


def test_policy_state_binary_gripper_uses_thresholded_scalar():
    state = ArmState(
        timestamp=0.0,
        joint_positions=[0.0] * 7,
        eef_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
    )

    closed = build_policy_state_vector(
        state,
        action_space=ActionSpace.CARTESIAN,
        gripper_width=0.02,
        gripper_max_width=0.10,
        gripper_mode="binary",
        gripper_threshold=0.5,
    )
    open_ = build_policy_state_vector(
        state,
        action_space=ActionSpace.JOINT_POSITION,
        gripper_width=0.04,
        gripper_max_width=0.08,
        gripper_mode="binary",
        gripper_threshold=0.5,
        arm_dof=7,
        gripper_action_index=7,
    )

    assert closed[-1] == 0.0
    assert open_[-1] == 1.0


def test_cartesian_policy_state_bimanual_is_two_eef_rot6d_segments():
    state = ArmState(
        timestamp=0.0,
        joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05],
        eef_pose=[
            0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0,
            0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0,
        ],
    )

    vec = build_policy_state_vector(
        state,
        action_space=ActionSpace.CARTESIAN,
        gripper_mode="width",
    )

    assert len(vec) == 20
    assert vec == pytest.approx(
        [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04,
         0.4, 0.5, 0.6, 1, 0, 0, 0, 1, 0, 0.05]
    )


def test_single_arm_eef_action_converts_rotation6d_to_pose7_with_gripper():
    action = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04]

    converted = canonicalize_action_values(action, ActionSpace.CARTESIAN)

    assert converted == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04])


def test_bimanual_eef_action_converts_two_rotation6d_actions_to_pose7():
    left = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04]
    right = [0.4, 0.5, 0.6, 1, 0, 0, 0, 1, 0, 0.05]

    converted = canonicalize_action_values(left + right, ActionSpace.CARTESIAN)

    assert converted == pytest.approx(
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04,
         0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.05]
    )


def test_canonical_cartesian_action_uses_single_arm_prefix_from_32d():
    raw = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04] + [99.0] * 22

    converted = canonicalize_action_values(
        raw,
        ActionSpace.CARTESIAN,
        policy_format="canonical",
        canonical_dim=32,
        effective_action_dim=10,
    )

    assert converted == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04])


def test_canonical_cartesian_action_uses_bimanual_prefix_from_32d():
    left = [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04]
    right = [0.4, 0.5, 0.6, 1, 0, 0, 0, 1, 0, 0.05]
    raw = left + right + [99.0] * 12

    converted = canonicalize_action_values(
        raw,
        ActionSpace.CARTESIAN,
        policy_format="canonical",
        canonical_dim=32,
        effective_action_dim=20,
    )

    assert converted == pytest.approx(
        [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04,
         0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.05]
    )


def test_canonical_cartesian_action_requires_32d_chunk_rows():
    with pytest.raises(ValueError, match="canonical policy action expects 32 values"):
        canonicalize_action_values(
            [0.0] * 20,
            ActionSpace.CARTESIAN,
            policy_format="canonical",
            canonical_dim=32,
            effective_action_dim=20,
        )


def test_eef_pose_delta_reports_translation_and_rotation():
    pose_a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    half = math.sqrt(0.5)
    pose_b = [0.03, 0.04, 0.0, half, 0.0, 0.0, half]

    trans, rot = eef_pose_delta(pose_a, pose_b)

    assert trans == pytest.approx(0.05)
    assert rot == pytest.approx(math.pi / 2)


def test_temporal_smoother_blends_new_chunk_into_remaining_old_actions():
    smoother = TemporalActionSmoother(action_space=ActionSpace.JOINT_POSITION)
    smoother.integrate_chunk([[0.0], [0.0]], latency_steps=0, overlap_steps=2)
    assert smoother.peek_next() == pytest.approx([0.0])
    assert smoother.pop_next() == pytest.approx([0.0])

    smoother.integrate_chunk([[10.0], [20.0], [30.0]], latency_steps=0, overlap_steps=2)

    assert smoother.pop_next() == pytest.approx([0.0])
    assert smoother.pop_next() == pytest.approx([20.0])
    assert smoother.pop_next() == pytest.approx([30.0])
    assert smoother.pop_next() is None


def test_temporal_smoother_drops_latency_steps_from_new_chunk():
    smoother = TemporalActionSmoother(action_space=ActionSpace.JOINT_POSITION)

    smoother.integrate_chunk([[1.0], [2.0], [3.0]], latency_steps=2, overlap_steps=2)

    assert smoother.pop_next() == pytest.approx([3.0])
    assert smoother.pop_next() is None


def test_temporal_smoother_async_latency_uses_published_steps_with_cap():
    smoother = TemporalActionSmoother(action_space=ActionSpace.JOINT_POSITION)

    smoother.integrate_chunk([[0.0], [0.0], [0.0]], latency_steps=0, overlap_steps=2)
    smoother.pop_next()
    smoother.pop_next()
    smoother.integrate_chunk(
        [[10.0], [20.0], [30.0], [40.0]],
        latency_steps=None,
        max_latency_steps=1,
        overlap_steps=2,
    )

    assert smoother.pop_next() == pytest.approx([0.0])
    assert smoother.pop_next() == pytest.approx([30.0])
    assert smoother.pop_next() == pytest.approx([40.0])
    assert smoother.pop_next() is None


def test_temporal_smoother_normalizes_cartesian_quaternion_blends():
    smoother = TemporalActionSmoother(action_space=ActionSpace.CARTESIAN)
    old_action = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    new_action = [0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]

    smoother.integrate_chunk([old_action, old_action], latency_steps=0, overlap_steps=2)
    smoother.pop_next()
    smoother.integrate_chunk([new_action, new_action], latency_steps=0, overlap_steps=2)
    smoother.pop_next()
    blended = np.array(smoother.pop_next())

    assert np.linalg.norm(blended[3:7]) == pytest.approx(1.0)


def test_cartesian_blend_uses_quaternion_slerp_for_rotation():
    old_action = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    new_action = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    blended = blend_action_values(
        old_action,
        new_action,
        weight_old=0.5,
        action_space=ActionSpace.CARTESIAN,
    )

    trans, rot = eef_pose_delta(old_action, blended)
    assert trans == pytest.approx(0.1)
    assert rot == pytest.approx(math.pi / 2)
    assert np.linalg.norm(blended[3:7]) == pytest.approx(1.0)


class _FakeRobot:
    dof = 6

    def __init__(self):
        self.calls = []
        self._params = RobotParams(
            dof=6,
            joint_position_min=[-1.0] * 6,
            joint_position_max=[1.0] * 6,
            joint_velocity_max=[1.0] * 6,
            joint_acceleration_max=[1.0] * 6,
            joint_torque_max=[1.0] * 6,
        )

    def act(self, action, *, state=None):
        self.calls.append((action, state))

    def get_params(self):
        return self._params


class _FakeBimanualRobot(_FakeRobot):
    dof = 14

    def __init__(self):
        super().__init__()
        self._params = RobotParams(
            dof=14,
            joint_position_min=[-1.0] * 14,
            joint_position_max=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.08,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09],
            joint_velocity_max=[1.0] * 14,
            joint_acceleration_max=[1.0] * 14,
            joint_torque_max=[1.0] * 14,
        )


class _FakeGripper:
    def __init__(self, max_width=0.1):
        self.max_width = max_width
        self.set_calls = []
        self.move_calls = []

    def observe(self):
        return type("State", (), {"max_width": self.max_width})()

    def set(self, open):
        self.set_calls.append(open)

    def move(self, width):
        self.move_calls.append(width)


def test_dispatcher_canonicalizes_eef_model_action_and_passes_state_to_robot():
    robot = _FakeRobot()
    dispatcher = ActionDispatcher(robot, action_space=ActionSpace.CARTESIAN, gripper_mode="width")
    state = object()

    dispatcher.dispatch([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.04], state=state)

    action, passed_state = robot.calls[-1]
    assert action.space == ActionSpace.CARTESIAN
    assert action.values == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04])
    assert passed_state is state


def test_dispatcher_external_gripper_binary_and_width_modes():
    robot = _FakeRobot()
    gripper = _FakeGripper(max_width=0.08)

    binary = ActionDispatcher(
        robot,
        gripper,
        arm_dim=6,
        gripper_index=6,
        gripper_mode="binary",
        gripper_threshold=0.5,
    )
    binary.dispatch([0, 0, 0, 0, 0, 0, 0.7])

    width = ActionDispatcher(
        robot,
        gripper,
        arm_dim=6,
        gripper_index=6,
        gripper_mode="width",
    )
    width.dispatch([0, 0, 0, 0, 0, 0, 0.2])

    assert gripper.set_calls == [True]
    assert gripper.move_calls == pytest.approx([0.08])


def test_dispatcher_embedded_bimanual_gripper_modes_convert_channels():
    robot = _FakeBimanualRobot()

    binary = ActionDispatcher(
        robot,
        gripper=None,
        arm_dim=14,
        gripper_index=-1,
        gripper_mode="binary",
        gripper_threshold=0.5,
    )
    binary.dispatch([0, 0, 0, 0, 0, 0, 0.2,
                     0, 0, 0, 0, 0, 0, 0.8])

    width = ActionDispatcher(
        robot,
        gripper=None,
        arm_dim=14,
        gripper_index=-1,
        gripper_mode="width",
    )
    width.dispatch([0, 0, 0, 0, 0, 0, 0.2,
                    0, 0, 0, 0, 0, 0, 0.2])

    binary_action = robot.calls[-2][0]
    width_action = robot.calls[-1][0]
    assert binary_action.values[6] == pytest.approx(0.0)
    assert binary_action.values[13] == pytest.approx(0.09)
    assert width_action.values[6] == pytest.approx(0.08)
    assert width_action.values[13] == pytest.approx(0.09)
