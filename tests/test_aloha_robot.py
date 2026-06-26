from __future__ import annotations

import math

import pytest

from Core import Action, ActionSpace
from Core.config_schema import SystemConfig
from Robot import AlohaRobot, BaseRobot


class _JointState:
    joint_1 = 1000
    joint_2 = 2000
    joint_3 = 3000
    joint_4 = 4000
    joint_5 = 5000
    joint_6 = 6000


class _JointMsgs:
    joint_state = _JointState()


class _GripperState:
    grippers_angle = 300000


class _GripperMsgs:
    gripper_state = _GripperState()


class _EndPose:
    X_axis = 100000
    Y_axis = 200000
    Z_axis = 300000
    RX_axis = 0
    RY_axis = 0
    RZ_axis = 0


class _EndPoseMsgs:
    end_pose = _EndPose()


class _FakePiper:
    def __init__(self):
        self.motion_calls = []
        self.joint_calls = []
        self.gripper_calls = []
        self.end_pose_calls = []

    def MotionCtrl_2(self, *args):
        self.motion_calls.append(args)

    def JointCtrl(self, *args):
        self.joint_calls.append(args)

    def GripperCtrl(self, *args):
        self.gripper_calls.append(args)

    def EndPoseCtrl(self, *args):
        self.end_pose_calls.append(args)

    def GetArmJointMsgs(self):
        return _JointMsgs()

    def GetArmGripperMsgs(self):
        return _GripperMsgs()

    def GetArmEndPoseMsgs(self):
        return _EndPoseMsgs()


def _connected_robot() -> AlohaRobot:
    robot = AlohaRobot(speed_percent=80)
    robot._left.piper = _FakePiper()
    robot._right.piper = _FakePiper()
    robot._connected = True
    return robot


def test_aloha_config_is_accepted_and_default_import_registers_robot():
    cfg = SystemConfig.model_validate(
        {
            "robot": {
                "type": "aloha",
                "can_interface_left": "can0",
                "can_interface_right": "can1",
                "control": {"frequency_hz": 30},
            }
        }
    )

    assert cfg.robot.type == "aloha"
    assert "aloha" in BaseRobot._registry
    robot = BaseRobot._registry["aloha"]._from_config_dict(cfg.robot)
    assert isinstance(robot, AlohaRobot)
    assert robot.robot_type == "aloha"
    assert robot.get_params().control_frequency_hz == 30


def test_aloha_joint_action_switches_movej_and_sends_joint_and_gripper_commands():
    robot = _connected_robot()

    robot.act(Action(ActionSpace.JOINT_POSITION, [0.0] * 14))

    assert robot._left.piper.motion_calls[-1] == (0x01, 0x01, 80, 0x00)
    assert robot._right.piper.motion_calls[-1] == (0x01, 0x01, 80, 0x00)
    assert robot._left.piper.joint_calls[-1] == (0, 0, 0, 0, 0, 0)
    assert robot._right.piper.joint_calls[-1] == (0, 0, 0, 0, 0, 0)
    assert robot._left.piper.gripper_calls[-1] == (0, 1000, 0x01, 0)


def test_aloha_observe_populates_joint_positions_and_bimanual_eef_pose():
    robot = _connected_robot()

    state = robot.observe()

    assert len(state.joint_positions) == 14
    assert state.joint_positions[:7] == pytest.approx(
        [math.radians(v / 1000.0) for v in [1000, 2000, 3000, 4000, 5000, 6000]]
        + [0.03]
    )
    assert len(state.eef_pose) == 14
    assert state.eef_pose[:7] == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])


def test_aloha_cartesian_action_switches_movel_and_sends_end_pose_and_gripper():
    robot = _connected_robot()
    action = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.04,
              0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0, 0.05]

    robot.act(Action(ActionSpace.CARTESIAN, action))

    assert robot._left.piper.motion_calls[-1] == (0x01, 0x02, 80, 0x00)
    assert robot._right.piper.motion_calls[-1] == (0x01, 0x02, 80, 0x00)
    assert robot._left.piper.end_pose_calls[-1] == (100000, 200000, 300000, 0, 0, 0)
    assert robot._right.piper.end_pose_calls[-1] == (400000, 500000, 600000, 0, 0, 0)
    assert robot._left.piper.gripper_calls[-1] == (400000, 1000, 0x01, 0)
    assert robot._right.piper.gripper_calls[-1] == (500000, 1000, 0x01, 0)
