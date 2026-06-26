from __future__ import annotations

import numpy as np
import pytest

from Robot.arx5_bimanual import Arx5BimanualRobot


class _FakeJointState:
    gripper_pos = 0.07
    gripper_vel = 0.0
    gripper_torque = 0.0

    def pos(self):
        return np.zeros(6)

    def vel(self):
        return np.zeros(6)

    def torque(self):
        return np.zeros(6)


class _FakeEefState:
    def __init__(self, pose6d):
        self._pose6d = pose6d

    def pose_6d(self):
        return np.asarray(self._pose6d, dtype=float)


class _FakeController:
    def __init__(self, pose6d):
        self._pose6d = pose6d

    def get_joint_state(self):
        return _FakeJointState()

    def get_eef_state(self):
        return _FakeEefState(self._pose6d)

    def reset_to_home(self):
        return None

    def set_to_damping(self):
        return None


def test_bimanual_observe_populates_left_and_right_eef_pose7():
    robot = Arx5BimanualRobot(
        left_model="X5",
        left_interface="can0",
        right_model="X5",
        right_interface="can1",
    )
    robot._connected = True
    robot._ctrls = (
        _FakeController([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
        _FakeController([0.4, 0.5, 0.6, 0.0, 0.0, 0.0]),
    )

    state = robot.observe()

    assert len(state.eef_pose) == 14
    assert state.eef_pose[:7] == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    assert state.eef_pose[7:14] == pytest.approx([0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0])
