from __future__ import annotations

import math
from types import MethodType

import pytest

from Core import RobotParams
from Core.config_schema import FlexivRobotConfig
from Robot.flexiv import FlexivRobot


def test_flexiv_build_params_uses_configured_home_position_deg():
    cfg = FlexivRobotConfig.model_validate(
        {
            "type": "flexiv",
            "serial_number": "Rizon4-test",
            "control": {
                "home_position_deg": [0, -10, 20, 30, 40, 50, 60],
            },
        }
    )

    params = FlexivRobot._build_params(cfg)

    assert params.home_position == pytest.approx(
        [math.radians(v) for v in [0, -10, 20, 30, 40, 50, 60]]
    )


def test_flexiv_go_home_moves_to_configured_home_position():
    robot = FlexivRobot.__new__(FlexivRobot)
    robot._connected = False
    robot._robot = None
    robot._params = RobotParams(
        dof=7,
        joint_position_min=[-10.0] * 7,
        joint_position_max=[10.0] * 7,
        joint_velocity_max=[1.0] * 7,
        joint_acceleration_max=[1.0] * 7,
        joint_torque_max=[1.0] * 7,
        home_position=[0.0, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    robot._control_config = {"home_velocity_scale": 30}
    calls = []

    def fake_move_joint_position(self, positions, *, velocity=None, timeout_s=30.0):
        calls.append((list(positions), velocity, timeout_s))
        return True

    robot.move_joint_position = MethodType(fake_move_joint_position, robot)

    assert FlexivRobot.go_home(robot, timeout_s=12.0) is True
    assert calls == [
        ([0.0, -0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 0.3, 12.0),
    ]


def test_flexiv_go_home_explicit_velocity_overrides_home_velocity_scale():
    robot = FlexivRobot.__new__(FlexivRobot)
    robot._connected = False
    robot._robot = None
    robot._params = RobotParams(
        dof=7,
        joint_position_min=[-10.0] * 7,
        joint_position_max=[10.0] * 7,
        joint_velocity_max=[1.0] * 7,
        joint_acceleration_max=[1.0] * 7,
        joint_torque_max=[1.0] * 7,
        home_position=[0.0] * 7,
    )
    robot._control_config = {"home_velocity_scale": 30}
    calls = []

    def fake_move_joint_position(self, positions, *, velocity=None, timeout_s=30.0):
        calls.append((list(positions), velocity, timeout_s))
        return True

    robot.move_joint_position = MethodType(fake_move_joint_position, robot)

    assert FlexivRobot.go_home(robot, velocity=0.8, timeout_s=5.0) is True
    assert calls == [([0.0] * 7, 0.8, 5.0)]
