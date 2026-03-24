from __future__ import annotations

import unittest
from types import SimpleNamespace

from Core import InferenceCommand
from Robot.flexiv import FlexivRobot


class FakeEnumValue:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeGripperStates:
    def __init__(self) -> None:
        self.width = 0.05
        self.force = 0.0
        self.is_moving = False


class FakeGripper:
    def __init__(self, robot: object) -> None:
        self._states = FakeGripperStates()
        self.calls: list[tuple[str, object]] = []

    def Init(self) -> None:
        self.calls.append(("Init", None))

    def Move(self, width: float, velocity: float, force: float) -> None:
        self._states.width = width
        self.calls.append(("Move", (width, velocity, force)))

    def Grasp(self, force: float) -> None:
        self._states.width = 0.0
        self.calls.append(("Grasp", force))

    def states(self) -> FakeGripperStates:
        return self._states


class FakeRobotHandle:
    def __init__(self) -> None:
        self._mode = FakeEnumValue("IDLE")
        self._operational_status = FakeEnumValue("READY")
        self._info = SimpleNamespace(
            serial_num="Rizon4-123456", model_name="Rizon 4",
            software_ver="v3.11", license_type="full", DoF=7,
        )
        self._states = SimpleNamespace(
            q=[0.0, -0.6981, 0.0, 1.5708, 0.0, 0.6981, 0.0],
            dq=[0.0] * 7,
            theta=[0.0, -0.6981, 0.0, 1.5708, 0.0, 0.6981, 0.0],
            tau=[0.0] * 7, tau_ext=[0.0] * 7,
            tcp_pose=[0.687, -0.114, 0.178, 1.0, 0.0, 0.0, 0.0],
            tcp_vel=[0.0] * 6,
            ext_wrench_in_tcp=[0.0] * 6,
            ext_wrench_in_world=[0.0] * 6,
            timestamp=0.0,
        )
        self._primitive_states: dict[str, object] = {"reachedTarget": 1}
        self.calls: list[tuple[str, object]] = []

    def connected(self) -> bool: return True
    def operational(self) -> bool: return True
    def busy(self) -> bool: return False
    def fault(self) -> bool: return False
    def stopped(self) -> bool: return False
    def mode(self) -> FakeEnumValue: return self._mode
    def operational_status(self) -> FakeEnumValue: return self._operational_status
    def states(self) -> SimpleNamespace: return self._states
    def info(self) -> SimpleNamespace: return self._info
    def primitive_states(self) -> dict: return self._primitive_states

    def Stop(self) -> None: self.calls.append(("Stop", None))
    def ClearFault(self) -> None: self.calls.append(("ClearFault", None))
    def Enable(self) -> None: self.calls.append(("Enable", None))
    def RunAutoRecovery(self) -> None: self.calls.append(("RunAutoRecovery", None))

    def SwitchMode(self, mode: object) -> None:
        self._mode = mode
        self.calls.append(("SwitchMode", mode))

    def SetVelocityScale(self, value: int) -> None:
        self.calls.append(("SetVelocityScale", value))

    def ExecutePrimitive(self, name: str, params: dict, block: bool) -> None:
        self.calls.append(("ExecutePrimitive", (name, params, block)))

    def ExecutePlan(self, plan: object, cont: bool, block: bool) -> None:
        self.calls.append(("ExecutePlan", (plan, cont, block)))

    def SendJointPosition(self, t: list, v: list, a: list, ff: list) -> None:
        self.calls.append(("SendJointPosition", (t, v, a, ff)))


class FakeJPos:
    def __init__(self, q: list[float], q_e: list[float] | None = None) -> None:
        self.q = q


class FakeCoord:
    def __init__(self, pos: list, ori: list, ref: list,
                 ref_q_m: list | None = None, ref_q_e: list | None = None) -> None:
        self.position = pos
        self.orientation = ori


class FlexivRobotTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_mode = SimpleNamespace(
            __members__={
                "IDLE": FakeEnumValue("IDLE"),
                "NRT_PRIMITIVE_EXECUTION": FakeEnumValue("NRT_PRIMITIVE_EXECUTION"),
            },
            IDLE=FakeEnumValue("IDLE"),
            NRT_PRIMITIVE_EXECUTION=FakeEnumValue("NRT_PRIMITIVE_EXECUTION"),
        )
        self.fake_rdk = SimpleNamespace(
            Mode=fake_mode,
            JPos=FakeJPos,
            Coord=FakeCoord,
            Gripper=FakeGripper,
        )
        self.handle = FakeRobotHandle()
        self.robot = FlexivRobot(
            "Rizon4-123456",
            robot_handle=self.handle,
            rdk_module=self.fake_rdk,
        )

    # ── 状态读取 ──────────────────────────────────────────────

    def test_get_state(self) -> None:
        state = self.robot.get_state()
        self.assertEqual(state.robot_name, "Rizon4-123456")
        self.assertAlmostEqual(state.pose.x, 0.687)
        self.assertEqual(state.metadata["mode"], "IDLE")

    def test_get_joint_positions(self) -> None:
        q = self.robot.get_joint_positions()
        self.assertEqual(len(q), 7)
        self.assertAlmostEqual(q[3], 1.5708)

    def test_get_joint_velocities(self) -> None:
        dq = self.robot.get_joint_velocities()
        self.assertEqual(dq, [0.0] * 7)

    def test_get_eef_pose(self) -> None:
        tcp = self.robot.get_eef_pose()
        self.assertEqual(len(tcp), 7)
        self.assertAlmostEqual(tcp[0], 0.687)

    def test_dof(self) -> None:
        self.assertEqual(self.robot.dof, 7)

    # ── 关节位置控制 ──────────────────────────────────────────

    def test_move_joint(self) -> None:
        target = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        self.robot.move_joint(target)
        # 应切换模式 + ExecutePrimitive MoveJ
        names = [c[0] for c in self.handle.calls]
        self.assertIn("SwitchMode", names)
        self.assertIn("ExecutePrimitive", names)
        ep = [c for c in self.handle.calls if c[0] == "ExecutePrimitive"][0]
        self.assertEqual(ep[1][0], "MoveJ")

    def test_move_joint_with_velocity(self) -> None:
        target = [0.0] * 7
        self.robot.move_joint(target, velocity=0.5)
        vel_calls = [c for c in self.handle.calls if c[0] == "SetVelocityScale"]
        self.assertEqual(len(vel_calls), 1)
        self.assertEqual(vel_calls[0][1], 50)

    # ── 末端位姿控制 ──────────────────────────────────────────

    def test_move_eef(self) -> None:
        self.robot.move_eef([0.5, 0.0, 0.3], [0.0, 0.0, 0.0])
        ep = [c for c in self.handle.calls if c[0] == "ExecutePrimitive"][0]
        self.assertEqual(ep[1][0], "MoveL")

    def test_move_eef_keep_orientation(self) -> None:
        self.robot.move_eef([0.5, 0.0, 0.3])
        ep = [c for c in self.handle.calls if c[0] == "ExecutePrimitive"][0]
        self.assertEqual(ep[1][0], "MoveL")

    # ── 夹爪控制 ──────────────────────────────────────────────

    def test_gripper_set_open(self) -> None:
        self.robot.gripper_set(True)
        gs = self.robot.get_gripper_state()
        self.assertAlmostEqual(gs.width, 0.09)

    def test_gripper_set_close(self) -> None:
        self.robot.gripper_set(False)
        gs = self.robot.get_gripper_state()
        self.assertAlmostEqual(gs.width, 0.0)

    def test_gripper_move(self) -> None:
        self.robot.gripper_move(0.04, velocity=0.05, force=20.0)
        gs = self.robot.get_gripper_state()
        self.assertAlmostEqual(gs.width, 0.04)

    # ── 其他 ──────────────────────────────────────────────────

    def test_apply_command_move_joint(self) -> None:
        self.robot.apply_command(InferenceCommand(
            action="move_joint",
            parameters={"positions": [0.0] * 7},
        ))
        ep = [c for c in self.handle.calls if c[0] == "ExecutePrimitive"][0]
        self.assertEqual(ep[1][0], "MoveJ")

    def test_apply_command_gripper_set(self) -> None:
        self.robot.apply_command(InferenceCommand(
            action="gripper_set",
            parameters={"open": True},
        ))
        gs = self.robot.get_gripper_state()
        self.assertAlmostEqual(gs.width, 0.09)

    def test_context_manager(self) -> None:
        with self.robot as r:
            self.assertIs(r, self.robot)

    def test_unknown_command_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.robot.apply_command(InferenceCommand(action="NOT_REAL"))


if __name__ == "__main__":
    unittest.main()
