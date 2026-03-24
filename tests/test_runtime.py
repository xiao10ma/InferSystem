from __future__ import annotations

import unittest

from Inference.rule_engine import RuleBasedInferenceEngine
from Robot.mock_mobile import MockMobileRobot
from SDK.fleet import FleetManager
from SDK.runtime import RobotRuntime
from Sensor.rgb_camera.mock_camera import MockRGBCamera


class RuntimeTests(unittest.TestCase):
    def test_rgb_camera_lifecycle_and_params(self) -> None:
        camera = MockRGBCamera(auto_open=False, with_depth=True)

        self.assertFalse(camera.is_open)
        self.assertTrue(camera.supports_depth)
        camera.set_param("gain", 2.0)
        camera.set_params({"exposure.auto": False}, **{"exposure.value": 900})
        camera.configure_stream("color", width=1280, height=720, fps=60)
        camera.configure_stream("depth", width=640, height=480, fps=30, encoding="z16")
        self.assertEqual(camera.get_param("gain"), 2.0)
        self.assertFalse(camera.get_params()["exposure.auto"])
        self.assertEqual(camera.get_stream_config("color")["width"], 1280)
        self.assertEqual(camera.get_stream_config("color")["height"], 720)

        with self.assertRaises(RuntimeError):
            camera.read_frame()

        camera.open()
        frame = camera.read_frame()
        self.assertTrue(camera.is_open)
        self.assertEqual(frame.payload["streams"]["color"]["width"], 1280)
        self.assertEqual(frame.payload["streams"]["color"]["height"], 720)
        self.assertIn("depth", frame.payload["streams"])

        camera.restart()
        self.assertTrue(camera.is_open)

        camera.close()
        self.assertFalse(camera.is_open)

    def test_robot_moves_when_path_is_clear(self) -> None:
        robot = MockMobileRobot(name="clear_bot")
        runtime = RobotRuntime(
            robot=robot,
            sensors=[MockRGBCamera(robot_name=robot.name, distance_sequence=(2.0,))],
            engine=RuleBasedInferenceEngine(obstacle_threshold=1.0, step_size=0.5),
        )

        result = runtime.step()

        self.assertEqual(result.command.action, "move")
        self.assertAlmostEqual(result.state_after.pose.x, 0.5)

    def test_robot_stops_when_obstacle_is_close(self) -> None:
        robot = MockMobileRobot(name="safe_bot")
        runtime = RobotRuntime(
            robot=robot,
            sensors=[MockRGBCamera(robot_name=robot.name, distance_sequence=(0.4,))],
            engine=RuleBasedInferenceEngine(obstacle_threshold=1.0, step_size=0.5),
        )

        result = runtime.step()

        self.assertEqual(result.command.action, "stop")
        self.assertAlmostEqual(result.state_after.pose.x, 0.0)
        self.assertEqual(result.state_after.metadata["stop_reason"], "obstacle")

    def test_fleet_manager_steps_each_robot(self) -> None:
        fleet = FleetManager()

        mover = RobotRuntime(
            robot=MockMobileRobot(name="mover"),
            sensors=[MockRGBCamera(robot_name="mover", distance_sequence=(2.0,))],
            engine=RuleBasedInferenceEngine(obstacle_threshold=1.0, step_size=0.3),
        )
        stopper = RobotRuntime(
            robot=MockMobileRobot(name="stopper"),
            sensors=[MockRGBCamera(robot_name="stopper", distance_sequence=(0.3,))],
            engine=RuleBasedInferenceEngine(obstacle_threshold=1.0, step_size=0.3),
        )

        fleet.register(mover)
        fleet.register(stopper)

        results = fleet.step_all()

        self.assertEqual(results["mover"].command.action, "move")
        self.assertEqual(results["stopper"].command.action, "stop")


if __name__ == "__main__":
    unittest.main()
