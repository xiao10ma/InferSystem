from __future__ import annotations

from collections.abc import Iterable

from Core import InferenceCommand, RobotState, SensorFrame
from Inference.base import BaseInferenceEngine


class RuleBasedInferenceEngine(BaseInferenceEngine):
    def __init__(self, obstacle_threshold: float = 1.0, step_size: float = 0.5) -> None:
        self.obstacle_threshold = obstacle_threshold
        self.step_size = step_size

    def infer(
        self,
        sensor_frames: Iterable[SensorFrame],
        robot_state: RobotState,
    ) -> InferenceCommand:
        nearest_obstacle: float | None = None

        for frame in sensor_frames:
            distance = frame.payload.get("obstacle_distance")
            if distance is None:
                continue
            distance = float(distance)
            if nearest_obstacle is None or distance < nearest_obstacle:
                nearest_obstacle = distance

        if nearest_obstacle is not None and nearest_obstacle < self.obstacle_threshold:
            return InferenceCommand(
                action="stop",
                parameters={"reason": "obstacle", "distance": nearest_obstacle},
                target_robot=robot_state.robot_name,
                priority=10,
            )

        return InferenceCommand(
            action="move",
            parameters={"dx": self.step_size, "dy": 0.0},
            target_robot=robot_state.robot_name,
        )

