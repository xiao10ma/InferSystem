from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from Core import SensorFrame
from Sensor.rgb_camera.base import BaseRGBCamera, CameraParamSpec


class MockRGBCamera(BaseRGBCamera):
    def __init__(
        self,
        name: str = "front_camera",
        robot_name: str | None = None,
        distance_sequence: Sequence[float] = (2.5, 0.5, 1.5, 3.0),
        params: Mapping[str, Any] | None = None,
        auto_open: bool = True,
        with_depth: bool = False,
    ) -> None:
        default_params: dict[str, Any] = {
            "exposure.auto": True,
            "exposure.value": 1200,
            "gain": 1.0,
            "white_balance.auto": True,
        }
        if params:
            default_params.update(params)

        streams: dict[str, dict[str, Any]] = {
            "color": {
                "enabled": True,
                "width": 640,
                "height": 480,
                "fps": 30,
                "encoding": "rgb8",
            }
        }
        if with_depth:
            streams["depth"] = {
                "enabled": True,
                "width": 640,
                "height": 480,
                "fps": 30,
                "encoding": "z16",
                "aligned_to": "color",
            }

        super().__init__(
            name=name,
            robot_name=robot_name,
            params=default_params,
            streams=streams,
            param_specs={
                "exposure.auto": CameraParamSpec(
                    key="exposure.auto",
                    default=True,
                    value_types=(bool,),
                    description="Enable or disable auto exposure.",
                ),
                "exposure.value": CameraParamSpec(
                    key="exposure.value",
                    default=1200,
                    value_types=(int, float),
                    minimum=1,
                    description="Manual exposure value.",
                ),
                "gain": CameraParamSpec(
                    key="gain",
                    default=1.0,
                    value_types=(int, float),
                    minimum=0.0,
                    description="Analog or digital gain.",
                ),
                "white_balance.auto": CameraParamSpec(
                    key="white_balance.auto",
                    default=True,
                    value_types=(bool,),
                    description="Enable or disable auto white balance.",
                ),
            },
        )
        self._distance_sequence = tuple(distance_sequence)
        self._frame_index = 0

        if auto_open:
            self.open()

    def _open_device(self) -> None:
        return None

    def _close_device(self) -> None:
        return None

    def read_frame(self) -> SensorFrame:
        self._ensure_open()

        distance = self._distance_sequence[self._frame_index % len(self._distance_sequence)]
        self._frame_index += 1

        color_config = self.get_stream_config("color")
        streams: dict[str, dict[str, Any]] = {
            "color": {
                "encoding": color_config["encoding"],
                "width": color_config["width"],
                "height": color_config["height"],
                "channels": 3,
            }
        }

        if self.supports_depth:
            depth_config = self.get_stream_config("depth")
            if depth_config["enabled"]:
                streams["depth"] = {
                    "encoding": depth_config["encoding"],
                    "width": depth_config["width"],
                    "height": depth_config["height"],
                    "aligned_to": depth_config["aligned_to"],
                    "unit": "meter",
                    "distance_estimate": distance,
                }

        return self.build_frame(
            frame_id=self._frame_index,
            streams=streams,
            metadata={
                "obstacle_distance": distance,
                "active_params": self.get_params(),
            },
        )
