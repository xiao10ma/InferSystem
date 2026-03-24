from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

from Core import SensorFrame
from Sensor.rgb_camera.base import BaseRGBCamera, CameraParamSpec, CameraStreamConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


# RealSense option name -> pyrealsense2 option enum
_RS_OPTION_MAP: dict[str, rs.option] = {
    "exposure.auto": rs.option.enable_auto_exposure,
    "exposure.value": rs.option.exposure,
    "gain": rs.option.gain,
    "white_balance.auto": rs.option.enable_auto_white_balance,
    "white_balance.value": rs.option.white_balance,
    "brightness": rs.option.brightness,
    "contrast": rs.option.contrast,
    "gamma": rs.option.gamma,
    "hue": rs.option.hue,
    "saturation": rs.option.saturation,
    "sharpness": rs.option.sharpness,
    "backlight_compensation": rs.option.backlight_compensation,
    "laser_power": rs.option.laser_power,
    "emitter_enabled": rs.option.emitter_enabled,
    "visual_preset": rs.option.visual_preset,
    "power_line_frequency": rs.option.power_line_frequency,
}

# Which sensor each option belongs to: "color" for RGB Camera, "depth" for Stereo Module
_OPTION_SENSOR_MAP: dict[str, str] = {
    "exposure.auto": "color",
    "exposure.value": "color",
    "gain": "color",
    "white_balance.auto": "color",
    "white_balance.value": "color",
    "brightness": "color",
    "contrast": "color",
    "gamma": "color",
    "hue": "color",
    "saturation": "color",
    "sharpness": "color",
    "backlight_compensation": "color",
    "power_line_frequency": "color",
    "laser_power": "depth",
    "emitter_enabled": "depth",
    "visual_preset": "depth",
}

_PARAM_SPECS: dict[str, CameraParamSpec] = {
    "exposure.auto": CameraParamSpec(
        key="exposure.auto", default=True, value_types=(bool, int, float),
        description="Enable/disable auto exposure for RGB sensor.",
    ),
    "exposure.value": CameraParamSpec(
        key="exposure.value", default=166, value_types=(int, float),
        minimum=1, maximum=10000,
        description="Manual exposure value for RGB sensor (1-10000).",
    ),
    "gain": CameraParamSpec(
        key="gain", default=64, value_types=(int, float),
        minimum=0, maximum=128,
        description="Gain for RGB sensor (0-128).",
    ),
    "white_balance.auto": CameraParamSpec(
        key="white_balance.auto", default=True, value_types=(bool, int, float),
        description="Enable/disable auto white balance.",
    ),
    "white_balance.value": CameraParamSpec(
        key="white_balance.value", default=4600, value_types=(int, float),
        minimum=2800, maximum=6500,
        description="Manual white balance value (2800-6500).",
    ),
    "brightness": CameraParamSpec(
        key="brightness", default=0, value_types=(int, float),
        minimum=-64, maximum=64,
        description="Brightness (-64 to 64).",
    ),
    "contrast": CameraParamSpec(
        key="contrast", default=50, value_types=(int, float),
        minimum=0, maximum=100,
        description="Contrast (0-100).",
    ),
    "gamma": CameraParamSpec(
        key="gamma", default=300, value_types=(int, float),
        minimum=100, maximum=500,
        description="Gamma (100-500).",
    ),
    "saturation": CameraParamSpec(
        key="saturation", default=64, value_types=(int, float),
        minimum=0, maximum=100,
        description="Saturation (0-100).",
    ),
    "sharpness": CameraParamSpec(
        key="sharpness", default=50, value_types=(int, float),
        minimum=0, maximum=100,
        description="Sharpness (0-100).",
    ),
    "laser_power": CameraParamSpec(
        key="laser_power", default=150, value_types=(int, float),
        minimum=0, maximum=360,
        description="IR laser power for depth (0-360).",
    ),
    "emitter_enabled": CameraParamSpec(
        key="emitter_enabled", default=1, value_types=(int, float),
        minimum=0, maximum=2,
        description="IR emitter: 0=off, 1=on, 2=auto.",
    ),
}

# D405 has no separate RGB sensor; color comes from stereo module
_D405_PARAM_SPECS: dict[str, CameraParamSpec] = {
    "exposure.auto": CameraParamSpec(
        key="exposure.auto", default=True, value_types=(bool, int, float),
        description="Enable/disable auto exposure (stereo module).",
    ),
    "exposure.value": CameraParamSpec(
        key="exposure.value", default=33000, value_types=(int, float),
        minimum=1, maximum=165000,
        description="Manual exposure for stereo module (1-165000).",
    ),
    "gain": CameraParamSpec(
        key="gain", default=16, value_types=(int, float),
        minimum=16, maximum=248,
        description="Gain for stereo module (16-248).",
    ),
    "white_balance.auto": CameraParamSpec(
        key="white_balance.auto", default=True, value_types=(bool, int, float),
        description="Enable/disable auto white balance (stereo module).",
    ),
    "white_balance.value": CameraParamSpec(
        key="white_balance.value", default=4600, value_types=(int, float),
        minimum=2800, maximum=6500,
        description="Manual white balance (stereo module).",
    ),
    "brightness": CameraParamSpec(
        key="brightness", default=0, value_types=(int, float),
        minimum=-64, maximum=64,
        description="Brightness (stereo module).",
    ),
    "contrast": CameraParamSpec(
        key="contrast", default=50, value_types=(int, float),
        minimum=0, maximum=100,
        description="Contrast (stereo module).",
    ),
    "saturation": CameraParamSpec(
        key="saturation", default=64, value_types=(int, float),
        minimum=0, maximum=100,
        description="Saturation (stereo module).",
    ),
    "sharpness": CameraParamSpec(
        key="sharpness", default=50, value_types=(int, float),
        minimum=0, maximum=100,
        description="Sharpness (stereo module).",
    ),
    "laser_power": CameraParamSpec(
        key="laser_power", default=150, value_types=(int, float),
        minimum=0, maximum=360,
        description="IR laser power for depth (0-360).",
    ),
    "emitter_enabled": CameraParamSpec(
        key="emitter_enabled", default=1, value_types=(int, float),
        minimum=0, maximum=2,
        description="IR emitter: 0=off, 1=on, 2=auto.",
    ),
}


class RealSenseCamera(BaseRGBCamera):
    """Intel RealSense RGB-D camera driver.

    Supports D435/D435i/D405 and other librealsense-compatible devices.
    Can be constructed directly or from a YAML config file.
    """

    def __init__(
        self,
        name: str = "realsense",
        robot_name: str | None = None,
        serial_number: str | None = None,
        params: Mapping[str, Any] | None = None,
        streams: Mapping[str, CameraStreamConfig | Mapping[str, Any]] | None = None,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
    ) -> None:
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._align_depth_to_color = align_depth_to_color

        self._pipeline: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None
        self._align: rs.align | None = None
        self._device: rs.device | None = None
        self._color_sensor: rs.sensor | None = None
        self._depth_sensor: rs.sensor | None = None
        self._is_d405 = False
        self._frame_index = 0

        default_streams: dict[str, dict[str, Any]] = {
            "color": {
                "enabled": True,
                "width": 640,
                "height": 480,
                "fps": 30,
                "encoding": "bgr8",
            },
        }
        if enable_depth:
            default_streams["depth"] = {
                "enabled": True,
                "width": 640,
                "height": 480,
                "fps": 30,
                "encoding": "z16",
                "aligned_to": "color" if align_depth_to_color else None,
            }
        if streams:
            default_streams.update(streams)

        super().__init__(
            name=name,
            robot_name=robot_name,
            params=params,
            streams=default_streams,
            param_specs=_PARAM_SPECS,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> RealSenseCamera:
        """Create a RealSenseCamera from a YAML config file."""
        cfg = _load_yaml(config_path)

        stream_dicts: dict[str, dict[str, Any]] = {}
        for s_cfg in cfg.get("streams", []):
            s_name = s_cfg.pop("name", "color")
            stream_dicts[s_name] = s_cfg

        return cls(
            name=cfg.get("name", "realsense"),
            robot_name=cfg.get("robot_name"),
            serial_number=cfg.get("serial_number"),
            params=cfg.get("params"),
            streams=stream_dicts or None,
            enable_depth=cfg.get("enable_depth", True),
            align_depth_to_color=cfg.get("align_depth_to_color", True),
        )

    @property
    def serial_number(self) -> str | None:
        return self._serial_number

    @property
    def device_name(self) -> str | None:
        if self._device is None:
            return None
        return self._device.get_info(rs.camera_info.name)

    def _open_device(self) -> None:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError("No RealSense device detected.")

        if self._serial_number:
            found = False
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == self._serial_number:
                    self._device = dev
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"RealSense device with serial {self._serial_number} not found."
                )
        else:
            self._device = devices[0]
            self._serial_number = self._device.get_info(rs.camera_info.serial_number)

        dev_name = self._device.get_info(rs.camera_info.name)
        self._is_d405 = "D405" in dev_name

        # Detect sensors
        for sensor in self._device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor_name == "RGB Camera":
                self._color_sensor = sensor
            elif sensor_name == "Stereo Module":
                self._depth_sensor = sensor

        # D405: color is from stereo module, update param specs
        if self._is_d405:
            self._color_sensor = self._depth_sensor
            self._param_specs.clear()
            for spec in _D405_PARAM_SPECS.values():
                self.register_param_spec(spec)

        # Build pipeline config
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self._serial_number)

        color_cfg = self._streams.get("color")
        if color_cfg and color_cfg.enabled:
            if self._is_d405:
                rs_config.enable_stream(
                    rs.stream.color,
                    color_cfg.width, color_cfg.height,
                    rs.format.bgr8, int(color_cfg.fps),
                )
            else:
                rs_config.enable_stream(
                    rs.stream.color,
                    color_cfg.width, color_cfg.height,
                    rs.format.bgr8, int(color_cfg.fps),
                )

        depth_cfg = self._streams.get("depth")
        if depth_cfg and depth_cfg.enabled:
            rs_config.enable_stream(
                rs.stream.depth,
                depth_cfg.width, depth_cfg.height,
                rs.format.z16, int(depth_cfg.fps),
            )

        self._profile = self._pipeline.start(rs_config)

        if self._align_depth_to_color and self._enable_depth:
            self._align = rs.align(rs.stream.color)

        self._frame_index = 0

    def _close_device(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._profile = None
        self._align = None
        self._device = None
        self._color_sensor = None
        self._depth_sensor = None
        self._frame_index = 0

    def _get_sensor_for_option(self, key: str) -> rs.sensor | None:
        if self._is_d405:
            return self._depth_sensor
        target = _OPTION_SENSOR_MAP.get(key, "color")
        return self._color_sensor if target == "color" else self._depth_sensor

    def _apply_param_to_device(self, key: str, value: Any) -> None:
        rs_opt = _RS_OPTION_MAP.get(key)
        if rs_opt is None:
            return
        sensor = self._get_sensor_for_option(key)
        if sensor is None:
            return
        try:
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            sensor.set_option(rs_opt, float(value))
        except RuntimeError:
            pass  # option not supported by this device

    def _read_param_from_device(self, key: str, default: Any = None) -> Any:
        rs_opt = _RS_OPTION_MAP.get(key)
        if rs_opt is None:
            return self._params.get(key, default)
        sensor = self._get_sensor_for_option(key)
        if sensor is None:
            return self._params.get(key, default)
        try:
            val = sensor.get_option(rs_opt)
            if key.endswith(".auto"):
                return bool(val)
            return val
        except RuntimeError:
            return self._params.get(key, default)

    def read_frame(self) -> SensorFrame:
        self._ensure_open()
        assert self._pipeline is not None

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        self._frame_index += 1
        streams: dict[str, dict[str, Any]] = {}

        color_frame = frames.get_color_frame()
        if color_frame:
            color_data = np.asanyarray(color_frame.get_data())
            streams["color"] = {
                "data": color_data,
                "encoding": "bgr8",
                "width": color_frame.get_width(),
                "height": color_frame.get_height(),
            }

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_data = np.asanyarray(depth_frame.get_data())
            w, h = depth_frame.get_width(), depth_frame.get_height()
            center_dist = depth_frame.get_distance(w // 2, h // 2)
            streams["depth"] = {
                "data": depth_data,
                "encoding": "z16",
                "width": w,
                "height": h,
                "unit": "meter",
                "distance_estimate": center_dist,
            }

        obstacle_distance = (
            streams["depth"]["distance_estimate"]
            if "depth" in streams
            else None
        )

        return self.build_frame(
            frame_id=self._frame_index,
            streams=streams,
            metadata={"obstacle_distance": obstacle_distance},
        )


class MultiRealSenseManager:
    """Manage multiple RealSense cameras: open, read, visualize, close."""

    def __init__(
        self,
        cameras: Sequence[RealSenseCamera] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self._cameras: list[RealSenseCamera] = []
        self._running = False

        if config_path is not None:
            self._cameras = self._from_config(config_path)
        if cameras:
            self._cameras.extend(cameras)

    @staticmethod
    def _from_config(config_path: str | Path) -> list[RealSenseCamera]:
        cfg = _load_yaml(config_path)
        cam_list: list[RealSenseCamera] = []
        for cam_cfg in cfg.get("cameras", []):
            stream_dicts: dict[str, dict[str, Any]] = {}
            for s_cfg in cam_cfg.get("streams", []):
                s_name = s_cfg.pop("name", "color")
                stream_dicts[s_name] = s_cfg

            cam = RealSenseCamera(
                name=cam_cfg.get("name", "realsense"),
                robot_name=cam_cfg.get("robot_name"),
                serial_number=cam_cfg.get("serial_number"),
                params=cam_cfg.get("params"),
                streams=stream_dicts or None,
                enable_depth=cam_cfg.get("enable_depth", True),
                align_depth_to_color=cam_cfg.get("align_depth_to_color", True),
            )
            cam_list.append(cam)
        return cam_list

    @staticmethod
    def discover() -> list[dict[str, str]]:
        """List all connected RealSense devices."""
        ctx = rs.context()
        result = []
        for dev in ctx.query_devices():
            result.append({
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            })
        return result

    @classmethod
    def from_all_connected(
        cls,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        params: Mapping[str, Any] | None = None,
    ) -> MultiRealSenseManager:
        """Auto-detect all connected RealSense cameras and create a manager."""
        devices = cls.discover()
        if not devices:
            raise RuntimeError("No RealSense devices detected.")

        cameras = []
        for i, dev_info in enumerate(devices):
            cam = RealSenseCamera(
                name=f"realsense_{i}",
                serial_number=dev_info["serial_number"],
                params=params,
                enable_depth=enable_depth,
                align_depth_to_color=align_depth_to_color,
            )
            cameras.append(cam)
        return cls(cameras=cameras)

    @property
    def cameras(self) -> list[RealSenseCamera]:
        return list(self._cameras)

    def add_camera(self, camera: RealSenseCamera) -> None:
        self._cameras.append(camera)

    def open_all(self) -> None:
        for cam in self._cameras:
            cam.open()

    def close_all(self) -> None:
        for cam in self._cameras:
            cam.close()

    def read_all(self) -> dict[str, SensorFrame]:
        """Read one frame from each camera."""
        return {cam.name: cam.read_frame() for cam in self._cameras if cam.is_open}

    def visualize(self, show_depth: bool = True) -> None:
        """Open all cameras and display live feeds in OpenCV windows.

        Press 'q' or Esc to exit.
        """
        import cv2

        self.open_all()
        self._running = True

        try:
            while self._running:
                for cam in self._cameras:
                    if not cam.is_open:
                        continue
                    frame = cam.read_frame()
                    streams = frame.payload.get("streams", {})

                    color_stream = streams.get("color")
                    if color_stream is not None:
                        color_img = color_stream["data"]
                        title = f"{cam.name} ({cam.serial_number})"
                        cv2.imshow(f"{title} - Color", color_img)

                    depth_stream = streams.get("depth")
                    if show_depth and depth_stream is not None:
                        depth_img = depth_stream["data"]
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_img, alpha=0.03),
                            cv2.COLORMAP_JET,
                        )
                        dist = depth_stream.get("distance_estimate", 0)
                        cv2.putText(
                            depth_colormap,
                            f"{dist:.2f}m",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow(f"{title} - Depth", depth_colormap)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.close_all()
            cv2.destroyAllWindows()

    def stop(self) -> None:
        self._running = False

    def __enter__(self) -> MultiRealSenseManager:
        self.open_all()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close_all()
