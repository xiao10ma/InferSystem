from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

from Core import SensorFrame, load_yaml
from Sensor.rgb_camera.base import BaseRGBCamera


# 统一参数名 → pyrealsense2 option enum
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

# 每个参数属于哪个硬件 sensor: "color" = RGB Camera, "depth" = Stereo Module
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


@BaseRGBCamera.register("realsense")
class RealSenseCamera(BaseRGBCamera):
    """Intel RealSense RGB-D 相机驱动。

    支持 D435/D435i/D405 等 librealsense 兼容设备。

    用法::

        cam = RealSenseCamera("main_cam", serial_number="409122273675")
        with cam:
            frame = cam.read_frame()
            color = frame.payload["streams"]["color"]["data"]  # numpy BGR
    """

    def __init__(
        self,
        name: str = "realsense",
        *,
        serial_number: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        depth_width: int | None = None,
        depth_height: int | None = None,
        depth_fps: int | None = None,
        params: dict[str, Any] | None = None,
        robot_name: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            width=width,
            height=height,
            fps=fps,
            params=params,
            robot_name=robot_name,
        )
        self._serial_number = serial_number
        self._enable_depth = enable_depth
        self._align_depth_to_color = align_depth_to_color
        self._depth_width = depth_width or width
        self._depth_height = depth_height or height
        self._depth_fps = depth_fps or fps

        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._device: rs.device | None = None
        self._color_sensor: rs.sensor | None = None
        self._depth_sensor: rs.sensor | None = None
        self._is_d405 = False

    @classmethod
    def _from_config_dict(
        cls,
        name: str,
        cfg: dict[str, Any],
        robot_name: str | None = None,
    ) -> RealSenseCamera:
        streams = cfg.get("streams", {})
        color_cfg = streams.get("color", {})
        depth_cfg = streams.get("depth", {})
        return cls(
            name=name,
            serial_number=cfg.get("serial_number"),
            width=color_cfg.get("width", 640),
            height=color_cfg.get("height", 480),
            fps=color_cfg.get("fps", 30),
            enable_depth=cfg.get("enable_depth", True),
            align_depth_to_color=cfg.get("align_depth_to_color", True),
            depth_width=depth_cfg.get("width"),
            depth_height=depth_cfg.get("height"),
            depth_fps=depth_cfg.get("fps"),
            params=cfg.get("params"),
            robot_name=robot_name,
        )

    @property
    def serial_number(self) -> str | None:
        return self._serial_number

    @property
    def device_name(self) -> str | None:
        if self._device is None:
            return None
        return self._device.get_info(rs.camera_info.name)

    @property
    def enable_depth(self) -> bool:
        return self._enable_depth

    def _open_device(self) -> None:
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError("未检测到 RealSense 设备。")

        # 按序列号查找设备
        if self._serial_number:
            self._device = None
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == self._serial_number:
                    self._device = dev
                    break
            if self._device is None:
                raise RuntimeError(
                    f"未找到序列号为 {self._serial_number} 的 RealSense 设备。"
                )
        else:
            self._device = devices[0]
            self._serial_number = self._device.get_info(rs.camera_info.serial_number)

        # 检测型号和 sensor
        dev_name = self._device.get_info(rs.camera_info.name)
        self._is_d405 = "D405" in dev_name

        for sensor in self._device.sensors:
            sensor_name = sensor.get_info(rs.camera_info.name)
            if sensor_name == "RGB Camera":
                self._color_sensor = sensor
            elif sensor_name == "Stereo Module":
                self._depth_sensor = sensor

        # D405: color 来自 stereo module
        if self._is_d405:
            self._color_sensor = self._depth_sensor

        # 配置 pipeline
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self._serial_number)

        rs_config.enable_stream(
            rs.stream.color,
            self.width, self.height,
            rs.format.bgr8, self.fps,
        )

        if self._enable_depth:
            rs_config.enable_stream(
                rs.stream.depth,
                self._depth_width, self._depth_height,
                rs.format.z16, self._depth_fps,
            )

        self._pipeline.start(rs_config)

        if self._enable_depth and self._align_depth_to_color:
            self._align = rs.align(rs.stream.color)

    def _close_device(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None
        self._device = None
        self._color_sensor = None
        self._depth_sensor = None

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

    def _grab_streams(self) -> dict[str, dict[str, Any]]:
        assert self._pipeline is not None

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        streams: dict[str, dict[str, Any]] = {}

        color_frame = frames.get_color_frame()
        if color_frame:
            raw = np.asanyarray(color_frame.get_data())
            streams["color"] = {
                "data": raw if raw.dtype == np.uint8 else raw.astype(np.uint8),
                "encoding": "bgr8",
                "width": color_frame.get_width(),
                "height": color_frame.get_height(),
            }

        depth_frame = frames.get_depth_frame()
        if depth_frame:
            w, h = depth_frame.get_width(), depth_frame.get_height()
            streams["depth"] = {
                "data": np.asanyarray(depth_frame.get_data()),
                "encoding": "z16",
                "width": w,
                "height": h,
                "unit": "meter",
                "center_distance": depth_frame.get_distance(w // 2, h // 2),
            }

        return streams


class MultiRealSenseManager:
    """管理多台 RealSense 相机: 批量打开、读取、可视化、关闭。"""

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
        cfg = load_yaml(config_path)
        cameras_cfg = cfg.get("cameras", {})

        # 支持 dict (name -> config) 和 list 两种格式
        if isinstance(cameras_cfg, dict):
            items = cameras_cfg.items()
        else:
            items = ((c.get("name", f"cam_{i}"), c) for i, c in enumerate(cameras_cfg))

        cam_list: list[RealSenseCamera] = []
        for cam_name, cam_cfg in items:
            if cam_cfg.get("type", "realsense") != "realsense":
                continue
            cam = RealSenseCamera._from_config_dict(cam_name, cam_cfg)
            cam_list.append(cam)
        return cam_list

    @staticmethod
    def discover() -> list[dict[str, str]]:
        """列出所有已连接的 RealSense 设备。"""
        ctx = rs.context()
        return [
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial_number": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
            for dev in ctx.query_devices()
        ]

    @classmethod
    def from_all_connected(
        cls,
        enable_depth: bool = True,
        align_depth_to_color: bool = True,
        params: dict[str, Any] | None = None,
    ) -> MultiRealSenseManager:
        """自动检测所有已连接的 RealSense 相机并创建管理器。"""
        devices = cls.discover()
        if not devices:
            raise RuntimeError("未检测到 RealSense 设备。")

        cameras = [
            RealSenseCamera(
                name=f"realsense_{i}",
                serial_number=dev["serial_number"],
                enable_depth=enable_depth,
                align_depth_to_color=align_depth_to_color,
                params=params,
            )
            for i, dev in enumerate(devices)
        ]
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
        """从每台相机读取一帧。"""
        return {cam.name: cam.read_frame() for cam in self._cameras if cam.is_open}

    def visualize(self, show_depth: bool = True) -> None:
        """打开所有相机并在 OpenCV 窗口中显示实时画面。

        按 'q' 或 Esc 退出。
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
                    title = f"{cam.name} ({cam.serial_number})"

                    color_stream = streams.get("color")
                    if color_stream is not None:
                        cv2.imshow(f"{title} - Color", color_stream["data"])

                    depth_stream = streams.get("depth")
                    if show_depth and depth_stream is not None:
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_stream["data"], alpha=0.03),
                            cv2.COLORMAP_JET,
                        )
                        dist = depth_stream.get("center_distance", 0)
                        cv2.putText(
                            depth_colormap, f"{dist:.2f}m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
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
