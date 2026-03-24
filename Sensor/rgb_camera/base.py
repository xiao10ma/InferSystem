from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from Core import SensorFrame, utc_now
from Sensor.base import BaseSensor


@dataclass(slots=True, frozen=True)
class CameraParamSpec:
    key: str
    default: Any = None
    readable: bool = True
    writable: bool = True
    value_types: tuple[type[Any], ...] = ()
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    requires_restart: bool = False
    description: str = ""


@dataclass(slots=True)
class CameraStreamConfig:
    name: str
    enabled: bool = True
    width: int = 640
    height: int = 480
    fps: float = 30.0
    encoding: str = "rgb8"
    aligned_to: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> CameraStreamConfig:
        known_keys = {
            "name",
            "enabled",
            "width",
            "height",
            "fps",
            "encoding",
            "aligned_to",
        }
        known_args = {key: data[key] for key in known_keys if key in data and key != "name"}
        extra = {key: value for key, value in data.items() if key not in known_keys}
        return cls(name=name, extra=extra, **known_args)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "encoding": self.encoding,
            "aligned_to": self.aligned_to,
            **deepcopy(self.extra),
        }


class BaseRGBCamera(BaseSensor, ABC):
    def __init__(
        self,
        name: str,
        robot_name: str | None = None,
        params: Mapping[str, Any] | None = None,
        streams: Mapping[str, CameraStreamConfig | Mapping[str, Any]] | None = None,
        param_specs: Mapping[str, CameraParamSpec] | None = None,
    ) -> None:
        super().__init__(name=name, sensor_type="rgb_camera", robot_name=robot_name)
        self._is_open = False
        self._params: dict[str, Any] = {}
        self._param_specs: dict[str, CameraParamSpec] = {}
        self._pending_restart_params: set[str] = set()
        self._streams: dict[str, CameraStreamConfig] = {}

        for spec in (param_specs or {}).values():
            self.register_param_spec(spec)

        initial_streams = streams or {"color": CameraStreamConfig(name="color")}
        for stream_name, stream_config in initial_streams.items():
            self.register_stream(stream_name, stream_config)

        if "color" not in self._streams:
            self.register_stream("color", CameraStreamConfig(name="color"))

        if params:
            self.set_params(params)

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def supported_streams(self) -> tuple[str, ...]:
        return tuple(self._streams.keys())

    @property
    def supports_depth(self) -> bool:
        return "depth" in self._streams

    def open(self) -> None:
        if self._is_open:
            return

        self._open_device()
        self._is_open = True

        try:
            self._apply_cached_configuration()
        except Exception:
            try:
                self._close_device()
            finally:
                self._is_open = False
            raise

    def close(self) -> None:
        if not self._is_open:
            return

        try:
            self._close_device()
        finally:
            self._is_open = False

    def restart(self) -> None:
        if self._is_open:
            self.close()
        self.open()

    def register_param_spec(self, spec: CameraParamSpec) -> None:
        self._param_specs[spec.key] = spec
        if spec.key not in self._params and spec.default is not None:
            self._params[spec.key] = deepcopy(spec.default)

    def get_param_spec(self, key: str) -> CameraParamSpec | None:
        return self._param_specs.get(key)

    def get_param_specs(self) -> dict[str, CameraParamSpec]:
        return dict(self._param_specs)

    def get_pending_restart_params(self) -> set[str]:
        return set(self._pending_restart_params)

    def register_stream(
        self,
        stream_name: str,
        config: CameraStreamConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        normalized = self._normalize_stream_config(stream_name, config, **kwargs)
        self._validate_stream_config(normalized)
        self._streams[stream_name] = normalized

        if self._is_open:
            self._apply_stream_config_to_device(stream_name, normalized)

    def configure_stream(
        self,
        stream_name: str,
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if stream_name not in self._streams:
            raise KeyError(f"Unknown stream: {stream_name}")

        updates = dict(config or {})
        updates.update(kwargs)
        current = self._streams[stream_name]
        merged = CameraStreamConfig.from_mapping(
            stream_name,
            {
                **current.as_dict(),
                **updates,
            },
        )
        self._validate_stream_config(merged)
        self._streams[stream_name] = merged

        if self._is_open:
            self._apply_stream_config_to_device(stream_name, merged)

    def get_stream_config(self, stream_name: str) -> dict[str, Any]:
        if stream_name not in self._streams:
            raise KeyError(f"Unknown stream: {stream_name}")
        return self._streams[stream_name].as_dict()

    def get_stream_configs(self) -> dict[str, dict[str, Any]]:
        return {
            stream_name: stream_config.as_dict()
            for stream_name, stream_config in self._streams.items()
        }

    # 参数设置建议:
    # 1. `set_param()` 管设备控制参数, 例如曝光、增益、白平衡、触发模式。
    # 2. `configure_stream()` 管视频流参数, 例如分辨率、帧率、编码格式。
    # 3. 推荐使用“功能域.字段”命名, 便于兼容不同厂商 SDK:
    #    camera.set_param("exposure.auto", False)
    #    camera.set_param("exposure.value", 1200)
    #    camera.set_param("white_balance.auto", True)
    # 4. 如果是 RGB-D 相机, 把 depth 当成独立流配置, 不要混进 RGB 参数:
    #    camera.configure_stream("color", width=1280, height=720, fps=30, encoding="rgb8")
    #    camera.configure_stream("depth", enabled=True, width=640, height=480, fps=30, encoding="z16", aligned_to="color")
    def set_param(self, key: str, value: Any) -> None:
        spec = self._param_specs.get(key)
        if spec is not None:
            if not spec.writable:
                raise PermissionError(f"Camera parameter is not writable: {key}")
            self._validate_param_value(spec, value)

        self._params[key] = deepcopy(value)

        if not self._is_open:
            return

        if spec is not None and spec.requires_restart:
            self._pending_restart_params.add(key)
            return

        self._apply_param_to_device(key, value)
        self._pending_restart_params.discard(key)

    def set_params(self, params: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        merged = dict(params or {})
        merged.update(kwargs)
        for key, value in merged.items():
            self.set_param(key, value)

    def get_param(self, key: str, default: Any = None, refresh_from_device: bool = False) -> Any:
        spec = self._param_specs.get(key)
        if spec is not None and not spec.readable:
            raise PermissionError(f"Camera parameter is not readable: {key}")

        if refresh_from_device and self._is_open:
            value = self._read_param_from_device(key, default)
            self._params[key] = deepcopy(value)

        return deepcopy(self._params.get(key, default))

    def get_params(self, refresh_from_device: bool = False) -> dict[str, Any]:
        if refresh_from_device and self._is_open:
            known_keys = set(self._params) | set(self._param_specs)
            for key in known_keys:
                spec = self._param_specs.get(key)
                if spec is not None and not spec.readable:
                    continue
                self._params[key] = deepcopy(
                    self._read_param_from_device(key, self._params.get(key))
                )

        return deepcopy(self._params)

    def read(self) -> SensorFrame:
        return self.read_frame()

    def build_frame(
        self,
        *,
        frame_id: int | None = None,
        streams: Mapping[str, Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SensorFrame:
        payload = dict(metadata or {})
        payload["streams"] = deepcopy(dict(streams or {}))
        payload["stream_configs"] = self.get_stream_configs()

        # 推荐的多流结构:
        # payload["streams"]["color"] = {"data": ..., "encoding": "rgb8"}
        # payload["streams"]["depth"] = {"data": ..., "encoding": "z16", "unit": "mm"}
        # 这样上层推理代码可以在一帧里同时拿到 RGB 和 depth, 也便于扩展 IR/stereo。
        if frame_id is not None:
            payload["frame_id"] = frame_id

        return SensorFrame(
            sensor_name=self.name,
            sensor_type=self.sensor_type,
            timestamp=utc_now(),
            robot_name=self.robot_name,
            payload=payload,
        )

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError(f"RGB camera is not open: {self.name}")

    def _apply_cached_configuration(self) -> None:
        for stream_name, stream_config in self._streams.items():
            self._apply_stream_config_to_device(stream_name, stream_config)

        for key, value in self._params.items():
            self._apply_param_to_device(key, value)
            self._pending_restart_params.discard(key)

    def _normalize_stream_config(
        self,
        stream_name: str,
        config: CameraStreamConfig | Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> CameraStreamConfig:
        if config is None:
            normalized = CameraStreamConfig(name=stream_name)
        elif isinstance(config, CameraStreamConfig):
            normalized = deepcopy(config)
            normalized.name = stream_name
        else:
            normalized = CameraStreamConfig.from_mapping(stream_name, config)

        if kwargs:
            normalized = CameraStreamConfig.from_mapping(
                stream_name,
                {
                    **normalized.as_dict(),
                    **kwargs,
                },
            )

        return normalized

    def _validate_stream_config(self, config: CameraStreamConfig) -> None:
        if config.width <= 0:
            raise ValueError(f"Stream width must be positive: {config.width}")
        if config.height <= 0:
            raise ValueError(f"Stream height must be positive: {config.height}")
        if config.fps <= 0:
            raise ValueError(f"Stream fps must be positive: {config.fps}")

    def _validate_param_value(self, spec: CameraParamSpec, value: Any) -> None:
        if spec.value_types and not isinstance(value, spec.value_types):
            expected_types = ", ".join(type_.__name__ for type_ in spec.value_types)
            raise TypeError(f"Camera parameter {spec.key} must be one of: {expected_types}")

        if spec.choices and value not in spec.choices:
            choices = ", ".join(repr(choice) for choice in spec.choices)
            raise ValueError(f"Camera parameter {spec.key} must be one of: {choices}")

        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"Camera parameter {spec.key} must be >= {spec.minimum}")

        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"Camera parameter {spec.key} must be <= {spec.maximum}")

    @abstractmethod
    def _open_device(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _close_device(self) -> None:
        raise NotImplementedError

    def _apply_param_to_device(self, key: str, value: Any) -> None:
        return None

    def _read_param_from_device(self, key: str, default: Any = None) -> Any:
        return self._params.get(key, default)

    def _apply_stream_config_to_device(
        self,
        stream_name: str,
        config: CameraStreamConfig,
    ) -> None:
        return None

    @abstractmethod
    def read_frame(self) -> SensorFrame:
        raise NotImplementedError
