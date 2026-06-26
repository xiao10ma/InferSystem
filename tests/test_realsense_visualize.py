from __future__ import annotations

import importlib

import numpy as np

from Core import SensorFrame


def _write_config(tmp_path, enabled_key: str = "enabled_cameras"):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
cameras:
  front:
    type: realsense
    serial_number: "123"
  wrist:
    type: realsense
    serial_number: "456"
tactile:
  touch:
    type: opencv
    device_path: "/dev/video9"
inference:
  {enabled_key}: [front, touch]
""",
        encoding="utf-8",
    )
    return path


def test_module_imports_without_realsense_sdk():
    module = importlib.import_module("Example.realsense_visualize")

    assert module is not None


def test_sensors_from_config_uses_enabled_cameras(monkeypatch, tmp_path):
    module = importlib.import_module("Example.realsense_visualize")
    calls = []

    class FakeSensorManager:
        @classmethod
        def from_config(cls, config, *, strict=True, enabled_names=None):
            calls.append((config, strict, enabled_names))
            return "manager"

    monkeypatch.setattr(module, "SensorManager", FakeSensorManager)

    manager = module.sensors_from_config(_write_config(tmp_path))

    assert manager == "manager"
    assert calls[0][2] == ["front", "touch"]


def test_sensors_from_config_supports_legacy_enable_cameras_key(monkeypatch, tmp_path):
    module = importlib.import_module("Example.realsense_visualize")
    calls = []

    class FakeSensorManager:
        @classmethod
        def from_config(cls, config, *, strict=True, enabled_names=None):
            calls.append(enabled_names)
            return "manager"

    monkeypatch.setattr(module, "SensorManager", FakeSensorManager)

    module.sensors_from_config(_write_config(tmp_path, enabled_key="enable_cameras"))

    assert calls == [["front", "touch"]]


def test_sensors_from_config_all_ignores_enabled_cameras(monkeypatch, tmp_path):
    module = importlib.import_module("Example.realsense_visualize")
    calls = []

    class FakeSensorManager:
        @classmethod
        def from_config(cls, config, *, strict=True, enabled_names=None):
            calls.append(enabled_names)
            return "manager"

    monkeypatch.setattr(module, "SensorManager", FakeSensorManager)

    module.sensors_from_config(_write_config(tmp_path), all_sensors=True)

    assert calls == [None]


def test_display_streams_include_tactile_image():
    module = importlib.import_module("Example.realsense_visualize")
    tactile_img = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = SensorFrame(
        sensor_name="touch",
        sensor_type="tactile",
        payload={"streams": {"tactile": {"data": tactile_img}}},
    )

    displays = module.display_streams_from_frame(frame, show_depth=True)

    assert displays == [("touch - Tactile", tactile_img)]
