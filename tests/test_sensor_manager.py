from __future__ import annotations

from typing import Any

import numpy as np

from Core import SensorFrame
from Sensor.manager import SensorManager
from Sensor.rgb_camera.base import BaseRGBCamera
from Sensor.tactile.base import BaseTactileSensor


class _FakeCamera(BaseRGBCamera):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _open_device(self) -> None:
        pass

    def _close_device(self) -> None:
        pass

    def _grab_streams(self) -> dict[str, Any]:
        return {
            "color": {
                "data": np.zeros((2, 2, 3), dtype=np.uint8),
            },
        }


class _FakeTactile(BaseTactileSensor):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _open_device(self) -> None:
        pass

    def _close_device(self) -> None:
        pass

    def _grab_frame(self) -> np.ndarray:
        return np.ones((2, 2, 3), dtype=np.uint8)


def test_sensor_manager_from_config_uses_enabled_image_sensor_whitelist(monkeypatch):
    created: list[str] = []

    def fake_camera_from_config(cls, name, cfg):
        created.append(name)
        return _FakeCamera(name)

    def fake_tactile_from_config(cls, name, cfg):
        created.append(name)
        return _FakeTactile(name)

    monkeypatch.setattr(BaseRGBCamera, "from_config", classmethod(fake_camera_from_config))
    monkeypatch.setattr(BaseTactileSensor, "from_config", classmethod(fake_tactile_from_config))

    manager = SensorManager.from_config(
        {
            "robot": {"type": "aloha"},
            "cameras": {
                "front": {"type": "realsense"},
                "wrist": {"type": "realsense"},
            },
            "tactile": {
                "left_touch": {"type": "opencv"},
                "right_touch": {"type": "opencv"},
            },
        },
        enabled_names=["front", "left_touch"],
    )

    assert sorted(manager.sensors) == ["front", "left_touch"]
    assert created == ["front", "left_touch"]

    manager.open_all()
    images = manager.read_images()
    assert sorted(images) == ["front", "left_touch"]


def test_sensor_manager_from_config_rejects_unknown_enabled_image_sensor():
    try:
        SensorManager.from_config(
            {
                "robot": {"type": "aloha"},
                "cameras": {"front": {"type": "realsense"}},
            },
            enabled_names=["missing"],
        )
    except ValueError as exc:
        assert "enabled_cameras contains unknown image sensor keys" in str(exc)
    else:
        raise AssertionError("expected unknown enabled image sensor to be rejected")
