from .base import BaseRGBCamera
from .mock_camera import MockRGBCamera

__all__ = [
    "BaseRGBCamera",
    "MockRGBCamera",
]

# RealSense 驱动按需导入 (需要 pyrealsense2)
try:
    from .realsense_camera import MultiRealSenseManager, RealSenseCamera

    __all__ += ["MultiRealSenseManager", "RealSenseCamera"]
except ImportError:
    pass
