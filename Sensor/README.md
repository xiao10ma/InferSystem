# Sensor

This folder owns sensor abstraction and hardware-specific drivers.

Current layout:

- `base.py`: shared top-level sensor contract
- `rgb_camera/`: rgb camera base class and concrete implementations
- `tactile/`: tactile sensor placeholder area
- `lidar/`: lidar placeholder area

Every driver should output a shared `SensorFrame` so the inference layer does
not care which vendor or transport produced the data.

Recommended next additions:

- `imu/`: inertial sensors
- `localization/`: gps, encoder, slam outputs
- `fusion/`: sensor fusion and normalization

