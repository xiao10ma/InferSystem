# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InferSystem is a multi-robot inference control system implementing a **Perception → Inference → Execution** loop. It supports multiple robot arms and two-finger grippers through unified abstractions, with remote model inference over ZMQ.

## Setup & Commands

```bash
# Environment setup
conda create -n infersystem python=3.10 -y && conda activate infersystem
pip install -r requirements.txt

# Robot-specific SDKs (pick one)
pip install flexivrdk spdlog          # Flexiv Rizon
# Arx5: compile from third_party/arx5-sdk (see README.md)

# Optional sensors
pip install pyrealsense2              # RealSense cameras
pip install pyarrow                   # Trajectory replay

# Run tests
pytest                                # testpaths = ["tests"]

# Run examples (all take YAML config as first arg)
python Example/robot_inference.py Config/rizon4_example.yaml --prompt "pick up the cup"
python Example/flexiv/probe.py Config/rizon4_example.yaml --polls 3 --enable
python Example/flexiv/go_home.py Config/rizon4_example.yaml --velocity 50
```

## Architecture

### Core Primitives (Core/)

Three fundamental operations drive everything:
- **observe()** → atomic state snapshot (one hardware read, all fields time-coherent)
- **act(Action)** → single non-blocking control step
- **Lifecycle** → connect/disconnect/enable/stop

Key types in `Core/types.py`: `ArmState`, `Action` (with `ActionSpace` enum), `Observation`, `SensorFrame`, `RobotParams`. Units: radians for angles (YAML uses degrees, auto-converted), meters for distances, N/Nm for forces.

### Registry Pattern (Core/registry.py)

All hardware drivers use a `Registrable` mixin for zero-coupling plugin registration:
```python
@BaseRobot.register("flexiv")
class FlexivRobot(BaseRobot): ...

robot = BaseRobot.from_config("config.yaml")  # dispatches by type field
```

Each registrable class implements `_from_config_dict(cls, name, cfg, **kw)`. The same pattern applies to `BaseRGBCamera`, `BaseTactileSensor`, etc.

### Robot Layer (Robot/)

- `BaseRobot` (base.py): abstract `observe()`/`act()` + lifecycle + task-level convenience methods (`move_joint_position`, `go_home`, etc.) with input validation against `RobotParams`
- `BaseGripper` (gripper.py): independent lifecycle, `observe() → GripperState`, `set(open: bool)`, `move(width)`
- Drivers: `FlexivRobot` ("flexiv"), `Arx5Robot` ("arx5"), `Arx5BimanualRobot` ("arx5_bimanual" — unified 14-DOF interface over two CAN buses)
- Grippers are separate objects created via `robot.create_gripper(config)`

### Sensor Layer (Sensor/)

- `BaseSensor` → `BaseRGBCamera` → RealSenseCamera, MockRGBCamera
- `BaseSensor` → `BaseTactileSensor` → OpenCVTactileSensor
- `SensorManager` (manager.py): orchestrates all sensors from config, provides `read_all()` and `read_images()`

### Inference Layer (Inference/)

- `InferenceClient`: ZMQ REQ socket, msgpack protocol. Two modes: `predict_chunk()` (batch) or `get_action()` (streaming with internal queue). Has automatic retry/reconnect.
- `InferenceServer`: ZMQ REP socket, deployed on GPU machine. Override `predict(images, state)`.
- `ActionDispatcher` (dispatch.py): splits action vectors into arm portion → `robot.act()` and gripper portion → `gripper.set()` (binary threshold, stateful — only commands on state change)
- `build_state_vector()`: assembles `[joint_positions..., gripper_normalized]`

### Configuration

One YAML file configures the entire system (robot + sensors + inference). Sections: `robot:` (type, serial, gripper, control, joint_limits), `cameras:`, `tactile:`, `inference:` (server address, fps, enabled_cameras, gripper_index/threshold).

## Adding a New Robot

1. Subclass `BaseRobot`, implement `observe()`, `act()`, and lifecycle methods
2. Decorate with `@BaseRobot.register("your_type")`
3. Implement `_from_config_dict(cls, name, cfg)` for YAML factory
4. Override `create_gripper()` if the robot has a gripper
5. Add a YAML config template in `Config/`

## Key Design Decisions

- Grippers are physically independent devices with their own lifecycle — not robot sub-components
- `_rt_sleep_until()` in BaseRobot uses hybrid sleep (yield CPU if >2ms, then busy-wait) for timing precision
- The inference server has zero InferSystem dependencies — can be deployed standalone on GPU machines
- All examples use YAML config as the entry point (first CLI argument)
