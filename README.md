# InferSystem

Scaffold for a multi-robot inference and control codebase.

## Recommended layout

```text
InferSystem/
├── Core/        # Shared data types and cross-module contracts
├── Sensor/      # Sensor abstraction and sensor drivers
├── Robot/       # Robot abstraction and concrete robot adapters
├── Inference/   # Decision, planning, and reasoning engines
├── SDK/         # Runtime orchestration and external-facing APIs
├── Example/     # Minimal examples and integration demos
├── Config/      # Declarative robot/sensor/runtime configuration
└── tests/       # Unit tests for core flows
```

## Module roles

- `Sensor`: normalizes raw hardware input into a common frame format.
- `Robot`: adapts different robot models behind the same command interface.
- `Inference`: converts sensor frames plus robot state into commands.
- `SDK`: wires sensors, robots, and inference into a runnable control loop.
- `Example`: shows how to build a single-robot or multi-robot runtime.
- `Core`: prevents shared types from being duplicated across modules.
- `Config`: stores robot, sensor, and runtime templates for deployment.
- `tests`: verifies behavior before hardware integration.

## Data flow

1. `Sensor` reads hardware and emits `SensorFrame`.
2. `Robot` exposes `RobotState`.
3. `Inference` decides the next `InferenceCommand`.
4. `SDK` dispatches the command back to the robot.
5. `Example` demonstrates end-to-end composition.

## What you still need next

- Real robot adapters for each vendor or hardware protocol.
- Real sensor drivers for camera, lidar, imu, gps, encoder, and custom IO.
- Task planning modules if commands depend on goals or missions.
- Config loading if runtime composition should come from YAML or JSON.
- Logging, telemetry, and replay if you need debugging on real devices.
- Safety guards such as emergency stop, command rate limit, and health checks.

## Quick start

Run the examples from the repository root:

```bash
python Example/simple_runtime.py
python Example/fleet_demo.py
python -m unittest discover -s tests
```

