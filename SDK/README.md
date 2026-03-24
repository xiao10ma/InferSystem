# SDK

This folder owns the runtime and the public API that external services or apps
use to drive the system.

The current scaffold includes:

- `RobotRuntime`: single robot control loop
- `FleetManager`: multi-robot orchestration wrapper

Likely additions:

- REST or gRPC gateway
- Command queue or message bus adapter
- Event logging and replay
- Health monitoring and watchdogs
- Configuration loader and dependency injection

