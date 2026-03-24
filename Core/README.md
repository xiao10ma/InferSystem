# Core

Shared contracts live here so `Sensor`, `Robot`, `Inference`, and `SDK`
depend on the same data model.

Additions that usually belong here:

- Pose, velocity, map, and task data structures
- Command and status enums
- Exceptions and shared validation helpers
- Event envelopes for logging or message buses

