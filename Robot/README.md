# Robot

This folder owns robot model abstraction and vendor-specific robot adapters.

Current layout:

- `base.py`: shared robot facade, robot state reader abstraction, controller abstraction, and control-mode metadata.
- `flexiv.py`: Flexiv robot facade plus `FlexivStateReader` for raw state/data snapshots.
- `flexiv_control.py`: standalone `FlexivController` with multiple control profiles and per-profile runtime parameters.
- `mock_mobile.py`: lightweight test robot used by runtime examples and unit tests.

Flexiv control is now split into two independent concerns:

- data reading: `FlexivStateReader.read_state()` and `FlexivStateReader.read_data()`
- command/control: `FlexivController`

`FlexivController` supports multiple control modes with isolated parameters:

- `maintenance`
- `primitive_execution`
- `plan_execution`
- `joint_position`
- `cartesian_motion_force`

Each mode keeps its own parameter dictionary and can be configured independently
through `configure_control_mode()` / `set_control_mode()` before commands are sent.

Recommended growth path:

- `mobile/`: wheeled or tracked platforms
- `arm/`: manipulators and end effectors
- `drone/`: aerial robot wrappers
- `humanoid/`: joint groups, gait, and balance adapters
- `sim/`: simulation-only robots for development

Each concrete robot should implement the same command interface so the control
loop can switch robot types without rewriting inference logic.
