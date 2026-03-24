# Robot

This folder owns robot model abstraction and vendor-specific robot adapters.

Recommended growth path:

- `mobile/`: wheeled or tracked platforms
- `arm/`: manipulators and end effectors
- `drone/`: aerial robot wrappers
- `humanoid/`: joint groups, gait, and balance adapters
- `sim/`: simulation-only robots for development

Each concrete robot should implement the same command interface so the control
loop can switch robot types without rewriting inference logic.

