# Inference

This folder owns decision logic that maps perception plus robot state into
robot commands.

Typical submodules to add later:

- Goal planner
- Behavior tree or task graph
- Model inference wrappers
- Multi-robot dispatcher
- Safety policy and fallback policy

Keep the interface stable: input is normalized frames plus state, output is a
shared command object.

