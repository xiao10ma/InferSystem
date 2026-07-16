#!/usr/bin/env bash
set -euo pipefail

GRIPPER_SCALE="${GRIPPER_SCALE:-10}"
CLAMP_GRIP="${CLAMP_GRIP:-0}"

python Example/robot_inference.py Config/flexiv.yaml \
    --prompt "pick up the cord from the base and then plug into the outlet" \
    --max-steps 0 \
    --n-execute 48 \
