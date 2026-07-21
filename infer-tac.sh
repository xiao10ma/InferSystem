#!/usr/bin/env bash
set -euo pipefail

GRIPPER_SCALE="${GRIPPER_SCALE:-10}"
CLAMP_GRIP="${CLAMP_GRIP:-0}"

python Example/robot_inference.py Config/flexiv-4view.yaml \
    --prompt "pick the chip on the plate and place it on the other plate" \
    --max-steps 0 \
    --n-execute 48 \
