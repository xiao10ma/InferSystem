#!/usr/bin/env bash
set -euo pipefail

# Find and stop processes that are using Linux camera/video devices.
# Usage:
#   scripts/kill_camera_processes.sh
#   scripts/kill_camera_processes.sh --dry-run
#
# Run with sudo if camera devices are held by processes owned by another user.

readonly TERM_TIMEOUT_SECONDS="${CAMERA_KILL_TIMEOUT:-3}"
dry_run=false

usage() {
  cat <<'EOF'
Usage: scripts/kill_camera_processes.sh [--dry-run]

Find processes using /dev/video*, /dev/media*, or /dev/v4l-subdev*, then
send SIGTERM. Processes that still hold a camera after the timeout receive
SIGKILL.

Environment:
  CAMERA_KILL_TIMEOUT  Seconds to wait after SIGTERM (default: 3)
EOF
}

case "${1:-}" in
  "") ;;
  --dry-run) dry_run=true ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! "${TERM_TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CAMERA_KILL_TIMEOUT must be a non-negative integer." >&2
  exit 2
fi

if ! command -v lsof >/dev/null 2>&1 && ! command -v fuser >/dev/null 2>&1; then
  echo "ERROR: neither lsof nor fuser is installed; cannot inspect camera usage." >&2
  exit 1
fi

shopt -s nullglob
camera_devices=(/dev/video* /dev/media* /dev/v4l-subdev*)
shopt -u nullglob

if ((${#camera_devices[@]} == 0)); then
  echo "No camera devices found."
  exit 0
fi

find_camera_pids() {
  local device

  if command -v lsof >/dev/null 2>&1; then
    lsof -t -- "${camera_devices[@]}" 2>/dev/null || true
  else
    for device in "${camera_devices[@]}"; do
      fuser "${device}" 2>/dev/null || true
    done
  fi | tr ' ' '\n' | awk -v self="$$" '/^[0-9]+$/ && $1 != self' | sort -un
}

show_processes() {
  local pid

  echo "Camera processes:"
  for pid in "$@"; do
    ps -ww -o pid=,user=,stat=,etime=,args= -p "${pid}" || true
  done
}

signal_processes() {
  local signal="$1"
  local pid
  shift

  for pid in "$@"; do
    if kill "-${signal}" "${pid}" 2>/dev/null; then
      echo "Sent SIG${signal} to PID ${pid}."
    elif kill -0 "${pid}" 2>/dev/null; then
      echo "WARNING: cannot signal PID ${pid}; try running this script with sudo." >&2
    fi
  done
}

wait_for_release() {
  local attempts="$1"
  local attempt
  local -a current_pids=()

  for ((attempt = 0; attempt < attempts; attempt++)); do
    mapfile -t current_pids < <(find_camera_pids)
    ((${#current_pids[@]} == 0)) && return 0
    sleep 0.1
  done
  return 1
}

mapfile -t pids < <(find_camera_pids)

if ((${#pids[@]} == 0)); then
  echo "No process is using a camera device."
  exit 0
fi

show_processes "${pids[@]}"

if [[ "${dry_run}" == true ]]; then
  echo "Dry run: no process was killed."
  exit 0
fi

signal_processes TERM "${pids[@]}"

if wait_for_release "$((TERM_TIMEOUT_SECONDS * 10))"; then
  echo "All camera devices have been released."
  exit 0
fi

mapfile -t pids < <(find_camera_pids)
if ((${#pids[@]} > 0)); then
  echo "Camera devices are still busy; forcing remaining processes to stop."
  signal_processes KILL "${pids[@]}"
fi

wait_for_release 10 || true
mapfile -t pids < <(find_camera_pids)

if ((${#pids[@]} > 0)); then
  echo "ERROR: camera devices are still occupied by PID(s): ${pids[*]}" >&2
  echo "Try: sudo $0" >&2
  exit 1
fi

echo "All camera devices have been released."
