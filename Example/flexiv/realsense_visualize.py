"""Detect all connected RealSense cameras and visualize their feeds.

Usage:
    # Visualize all cameras (auto-detect)
    python Example/flexiv/realsense_visualize.py

    # Visualize from config file
    python Example/flexiv/realsense_visualize.py --config Config/realsense_camera.yaml

    # Visualize color only (no depth)
    python Example/flexiv/realsense_visualize.py --no-depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Sensor.rgb_camera import MultiRealSenseManager, RealSenseCamera


def print_device_info() -> None:
    devices = MultiRealSenseManager.discover()
    if not devices:
        print("No RealSense devices detected.")
        return
    print(f"Found {len(devices)} RealSense device(s):")
    for dev in devices:
        print(f"  - {dev['name']}  serial={dev['serial_number']}  fw={dev['firmware']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense camera detection and visualization")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (e.g. Config/realsense_camera.yaml)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable depth stream visualization")
    parser.add_argument("--serial", type=str, default=None,
                        help="Only open a single camera by serial number")
    args = parser.parse_args()

    print_device_info()

    if args.serial:
        cam = RealSenseCamera(
            name="single_cam",
            serial_number=args.serial,
            enable_depth=not args.no_depth,
        )
        manager = MultiRealSenseManager(cameras=[cam])
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        manager = MultiRealSenseManager(config_path=config_path)
    else:
        manager = MultiRealSenseManager.from_all_connected(
            enable_depth=not args.no_depth,
        )

    print(f"Opening {len(manager.cameras)} camera(s)...")
    for cam in manager.cameras:
        print(f"  - {cam.name} (serial={cam.serial_number})")
    print("Press 'q' or Esc to exit.\n")

    manager.visualize(show_depth=not args.no_depth)


if __name__ == "__main__":
    main()
