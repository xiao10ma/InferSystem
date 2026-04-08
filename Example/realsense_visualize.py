"""RealSense 相机检测与可视化。

用法:
    # 从配置文件加载相机
    python Example/flexiv/realsense_visualize.py Config/rizon4_example.yaml

    # 自动检测所有相机
    python Example/flexiv/realsense_visualize.py --auto

    # 不显示深度
    python Example/flexiv/realsense_visualize.py Config/rizon4_example.yaml --no-depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Sensor.rgb_camera import MultiRealSenseManager


def print_device_info() -> None:
    devices = MultiRealSenseManager.discover()
    if not devices:
        print("未检测到 RealSense 设备。")
        return
    print(f"检测到 {len(devices)} 台 RealSense 设备:")
    for dev in devices:
        print(f"  - {dev['name']}  serial={dev['serial_number']}  fw={dev['firmware']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense 相机检测与可视化")
    parser.add_argument("config", nargs="?", default=None, help="YAML 配置文件路径")
    parser.add_argument("--auto", action="store_true", help="自动检测所有相机 (无需配置文件)")
    parser.add_argument("--no-depth", action="store_true", help="不显示深度流")
    args = parser.parse_args()

    print_device_info()

    if args.auto or args.config is None:
        manager = MultiRealSenseManager.from_all_connected(
            enable_depth=not args.no_depth,
        )
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        manager = MultiRealSenseManager(config_path=config_path)

    print(f"打开 {len(manager.cameras)} 台相机...")
    for cam in manager.cameras:
        print(f"  - {cam.name} (serial={cam.serial_number})")
    print("按 'q' 或 Esc 退出。\n")

    manager.visualize(show_depth=not args.no_depth)


if __name__ == "__main__":
    main()
