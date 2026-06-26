import cv2
from Core import load_yaml
from Sensor.manager import SensorManager

cfg = load_yaml("Config/piper.yaml")
enabled = cfg["inference"]["enabled_cameras"]
warmup_frames = cfg.get("inference", {}).get("camera_warmup_frames", 100)

sensors = SensorManager.from_config(cfg)
sensors.open_all()
try:
    for _ in range(warmup_frames):
        sensors.read_images(enabled)

    images = sensors.read_images(enabled)
    print("read images:", {k: v.shape for k, v in images.items()})
    for name, img in images.items():
        path = name.replace(".", "_") + ".jpg"
        cv2.imwrite(path, img)
        print("saved", path)
finally:
    sensors.close_all()
