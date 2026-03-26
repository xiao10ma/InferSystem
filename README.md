# InferSystem

多机器人兼容的模型推理控制系统。支持多种机械臂和二指夹爪，通过统一抽象实现 **感知 → 推理 → 执行** 控制循环。

## 架构

```text
InferSystem/
├── Core/           共享类型 + 注册器 Mixin
├── Robot/          机器人抽象 + 驱动 (Flexiv, ...)
├── Sensor/         传感器抽象 + 驱动 (RealSense, 触觉, ...)
├── Inference/      推理客户端/服务器 + Action 分发
├── Example/        示例脚本
├── Config/         YAML 配置文件
└── SDK/            (预留) 运行时编排
```

## 数据流

```
Sensor.read_images()  ─┐
Robot.observe()        ─┼─→  Observation  ─→  InferenceClient.predict_chunk()
prompt (语言指令)      ─┘                          │
                                                   ▼
                                            action chunk (N x dim)
                                                   │
                                                   ▼
                                         ActionDispatcher.dispatch()
                                           ├─→ Robot.act()      (手臂)
                                           └─→ Gripper.set()    (夹爪)
```

## 环境安装

### 1. 创建 Conda 环境

```bash
conda create -n infersystem python=3.10 -y
conda activate infersystem
```

### 2. 安装公共依赖

```bash
pip install -r requirements.txt
```

公共依赖 (`requirements.txt`):

| 包 | 用途 |
|---|------|
| `pyyaml` | YAML 配置解析 |
| `numpy` | 数值计算、图像数组 |
| `opencv-python` | 图像编解码、触觉传感器、可视化 |
| `pyzmq` | 推理客户端/服务器通信 |
| `msgpack` | 推理协议序列化 |

### 3. 安装机器人特有依赖

根据你使用的机器人型号，安装对应的 SDK:

#### Flexiv Rizon 系列

```bash
pip install flexivrdk spdlog
```

- 支持 Python: 3.8, 3.10, 3.12 (RDK 1.9.0)
- 文档: [Flexiv RDK 手册](https://www.flexiv.com/software/rdk/manual/)
- 兼容性: [robot_software_compatibility](https://www.flexiv.com/software/rdk/manual/robot_software_compatibility.html)

#### (其他机器人)

接入新机器人时，在此添加对应 SDK 的安装说明。

### 4. 安装传感器特有依赖 (按需)

#### Intel RealSense 相机

```bash
pip install pyrealsense2
```

#### 轨迹回放 (parquet 数据)

```bash
pip install pyarrow
```

## 核心类型 (Core/)

| 类型 | 用途 |
|------|------|
| `Observation` | 推理统一输入: images + state + prompt + extra |
| `ArmState` | 机械臂原子状态快照 (关节位置/速度/力矩 + 末端位姿) |
| `Action` | 统一动作指令 (ActionSpace + values + extra) |
| `SensorFrame` | 传感器统一输出帧 |
| `RobotParams` | 机器人硬件参数 (自由度、关节限位等) |

## 模块职责

### Robot — 观测 + 执行 + 生命周期

```python
from Robot import BaseRobot

with BaseRobot.from_config("Config/rizon4_example.yaml") as robot:
    robot.enable()
    robot.wait_until_operational()

    state = robot.observe()           # 原子状态快照
    robot.act(Action(...))            # 非阻塞单步执行
    robot.move_joint_position(q)      # 阻塞移动 (任务层)
    robot.go_home()                   # 回 Home

    gripper = robot.create_gripper()  # 夹爪 (独立设备)
```

接入新机器人: 继承 `BaseRobot`，实现 `observe()` + `act()` + 生命周期方法，用 `@BaseRobot.register("type")` 注册。详见 [Robot/README.md](Robot/README.md)。

### Sensor — 统一传感器管理

```python
from Sensor.manager import SensorManager

config = load_yaml("Config/rizon4_example.yaml")
with SensorManager.from_config(config) as sensors:
    frames = sensors.read_all()              # {name: SensorFrame}
    images = sensors.read_images(["main_cam", "wrist_left"])  # {name: ndarray}
```

支持的传感器: RealSense RGB-D 相机、OpenCV 视触觉传感器、Mock 相机 (测试用)。

### Inference — 远程推理 + Action 分发

```python
from Inference import InferenceClient
from Inference.dispatch import ActionDispatcher, build_state_vector

# 客户端
with InferenceClient("192.168.50.225:5555") as client:
    actions = client.predict_chunk(
        images, state,
        prompt="pick up the red cup",    # VLA 语言指令
    )

# Action 分发 (推理输出 → 机器人 + 夹爪)
dispatcher = ActionDispatcher.from_config(robot, gripper, config)
for action_vec in actions:
    dispatcher.dispatch(action_vec)     # 自动拆分手臂 + 夹爪
```

服务器端可独立部署到 GPU 机器，无 InferSystem 依赖。

## 运行示例

所有示例统一以 YAML 配置文件作为第一个参数:

```bash
# 探测机器人连接
python Example/flexiv/probe.py Config/rizon4_example.yaml --polls 3 --enable

# 回 Home
python Example/flexiv/go_home.py Config/rizon4_example.yaml --velocity 50

# 夹爪控制
python Example/flexiv/gripper.py Config/rizon4_example.yaml open close

# 配置驱动的推理控制 (完整控制循环)
python Example/robot_inference.py Config/rizon4_example.yaml --prompt "pick up the cup"

# Replay 录制轨迹
python Example/flexiv/replay_parquet.py Config/rizon4_example.yaml /path/to/data.parquet

# RealSense 相机可视化
python Example/flexiv/realsense_visualize.py Config/rizon4_example.yaml
```

## YAML 配置

一个 YAML 文件配置整套系统 (机器人 + 传感器 + 推理):

```yaml
robot:
  type: flexiv
  serial_number: "Rizon4-063609"
  gripper:
    name: "Flexiv-GN01"
    max_width: 0.1
  control:
    frequency_hz: 1000.0
    home_position_deg: [0, -20, 0, 90, 0, 20, 0]

cameras:
  main_cam:
    type: realsense
    serial_number: "409122273675"
    enable_depth: true

tactile:
  wrist_left:
    type: opencv
    device_path: "/dev/v4l/by-path/..."

inference:
  server: "192.168.50.225:5555"
  fps: 30
  enabled_cameras: [main_cam, wrist_left]
  gripper_index: 7
  gripper_threshold: 0.5
```

## 设计原则

1. **第一性原理** — 机器人只做三件事: observe / act / lifecycle
2. **原子观测** — `observe()` 一次读取全部传感器，保证时间一致性
3. **统一动作** — `act(Action)` 是唯一的实时控制入口
4. **夹爪独立** — 物理上独立的设备，通过 `create_gripper()` 工厂注入 handle
5. **配置驱动** — 一个 YAML 启动整套系统
6. **注册器模式** — `@register()` 装饰器接入新硬件，零修改基类
