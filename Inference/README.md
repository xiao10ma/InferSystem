# Inference — 推理通信层

## 设计原则

推理通信层解决一个问题: **机器人端的观测 → 网络 → GPU 端的模型 → 网络 → 机器人端的动作**。

不做模型推理本身，不依赖机器人/传感器代码。Server 可以直接部署到任何有 GPU 的机器上。

## 架构

```
InferenceServer (server.py)   — GPU 机器上运行，接收观测，调用模型，返回 action chunk
InferenceClient (client.py)   — 机器人端运行，发送观测，获取 action
```

## 协议 (ZMQ REP/REQ, msgpack)

```
客户端 → 服务器:
  {"cmd": "predict", "state": [float, ...], "<cam_name>": bytes(JPEG), ...}
  {"cmd": "reset"}

服务器 → 客户端:
  {"status": "ok", "actions": [[float, ...], ...], "infer_time_ms": float}
  {"status": "ok"}
  {"status": "error", "message": str}
```

## 服务器用法

```python
from Inference import InferenceServer

# 函数式
def my_predict(images: dict[str, np.ndarray], state: list[float]) -> list[list[float]]:
    return my_model.infer(images, state)

server = InferenceServer("tcp://*:5555", predict_fn=my_predict)
server.run()

# 子类式
class MyServer(InferenceServer):
    def predict(self, images, state):
        return self.model(images, state)

    def on_reset(self):
        self.model.reset()
```

## 客户端用法

### 方式 1: 获取完整 chunk

一次拿到全部 action，自行控制执行节奏:

```python
from Inference import InferenceClient

client = InferenceClient("192.168.50.225:5555")
with client:
    client.reset()
    actions = client.predict_chunk(images, state)  # list[list[float]]
    for action in actions[:n_execute]:
        robot.send_joint_position(action[:7])
```

### 方式 2: 逐帧获取 (推荐)

每次返回一帧 action，队列空时自动用最新观测请求新 chunk:

```python
with InferenceClient("192.168.50.225:5555") as client:
    client.reset()
    while running:
        images = cameras.read()
        state = robot.get_state()
        action = client.get_action(images, state)   # list[float], 单帧
        robot.send_joint_position(action[:7])
```

**方式 2 的关键**: 每次调用 `get_action()` 都会存储最新的 `images` 和 `state`。
当队列用完时，会用**最新存储的观测**（而不是旧观测）去请求新 chunk。
这对于视觉伺服等需要持续更新观测的场景至关重要。

## 网络错误处理

- 发送/接收超时自动重试 (默认 3 次)
- ZMQ REQ socket 超时后状态损坏时自动重建连接
- 服务器返回 `{"status": "error"}` 时抛出 `RuntimeError`
- 所有重试耗尽后抛出 `ConnectionError`

## 依赖

Server 和 Client 只依赖: `zmq`, `msgpack`, `numpy`, `cv2` — 常见 ML 环境默认可用。
