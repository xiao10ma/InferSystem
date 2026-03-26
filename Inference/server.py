"""独立推理服务器。

无 InferSystem 依赖，可直接部署到任何有 GPU 的机器上。

协议 (ZMQ REP, msgpack):

  请求:
    {"cmd": "predict", "state": [float, ...], "prompt": str, "extra": {...},
     "<cam_name>": bytes(JPEG), ...}
    {"cmd": "reset"}

  响应:
    {"status": "ok", "actions": [[float, ...], ...], "infer_time_ms": float}
    {"status": "ok"}
    {"status": "error", "message": str}

用法::

    from Inference.server import InferenceServer

    def my_predict(images, state, prompt="", extra=None):
        # 你的模型推理
        return actions_chunk

    server = InferenceServer("tcp://*:5555", predict_fn=my_predict)
    server.run()  # 阻塞运行

    # 或者子类方式
    class MyServer(InferenceServer):
        def predict(self, images, state, prompt="", extra=None):
            return my_model(images, state, prompt)

        def on_reset(self):
            my_model.reset()
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import cv2
import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)

PredictFn = Callable[..., list[list[float]]]


class InferenceServer:
    """ZMQ 推理服务器。

    Args:
        addr: ZMQ 绑定地址，如 "tcp://*:5555"
        predict_fn: 推理回调。签名:
            (images: dict[str, ndarray], state: list[float],
             prompt: str, extra: dict | None) -> list[list[float]]
            也可通过子类覆盖 predict() 方法替代。
    """

    def __init__(
        self,
        addr: str = "tcp://*:5555",
        predict_fn: PredictFn | None = None,
    ) -> None:
        self._addr = addr
        self._predict_fn = predict_fn
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: list[float],
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """推理入口。子类覆盖此方法，或构造时传入 predict_fn。

        Args:
            images: {cam_name: BGR ndarray}
            state: 机器人状态向量
            prompt: 语言指令 (可为空)
            extra: 扩展输入 (可为 None)

        Returns:
            action chunk: N x action_dim
        """
        if self._predict_fn is not None:
            return self._predict_fn(images, state, prompt, extra)
        raise NotImplementedError("请传入 predict_fn 或覆盖 predict() 方法")

    def on_reset(self) -> None:
        """收到 reset 命令时调用。子类按需覆盖。"""

    def run(self) -> None:
        """阻塞运行，处理请求直到 KeyboardInterrupt。"""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(self._addr)
        logger.info("推理服务器启动: %s", self._addr)

        try:
            while True:
                raw = self._socket.recv()
                reply = self._handle_request(raw)
                self._socket.send(msgpack.packb(reply))
        except KeyboardInterrupt:
            logger.info("服务器收到中断信号，停止。")
        finally:
            self.close()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def _handle_request(self, raw: bytes) -> dict[str, Any]:
        try:
            msg = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            return {"status": "error", "message": f"msgpack 解码失败: {e}"}

        cmd = msg.get("cmd", "predict")

        if cmd == "reset":
            self.on_reset()
            logger.info("策略已重置")
            return {"status": "ok"}

        if cmd == "predict":
            return self._handle_predict(msg)

        return {"status": "error", "message": f"未知命令: {cmd}"}

    def _handle_predict(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            state = msg.get("state", [])
            prompt = msg.get("prompt", "")
            extra = msg.get("extra")
            images = self._decode_images(msg)

            t0 = time.perf_counter()
            actions = self.predict(images, state, prompt=prompt, extra=extra)
            infer_ms = (time.perf_counter() - t0) * 1000

            return {
                "status": "ok",
                "actions": actions,
                "infer_time_ms": infer_ms,
            }
        except Exception as e:
            logger.error("推理失败: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _decode_images(msg: dict[str, Any]) -> dict[str, np.ndarray]:
        """从 msgpack 消息中解码 JPEG 图像。"""
        skip_keys = {"cmd", "state", "prompt", "extra"}
        images: dict[str, np.ndarray] = {}
        for key, value in msg.items():
            if key in skip_keys:
                continue
            if not isinstance(value, (bytes, bytearray)):
                continue
            buf = np.frombuffer(value, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                images[key] = img
        return images
