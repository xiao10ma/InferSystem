"""Openpi ZMQ 推理服务器。

将 ZMQ REP (msgpack) 协议适配到 openpi 的 Policy.infer() 接口。

协议 (ZMQ REP, msgpack):

  请求:
    {"cmd": "predict", "state": [float, ...], "prompt": str, "extra": {...},
     "top_head": bytes(PNG), "hand_left": bytes(PNG), ...}
    {"cmd": "reset"}

  图像字段:
    - 当前 InferSystem client 发送 bytes(PNG)
    - 也兼容 RGB uint8 CHW list, shape=(3, 224, 224)
    - 也兼容 RGB HWC list/array，server 会转为 CHW

  响应:
    {"status": "ok", "actions": [[float, ...], ...], "infer_time_ms": float}
    {"status": "ok"}
    {"status": "error", "message": str}

用法::

    # 直接运行
    .venv/bin/python openpi_zmq_server.py --addr tcp://*:5555 \\
        --config fold_clothes_delta \\
        --checkpoint ~/kai0/checkpoints/fold_clothes_delta/run1/20000

    # 或导入使用
    from openpi_zmq_server import PIZmqServer
    server = PIZmqServer(
        addr="tcp://*:5555",
        config_name="fold_clothes_delta",
        checkpoint_dir="checkpoints/fold_clothes_delta/run1/20000",
    )
    server.run()
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from Inference.server import (
    IMAGE_SKIP_KEYS,
    decode_images_from_msg,
    image_keys_from_msg,
)

logger = logging.getLogger(__name__)

# 非图像字段：不会被当作相机帧解码。其余图像 payload key
# 并以原 key 名（如 observation/image）透传给 openpi 的 *Inputs transform。
SKIP_KEYS = IMAGE_SKIP_KEYS


def _build_obs(msg: dict[str, Any]) -> dict:
    """将 ZMQ 消息转换为 openpi Policy.infer() 期望的 obs dict。

    openpi 的 *Inputs transform（如 FlexivInputs）直接从顶层读取原始 key，
    例如 data["observation/image"]、data["state"]，因此这里把每个图像 key
    按原名放到顶层，state/action 同布局直接透传。
    """
    state = np.asarray(msg.get("state", []), dtype=np.float32)
    prompt = msg.get("prompt", "")

    obs: dict[str, Any] = {"state": state}

    obs.update(decode_images_from_msg(msg, skip_keys=SKIP_KEYS, strict=True))

    if prompt:
        obs["prompt"] = prompt

    return obs


class PIZmqServer:
    """ZMQ 推理服务器，内部使用 openpi Policy 进行推理。"""

    def __init__(
        self,
        addr: str = "tcp://*:5555",
        config_name: str = "fold_clothes_delta",
        checkpoint_dir: str = "checkpoints/fold_clothes_delta/run1/20000",
        default_prompt: str | None = "Fold the clothes.",
        print_action_chunk: bool = False,
    ) -> None:
        self._addr = addr
        self._config_name = config_name
        self._checkpoint_dir = checkpoint_dir
        self._default_prompt = default_prompt
        self._print_action_chunk = print_action_chunk
        self._policy = None
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None

    def _load_policy(self):
        """加载 openpi policy。"""
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        logger.info("加载配置: %s", self._config_name)
        train_config = _config.get_config(self._config_name)

        logger.info("加载 checkpoint: %s", self._checkpoint_dir)
        self._policy = _policy_config.create_trained_policy(
            train_config,
            self._checkpoint_dir,
            default_prompt=self._default_prompt,
        )
        logger.info("模型加载完成")

        # 打印 transform 链供调试
        for i, t in enumerate(self._policy._input_transform.transforms):
            logger.info("  input_transform[%d]: %s", i, type(t).__name__)
        for i, t in enumerate(self._policy._output_transform.transforms):
            logger.info("  output_transform[%d]: %s", i, type(t).__name__)

    def predict(self, msg: dict[str, Any]) -> dict[str, Any]:
        """处理单次推理请求。"""
        obs = _build_obs(msg)
        result = self._policy.infer(obs)

        actions = result["actions"]
        actions_arr = np.asarray(actions)
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        if self._print_action_chunk:
            np.set_printoptions(precision=4, suppress=True, linewidth=200)
            logger.info("action_chunk shape=%s:\n%s", actions_arr.shape, actions_arr)

        response = {
            "status": "ok",
            "actions": actions,
            "infer_time_ms": result.get("policy_timing", {}).get("infer_ms", 0.0),
        }
        return response

    def on_reset(self) -> None:
        """重置策略状态（如有需要）。"""
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
        logger.info("策略已重置")

    def run(self) -> None:
        """阻塞运行服务器。"""
        self._load_policy()

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(self._addr)
        logger.info("ZMQ 推理服务器启动: %s", self._addr)

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
            return {"status": "ok"}

        if cmd == "predict":
            return self._handle_predict(msg)

        return {"status": "error", "message": f"未知命令: {cmd}"}

    def _handle_predict(self, msg: dict[str, Any]) -> dict[str, Any]:
        try:
            # 打印 client 发来的所有 key 及图像 payload key
            all_keys = list(msg.keys())
            img_keys = image_keys_from_msg(msg, skip_keys=SKIP_KEYS)
            logger.info("收到请求 keys=%s, 图像 keys=%s", all_keys, img_keys)

            t0 = time.perf_counter()
            result = self.predict(msg)
            total_ms = (time.perf_counter() - t0) * 1000

            # 用实际端到端时间（含解码）覆盖，原始模型推理时间保留在 model_infer_time_ms
            result["model_infer_time_ms"] = result.get("infer_time_ms", 0.0)
            result["infer_time_ms"] = total_ms

            logger.info(
                "推理完成: actions shape=%s, model=%.1fms, total=%.1fms",
                np.array(result["actions"]).shape if isinstance(result["actions"], list) else "?",
                result["model_infer_time_ms"],
                total_ms,
            )
            return result
        except Exception as e:
            logger.error("推理失败: %s", e, exc_info=True)
            return {"status": "error", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="OpenPi ZMQ 推理服务器")
    parser.add_argument("--addr", default="tcp://*:5555", help="ZMQ 绑定地址")
    parser.add_argument("--config", default="aloha_wipe_board", help="训练配置名")
    parser.add_argument("--checkpoint", default="/data/checkpoints/fold_clothes_delta/run1/20000", help="checkpoint 目录")
    parser.add_argument("--default-prompt", default="pick up the white board eraser and then clean the white board", help="默认语言指令")
    parser.add_argument("--print-action-chunk", action="store_true", help="打印每次推理输出的 action_chunk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    server = PIZmqServer(
        addr=args.addr,
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        default_prompt=args.default_prompt,
        print_action_chunk=args.print_action_chunk,
    )
    server.run()


if __name__ == "__main__":
    main()
