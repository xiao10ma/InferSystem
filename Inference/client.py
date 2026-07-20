"""推理客户端。

两种使用方式:

  方式 1 — 获取完整 chunk::

      client = InferenceClient("192.168.50.225:5555")
      with client:
          actions = client.predict_chunk(images, state)  # list[list[float]]
          for action in actions:
              robot.act(Action(ActionSpace.JOINT_POSITION, action[:7]))

  方式 2 — 逐帧获取 (自动管理队列)::

      client = InferenceClient("192.168.50.225:5555")
      with client:
          while running:
              action = client.get_action(images, state, prompt="pick up the cup")
              robot.act(Action(ActionSpace.JOINT_POSITION, action[:7]))

  方式 2 每次调用都存储最新观测。队列用完时自动用最新观测请求新 chunk。

  也可以直接传入 Observation 对象::

      from Core import Observation
      obs = Observation(images=images, state=state, prompt="pick up the cup")
      actions = client.predict_chunk_from_obs(obs)
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import cv2
import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class InferenceClient:
    """ZMQ 推理客户端。

    Args:
        server_addr: 服务器地址，如 "192.168.50.225:5555"
        recv_timeout_ms: 接收超时 (ms)，首次推理较慢建议 ≥ 30000
        send_timeout_ms: 发送超时 (ms)
        max_retries: 网络错误自动重试次数
    """

    def __init__(
        self,
        server_addr: str,
        *,
        recv_timeout_ms: int = 30000,
        send_timeout_ms: int = 5000,
        max_retries: int = 3,
    ) -> None:
        self._server_addr = server_addr
        self._recv_timeout_ms = recv_timeout_ms
        self._send_timeout_ms = send_timeout_ms
        self._max_retries = max_retries

        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._connected = False

        # 方式 2 的队列和存储
        self._action_queue: deque[list[float]] = deque()
        self._latest_images: dict[str, np.ndarray] = {}
        self._latest_state: list[float] = []
        self._latest_prompt: str = ""
        self._latest_extra: dict[str, Any] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── 生命周期 ──

    def connect(self) -> None:
        """连接到推理服务器。"""
        if self._connected:
            return
        self._context = zmq.Context()
        self._create_socket()
        self._connected = True
        logger.info("已连接到推理服务器: %s", self._server_addr)

    def close(self) -> None:
        """断开连接并释放资源。"""
        if not self._connected:
            return
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
        finally:
            self._connected = False
            self._action_queue.clear()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 方式 1: 获取完整 chunk ──

    def predict_chunk(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """发送观测，获取完整 action chunk。

        Args:
            images: {cam_name: BGR numpy (H,W,3)}
            state: 机器人状态向量
            prompt: 语言指令 (VLA 等模型使用)
            extra: 扩展输入 (点云、力等)

        Returns:
            action chunk: N x action_dim
        """
        self._ensure_connected()
        payload = self._encode_observation(images, state, prompt=prompt, extra=extra)
        resp = self._request(payload)
        return resp["actions"]

    def predict_chunk_from_obs(self, obs: Any) -> list[list[float]]:
        """从 Observation 对象获取完整 action chunk。

        Args:
            obs: Core.Observation 实例

        Returns:
            action chunk: N x action_dim
        """
        return self.predict_chunk(
            images=obs.images,
            state=obs.state,
            prompt=obs.prompt,
            extra=obs.extra if obs.extra else None,
        )

    # ── 方式 2: 逐帧获取 ──

    def get_action(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[float]:
        """获取单帧 action。

        每次调用都存储最新观测。队列为空时，
        用最新存储的观测自动请求新 chunk。

        Args:
            images: {cam_name: BGR numpy (H,W,3)}
            state: 机器人状态向量
            prompt: 语言指令
            extra: 扩展输入

        Returns:
            单帧 action: action_dim 维列表
        """
        # 始终存储最新观测
        self._latest_images = images
        self._latest_state = (
            state.tolist() if isinstance(state, np.ndarray) else list(state)
        )
        self._latest_prompt = prompt
        self._latest_extra = extra or {}

        if not self._action_queue:
            actions = self.predict_chunk(
                self._latest_images,
                self._latest_state,
                prompt=self._latest_prompt,
                extra=self._latest_extra or None,
            )
            self._action_queue.extend(actions)

        return self._action_queue.popleft()

    @property
    def actions_remaining(self) -> int:
        """队列中剩余的 action 数量。"""
        return len(self._action_queue)

    def clear_queue(self) -> None:
        """清空 action 队列，下次 get_action 时会请求新 chunk。"""
        self._action_queue.clear()

    # ── Stateful tactile (prepare/refine) ──

    def predict_stateful(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
        stateful_tactile: dict[str, Any],
        actions_prefix: list[list[float]] | None = None,
        request_id: str,
    ) -> dict[str, Any]:
        """发送 stateful tactile prepare/refine 请求，返回完整响应。

        Args:
            stateful_tactile: 协议控制字段 (op/plan_id/action_offset/tactile_seq...)
            actions_prefix: refine 时回传的已执行动作前缀
            request_id: 请求唯一 ID，服务器据此对超时重试做幂等重放

        Returns:
            完整响应 dict，含 "actions" (修正后的完整序列) 和 "plan" 元数据
        """
        self._ensure_connected()
        payload = self._encode_observation(images, state, prompt=prompt, extra=extra)
        payload["stateful_tactile"] = dict(stateful_tactile)
        payload["request_id"] = request_id
        if actions_prefix is not None:
            payload["actions"] = actions_prefix
        return self._request(payload)

    # ── 控制命令 ──

    def get_metadata(self) -> dict[str, Any]:
        """获取服务器元信息 (cmd=metadata)。"""
        self._ensure_connected()
        resp = self._request({"cmd": "metadata"})
        return resp.get("metadata", {})

    def reset(self) -> dict[str, Any]:
        """通知服务器重置策略。同时清空本地 action 队列。"""
        self._ensure_connected()
        resp = self._request({"cmd": "reset"})
        self._action_queue.clear()
        logger.info("策略已重置")
        return resp

    # ── 内部实现 ──

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(
                "未连接到推理服务器，请先调用 connect() 或使用 with 语句"
            )

    def _create_socket(self) -> None:
        """创建新的 REQ socket。"""
        assert self._context is not None
        if self._socket is not None:
            self._socket.close()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self._send_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._server_addr}")

    def _reconnect(self) -> None:
        """重建 socket 连接 (ZMQ REQ 在超时后状态会损坏)。"""
        logger.warning("重建连接: %s", self._server_addr)
        self._create_socket()

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送请求并接收响应，含重试逻辑。"""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                assert self._socket is not None
                self._socket.send(msgpack.packb(payload, use_bin_type=True))
                raw = self._socket.recv()
                resp = msgpack.unpackb(raw, raw=False)

                if resp.get("status") == "error":
                    raise RuntimeError(
                        f"服务器返回错误: {resp.get('message', '未知')}"
                    )
                return resp

            except zmq.Again as e:
                last_error = e
                logger.warning(
                    "请求超时 (第 %d/%d 次): %s",
                    attempt, self._max_retries, e,
                )
                self._reconnect()

            except zmq.ZMQError as e:
                last_error = e
                logger.warning(
                    "ZMQ 错误 (第 %d/%d 次): %s",
                    attempt, self._max_retries, e,
                )
                self._reconnect()

        raise ConnectionError(
            f"推理请求失败，已重试 {self._max_retries} 次: {last_error}"
        )

    def _encode_observation(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray | list[float],
        *,
        prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """将观测编码为 msgpack 可序列化的字典。"""
        state_list = np.asarray(state, dtype=np.float32).reshape(-1).tolist()
        payload: dict[str, Any] = {
            "cmd": "predict",
            "state": state_list,
        }

        # 语言指令
        if prompt:
            payload["prompt"] = prompt

        # 扩展输入 (确保可序列化)
        if extra:
            payload["extra"] = extra

        for cam_name, img_bgr in images.items():
            img_bgr = np.asarray(img_bgr)
            # Lossless PNG; low compression keeps latency lower at the cost of bandwidth.
            ok, encoded = cv2.imencode(
                ".png",
                img_bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )
            if not ok:
                raise ValueError(f"PNG 编码失败: {cam_name}")
            payload[cam_name] = encoded.tobytes()

        return payload
