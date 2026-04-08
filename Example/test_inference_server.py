"""推理服务端测试：启动 mock server，等待 client 连接。

收到 predict 请求后将 state 原样返回为 action chunk（机器人保持原位），
打印收到的 state 维度、图像数量和尺寸，用于验证远程数据链路。

在 GPU 机器上运行:
    python Example/test_inference_server.py
    python Example/test_inference_server.py --port 5555 --n-actions 20
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference import InferenceServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


class EchoServer(InferenceServer):
    """将 state 原样回传 n 次，用于测试连通性和数据完整性。"""

    def __init__(self, addr: str, n_actions: int) -> None:
        super().__init__(addr)
        self._n_actions = n_actions
        self._request_count = 0

    def predict(self, images, state, prompt="", extra=None):
        self._request_count += 1
        cam_info = {name: img.shape for name, img in images.items()}
        log.info(
            "[#%d] state_dim=%d  cameras=%s  prompt=%r",
            self._request_count, len(state), cam_info,
            prompt[:50] if prompt else "",
        )
        return [list(state)] * self._n_actions

    def on_reset(self):
        log.info("[reset] 策略已重置 (请求计数=%d)", self._request_count)
        self._request_count = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="推理服务端测试 (mock echo server)")
    parser.add_argument("--port", type=int, default=5555, help="监听端口")
    parser.add_argument("--n-actions", type=int, default=10, help="每次返回的 action 数量")
    args = parser.parse_args()

    addr = f"tcp://*:{args.port}"
    log.info("启动 echo server: %s (每次返回 %d actions)", addr, args.n_actions)
    log.info("等待 client 连接... (Ctrl-C 退出)")

    server = EchoServer(addr, args.n_actions)
    server.run()


if __name__ == "__main__":
    main()
