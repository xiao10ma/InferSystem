"""TactilePlanWorker 调度语义测试（mock client，无网络）。"""
import threading
import time

import numpy as np
import pytest

from Inference.async_worker import InferenceObservationSnapshot
from Inference.tactile_plan_worker import StatefulTactileProtocolError, TactilePlanWorker


def _snapshot(tag: float = 0.0) -> InferenceObservationSnapshot:
    return InferenceObservationSnapshot(
        images={"cam": np.full((2, 2, 3), int(tag) % 255, dtype=np.uint8)},
        state=[tag],
        prompt="pick",
    )


def _wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def _peek(worker: TactilePlanWorker, idx: int) -> list[float]:
    with worker._condition:
        return worker._plan.actions[idx].tolist()


class FakeStatefulClient:
    """同步返回的 stateful 服务器桩。

    prepare 返回 [100*n + t]（n 为第几个 plan），refine 返回
    [1000*tactile_seq + t]，便于按值区分动作来源。
    """

    def __init__(self, horizon: int = 6, dim: int = 2):
        self.horizon = horizon
        self.dim = dim
        self.calls: list[dict] = []
        self.reset_count = 0
        self.plan_counter = 0
        self.gate: threading.Event | None = None
        self.fail_ops: set[str] = set()

    def get_metadata(self):
        return {"supports_stateful_tactile": True}

    def reset(self):
        self.reset_count += 1

    def predict_stateful(
        self, images, state, *, prompt="", extra=None,
        stateful_tactile, actions_prefix=None, request_id,
    ):
        control = dict(stateful_tactile)
        control["n_prefix"] = 0 if actions_prefix is None else len(actions_prefix)
        self.calls.append(control)
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        if control["op"] in self.fail_ops:
            raise RuntimeError("boom")
        if control["op"] == "prepare":
            self.plan_counter += 1
            base = 100.0 * self.plan_counter
        else:
            base = 1000.0 * control["tactile_seq"]
        actions = [[base + t] * self.dim for t in range(self.horizon)]
        return {
            "status": "ok",
            "actions": actions,
            "plan": {
                "plan_id": control["plan_id"],
                "op": control["op"],
                "action_offset": control["action_offset"],
                "tactile_seq": control["tactile_seq"],
                "action_horizon": self.horizon,
                "prepared_slow_path": control["op"] == "prepare",
            },
        }


def test_rejects_server_without_stateful_support():
    class NoSupport(FakeStatefulClient):
        def get_metadata(self):
            return {}

    with pytest.raises(StatefulTactileProtocolError):
        TactilePlanWorker(NoSupport())


def test_prepare_then_pop_consumes_plan_in_order():
    client = FakeStatefulClient(horizon=4)
    worker = TactilePlanWorker(client)
    worker.start()
    try:
        assert worker.pop_action() is None
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: worker.remaining == 4)
        popped = [worker.pop_action() for _ in range(4)]
        assert [a[0] for a in popped] == [100.0, 101.0, 102.0, 103.0]
        assert worker.pop_action() is None
        time.sleep(0.05)
        assert client.plan_counter == 1  # 耗尽后没有更新的帧，不允许 replan
    finally:
        worker.stop()


def test_refine_merges_only_unexecuted_suffix():
    client = FakeStatefulClient(horizon=6)
    worker = TactilePlanWorker(client, delay_init=2)
    worker.start()
    try:
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: worker.remaining == 6)
        assert worker.pop_action()[0] == 100.0
        assert worker.pop_action()[0] == 101.0

        # 新帧触发 refine: refine_offset = offset(2) + delay_init(2) = 4
        worker.update_observation(_snapshot(2))
        _wait_until(lambda: any(c["op"] == "refine" for c in client.calls))
        refine = next(c for c in client.calls if c["op"] == "refine")
        assert refine["action_offset"] == 4
        assert refine["n_prefix"] == 4

        seq = refine["tactile_seq"]
        _wait_until(lambda: _peek(worker, 4)[0] == 1000.0 * seq + 4)
        # 已消费/预留前缀 (2,3) 不变，后缀 (4,5) 来自 refine
        assert worker.pop_action()[0] == 102.0
        assert worker.pop_action()[0] == 103.0
        assert worker.pop_action()[0] == 1000.0 * seq + 4
        assert worker.pop_action()[0] == 1000.0 * seq + 5
    finally:
        worker.stop()


def test_replan_uses_fresh_frame_after_exhaustion():
    client = FakeStatefulClient(horizon=2)
    worker = TactilePlanWorker(client)
    worker.start()
    try:
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: worker.remaining == 2)
        worker.pop_action()
        worker.pop_action()
        time.sleep(0.05)
        assert client.plan_counter == 1

        worker.update_observation(_snapshot(2))
        _wait_until(lambda: client.plan_counter == 2)
        prepare2 = [c for c in client.calls if c["op"] == "prepare"][1]
        assert prepare2["previous_action_offset"] == 2
        _wait_until(lambda: worker.remaining == 2)
        assert worker.pop_action()[0] == 200.0
    finally:
        worker.stop()


def test_pause_discards_inflight_result():
    client = FakeStatefulClient(horizon=4)
    client.gate = threading.Event()
    worker = TactilePlanWorker(client)
    worker.start()
    try:
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: len(client.calls) == 1)  # prepare 被 gate 挡在途中
        worker.pause()
        client.gate.set()
        time.sleep(0.1)
        assert worker.pop_action() is None
    finally:
        worker.stop()


def test_reset_episode_resets_server_and_sequence():
    client = FakeStatefulClient(horizon=3)
    worker = TactilePlanWorker(client)
    worker.start()
    try:
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: worker.remaining == 3)
        worker.reset_episode()
        assert client.reset_count == 1
        assert worker.pop_action() is None

        worker.update_observation(_snapshot(2))
        _wait_until(lambda: worker.remaining == 3)
        prepare2 = [c for c in client.calls if c["op"] == "prepare"][1]
        assert prepare2["tactile_seq"] == 0
        assert prepare2["previous_plan_id"] is None
    finally:
        worker.stop()


def test_worker_error_raises_on_pop():
    client = FakeStatefulClient()
    client.fail_ops = {"prepare"}
    worker = TactilePlanWorker(client)
    worker.start()
    try:
        worker.update_observation(_snapshot(1))
        _wait_until(lambda: len(client.calls) == 1)

        def _raises():
            try:
                worker.pop_action()
                return False
            except RuntimeError:
                return True

        _wait_until(_raises)
    finally:
        worker.stop()
