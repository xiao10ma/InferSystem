"""tactile expert 异步推理 worker（prepare/refine，真机版）。

一个后台线程持续对最新观测做 prepare/refine——

  - prepare: 用最新帧跑完整慢路径 plan（仅在无 plan 或旧 plan 全部消费完后触发）
  - refine:  复用服务器端 plan state，只对未执行后缀做 tactile expert 快路径修正，
             并回传已执行动作前缀做锚定

控制循环通过 update_observation() 发布观测、pop_action() 消费动作。
已经交给机器人执行的前缀永远不会被后续 refine 改写（commit 时取
max(当前 offset, refine_offset) 之后的后缀合并）。与仿真 lockstep 不同，
pop_action() 非阻塞：plan 耗尽且 prepare 在途时返回 None，由调用方 idle。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from Inference.async_worker import InferenceObservationSnapshot

logger = logging.getLogger(__name__)


class StatefulTactileProtocolError(RuntimeError):
    """推理服务器未实现或违反 stateful tactile 协议。"""


@dataclass(frozen=True)
class _ObservationFrame:
    seq: int
    snapshot: InferenceObservationSnapshot


@dataclass(frozen=True)
class _ActionPlan:
    generation: int
    plan_id: str
    actions: np.ndarray  # 服务器原始动作序列 (T, D)，canonicalize 由消费方处理
    offset: int
    horizon: int
    replan_after_frame_seq: int = -1


@dataclass(frozen=True)
class _InferenceJob:
    op: str
    episode_generation: int
    tactile_seq: int
    frame: _ObservationFrame
    plan: _ActionPlan | None
    refine_offset: int = 0


class TactilePlanWorker:
    """后台 prepare/refine 循环，对接 InferSystem ZMQ 协议。

    Args:
        client: InferenceClient 实例（已 connect），worker 持有其网络访问权；
                episode reset 必须走 reset_episode() 以便与在途请求串行化。
        delay_init: refine 在途期间被消费步数的初始估计（步）。
        raw_action_callback: 每次 prepare/refine 成功后回调 (request_count, actions)。
    """

    def __init__(
        self,
        client: Any,
        *,
        delay_init: int = 2,
        raw_action_callback: Callable[[int, Any], None] | None = None,
    ) -> None:
        metadata = client.get_metadata()
        if not metadata.get("supports_stateful_tactile"):
            raise StatefulTactileProtocolError(
                "推理服务器不支持 stateful tactile 协议 "
                "(metadata.supports_stateful_tactile 为空)，请确认模型与服务器版本。"
            )
        self._client = client
        self._delay_init = int(delay_init)
        self._raw_action_callback = raw_action_callback

        self._condition = threading.Condition()
        # 网络请求与 episode reset 在此串行化（ZMQ REQ socket 非线程安全）。
        self._network_lock = threading.Lock()
        self._latest_frame: _ObservationFrame | None = None
        self._frame_seq = 0
        self._episode_generation = 0
        self._plan_generation = 0
        self._plan: _ActionPlan | None = None
        self._tactile_seq = -1
        self._worker_error: Exception | None = None
        self._paused = False
        self._closed = False
        self._delay_history: deque[int] = deque([self._delay_init], maxlen=10)
        self._last_refine_offset = 0
        self._skip_logged_generation = -1
        self._thread: threading.Thread | None = None
        self.request_count = 0
        self.last_infer_ms: float | None = None

    # ── 生命周期 ──

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._inference_loop, name="tactile-plan", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        with self._condition:
            self._closed = True
            self._episode_generation += 1
            self._latest_frame = None
            self._plan = None
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def pause(self) -> None:
        """丢弃当前 plan 并使在途请求的结果失效（restart 流程用）。"""
        with self._condition:
            self._paused = True
            self._episode_generation += 1
            self._latest_frame = None
            self._plan = None
            self._worker_error = None
            self._condition.notify_all()

    def reset_episode(self) -> None:
        """新 episode：清空本地状态并重置服务器端 plan 会话。"""
        with self._condition:
            self._episode_generation += 1
            self._latest_frame = None
            self._plan = None
            self._tactile_seq = -1
            self._worker_error = None
            self._delay_history = deque([self._delay_init], maxlen=10)
            self._last_refine_offset = 0
            self._paused = False
            self._condition.notify_all()
        # 等待在途请求结束后再复用 socket。
        with self._network_lock:
            self._client.reset()

    # ── 主循环接口 ──

    def update_observation(self, obs: InferenceObservationSnapshot) -> None:
        """发布最新观测供后台 prepare/refine 使用（不等待推理）。"""
        snapshot = InferenceObservationSnapshot(
            images=dict(obs.images),
            state=list(obs.state),
            prompt=obs.prompt,
            extra=dict(obs.extra) if isinstance(obs.extra, dict) else obs.extra,
            meta=dict(obs.meta),
        )
        with self._condition:
            self._latest_frame = _ObservationFrame(seq=self._frame_seq, snapshot=snapshot)
            self._frame_seq += 1
            self._condition.notify_all()

    def pop_action(self) -> list[float] | None:
        """取下一帧已提交动作；plan 未就绪或已耗尽时返回 None。

        后台推理失败时抛出 RuntimeError（fail-fast，不静默降级）。
        """
        with self._condition:
            self._raise_worker_error()
            plan = self._plan
            if plan is None or plan.offset >= plan.horizon:
                return None
            action = plan.actions[plan.offset].tolist()
            next_offset = plan.offset + 1
            replan_after = plan.replan_after_frame_seq
            if next_offset == plan.horizon and self._latest_frame is not None:
                # 耗尽后必须等到比此刻更新的帧才允许 prepare 下一个 plan。
                replan_after = self._latest_frame.seq
            self._plan = replace(
                plan,
                offset=next_offset,
                replan_after_frame_seq=replan_after,
            )
            self._condition.notify_all()
            return action

    @property
    def remaining(self) -> int:
        with self._condition:
            plan = self._plan
            return 0 if plan is None else max(0, plan.horizon - plan.offset)

    # ── 后台调度 ──

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("异步 tactile 推理失败") from self._worker_error

    def _next_job(self, last_frame_seq: int) -> tuple[_InferenceJob | None, int]:
        with self._condition:
            while True:
                if self._closed:
                    return None, last_frame_seq
                frame = self._latest_frame
                plan = self._plan
                if not self._paused and self._worker_error is None and frame is not None:
                    refine_offset = 0
                    if plan is None:
                        op = "prepare"
                    elif plan.offset >= plan.horizon:
                        if frame.seq <= plan.replan_after_frame_seq:
                            self._condition.wait()
                            continue
                        op = "prepare"
                    elif frame.seq > last_frame_seq:
                        # 预留 refine 在途期间机器人会消费掉的步数。
                        refine_offset = max(
                            plan.offset + max(self._delay_history),
                            self._last_refine_offset,
                        )
                        if refine_offset >= plan.horizon:
                            if self._skip_logged_generation != plan.generation:
                                self._skip_logged_generation = plan.generation
                                logger.warning(
                                    "跳过 tactile refine: 预测 offset %d >= horizon %d "
                                    "(已执行 %d + 最大延迟 %d)",
                                    refine_offset,
                                    plan.horizon,
                                    plan.offset,
                                    max(self._delay_history),
                                )
                            self._condition.wait()
                            continue
                        op = "refine"
                        self._last_refine_offset = refine_offset
                    else:
                        self._condition.wait()
                        continue

                    self._tactile_seq += 1
                    return (
                        _InferenceJob(
                            op=op,
                            episode_generation=self._episode_generation,
                            tactile_seq=self._tactile_seq,
                            frame=frame,
                            plan=plan,
                            refine_offset=refine_offset,
                        ),
                        frame.seq,
                    )
                self._condition.wait()

    def _inference_loop(self) -> None:
        last_frame_seq = -1
        while True:
            job, last_frame_seq = self._next_job(last_frame_seq)
            if job is None:
                return
            try:
                with self._network_lock:
                    with self._condition:
                        if job.episode_generation != self._episode_generation:
                            continue
                    if job.op == "prepare":
                        self._run_prepare(job)
                    else:
                        self._run_refine(job)
            except Exception as exc:
                with self._condition:
                    if job.episode_generation == self._episode_generation:
                        self._worker_error = exc
                        self._condition.notify_all()

    def _run_prepare(self, job: _InferenceJob) -> None:
        previous = job.plan
        plan_id = uuid4().hex
        control = {
            "op": "prepare",
            "plan_id": plan_id,
            "action_offset": 0,
            "tactile_seq": job.tactile_seq,
            "previous_plan_id": None if previous is None else previous.plan_id,
            "previous_action_offset": None if previous is None else previous.offset,
            "force_replan": False,
        }
        actions, horizon = self._infer(job, control)

        with self._condition:
            if job.episode_generation != self._episode_generation:
                return
            if previous is not None:
                current = self._plan
                if current is None or current.generation != previous.generation:
                    return
            self._plan_generation += 1
            self._plan = _ActionPlan(
                generation=self._plan_generation,
                plan_id=plan_id,
                actions=actions,
                offset=0,
                horizon=horizon,
            )
            self._last_refine_offset = 0
            self._condition.notify_all()

    def _run_refine(self, job: _InferenceJob) -> None:
        plan = job.plan
        assert plan is not None
        control = {
            "op": "refine",
            "plan_id": plan.plan_id,
            "action_offset": job.refine_offset,
            "tactile_seq": job.tactile_seq,
        }
        actions, horizon = self._infer(
            job,
            control,
            actions_prefix=plan.actions[: job.refine_offset].tolist(),
        )
        if horizon != plan.horizon:
            raise StatefulTactileProtocolError("refine 返回的 action_horizon 与 plan 不一致。")

        with self._condition:
            current = self._plan
            if (
                job.episode_generation != self._episode_generation
                or current is None
                or current.generation != plan.generation
                or current.plan_id != plan.plan_id
            ):
                return
            self._delay_history.append(current.offset - plan.offset)
            commit_offset = max(current.offset, job.refine_offset)
            if commit_offset < current.horizon:
                # 请求在途期间已消费的动作保持不可变。
                merged = current.actions.copy()
                merged[commit_offset:] = actions[commit_offset:]
                self._plan = replace(current, actions=merged)
            self._condition.notify_all()

    def _infer(
        self,
        job: _InferenceJob,
        control: dict[str, Any],
        actions_prefix: list[list[float]] | None = None,
    ) -> tuple[np.ndarray, int]:
        snapshot = job.frame.snapshot
        t0 = time.perf_counter()
        resp = self._client.predict_stateful(
            snapshot.images,
            snapshot.state,
            prompt=snapshot.prompt,
            extra=snapshot.extra,
            stateful_tactile=control,
            actions_prefix=actions_prefix,
            request_id=uuid4().hex,
        )
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        self.request_count += 1

        plan_meta = resp.get("plan")
        if not isinstance(plan_meta, dict):
            raise StatefulTactileProtocolError(
                "stateful tactile 响应缺少 plan 元数据，服务器为旧版或不兼容。"
            )
        if (
            plan_meta["plan_id"] != control["plan_id"]
            or plan_meta["op"] != control["op"]
            or int(plan_meta["action_offset"]) != int(control["action_offset"])
            or int(plan_meta["tactile_seq"]) != int(control["tactile_seq"])
        ):
            raise StatefulTactileProtocolError(
                f"stateful tactile 响应与请求不匹配: {plan_meta}"
            )

        actions = np.asarray(resp["actions"], dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if self._raw_action_callback is not None:
            try:
                self._raw_action_callback(self.request_count, actions.tolist())
            except Exception:
                logger.warning("tactile raw action callback 失败", exc_info=True)
        logger.debug(
            "tactile %s #%d: offset=%d seq=%d infer=%.1fms",
            control["op"],
            self.request_count,
            control["action_offset"],
            control["tactile_seq"],
            self.last_infer_ms,
        )
        return actions, int(plan_meta["action_horizon"])
