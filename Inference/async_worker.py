"""Asynchronous inference worker for streaming action chunks into a smoother."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from Core import ActionSpace
from Inference.action_processing import canonicalize_action_chunk
from Inference.action_smoothing import TemporalActionSmoother

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceObservationSnapshot:
    """Observation snapshot sent to the inference client."""

    images: dict[str, np.ndarray]
    state: list[float]
    prompt: str = ""
    extra: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


ObservationFn = Callable[[], InferenceObservationSnapshot | None]


class AsyncInferenceWorker:
    """Background loop that requests chunks and integrates them into a smoother."""

    def __init__(
        self,
        *,
        client: Any,
        observation_fn: ObservationFn | None = None,
        smoother: TemporalActionSmoother,
        action_space: Any = ActionSpace.JOINT_POSITION,
        policy_format: Any = "normal",
        canonical_dim: int = 32,
        effective_action_dim: int | None = None,
        overlap_steps: int = 8,
        max_latency_steps: int = 8,
        latency_compensation: bool = True,
        smooth_enabled: bool = True,
        obs_fps: float = 30.0,
        action_fps: float | None = None,
        raw_action_callback: Callable[[int, Any], None] | None = None,
    ) -> None:
        self._client = client
        self._observation_fn = observation_fn
        self._smoother = smoother
        self._action_space = action_space
        self._policy_format = policy_format
        self._canonical_dim = int(canonical_dim)
        self._effective_action_dim = effective_action_dim
        self._overlap_steps = max(1, int(overlap_steps))
        self._max_latency_steps = max(0, int(max_latency_steps))
        self._latency_compensation = bool(latency_compensation)
        self._smooth_enabled = bool(smooth_enabled)
        self._obs_fps = max(float(obs_fps), 1e-6)
        self._action_fps = max(float(action_fps if action_fps is not None else obs_fps), 1e-6)
        self._raw_action_callback = raw_action_callback
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest_observation: InferenceObservationSnapshot | None = None
        self._paused = False
        self._epoch = 0
        self._thread: threading.Thread | None = None
        self.request_count = 0
        self.last_infer_ms: float | None = None
        self.last_error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stop_event.set()
        self.invalidate_pending()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def update_observation(self, obs: InferenceObservationSnapshot) -> None:
        """Publish the latest main-loop observation for the background thread."""
        snapshot = InferenceObservationSnapshot(
            images=dict(obs.images),
            state=list(obs.state),
            prompt=obs.prompt,
            extra=dict(obs.extra) if isinstance(obs.extra, dict) else obs.extra,
            meta=dict(obs.meta),
        )
        with self._state_lock:
            self._latest_observation = snapshot

    def pause(self) -> None:
        """Drop buffered work and reject any in-flight inference result."""
        with self._state_lock:
            self._paused = True
            self._epoch += 1
        self._smoother.clear()

    def resume(self) -> None:
        """Resume inference after pause, starting from fresh future chunks."""
        with self._state_lock:
            self._paused = False
            self._epoch += 1
        self._smoother.clear()

    def invalidate_pending(self) -> None:
        """Reject chunks returned by requests started before this call."""
        with self._state_lock:
            self._epoch += 1

    def _next_observation(self) -> tuple[InferenceObservationSnapshot | None, int]:
        with self._state_lock:
            if self._paused:
                return None, self._epoch
            epoch = self._epoch
            latest = self._latest_observation
        if self._observation_fn is not None:
            return self._observation_fn(), epoch
        return latest, epoch

    def _is_stale(self, epoch: int) -> bool:
        if self._stop_event.is_set():
            return True
        with self._state_lock:
            return self._paused or self._epoch != epoch

    def run_once(self) -> bool:
        """Request one chunk and integrate it. Returns True if a chunk was added."""
        obs, epoch = self._next_observation()
        if obs is None:
            return False

        t0 = time.perf_counter()
        raw_actions = self._client.predict_chunk(
            obs.images,
            obs.state,
            prompt=obs.prompt,
            extra=obs.extra,
        )
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        self.request_count += 1

        if raw_actions is None or len(raw_actions) == 0:
            logger.warning("async inference returned empty action chunk")
            return False
        if self._is_stale(epoch):
            logger.debug("async inference chunk dropped after pause/stop")
            return False

        if self._raw_action_callback is not None:
            try:
                self._raw_action_callback(self.request_count, raw_actions)
            except ValueError as exc:
                logger.debug("async raw action callback skipped: %s", exc)
            except Exception:
                logger.warning("async raw action callback failed", exc_info=True)

        actions = canonicalize_action_chunk(
            raw_actions,
            self._action_space,
            policy_format=self._policy_format,
            canonical_dim=self._canonical_dim,
            effective_action_dim=self._effective_action_dim,
        )
        latency_steps = None if self._latency_compensation else 0
        self._smoother.integrate_chunk(
            actions,
            latency_steps=latency_steps,
            max_latency_steps=self._max_latency_steps if latency_steps is None else None,
            overlap_steps=self._overlap_steps,
            smooth_enabled=self._smooth_enabled,
        )
        logger.debug(
            "async chunk #%d: actions=%d infer=%.1fms latency=%s max_latency=%d smooth=%s remaining=%d",
            self.request_count,
            len(actions),
            self.last_infer_ms,
            "published_steps" if latency_steps is None else latency_steps,
            self._max_latency_steps,
            self._smooth_enabled,
            self._smoother.remaining,
        )
        return True

    def _run_loop(self) -> None:
        period = 1.0 / self._obs_fps
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                self.run_once()
                self.last_error = None
            except Exception as e:
                self.last_error = e
                logger.warning("async inference worker error: %s", e, exc_info=True)

            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s <= 0:
                next_t = time.perf_counter()
                sleep_s = period
            self._stop_event.wait(sleep_s)
