from __future__ import annotations

import Inference

Inference.InferenceClient = object

import Example.robot_inference as robot_inference


class _KeyWorker:
    def __init__(self, key: int | None) -> None:
        self._key = key

    def poll_key(self) -> int | None:
        return self._key


def test_enter_starts_waiting_inference_from_stdin_flag():
    robot_inference._running = True
    robot_inference._enter_key_pressed = True
    robot_inference._reset_key_pressed = False

    assert robot_inference._consume_start_signal(None) == "start"
    assert robot_inference._enter_key_pressed is False


def test_reset_key_does_not_start_waiting_inference():
    robot_inference._running = True
    robot_inference._enter_key_pressed = False
    robot_inference._reset_key_pressed = True

    assert robot_inference._consume_start_signal(None) is None
    assert robot_inference._reset_key_pressed is False


def test_enter_starts_waiting_inference_from_camera_window():
    robot_inference._running = True
    robot_inference._enter_key_pressed = False
    robot_inference._reset_key_pressed = False

    assert robot_inference._consume_start_signal(_KeyWorker(13)) == "start"


def test_q_quits_waiting_inference_from_camera_window():
    robot_inference._running = True
    robot_inference._enter_key_pressed = False
    robot_inference._reset_key_pressed = False

    assert robot_inference._consume_start_signal(_KeyWorker(ord("q"))) == "quit"
    assert robot_inference._running is False
