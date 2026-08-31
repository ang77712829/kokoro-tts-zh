"""Freeze the current post-queue-done worker stream tail behavior.

The contract is behavioral: it permits the tail phase to remain inline or move
to one private helper.  Queue-level completion and semantic stream terminal
events deliberately remain distinct.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from kokoro_tts.contracts.errors import EngineError, WorkerFailureEnvelope
from kokoro_tts.workers import process_worker as process_worker_module
from kokoro_tts.workers.process_worker import EngineProcessClient


pytestmark = pytest.mark.contract

_CURRENT_REQUEST = "current-worker-request"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _TrackingLock:
    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.depth -= 1


@dataclass
class _QueuedResult:
    delay: float
    raw: tuple[str, str, object]


class _ResultQueue:
    def __init__(
        self,
        clock: _FakeClock,
        lock: _TrackingLock,
        values: list[_QueuedResult],
    ) -> None:
        self.clock = clock
        self.lock = lock
        self.values = list(values)
        self.get_timeouts: list[float] = []

    def get(self, *, timeout: float):
        assert self.lock.depth == 1, "stream tail must remain under the caller-owned request lock"
        timeout = float(timeout)
        self.get_timeouts.append(timeout)
        if not self.values:
            self.clock.advance(timeout)
            raise queue.Empty

        current = self.values[0]
        if current.delay > timeout:
            current.delay -= timeout
            self.clock.advance(timeout)
            raise queue.Empty

        self.values.pop(0)
        self.clock.advance(current.delay)
        return current.raw


def _queued(
    kind: str,
    payload: object = None,
    *,
    request_id: str = _CURRENT_REQUEST,
    delay: float = 0.0,
) -> _QueuedResult:
    return _QueuedResult(delay, (request_id, kind, payload))


def _client(
    monkeypatch: pytest.MonkeyPatch,
    values: list[_QueuedResult],
    *,
    drain_timeout: float = 0.1,
) -> tuple[EngineProcessClient, _ResultQueue, _FakeClock, _TrackingLock, list[int]]:
    clock = _FakeClock()
    lock = _TrackingLock()
    result_queue = _ResultQueue(clock, lock, values)
    soft_cancels: list[int] = []

    client = object.__new__(EngineProcessClient)
    client._request_lock = lock
    client._stream_generation = 0
    client._cancel_flag = SimpleNamespace(value=0)
    client.config = SimpleNamespace(
        engine_process_stream_idle_timeout_seconds=5.0,
        engine_process_stream_drain_seconds=drain_timeout,
    )
    client.engine_id = "kokoro"
    client.logger = None
    client._process = None
    client._send = lambda command, payload: _CURRENT_REQUEST
    client._raise_if_worker_exited = lambda: None
    client._require_result_queue = lambda: result_queue
    client._soft_cancel_worker = lambda generation=None: soft_cancels.append(int(generation or 0))
    client.close = lambda *, kill=False: pytest.fail("tail characterization must not kill the worker")

    monkeypatch.setattr(process_worker_module.time, "monotonic", clock.monotonic)
    return client, result_queue, clock, lock, soft_cancels


def test_tail_typed_failure_uses_the_existing_engine_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = WorkerFailureEnvelope(
        version=1,
        code="engine_runtime_failed",
        message="child synthesis failed",
    )
    client, _, _, lock, _ = _client(
        monkeypatch,
        [_queued("done"), _queued("error", envelope)],
    )

    with pytest.raises(EngineError) as captured:
        list(client.stream({}, timeout=1.0))

    assert captured.value.code == "engine_runtime_failed"
    assert captured.value.message == "child synthesis failed"
    assert lock.depth == 0


def test_duplicate_queue_done_is_tolerated_and_tail_drain_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = {"type": "done", "total_audio_chunks": 1}
    client, result_queue, _, _, _ = _client(
        monkeypatch,
        [
            _queued("done"),
            _queued("done"),
            _queued("event", terminal),
            _queued("event", {"type": "audio", "data": "must-remain"}),
        ],
    )

    assert list(client.stream({}, timeout=1.0)) == [terminal]
    assert len(result_queue.values) == 1


def test_stale_tail_request_is_discarded_without_hiding_current_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = {"type": "cancelled"}
    client, _, _, _, _ = _client(
        monkeypatch,
        [
            _queued("done"),
            _queued("event", {"type": "audio", "data": "stale"}, request_id="old-request"),
            _queued("event", terminal),
        ],
    )

    assert list(client.stream({}, timeout=1.0)) == [terminal]


def test_unknown_same_request_tail_kind_is_a_stable_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _, _, _ = _client(
        monkeypatch,
        [_queued("done"), _queued("mystery", {"unexpected": True})],
    )

    with pytest.raises(EngineError) as captured:
        list(client.stream({}, timeout=1.0))

    assert captured.value.code == "worker_protocol_failed"
    assert "mystery" in captured.value.message


def test_sliding_tail_grace_is_extended_by_events_but_capped_by_hard_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_count = 140
    events = [
        _queued("event", {"type": "audio", "index": index}, delay=0.04)
        for index in range(event_count)
    ]
    client, result_queue, clock, _, _ = _client(
        monkeypatch,
        [_queued("done"), *events],
    )

    frames = list(client.stream({}, timeout=1.0))

    # The original 0.5s grace would admit roughly 12 frames. Repeated events
    # slide that grace, while the fixed 5s hard limit prevents all 140 frames.
    assert 20 < len(frames) < event_count
    assert 4.9 <= clock.now <= 5.05
    assert len(result_queue.values) == event_count - len(frames)
    assert frames == [
        {"type": "audio", "index": index}
        for index in range(len(frames))
    ]


@pytest.mark.parametrize("terminal_type", ["done", "cancelled", "error", "segment_error"])
def test_tail_semantic_terminal_event_is_yielded_once_and_stops_the_phase(
    monkeypatch: pytest.MonkeyPatch,
    terminal_type: str,
) -> None:
    terminal = {"type": terminal_type, "message": "terminal"}
    client, result_queue, _, _, _ = _client(
        monkeypatch,
        [
            _queued("done"),
            _queued("event", terminal),
            _queued("event", {"type": "audio", "data": "after-terminal"}),
        ],
    )

    assert list(client.stream({}, timeout=1.0)) == [terminal]
    assert len(result_queue.values) == 1


def test_late_nonterminal_event_survives_queue_done_until_semantic_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = {"type": "audio", "data": "late-audio"}
    terminal = {"type": "done", "total_audio_chunks": 1}
    client, _, _, _, _ = _client(
        monkeypatch,
        [_queued("done"), _queued("event", audio), _queued("event", terminal)],
    )

    assert list(client.stream({}, timeout=1.0)) == [audio, terminal]


def test_cancel_drain_queue_done_short_circuits_before_tail_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, result_queue, _, lock, soft_cancels = _client(
        monkeypatch,
        [
            _queued("done"),
            _queued("event", {"type": "audio", "data": "late-after-cancel"}),
        ],
    )

    assert list(client.stream({}, timeout=1.0, cancel_check=lambda: True)) == []
    assert soft_cancels == [1]
    assert len(result_queue.values) == 1
    assert lock.depth == 0
