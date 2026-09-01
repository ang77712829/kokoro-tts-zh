"""Behavior contract for the child-side worker stream protocol phase."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import queue
from types import SimpleNamespace

import pytest

from kokoro_tts.contracts.errors import WorkerFailureEnvelope
from kokoro_tts.workers import process_worker
from kokoro_tts.workers.spec import EngineWorkerSpec


pytestmark = pytest.mark.contract


class _BaseEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True


class _KwargsEngine(_BaseEngine):
    def __init__(
        self,
        frames: Iterable[object] = (),
        *,
        producer: Callable[[dict[str, object]], Iterable[object]] | None = None,
    ) -> None:
        super().__init__()
        self.frames = tuple(frames)
        self.producer = producer
        self.received: dict[str, object] = {}

    def synthesize_stream(self, **kwargs) -> Iterator[object]:
        self.received = dict(kwargs)
        frames = self.producer(self.received) if self.producer is not None else self.frames
        yield from frames


class _ExplicitCancelEngine(_BaseEngine):
    def __init__(self, observer: Callable[[Callable[[], bool]], Iterable[object]]) -> None:
        super().__init__()
        self.observer = observer
        self.received: dict[str, object] = {}

    def synthesize_stream(
        self,
        text: str,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[object]:
        self.received = {"text": text, "cancel_check": cancel_check}
        assert cancel_check is not None
        yield from self.observer(cancel_check)


class _LegacyEngine(_BaseEngine):
    def __init__(self) -> None:
        super().__init__()
        self.received_text = ""

    def synthesize_stream(self, text: str) -> Iterator[object]:
        self.received_text = text
        yield {"type": "done", "source": "legacy"}


class _CallFailureEngine(_BaseEngine):
    def synthesize_stream(self, **kwargs):
        raise ValueError("call failure")


class _IterationFailureEngine(_BaseEngine):
    def synthesize_stream(self, **kwargs) -> Iterator[object]:
        raise ValueError("iteration failure")
        yield  # pragma: no cover - makes the failure occur during iteration


def _unused_factory(config: object, provider: str | None) -> object:
    raise AssertionError("the test replaces create_worker_engine")


def _run_stream(
    monkeypatch: pytest.MonkeyPatch,
    engine: _BaseEngine,
    *,
    payload: dict[str, object] | None = None,
    cancel_flag: SimpleNamespace | None = None,
) -> tuple[list[tuple[str, str, object]], SimpleNamespace]:
    commands: queue.Queue[tuple[str, str, dict[str, object]]] = queue.Queue()
    results: queue.Queue[tuple[str, str, object]] = queue.Queue()
    flag = cancel_flag or SimpleNamespace(value=0)
    commands.put(("stream-request", "synthesize_stream", dict(payload or {})))
    commands.put(("shutdown-request", "shutdown", {}))
    monkeypatch.setattr(process_worker, "create_worker_engine", lambda config, spec: engine)

    process_worker._worker_main(
        SimpleNamespace(),
        EngineWorkerSpec("contract", _unused_factory),
        commands,
        results,
        flag,
    )

    stream_results: list[tuple[str, str, object]] = []
    while not results.empty():
        item = results.get_nowait()
        if item[0] == "stream-request":
            stream_results.append(item)
    return stream_results, flag


@pytest.mark.parametrize("terminal_type", ("done", "cancelled", "error", "segment_error"))
def test_events_forward_in_order_and_semantic_terminal_suppresses_repair(
    monkeypatch: pytest.MonkeyPatch,
    terminal_type: str,
) -> None:
    audio = {"type": "audio", "index": 0}
    terminal = {"type": terminal_type, "marker": terminal_type}
    results, _ = _run_stream(monkeypatch, _KwargsEngine((audio, terminal)))

    assert results == [
        ("stream-request", "event", audio),
        ("stream-request", "event", terminal),
        ("stream-request", "done", None),
    ]


def test_missing_terminal_repair_preserves_order_and_observed_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = {"type": "started", "segments": 4}
    audio_zero = {"type": "audio", "index": 0}
    audio_one = {"type": "audio", "index": 1}
    results, _ = _run_stream(
        monkeypatch,
        _KwargsEngine((started, audio_zero, audio_one)),
    )

    assert [item[:2] for item in results] == [
        ("stream-request", "event"),
        ("stream-request", "event"),
        ("stream-request", "event"),
        ("stream-request", "event"),
        ("stream-request", "event"),
        ("stream-request", "done"),
    ]
    assert [item[2] for item in results[:3]] == [started, audio_zero, audio_one]
    repair_error = results[3][2]
    assert isinstance(repair_error, dict)
    assert repair_error["type"] == "segment_error"
    assert set(repair_error) == {"type", "message"}
    assert isinstance(repair_error["message"], str) and repair_error["message"]
    assert results[4][2] == {
        "type": "done",
        "total_audio_chunks": 2,
        "total_segments": 4,
    }
    assert results[5] == ("stream-request", "done", None)


def test_missing_terminal_without_started_omits_unknown_segment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results, _ = _run_stream(
        monkeypatch,
        _KwargsEngine(({"type": "audio", "index": 0},)),
    )

    synthetic_done = results[-2][2]
    assert synthetic_done == {"type": "done", "total_audio_chunks": 1}
    assert results[-1] == ("stream-request", "done", None)


@pytest.mark.parametrize("signature", ("explicit", "kwargs"))
def test_compatible_engine_receives_child_generation_callback_and_sanitized_payload(
    monkeypatch: pytest.MonkeyPatch,
    signature: str,
) -> None:
    flag = SimpleNamespace(value=0)
    callback_results: list[bool] = []

    def observe(callback: Callable[[], bool]) -> Iterable[object]:
        callback_results.append(callback())
        flag.value = 7
        callback_results.append(callback())
        flag.value = 6
        callback_results.append(callback())
        flag.value = 0
        callback_results.append(callback())
        yield {"type": "done"}

    if signature == "explicit":
        engine: _BaseEngine = _ExplicitCancelEngine(observe)
    else:
        engine = _KwargsEngine(producer=lambda kwargs: observe(kwargs["cancel_check"]))  # type: ignore[arg-type]
    caller_callback = lambda: True
    results, _ = _run_stream(
        monkeypatch,
        engine,
        payload={
            "text": "contract",
            "cancel_check": caller_callback,
            "_cancel_generation": 7,
        },
        cancel_flag=flag,
    )

    received = engine.received  # type: ignore[attr-defined]
    assert "_cancel_generation" not in received
    assert callable(received["cancel_check"])
    assert received["cancel_check"] is not caller_callback
    assert callback_results == [False, True, False, False]
    assert results == [
        ("stream-request", "event", {"type": "done"}),
        ("stream-request", "done", None),
    ]


def test_legacy_engine_receives_neither_private_generation_nor_cancel_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _LegacyEngine()
    results, _ = _run_stream(
        monkeypatch,
        engine,
        payload={
            "text": "legacy text",
            "cancel_check": lambda: True,
            "_cancel_generation": 3,
        },
    )

    assert engine.received_text == "legacy text"
    assert results == [
        ("stream-request", "event", {"type": "done", "source": "legacy"}),
        ("stream-request", "done", None),
    ]


def test_matching_generation_cancellation_stops_before_next_frame_and_resets_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flag = SimpleNamespace(value=0)

    def frames(kwargs: dict[str, object]) -> Iterable[object]:
        yield {"type": "audio", "index": 0}
        flag.value = 5
        yield {"type": "audio", "index": 1}

    results, returned_flag = _run_stream(
        monkeypatch,
        _KwargsEngine(producer=frames),
        payload={"text": "cancel", "_cancel_generation": 5},
        cancel_flag=flag,
    )

    assert results == [
        ("stream-request", "event", {"type": "audio", "index": 0}),
        ("stream-request", "done", None),
    ]
    assert returned_flag.value == 0


@pytest.mark.parametrize(
    ("stream_generation", "foreign_token"),
    ((5, 6), (0, 9)),
)
def test_foreign_or_zero_generation_does_not_clear_another_cancel_token(
    monkeypatch: pytest.MonkeyPatch,
    stream_generation: int,
    foreign_token: int,
) -> None:
    flag = SimpleNamespace(value=foreign_token)
    results, returned_flag = _run_stream(
        monkeypatch,
        _KwargsEngine(({"type": "done"},)),
        payload={"text": "isolation", "_cancel_generation": stream_generation},
        cancel_flag=flag,
    )

    assert results[-1] == ("stream-request", "done", None)
    assert returned_flag.value == foreign_token


@pytest.mark.parametrize("engine_type", (_CallFailureEngine, _IterationFailureEngine))
def test_stream_exception_uses_outer_runtime_envelope_without_queue_done(
    monkeypatch: pytest.MonkeyPatch,
    engine_type: type[_BaseEngine],
) -> None:
    results, _ = _run_stream(monkeypatch, engine_type(), payload={"text": "failure"})

    assert len(results) == 1
    request_id, kind, payload = results[0]
    assert request_id == "stream-request"
    assert kind == "error"
    assert isinstance(payload, WorkerFailureEnvelope)
    assert payload.version == 1
    assert payload.code == "engine_runtime_failed"
    assert isinstance(payload.message, str) and "failure" in payload.message
    assert not any(item[1] == "done" for item in results)
