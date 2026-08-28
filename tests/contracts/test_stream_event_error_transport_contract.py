"""Contracts for existing stream events and their transport semantics.

The classifiers and matrices in this module are test-only interpretation aids,
not production APIs or proposals for a new public event model.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import inspect
from pathlib import Path
import queue
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from kokoro_tts.contracts import StreamingRequest, StreamingResult, WorkerFailureEnvelope
from kokoro_tts.engines.adapters.kokoro import KokoroAdapter
from kokoro_tts.engines.adapters.moss import MossAdapter
from kokoro_tts.engines.adapters.zipvoice import ZipVoiceEngine as ExportedZipVoiceEngine
from kokoro_tts.engines.base import EngineAdapter
from kokoro_tts.services.streaming_service import StreamingService
from kokoro_tts.validation import websocket_error_frame_from_http
from kokoro_tts.workers import EngineWorkerSpec, process_worker
from kokoro_tts.workers.process_worker import EngineProcessClient, WorkerResult
from kokoro_tts.ws.cancel import CancelLifecycleMixin
from kokoro_tts.ws.session import TtsWebSocketSession
from kokoro_tts.ws.state import WsSessionState
from kokoro_tts.ws.streaming import StreamingLoopMixin
from kokoro_tts.zipvoice.engine import ZipVoiceEngine


pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


class _SemanticOutcome(str, Enum):
    """TEST-ONLY CLASSIFIER; NOT A PRODUCTION API."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    DISCONNECTED = "DISCONNECTED"
    NO_SEMANTIC_OUTCOME = "NO_SEMANTIC_OUTCOME"


@dataclass(frozen=True)
class _SequenceMeaning:
    """TEST-ONLY interpretation of the current WebSocket session behavior."""

    request_outcome: _SemanticOutcome
    failed_segments: tuple[int | None, ...]
    protocol_closed: bool


def _classify_ws_sequence(frames: Iterable[dict[str, Any]]) -> _SequenceMeaning:
    """Classify current WS state ownership without defining a public schema."""

    request_outcome = _SemanticOutcome.NO_SEMANTIC_OUTCOME
    failed_segments: list[int | None] = []
    protocol_closed = False
    for frame in frames:
        frame_type = str(frame.get("type") or "")
        if frame_type == "error":
            request_outcome = _SemanticOutcome.FAILURE
        elif frame_type == "segment_error":
            failed_segments.append(frame.get("index"))
            # The engine event is segment-scoped, while the current WS send loop
            # records both error kinds as request failure. Keep both facts.
            request_outcome = _SemanticOutcome.FAILURE
        elif frame_type == "cancelled":
            request_outcome = _SemanticOutcome.CANCELLED
            protocol_closed = True
        elif frame_type == "done":
            protocol_closed = True
            if request_outcome is _SemanticOutcome.NO_SEMANTIC_OUTCOME:
                request_outcome = _SemanticOutcome.SUCCESS
    return _SequenceMeaning(request_outcome, tuple(failed_segments), protocol_closed)


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _run_worker_stream(
    monkeypatch: pytest.MonkeyPatch,
    frames: Iterable[object],
    *,
    cancel_generation: int = 0,
) -> list[tuple[str, str, object]]:
    """Run the child entry point synchronously with in-memory queues."""

    class _WorkerEngine:
        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

        def synthesize_stream(self, **kwargs):
            yield from frames

    cancel_flag = SimpleNamespace(value=cancel_generation)
    command_queue: queue.Queue[tuple[str, str, dict[str, Any]]] = queue.Queue()
    result_queue: queue.Queue[tuple[str, str, object]] = queue.Queue()
    payload: dict[str, Any] = {"text": "contract text"}
    if cancel_generation:
        payload["_cancel_generation"] = cancel_generation
    command_queue.put(("request-stream", "synthesize_stream", payload))
    command_queue.put(("request-shutdown", "shutdown", {}))
    monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args: _WorkerEngine())

    process_worker._worker_main(
        SimpleNamespace(),
        EngineWorkerSpec("kokoro", str),
        command_queue,
        result_queue,
        cancel_flag,
    )

    results: list[tuple[str, str, object]] = []
    while not result_queue.empty():
        result = result_queue.get_nowait()
        if result[0] == "request-stream":
            results.append(result)
    return results


class _FakeEngine:
    def __init__(self, frames: Iterable[object]) -> None:
        self.frames = list(frames)
        self.received: dict[str, Any] = {}

    def synthesize_stream(self, text, voice, speed, fmt, *, cancel_check=None):
        self.received = {
            "text": text,
            "voice": voice,
            "speed": speed,
            "fmt": fmt,
            "cancel_check": cancel_check,
        }
        yield from self.frames


class _FakeManager:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine

    @contextmanager
    def borrow(self, model_id: str):
        assert model_id == "kokoro"
        yield self.engine


def _streaming_service(frames: Iterable[object]) -> tuple[StreamingService, _FakeEngine]:
    engine = _FakeEngine(frames)
    state = SimpleNamespace(cfg=SimpleNamespace(), model_manager=_FakeManager(engine))
    return StreamingService(state), engine


def _request() -> StreamingRequest:
    return StreamingRequest(
        text="contract text",
        model_id="kokoro",
        voice="voice",
        speed=1.0,
        request_id="request-contract",
    )


class _FakeState:
    def __init__(self) -> None:
        self.cancelled = False
        self.events: list[tuple[Any, ...]] = []

    def is_cancelled(self, request_id: str) -> bool:
        return self.cancelled

    def request_cancel(self, request_id: str) -> None:
        self.cancelled = True
        self.events.append(("request_cancel", request_id))

    def inc_stat(self, key: str, amount: float = 1) -> None:
        self.events.append(("inc_stat", key, amount))

    def mark_request(self, request_id: str, status: str, **extra) -> None:
        self.events.append(("mark_request", request_id, status, extra))

    def finish_request(self, request_id: str, status: str, **extra) -> None:
        self.events.append(("finish_request", request_id, status, extra))


class _FakeWebSocket:
    def __init__(self) -> None:
        self.json_frames: list[dict[str, Any]] = []
        self.binary_frames: list[bytes] = []

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.json_frames.append(dict(frame))

    async def send_bytes(self, payload: bytes) -> None:
        self.binary_frames.append(bytes(payload))


class _LoopHarness(StreamingLoopMixin, CancelLifecycleMixin):
    def __init__(self) -> None:
        self.websocket = _FakeWebSocket()
        self.state = _FakeState()
        self.cfg = SimpleNamespace(request_timeout_seconds=5, websocket_stream_idle_timeout_seconds=5)
        self.request_id = "request-ws"
        self.queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=4)
        self.done_marker = object()
        self.cancel_event = asyncio.Event()
        self.cancelled_by_client = False
        self.cancel_notified = False
        self.saw_stream_error = False
        self.saw_stream_terminal = False
        self.stream_error_counted = False
        self.phase = WsSessionState.RUNNING
        self.control_task = None
        self.producer_task = None

    def _transition(self, phase: WsSessionState, **extra) -> None:
        self.phase = phase


class TestStreamRepresentationOwnership:
    """STATIC OWNERSHIP CONTRACT."""

    def test_real_representation_owners_and_absent_transports(self) -> None:
        assert not (ROOT / "src/kokoro_tts/streaming").exists()
        assert "@router.websocket(\"/ws/v1/tts\")" in _source("src/kokoro_tts/routes/ws.py")
        route_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "src/kokoro_tts/routes").glob("*.py"))
        )
        assert "EventSourceResponse" not in route_sources
        assert "text/event-stream" not in route_sources
        assert inspect.isgeneratorfunction(StreamingService.iter_frames)
        assert tuple(WorkerResult.__dataclass_fields__) == ("request_id", "kind", "payload")

    def test_streaming_result_is_the_existing_project_mapping(self) -> None:
        native = {
            "type": "audio",
            "data": "YXVkaW8=",
            "index": 0,
            "model": "engine-model",
        }
        result = StreamingResult.from_frame(native, model_id="fallback-model", request_id="request-1")
        assert result.type == "audio"
        assert result.as_frame() == {**native, "request_id": "request-1"}
        assert native == {"type": "audio", "data": "YXVkaW8=", "index": 0, "model": "engine-model"}


class TestThreeEngineStreamingModes:
    """CURRENT-BEHAVIOR CHARACTERIZATION."""

    def test_capability_modes_and_implementation_boundaries(self) -> None:
        registry = _source("src/kokoro_tts/engines/registry.py")
        kokoro = _source("src/kokoro_tts/engine.py")
        moss = _source("src/kokoro_tts/moss_engine_streaming.py")
        zipvoice = _source("src/kokoro_tts/zipvoice/engine.py")
        assert 'stream_mode="segmented"' in registry
        assert 'stream_mode="native"' in registry
        assert "_synthesize_segment" in kokoro and "_split_stream_audio" in kokoro
        assert "_push_stream_waveforms" in moss and "_synthesize_stream_process_isolated" in moss
        assert "self.synthesize(segment" in zipvoice and '"stream_mode": "segmented"' in zipvoice

    def test_p3a1_adapter_boundary_remains_owned_by_existing_types(self) -> None:
        assert getattr(KokoroAdapter, "synthesize_stream") is not None
        assert getattr(MossAdapter, "synthesize_stream") is not None
        assert ExportedZipVoiceEngine is ZipVoiceEngine
        assert EngineAdapter.__name__ == "EngineAdapter"


class TestSemanticOutcomeAndProtocolClosure:
    """PUBLIC PROTOCOL COMPATIBILITY CONTRACT."""

    @pytest.mark.parametrize("error_type", ["error", "segment_error"])
    def test_error_then_done_closes_without_overwriting_failure(self, error_type: str) -> None:
        frames = [{"type": error_type, "index": 2}, {"type": "done"}]
        meaning = _classify_ws_sequence(frames)
        assert meaning.request_outcome is _SemanticOutcome.FAILURE
        assert meaning.protocol_closed is True
        assert meaning.failed_segments == ((2,) if error_type == "segment_error" else ())

    def test_current_ws_success_needs_absence_of_recorded_error_or_cancel(self) -> None:
        meaning = _classify_ws_sequence([{"type": "started"}, {"type": "audio"}, {"type": "done"}])
        assert meaning == _SequenceMeaning(_SemanticOutcome.SUCCESS, (), True)
        assert _classify_ws_sequence([{"type": "done"}]).request_outcome is _SemanticOutcome.SUCCESS
        assert _classify_ws_sequence([]).request_outcome is _SemanticOutcome.NO_SEMANTIC_OUTCOME

    def test_service_preserves_compatible_error_then_done_sequences(self) -> None:
        for error_type in ("error", "segment_error"):
            service, _ = _streaming_service([{"type": error_type, "message": "x"}, {"type": "done"}])
            frames = list(service.iter_frames(_request()))
            assert [frame["type"] for frame in frames] == [error_type, "done"]
            assert all(frame["request_id"] == "request-contract" for frame in frames)

    def test_generator_exhaustion_and_cancellation_have_distinct_service_frames(self) -> None:
        service, _ = _streaming_service([])
        exhausted = list(service.iter_frames(_request()))
        assert [frame["type"] for frame in exhausted] == ["segment_error"]
        assert _classify_ws_sequence(exhausted).request_outcome is _SemanticOutcome.FAILURE

        service, _ = _streaming_service([{"type": "audio", "data": "YQ=="}, {"type": "done"}])
        cancelled = list(service.iter_frames(_request(), cancel_check=lambda: True))
        assert [frame["type"] for frame in cancelled] == ["cancelled"]


class TestPublicWebSocketCompatibility:
    """BEHAVIORAL MAPPING and PUBLIC PROTOCOL COMPATIBILITY CONTRACT."""

    @pytest.mark.parametrize("error_type", ["error", "segment_error"])
    def test_send_loop_keeps_error_then_done_but_finish_remains_error(self, error_type: str) -> None:
        async def exercise() -> _LoopHarness:
            harness = _LoopHarness()
            await harness.queue.put({"type": error_type, "message": "engine failed", "index": 1})
            await harness.queue.put({"type": "done", "total_audio_chunks": 0})
            await harness.queue.put(harness.done_marker)
            await harness._send_loop(binary=False)
            harness._finish(0.0)
            return harness

        harness = asyncio.run(exercise())
        assert [frame["type"] for frame in harness.websocket.json_frames] == [error_type, "done"]
        assert harness.saw_stream_error is True
        assert harness.phase is WsSessionState.ERROR
        assert any(event[:3] == ("finish_request", "request-ws", "error") for event in harness.state.events)

    def test_binary_audio_is_json_metadata_followed_by_decoded_bytes(self) -> None:
        async def exercise() -> _LoopHarness:
            harness = _LoopHarness()
            await harness.queue.put({"type": "audio", "data": base64.b64encode(b"pcm").decode(), "index": 0})
            await harness.queue.put({"type": "done", "total_audio_chunks": 1})
            await harness.queue.put(harness.done_marker)
            await harness._send_loop(binary=True)
            return harness

        harness = asyncio.run(exercise())
        assert harness.websocket.json_frames[0] == {"type": "audio", "index": 0, "request_id": "request-ws"}
        assert harness.websocket.binary_frames == [b"pcm"]
        assert harness.websocket.json_frames[-1]["type"] == "done"

    def test_public_request_fields_close_codes_and_error_shape_are_unchanged(self) -> None:
        session = _source("src/kokoro_tts/ws/session.py")
        route = _source("src/kokoro_tts/routes/ws.py")
        for field in (
            "token", "model", "voice", "format", "binary", "prompt_audio", "prompt_audio_data",
            "reference_audio_data", "prompt_audio_filename", "prompt_text", "engine_params",
            "text_normalization", "tn_engine", "text", "speed",
        ):
            assert f'"{field}"' in session
        assert "code=1008" in session
        assert "code=1009" in session
        assert "code=1003" in session
        assert "code=1013" in route
        frame = websocket_error_frame_from_http(
            SimpleNamespace(detail={"code": "BAD_REQUEST", "message": "invalid"}),
            request_id="request-1",
        )
        assert frame == {"type": "error", "code": "BAD_REQUEST", "message": "invalid", "request_id": "request-1"}


class TestWorkerMessageMapping:
    """CURRENT-BEHAVIOR CHARACTERIZATION."""

    def test_worker_tuple_kinds_and_protocol_done_are_separate(self) -> None:
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        assert '(request_id, "event", item)' in worker
        assert '(request_id, "result",' in worker
        assert '(request_id, "error",' in worker
        assert '(request_id, "done", None)' in worker
        assert 'done_payload = {"type": "done"' in worker
        assert "始终发送终止消息" in worker

    def test_worker_error_uses_typed_transport_without_public_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failing_frames():
            raise ValueError("detail")
            yield  # pragma: no cover - makes this a generator

        results = _run_worker_stream(monkeypatch, failing_frames())
        assert len(results) == 1
        request_id, kind, payload = results[0]
        assert request_id == "request-stream"
        assert kind == "error"
        assert isinstance(payload, WorkerFailureEnvelope)
        assert (payload.version, payload.code, payload.message) == (
            1,
            "engine_runtime_failed",
            "ValueError: detail",
        )
        assert not isinstance(payload, BaseException)
        assert "Traceback" not in payload.message

    def test_missing_child_terminal_maps_to_failure_then_protocol_and_queue_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _run_worker_stream(
            monkeypatch,
            [{"type": "started", "segments": 1}, {"type": "audio", "index": 0}],
        )
        event_frames = [payload for _, kind, payload in results if kind == "event"]

        assert [frame["type"] for frame in event_frames] == [
            "started",
            "audio",
            "segment_error",
            "done",
        ]
        assert set(event_frames[2]) == {"type", "message"}
        assert isinstance(event_frames[2]["message"], str)
        assert event_frames[2]["message"]
        assert event_frames[3] == {"type": "done", "total_audio_chunks": 1, "total_segments": 1}
        assert results[-1] == ("request-stream", "done", None)
        assert _classify_ws_sequence(event_frames) == _SequenceMeaning(
            _SemanticOutcome.FAILURE,
            (None,),
            True,
        )

    def test_explicit_child_done_does_not_gain_a_failure_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _run_worker_stream(monkeypatch, [{"type": "done", "total_audio_chunks": 0}])

        assert results == [
            ("request-stream", "event", {"type": "done", "total_audio_chunks": 0}),
            ("request-stream", "done", None),
        ]

    def test_cancelled_generation_does_not_gain_a_missing_terminal_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        results = _run_worker_stream(
            monkeypatch,
            [{"type": "audio", "index": 0}],
            cancel_generation=7,
        )

        assert results == [("request-stream", "done", None)]


class TestCancellationDisconnectAndLateAudioBoundary:
    """BEHAVIORAL MAPPING CONTRACT plus EXPLICIT NON-GUARANTEE."""

    def test_cancel_notice_drains_queued_late_audio_and_is_idempotent(self) -> None:
        async def exercise() -> _LoopHarness:
            harness = _LoopHarness()
            await harness.queue.put({"type": "audio", "index": 99})
            await harness._notify_cancelled()
            await harness._notify_cancelled()
            return harness

        harness = asyncio.run(exercise())
        first = harness.queue.get_nowait()
        second = harness.queue.get_nowait()
        assert first == {"type": "cancelled", "request_id": "request-ws"}
        assert second is harness.done_marker
        assert harness.queue.empty()

    def test_current_sources_only_offer_best_effort_end_to_end_stop(self) -> None:
        service = _source("src/kokoro_tts/services/streaming_service.py")
        ws_streaming = _source("src/kokoro_tts/ws/streaming.py")
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        moss = _source("src/kokoro_tts/moss_engine_streaming.py")
        assert "cancel_check" in service
        assert "continue" in ws_streaming and "取消后的旧音频不再推给前端" in ws_streaming
        assert "_soft_cancel_worker" in worker and "drain_deadline" in worker
        assert "无法中断正在进行的 ONNX/CUDA 单帧推理" in moss

    def test_disconnect_and_send_failure_request_cancellation_without_acknowledged_engine_stop(self) -> None:
        streaming = _source("src/kokoro_tts/ws/streaming.py")
        cancel = _source("src/kokoro_tts/ws/cancel.py")
        assert "except WebSocketDisconnect" in streaming
        assert "self.state.request_cancel(self.request_id)" in streaming
        assert "self.cancel_event.set()" in streaming
        assert "await asyncio.wait_for(self.producer_task, timeout=5.0)" in cancel
        assert "producer_task.cancel()" in cancel


class TestQueueBackpressureAndCleanupBoundary:
    """EXPLICIT NON-GUARANTEE and per-request cleanup contract."""

    def test_channel_bounds_and_shutdown_markers_are_current_characterization(self) -> None:
        session = _source("src/kokoro_tts/ws/session.py")
        streaming = _source("src/kokoro_tts/ws/streaming.py")
        moss = _source("src/kokoro_tts/moss_engine_streaming.py")
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        assert "asyncio.Queue(maxsize=4)" in session
        assert "queue.Queue(maxsize=max(1, int(self.config.moss_stream_queue_max_items)))" in moss
        assert "item_queue.put(item, timeout=0.1)" in moss
        assert "self._ctx.Queue()" in worker
        assert "self.queue.put(item)" in streaming
        assert "fut.result(timeout=min(0.5" in streaming
        assert "self.done_marker" in streaming
        assert '(request_id, "done", None)' in worker

    def test_bounded_local_queues_do_not_claim_end_to_end_backpressure(self) -> None:
        worker = inspect.getsource(EngineProcessClient.start)
        producer = inspect.getsource(StreamingLoopMixin._thread_put)
        assert "Queue()" in worker and "maxsize" not in worker
        assert "run_coroutine_threadsafe" in producer
        assert "queue_wait_limit" in producer
        assert "return False" in producer

    def test_per_request_cleanup_owners_are_distinct_from_application_shutdown(self) -> None:
        session = _source("src/kokoro_tts/ws/session.py")
        cancel = _source("src/kokoro_tts/ws/cancel.py")
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        assert "delete_prompt_audio_path" in session
        assert "await self._cancel_background_tasks()" in session
        assert "await self.websocket.close()" in session
        assert "self.state.finish_request" in cancel
        assert "_stream_generation" in worker
        assert "丢弃过期流式消息" in worker
        for global_owner in ("uvicorn", "container restart", "close_all"):
            assert global_owner not in inspect.getsource(CancelLifecycleMixin)


def test_contract_source_shape_is_hermetic_and_contract_only() -> None:
    """STATIC OWNERSHIP CONTRACT for this contract's own execution seam."""

    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        ".write_" + "text(",
        "Test" + "Client(",
        "websocket_" + "connect(",
        "multiprocessing." + "Process(",
        "threading." + "Thread(",
        "requests." + "get(",
        "http" + "x.",
        "tor" + "ch.",
    )
    assert all(token not in source for token in forbidden)
    status_contract = ROOT / "tests/contracts/test_engine_adapter_conformance_contract.py"
    assert status_contract.exists()
    assert [base.__name__ for base in TtsWebSocketSession.__bases__] == [
        "MessageParsingMixin",
        "StreamingLoopMixin",
        "CancelLifecycleMixin",
    ]
