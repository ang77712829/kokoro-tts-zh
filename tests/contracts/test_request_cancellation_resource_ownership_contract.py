"""Freeze current request-cancellation and per-request ownership boundaries.

The classifiers and matrices below are test-only interpretation aids.  They
are not production APIs, public acknowledgement types, or implementation
proposals.  Known gaps are characterized without turning them into desired
behavior.
"""

from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
from enum import Enum
import inspect
from pathlib import Path
import queue
import textwrap
from types import SimpleNamespace
from typing import Any, Iterable

import pytest
from fastapi import APIRouter, HTTPException

from kokoro_tts.contracts import CancellationContext, StreamingRequest, StreamingResult
from kokoro_tts import moss_engine_streaming as moss_streaming_module
from kokoro_tts.moss_engine_streaming import MossStreamingMixin
from kokoro_tts.routes.status_parts import StatusRouteContext
from kokoro_tts.routes.status_parts.control import attach_control_routes
from kokoro_tts.services import synthesis_service as synthesis_module
from kokoro_tts.services.state_parts.request_registry import RequestRegistryMixin
from kokoro_tts.services import streaming_service as streaming_module
from kokoro_tts.services.streaming_service import StreamingService
from kokoro_tts.services.synthesis_service import SynthesisService
from kokoro_tts.service_state import ServiceState
from kokoro_tts.workers import EngineWorkerSpec, process_worker
from kokoro_tts.workers.process_worker import EngineProcessClient, WorkerResult
from kokoro_tts.ws.session import TtsWebSocketSession
from kokoro_tts.ws.state import WsSessionState


pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class _NoopLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _TrackingLock(_NoopLock):
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.exited += 1
        return False


class _Registry(RequestRegistryMixin):
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(queue_status_enabled=True)
        self.request_lock = _NoopLock()
        self.active_requests: dict[str, dict[str, Any]] = {}
        self.cancelled_requests: set[str] = set()
        self.stats: dict[str, int] = {}

    def inc_stat(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount


class _RouteState(_Registry):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(
            queue_status_enabled=True,
            request_timeout_seconds=5,
            websocket_stream_idle_timeout_seconds=5,
        )
        self.request_cancel_calls: list[str] = []
        self.model_manager = _ForbiddenRuntimeOwner()

    def new_request_id(self) -> str:
        return "public-request"

    def request_cancel(self, request_id: str) -> bool:
        self.request_cancel_calls.append(request_id)
        return super().request_cancel(request_id)

    def is_cancelled(self, request_id: str) -> bool:
        return super().is_cancelled(request_id)


class _ForbiddenRuntimeOwner:
    def __getattr__(self, name: str):
        raise AssertionError(f"HTTP cancel must not reach runtime owner: {name}")


async def _verify_nothing(*args, **kwargs) -> None:
    return None


def _cancel_endpoint(state: _RouteState):
    router = APIRouter()
    context = StatusRouteContext(
        state=state,
        cfg=state.cfg,
        verify_api_key=_verify_nothing,
        verify_admin=_verify_nothing,
    )
    attach_control_routes(router, context)
    route = next(route for route in router.routes if getattr(route, "path", "") == "/v1/audio/requests/{request_id}/cancel")
    return route


class _FakeSender:
    def __init__(self) -> None:
        self.json_frames: list[dict[str, Any]] = []

    async def send_json(self, frame: dict[str, Any]) -> None:
        self.json_frames.append(dict(frame))


class _SessionState(_RouteState):
    def __init__(self) -> None:
        super().__init__()
        self.tts_semaphore = asyncio.Semaphore(1)
        self.events: list[tuple[Any, ...]] = []
        self.statuses: dict[str, str] = {}

    def inc_stat(self, key: str, amount: int = 1) -> None:
        self.events.append(("inc_stat", key, amount))
        super().inc_stat(key, amount)

    def mark_request(self, request_id: str, status: str, **extra) -> None:
        self.statuses[request_id] = status
        self.events.append(("mark_request", request_id, status, extra))
        super().mark_request(request_id, status, **extra)

    def finish_request(self, request_id: str, status: str, **extra) -> None:
        self.statuses[request_id] = status
        self.events.append(("finish_request", request_id, status, extra))
        super().finish_request(request_id, status, **extra)


class _RuntimeSignals:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def cancel_model_request(self, model_id: str, *, force: bool = False) -> dict[str, Any]:
        self.calls.append((model_id, force))
        return {"model": model_id, "force_killed": force}


class _Latency:
    def __init__(self) -> None:
        self.values: list[float] = []

    def record(self, value: float) -> None:
        self.values.append(value)


class _NonstreamState:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.cfg = SimpleNamespace(request_timeout_seconds=2)
        self.tts_semaphore = asyncio.Semaphore(1)
        self.cancelled = cancelled
        self.events: list[tuple[Any, ...]] = []
        self.model_manager = _RuntimeSignals()
        self.latency_tracker = _Latency()
        self.saved_outputs: list[bytes] = []

    def inc_stat(self, key: str, amount: int = 1) -> None:
        self.events.append(("inc_stat", key, amount))

    def mark_request(self, request_id: str, status: str, **extra) -> None:
        self.events.append(("mark_request", request_id, status, extra))

    def finish_request(self, request_id: str, status: str, **extra) -> None:
        self.events.append(("finish_request", request_id, status, extra))

    def is_cancelled(self, request_id: str) -> bool:
        return self.cancelled

    def save_generated_output(self, **kwargs):
        self.saved_outputs.append(bytes(kwargs["audio_bytes"]))
        return None


def _synthesis_request():
    from kokoro_tts.contracts import SynthesisRequest

    return SynthesisRequest(
        text="contract text",
        model_id="kokoro",
        voice="voice",
        speed=1.0,
        request_id="public-request",
    )


class _SentinelIterator:
    def __init__(
        self,
        frames: Iterable[object],
        *,
        fail_after_frames: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self.frames = list(frames)
        self.fail_after_frames = fail_after_frames
        self.close_error = close_error
        self.failed = False
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frames:
            return self.frames.pop(0)
        if self.fail_after_frames and not self.failed:
            self.failed = True
            raise RuntimeError("sentinel iterator failure")
        raise StopIteration

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _SentinelEngine:
    def __init__(self, iterator: _SentinelIterator) -> None:
        self.iterator = iterator

    def synthesize_stream(self, *args, **kwargs):
        return self.iterator


class _BorrowManager:
    def __init__(self, engine: _SentinelEngine) -> None:
        self.engine = engine
        self.entered = 0
        self.exited = 0

    @contextmanager
    def borrow(self, model_id: str):
        assert model_id == "kokoro"
        self.entered += 1
        try:
            yield self.engine
        finally:
            self.exited += 1


def _streaming_service(iterator: object) -> tuple[StreamingService, _BorrowManager]:
    manager = _BorrowManager(_SentinelEngine(iterator))
    state = SimpleNamespace(cfg=SimpleNamespace(), model_manager=manager)
    return StreamingService(state), manager


def _streaming_request() -> StreamingRequest:
    return StreamingRequest(
        text="contract text",
        model_id="kokoro",
        voice="voice",
        speed=1.0,
        request_id="public-request",
    )


class _FakeProcessClient:
    def __init__(self, iterator: object) -> None:
        self.iterator = iterator
        self.is_loaded = True
        self.calls: list[dict[str, Any]] = []

    def stream(self, payload: dict[str, Any], *, timeout: float, cancel_check=None):
        self.calls.append({"payload": dict(payload), "timeout": timeout, "cancel_check": cancel_check})
        return self.iterator


class _MossProcessHarness(MossStreamingMixin):
    def __init__(self, iterator: object) -> None:
        self._process_client = _FakeProcessClient(iterator)
        self.config = SimpleNamespace(
            request_timeout_seconds=5,
            engine_process_stream_idle_timeout_seconds=5,
        )
        self.engine_id = "moss"
        self.failures: list[dict[str, Any]] = []

    def load(self) -> None:
        raise AssertionError("hermetic MOSS process seam must not load a model")

    def _mark_process_failure(self, **details) -> None:
        self.failures.append(dict(details))


def _moss_process_stream(harness: _MossProcessHarness):
    return harness._synthesize_stream_process_isolated(
        text="contract text",
        voice="voice",
        speed=1.0,
        fmt="pcm_s16le",
        prompt_audio_path=None,
        cancel_check=None,
    )


class _ResultQueue:
    def __init__(self, values: Iterable[tuple[str, str, object]] = ()) -> None:
        self.values = list(values)
        self.get_calls = 0

    def get(self, *, timeout: float):
        self.get_calls += 1
        if not self.values:
            raise queue.Empty
        return self.values.pop(0)


def _worker_client(result_queue: _ResultQueue) -> tuple[EngineProcessClient, _TrackingLock, dict[str, Any]]:
    client = EngineProcessClient.__new__(EngineProcessClient)
    lock = _TrackingLock()
    sent: dict[str, Any] = {}
    client._request_lock = lock
    client._stream_generation = 0
    client._cancel_flag = SimpleNamespace(value=0)
    client.config = SimpleNamespace(
        engine_process_stream_idle_timeout_seconds=5,
        engine_process_stream_drain_seconds=0.1,
    )
    client.engine_id = "kokoro"
    client.logger = None
    client._process = None

    def send(command: str, payload: dict[str, Any]) -> str:
        sent.update(command=command, payload=dict(payload))
        return "worker-request"

    client._send = send
    client._raise_if_worker_exited = lambda: None
    client._require_result_queue = lambda: result_queue
    return client, lock, sent


class _Acknowledgement(str, Enum):
    """TEST-ONLY INTERPRETATION; NOT A PRODUCTION API."""

    STATE_ACCEPTED = "STATE ACCEPTED"
    PUBLIC_DELIVERY_STOPPED = "PUBLIC DELIVERY STOPPED"
    PRODUCER_TASK_FINISHED = "PRODUCER TASK FINISHED"
    GENERATOR_CLOSED = "GENERATOR CLOSED"
    WORKER_REQUEST_FINISHED = "WORKER REQUEST FINISHED"
    ENGINE_COMPUTATION_STOPPED = "ENGINE COMPUTATION STOPPED"


def _classify_acknowledgement(signal: str) -> _Acknowledgement:
    """TEST-ONLY INTERPRETATION; NOT A PRODUCTION API."""

    return {
        "request_cancel_returned": _Acknowledgement.STATE_ACCEPTED,
        "cancelled_frame_queued": _Acknowledgement.PUBLIC_DELIVERY_STOPPED,
        "producer_task_done": _Acknowledgement.PRODUCER_TASK_FINISHED,
        "outer_generator_closed": _Acknowledgement.GENERATOR_CLOSED,
        "worker_queue_done": _Acknowledgement.WORKER_REQUEST_FINISHED,
        "engine_stop_ack": _Acknowledgement.ENGINE_COMPUTATION_STOPPED,
    }[signal]


class TestCancellationScopeIdentity:
    """RESOURCE OWNERSHIP CONTRACT."""

    def test_public_worker_and_generation_identities_have_distinct_owners(self) -> None:
        public = CancellationContext("public-request", lambda: False)
        client = EngineProcessClient.__new__(EngineProcessClient)
        client.start = lambda: None
        client._command_queue = queue.Queue()
        worker_request = EngineProcessClient._send(client, "synthesize_stream", {})
        generation = 7

        assert public.request_id == "public-request"
        assert worker_request != public.request_id
        assert isinstance(worker_request, str) and len(worker_request) == 32
        assert isinstance(generation, int)
        assert generation not in {public.request_id, worker_request}
        command = client._command_queue.get_nowait()
        assert command == (worker_request, "synthesize_stream", {})

        stream_source = inspect.getsource(EngineProcessClient.stream)
        assert "self._stream_generation" in stream_source
        assert 'stream_payload["_cancel_generation"] = stream_generation' in stream_source


class TestHttpCancellationStateOwnership:
    """BEHAVIOR GUARANTEE and PUBLIC COMPATIBILITY CONTRACT."""

    def test_cancel_route_is_state_only_with_the_exact_existing_result(self) -> None:
        state = _RouteState()
        state.mark_request("public-request", "running")
        route = _cancel_endpoint(state)

        result = asyncio.run(route.endpoint("public-request", None))

        assert route.path == "/v1/audio/requests/{request_id}/cancel"
        assert route.methods == {"POST"}
        assert result == {
            "ok": True,
            "request_id": "public-request",
            "known": True,
            "status": "cancelling",
        }
        assert state.request_cancel_calls == ["public-request"]
        assert state.is_cancelled("public-request") is True

        owner = inspect.getsource(attach_control_routes)
        cancel_block = owner[owner.index('@router.post("/v1/audio/requests/{request_id}/cancel")') :]
        assert "state.request_cancel(request_id)" in cancel_block
        for forbidden_owner in ("model_manager", "runtime", "worker", "run_in_threadpool"):
            assert forbidden_owner not in cancel_block


class TestRequestRegistryEdgeLifecycle:
    """CURRENT CHARACTERIZATION and P3B3 CANDIDATE CHARACTERIZATION."""

    @pytest.mark.parametrize("initial", ["running", None, "done", "error", "timeout", "cancelled"])
    def test_active_unknown_and_terminal_cancel_edge_matrix(self, initial: str | None) -> None:
        registry = _Registry()
        request_id = f"request-{initial or 'unknown'}"
        if initial is not None:
            registry.active_requests[request_id] = {"id": request_id, "status": initial}

        first = registry.request_cancel(request_id)
        first_marker = registry.is_cancelled(request_id)
        first_history = registry.request_info(request_id)
        repeated = registry.request_cancel(request_id)

        assert first is (initial is not None)
        assert repeated is True
        assert first_marker is True
        assert registry.is_cancelled(request_id) is True
        assert first_history is not None and first_history["status"] == "cancelling"
        assert registry.request_info(request_id)["status"] == "cancelling"
        assert registry.stats["ws_cancelled_total"] == 2

    def test_terminal_finish_discards_marker_but_a_later_cancel_recreates_it(self) -> None:
        registry = _Registry()
        registry.mark_request("request-terminal", "running")
        registry.request_cancel("request-terminal")
        registry.finish_request("request-terminal", "done")
        assert registry.is_cancelled("request-terminal") is False
        assert registry.request_info("request-terminal")["status"] == "done"

        assert registry.request_cancel("request-terminal") is True
        assert registry.is_cancelled("request-terminal") is True
        assert registry.request_info("request-terminal")["status"] == "cancelling"


class TestNonstreamCancellationBoundary:
    """BEHAVIOR GUARANTEE plus EXPLICIT NON-GUARANTEE."""

    def test_pre_inference_cancel_prevents_the_synchronous_owner_from_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _NonstreamState(cancelled=True)
        service = SynthesisService(state)

        def forbidden_call(request):
            raise AssertionError("pre-cancelled inference ran")

        async def inline(callable_, *args):
            return callable_(*args)

        service.response_bytes = forbidden_call
        monkeypatch.setattr(synthesis_module, "run_in_threadpool", inline)
        with pytest.raises(HTTPException) as raised:
            asyncio.run(service.response_threaded(_synthesis_request()))

        assert raised.value.status_code == 499
        assert state.model_manager.calls == []
        assert any(event[:3] == ("finish_request", "public-request", "cancelled") for event in state.events)

    def test_active_synchronous_call_finishes_before_post_check_discards_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _NonstreamState()
        service = SynthesisService(state)
        synchronous_owner: list[str] = []

        def complete_native_owner(request):
            synchronous_owner.append("entered")
            state.cancelled = True
            synchronous_owner.append("returned")
            return b"discarded", "audio/wav"

        async def inline(callable_, *args):
            return callable_(*args)

        service.response_bytes = complete_native_owner
        monkeypatch.setattr(synthesis_module, "run_in_threadpool", inline)
        with pytest.raises(HTTPException) as raised:
            asyncio.run(service.response_threaded(_synthesis_request()))

        assert raised.value.status_code == 499
        assert synchronous_owner == ["entered", "returned"]
        assert state.saved_outputs == []
        assert state.model_manager.calls == []

    def test_timeout_is_the_distinct_force_drop_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        state = _NonstreamState()
        service = SynthesisService(state)

        async def time_out(*args):
            raise asyncio.TimeoutError

        monkeypatch.setattr(synthesis_module, "run_in_threadpool", time_out)
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(service.response_threaded(_synthesis_request()))

        assert state.model_manager.calls == [("kokoro", True)]
        assert any(event[:3] == ("finish_request", "public-request", "timeout") for event in state.events)


class TestWebSocketLocalCancellation:
    """BEHAVIOR GUARANTEE and RESOURCE OWNERSHIP CONTRACT."""

    def test_first_cancel_drains_local_queue_and_repeat_is_idempotent(self) -> None:
        async def exercise():
            state = _SessionState()
            session = TtsWebSocketSession(websocket=_FakeSender(), state=state)
            await session.queue.put({"type": "audio", "index": 1})
            await session.queue.put({"type": "audio", "index": 2})
            await session._mark_client_cancelled()
            await session._mark_client_cancelled()
            return state, session

        state, session = asyncio.run(exercise())
        assert session.queue.maxsize == 4
        assert session.phase is WsSessionState.CANCELLING
        assert session.cancel_event.is_set()
        assert session.cancelled_by_client is True
        assert state.request_cancel_calls == ["public-request", "public-request"]
        assert session.queue.get_nowait() == {"type": "cancelled", "request_id": "public-request"}
        assert session.queue.get_nowait() is session.done_marker
        assert session.queue.empty()

    def test_late_frames_observed_after_cancel_are_not_publicly_enqueued(self) -> None:
        state = _SessionState()
        session = TtsWebSocketSession(websocket=_FakeSender(), state=state)
        published: list[dict[str, Any]] = []
        state.streaming = SimpleNamespace(
            iter_frames=lambda request, cancel_check: iter(
                ({"type": "audio", "index": 8}, {"type": "done"})
            )
        )
        session.cancel_event.set()
        session._thread_put = lambda frame: published.append(frame) or True

        session._producer(_streaming_request())

        assert published == []

    def test_disconnect_delivery_and_fixed_grace_are_not_engine_stop_acknowledgements(self) -> None:
        streaming = _source("src/kokoro_tts/ws/streaming.py")
        cancel = _source("src/kokoro_tts/ws/cancel.py")
        assert "except WebSocketDisconnect" in streaming
        assert "except Exception" in streaming
        assert "self.state.request_cancel(self.request_id)" in streaming
        assert "self.cancel_event.set()" in streaming
        assert "await asyncio.wait_for(self.producer_task, timeout=5.0)" in cancel
        assert "engine" not in inspect.getsource(TtsWebSocketSession._cancel_background_tasks)


class TestHttpToWebSocketQueueOwnership:
    """CURRENT CHARACTERIZATION and P3B3 CANDIDATE CHARACTERIZATION."""

    def test_registry_only_http_cancel_does_not_synchronously_drain_session_queue(self) -> None:
        async def exercise():
            state = _SessionState()
            session = TtsWebSocketSession(websocket=_FakeSender(), state=state)
            queued_audio = {"type": "audio", "index": 4}
            await session.queue.put(queued_audio)
            state.mark_request(session.request_id, "running")
            route = _cancel_endpoint(state)
            response = await route.endpoint(session.request_id, None)
            return state, session, queued_audio, response

        state, session, queued_audio, response = asyncio.run(exercise())
        assert response["known"] is True
        assert state.is_cancelled(session.request_id) is True
        assert session.cancel_event.is_set() is False
        assert session.queue.get_nowait() == queued_audio
        assert session.queue.empty()


class TestStreamingGeneratorOwnership:
    """RESOURCE OWNERSHIP CONTRACT for immediate synchronous iterators."""

    def test_normal_exhaustion_releases_borrow(self) -> None:
        inner = _SentinelIterator(({"type": "done"},))
        service, manager = _streaming_service(inner)

        assert [frame["type"] for frame in service.iter_frames(_streaming_request())] == ["done"]
        assert (manager.entered, manager.exited) == (1, 1)
        assert inner.close_calls == 1

    def test_explicit_outer_close_releases_borrow(self) -> None:
        inner = _SentinelIterator(({"type": "started"}, {"type": "audio"}))
        service, manager = _streaming_service(inner)
        outer = service.iter_frames(_streaming_request())

        assert next(outer)["type"] == "started"
        assert manager.exited == 0
        outer.close()

        assert manager.exited == 1
        assert inner.close_calls == 1

    def test_exception_unwind_releases_borrow(self) -> None:
        inner = _SentinelIterator(({"type": "started"},), fail_after_frames=True)
        service, manager = _streaming_service(inner)

        with pytest.raises(RuntimeError, match="sentinel iterator failure"):
            list(service.iter_frames(_streaming_request()))

        assert manager.exited == 1
        assert inner.close_calls == 1

    def test_terminal_frame_path_closes_inner_once(self) -> None:
        inner = _SentinelIterator(({"type": "started"}, {"type": "done"}))
        service, manager = _streaming_service(inner)

        assert [frame["type"] for frame in service.iter_frames(_streaming_request())] == ["started", "done"]
        assert (manager.entered, manager.exited) == (1, 1)
        assert inner.close_calls == 1

    def test_cancellation_break_closes_inner_before_cancelled_fallback(self) -> None:
        inner = _SentinelIterator(({"type": "audio"}, {"type": "done"}))
        service, manager = _streaming_service(inner)

        frames = list(service.iter_frames(_streaming_request(), cancel_check=lambda: True))

        assert [frame["type"] for frame in frames] == ["cancelled"]
        assert (manager.entered, manager.exited) == (1, 1)
        assert inner.close_calls == 1

    def test_stream_exception_wins_when_inner_close_raises(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"},),
            fail_after_frames=True,
            close_error=RuntimeError("sentinel close failure"),
        )
        service, manager = _streaming_service(inner)

        with pytest.raises(RuntimeError, match="sentinel iterator failure"):
            list(service.iter_frames(_streaming_request()))

        assert manager.exited == 1
        assert inner.close_calls == 1

    def test_outer_close_suppresses_ordinary_inner_close_failure(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"}, {"type": "audio"}),
            close_error=RuntimeError("sentinel close failure"),
        )
        service, manager = _streaming_service(inner)
        outer = service.iter_frames(_streaming_request())

        assert next(outer)["type"] == "started"
        outer.close()

        assert manager.exited == 1
        assert inner.close_calls == 1

    def test_inner_close_baseexception_is_not_swallowed(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"}, {"type": "audio"}),
            close_error=KeyboardInterrupt("sentinel base exception"),
        )
        service, manager = _streaming_service(inner)
        outer = service.iter_frames(_streaming_request())

        assert next(outer)["type"] == "started"
        with pytest.raises(KeyboardInterrupt, match="sentinel base exception"):
            outer.close()

        assert manager.exited == 1
        assert inner.close_calls == 1

    def test_iterator_without_close_keeps_existing_stream_behavior(self) -> None:
        inner = iter(({"type": "done"},))
        service, manager = _streaming_service(inner)

        assert [frame["type"] for frame in service.iter_frames(_streaming_request())] == ["done"]
        assert (manager.entered, manager.exited) == (1, 1)

    def test_streaming_service_explicitly_owns_inner_iterator_close(self) -> None:
        source = inspect.getsource(StreamingService.iter_frames)
        tree = ast.parse(textwrap.dedent(source))
        assert "for chunk in engine_iterator" in source
        assert "_close_stream_iterator" in source
        assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
        helper = inspect.getsource(streaming_module._close_stream_iterator)
        assert 'getattr(iterator, "close", None)' in helper
        assert "except Exception:" in helper


class TestMossNestedProcessIteratorOwnership:
    """RESOURCE OWNERSHIP CONTRACT for the nested process stream iterator."""

    def test_normal_exhaustion_closes_nested_iterator_once(self) -> None:
        inner = _SentinelIterator(({"type": "done"},))
        harness = _MossProcessHarness(inner)

        assert [frame["type"] for frame in _moss_process_stream(harness)] == ["done"]
        assert inner.close_calls == 1
        assert harness.failures == []

    def test_outer_wrapper_close_closes_nested_iterator_once(self) -> None:
        inner = _SentinelIterator(({"type": "started"}, {"type": "audio"}))
        harness = _MossProcessHarness(inner)
        outer = _moss_process_stream(harness)

        assert next(outer)["type"] == "started"
        outer.close()

        assert inner.close_calls == 1

    def test_outer_wrapper_close_suppresses_ordinary_nested_close_failure(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"}, {"type": "audio"}),
            close_error=RuntimeError("nested close failure"),
        )
        harness = _MossProcessHarness(inner)
        outer = _moss_process_stream(harness)

        assert next(outer)["type"] == "started"
        outer.close()

        assert inner.close_calls == 1

    def test_nested_close_baseexception_is_not_swallowed(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"}, {"type": "audio"}),
            close_error=KeyboardInterrupt("nested base exception"),
        )
        harness = _MossProcessHarness(inner)
        outer = _moss_process_stream(harness)

        assert next(outer)["type"] == "started"
        with pytest.raises(KeyboardInterrupt, match="nested base exception"):
            outer.close()

        assert inner.close_calls == 1

    def test_inner_exception_closes_nested_iterator_once(self) -> None:
        inner = _SentinelIterator(({"type": "started"},), fail_after_frames=True)
        harness = _MossProcessHarness(inner)

        frames = list(_moss_process_stream(harness))

        assert [frame["type"] for frame in frames] == ["started", "segment_error", "done"]
        assert inner.close_calls == 1
        assert harness.failures == [{"timeout": 5.0, "reason": "stream_error"}]

    def test_nested_close_failure_does_not_replace_stream_error_mapping(self) -> None:
        inner = _SentinelIterator(
            ({"type": "started"},),
            fail_after_frames=True,
            close_error=RuntimeError("nested close failure"),
        )
        harness = _MossProcessHarness(inner)

        frames = list(_moss_process_stream(harness))

        assert [frame["type"] for frame in frames] == ["started", "segment_error", "done"]
        assert "sentinel iterator failure" in frames[1]["message"]
        assert inner.close_calls == 1

    def test_nested_iterator_without_close_keeps_existing_behavior(self) -> None:
        harness = _MossProcessHarness(iter(({"type": "done"},)))

        assert [frame["type"] for frame in _moss_process_stream(harness)] == ["done"]

    def test_moss_wrapper_explicitly_owns_nested_iterator_close(self) -> None:
        source = inspect.getsource(MossStreamingMixin._synthesize_stream_process_isolated)
        tree = ast.parse(textwrap.dedent(source))
        assert "for event in process_iterator" in source
        assert "_close_process_iterator" in source
        assert any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
        helper = inspect.getsource(moss_streaming_module._close_process_iterator)
        assert 'getattr(iterator, "close", None)' in helper
        assert "except Exception:" in helper


class TestExternalSessionTaskCancellation:
    """CURRENT CHARACTERIZATION and P3B3 CANDIDATE CHARACTERIZATION."""

    def test_external_task_cancel_runs_cleanup_but_bypasses_finish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def exercise():
            state = _SessionState()
            session = TtsWebSocketSession(websocket=_FakeSender(), state=state)
            send_loop_started = asyncio.Event()

            async def control_listener():
                await asyncio.Event().wait()

            async def send_loop(*, binary: bool):
                send_loop_started.set()
                await asyncio.Event().wait()

            async def producer_without_thread():
                while not session.cancel_event.is_set():
                    await asyncio.sleep(0)

            def fake_to_thread(*args, **kwargs):
                return producer_without_thread()

            session._control_listener = control_listener
            session._send_loop = send_loop
            monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

            task = asyncio.create_task(session._stream(_streaming_request()))
            await send_loop_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            return state, session

        state, session = asyncio.run(exercise())
        assert session.cancel_event.is_set() is True
        assert session.producer_task is not None and session.producer_task.done()
        assert session.control_task is not None and session.control_task.cancelled()
        assert state.statuses[session.request_id] == "running"
        assert not any(event[0] == "finish_request" for event in state.events)


class TestWorkerGenerationCancellation:
    """RESOURCE OWNERSHIP CONTRACT and EXPLICIT NON-GUARANTEE."""

    def test_soft_cancel_is_generation_scoped_and_older_token_misses_new_generation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Engine:
            def load(self) -> None:
                return None

            def unload(self) -> None:
                return None

            def synthesize_stream(self, **kwargs):
                yield {"type": "audio", "index": 0}
                yield {"type": "done"}

        commands: queue.Queue[tuple[str, str, dict[str, Any]]] = queue.Queue()
        results: queue.Queue[tuple[str, str, object]] = queue.Queue()
        cancel_flag = SimpleNamespace(value=1)
        commands.put(("worker-one", "synthesize_stream", {"_cancel_generation": 1}))
        commands.put(("worker-two", "synthesize_stream", {"_cancel_generation": 2}))
        commands.put(("worker-stop", "shutdown", {}))
        monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args: _Engine())

        process_worker._worker_main(
            SimpleNamespace(),
            EngineWorkerSpec("kokoro", str),
            commands,
            results,
            cancel_flag,
        )

        captured = []
        while not results.empty():
            captured.append(results.get_nowait())
        first = [item for item in captured if item[0] == "worker-one"]
        second = [item for item in captured if item[0] == "worker-two"]
        assert first == [("worker-one", "done", None)]
        assert [kind for _, kind, _ in second] == ["event", "event", "done"]
        assert second[0][2]["type"] == "audio"
        assert cancel_flag.value == 0

    def test_stale_worker_uuid_is_discarded_and_queue_done_releases_request_lock(self) -> None:
        result_queue = _ResultQueue(
            (
                ("stale-worker-request", "event", {"type": "audio"}),
                ("worker-request", "done", None),
            )
        )
        client, lock, sent = _worker_client(result_queue)

        assert list(client.stream({}, timeout=5, cancel_check=lambda: False)) == []
        assert sent["payload"]["_cancel_generation"] == 1
        assert (lock.entered, lock.exited) == (1, 1)
        assert result_queue.get_calls >= 2

    def test_drain_deadline_can_release_parent_lock_before_child_queue_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result_queue = _ResultQueue()
        client, lock, sent = _worker_client(result_queue)
        ticks = iter((0.0, 0.0, 0.0, 2.0))
        monkeypatch.setattr(process_worker.time, "monotonic", lambda: next(ticks))

        assert list(client.stream({}, timeout=5, cancel_check=lambda: True)) == []
        assert sent["payload"]["_cancel_generation"] == 1
        assert client._cancel_flag.value == 1
        assert result_queue.get_calls == 0
        assert (lock.entered, lock.exited) == (1, 1)


class TestCancellationAcknowledgementLevels:
    """CURRENT CHARACTERIZATION and EXPLICIT NON-GUARANTEE."""

    def test_six_acknowledgement_levels_are_not_interchangeable(self) -> None:
        mapped = {
            signal: _classify_acknowledgement(signal)
            for signal in (
                "request_cancel_returned",
                "cancelled_frame_queued",
                "producer_task_done",
                "outer_generator_closed",
                "worker_queue_done",
                "engine_stop_ack",
            )
        }
        assert len(set(mapped.values())) == 6
        assert mapped["worker_queue_done"] is _Acknowledgement.WORKER_REQUEST_FINISHED
        assert mapped["worker_queue_done"] is not _Acknowledgement.ENGINE_COMPUTATION_STOPPED

        production = "\n".join(
            _source(path)
            for path in (
                "src/kokoro_tts/ws/cancel.py",
                "src/kokoro_tts/ws/streaming.py",
                "src/kokoro_tts/workers/process_worker.py",
            )
        )
        assert "engine_stop_ack" not in production
        assert '(request_id, "done", None)' in production


class TestPerRequestResourceOwnership:
    """RESOURCE OWNERSHIP CONTRACT."""

    def test_normal_error_cancel_disconnect_and_timeout_owner_matrix(self) -> None:
        matrix = {
            "request_registry": ("finish", "finish", "cancel/finish", "cancel/finish", "finish"),
            "cancelled_marker": ("discard", "discard", "edge-gap", "discard", "discard"),
            "engine_borrow": ("finally", "finally", "unwind", "unwind", "unwind/force"),
            "request_lock": ("with", "with", "drain-bounded", "drain-bounded", "timeout/force"),
            "producer_task": ("complete", "cleanup", "grace/cancel", "grace/cancel", "cleanup"),
            "control_task": ("cancel", "cancel", "cancel", "exit", "cancel"),
            "engine_iterator": (
                "exhaust/finally-close",
                "unwind/finally-close",
                "unwind/finally-close",
                "unwind/finally-close",
                "path-specific/finally-close",
            ),
            "queue_done": ("command", "command", "may-arrive-late", "may-arrive-late", "force-differs"),
            "prompt_temp": ("finally", "finally", "finally", "finally", "finally"),
        }
        assert tuple(matrix) == (
            "request_registry",
            "cancelled_marker",
            "engine_borrow",
            "request_lock",
            "producer_task",
            "control_task",
            "engine_iterator",
            "queue_done",
            "prompt_temp",
        )
        assert all(len(outcomes) == 5 for outcomes in matrix.values())
        assert matrix["engine_iterator"][2:4] == (
            "unwind/finally-close",
            "unwind/finally-close",
        )
        assert matrix["queue_done"][2] == "may-arrive-late"

        manager = _source("src/kokoro_tts/engine_manager.py")
        session = _source("src/kokoro_tts/ws/session.py")
        audio = _source("src/kokoro_tts/routes/audio.py")
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        assert "try:\n            yield engine\n        finally:" in manager
        assert "with self._request_lock:" in worker
        assert "if self.prompt_audio_path:\n                delete_prompt_audio_path" in session
        assert "if prompt_audio_path:\n                delete_prompt_audio_path" in audio


class TestProcessLocalRegistryBoundary:
    """EXPLICIT NON-GUARANTEE and OUT-OF-SCOPE RATCHET."""

    def test_service_state_constructor_owns_process_local_mutable_state(self) -> None:
        source = inspect.getsource(ServiceState.__init__)
        tree = ast.parse(textwrap.dedent(source))
        assigned = {
            node.targets[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        }
        assigned.update(
            node.target.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Attribute)
            and isinstance(node.target.value, ast.Name)
            and node.target.value.id == "self"
        )
        assert {"model_manager", "active_requests", "cancelled_requests", "request_lock"} <= assigned
        assert not hasattr(ServiceState, "active_requests")
        assert not hasattr(ServiceState, "cancelled_requests")
        assert "shared" not in source.casefold()

    def test_cross_worker_http_cancel_requires_new_ipc_or_routing_and_is_out_of_scope(self) -> None:
        registry = _source("src/kokoro_tts/services/state_parts/request_registry.py")
        control = _source("src/kokoro_tts/routes/status_parts/control.py")
        assert "self.cancelled_requests.add(request_id)" in registry
        assert "state.request_cancel(request_id)" in control
        for shared_store in ("redis", "pubsub", "broadcast", "sticky"):
            assert shared_store not in (registry + control).casefold()


class TestPublicCompatibility:
    """PUBLIC COMPATIBILITY CONTRACT without reopening the P3A event matrix."""

    def test_existing_request_result_and_worker_shapes_remain_owned_by_p3a_and_p2g(self) -> None:
        assert tuple(StreamingRequest.__dataclass_fields__) == (
            "text",
            "model_id",
            "voice",
            "speed",
            "audio_format",
            "binary",
            "condition",
            "generation",
            "request_id",
        )
        assert tuple(StreamingResult.__dataclass_fields__) == ("type", "payload")
        assert tuple(WorkerResult.__dataclass_fields__) == ("request_id", "kind", "payload")

        ws_sources = _source("src/kokoro_tts/ws/cancel.py") + _source("src/kokoro_tts/ws/streaming.py")
        for existing_type in ("cancelled", "done", "error", "segment_error"):
            assert f'"{existing_type}"' in ws_sources
        worker = _source("src/kokoro_tts/workers/process_worker.py")
        for existing_kind in ("event", "result", "error", "done"):
            assert f'"{existing_kind}"' in worker
        assert "acknowledgement" not in ws_sources.casefold()


class TestP3B3CandidateCharacterization:
    """Preserve all non-selected P3B3 characterization boundaries."""

    def test_candidates_are_classified_without_selecting_or_implementing_one(self) -> None:
        candidates = {
            "inner iterator explicit close": "SELECTED P3B NARROW IMPLEMENTATION",
            "HTTP to WS local queue gap": "POTENTIAL P3B3 / ACCEPTED LIMITATION",
            "unknown or terminal marker lifetime": "POTENTIAL P3B3 / CONTRACT-ONLY",
            "external async session terminalization": "POTENTIAL P3B3 / NO IMPLEMENTATION",
            "control task completion": "CONTRACT-ONLY / POTENTIAL P3B3",
            "active to_thread continuation": "ACCEPTED LIMITATION",
            "cross-worker HTTP cancel": "REQUIRES NEW IPC / OUT OF SCOPE",
        }
        assert len(candidates) == 7
        assert candidates["cross-worker HTTP cancel"].endswith("OUT OF SCOPE")
        assert sum("SELECTED" in disposition for disposition in candidates.values()) == 1


def test_contract_source_shape_is_hermetic_contract_only_and_single_file() -> None:
    """OUT-OF-SCOPE RATCHET."""

    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        ".write_" + "text(",
        "Test" + "Client(",
        "websocket_" + "connect(",
        "threading." + "Thread(",
        "multiprocessing." + "Process(",
        "sub" + "process.",
        "requests." + "get(",
        "http" + "x.",
        "socket" + ".socket(",
        "torch" + ".",
        "os." + "environ",
        "create_" + "app(",
    )
    assert all(token not in source for token in forbidden)
    assert "close_" + "all" not in source
