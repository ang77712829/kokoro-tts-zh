"""Freeze the minimal P3 engine/worker failure boundary.

Current-state characterization preserves RuntimeError, timeout, cancellation,
and HTTP/WebSocket compatibility.  The future gates intentionally remain red
until a narrow implementation adds a spawn-safe data envelope and maps it to a
minimal in-process EngineError without changing transport response shapes.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pickle
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from kokoro_tts.workers.process_worker import (
    EngineProcessClient,
    EngineProcessTimeoutError,
    WorkerResult,
    _worker_main,
)
from kokoro_tts.workers.spec import EngineWorkerSpec


pytestmark = pytest.mark.contract


PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "kokoro_tts"
PROCESS_WORKER_PATH = PACKAGE_ROOT / "workers" / "process_worker.py"
ERROR_OWNER_PATH = PACKAGE_ROOT / "contracts" / "errors.py"
HTTP_ROUTE_PATH = PACKAGE_ROOT / "routes" / "audio.py"
WS_STREAMING_PATH = PACKAGE_ROOT / "ws" / "streaming.py"
ERROR_OWNER_MODULE = "kokoro_tts.contracts.errors"

ENGINE_LOAD_FAILED = "engine_load_failed"
ENGINE_RUNTIME_FAILED = "engine_runtime_failed"
WORKER_TIMEOUT = "worker_timeout"
WORKER_PROCESS_FAILED = "worker_process_failed"
WORKER_PROTOCOL_FAILED = "worker_protocol_failed"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _client_with_result(result: WorkerResult) -> EngineProcessClient:
    client = object.__new__(EngineProcessClient)
    client.engine_id = "kokoro"
    client._request_lock = threading.RLock()
    client._last_metadata = {}
    client._loaded = False
    client._send = lambda command, payload: result.request_id
    client._wait_for = lambda request_id, *, timeout: result
    return client


def _future_owner():
    try:
        return importlib.import_module(ERROR_OWNER_MODULE)
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Gate A: src/kokoro_tts/contracts/errors.py must own the minimal "
            "EngineError and WorkerFailureEnvelope contracts"
        )
        raise AssertionError("unreachable") from exc


@dataclasses.dataclass(frozen=True)
class _ContractEnvelope:
    version: int
    code: str
    message: str


class _ContractEngineError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _owner_or_contract_stub():
    try:
        return importlib.import_module(ERROR_OWNER_MODULE)
    except ModuleNotFoundError:
        return SimpleNamespace(
            WorkerFailureEnvelope=_ContractEnvelope,
            EngineError=_ContractEngineError,
        )


def _unused_factory(config: object, requested_provider: str | None) -> object:
    raise AssertionError("EngineWorkerSpec factory should be replaced by the test seam")


class _Commands:
    def __init__(self, *commands: tuple[str, str, dict]) -> None:
        self._commands = list(commands)

    def get(self):
        return self._commands.pop(0)


class _Results:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, object]] = []

    def put(self, item) -> None:
        self.items.append(item)


class _FakeEngine:
    def __init__(self, *, load_error: BaseException | None = None, runtime_error: BaseException | None = None) -> None:
        self.load_error = load_error
        self.runtime_error = runtime_error

    def load(self) -> None:
        if self.load_error is not None:
            raise self.load_error

    def unload(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {"loaded": True}

    def synthesize(self, **payload) -> bytes:
        if self.runtime_error is not None:
            raise self.runtime_error
        return b"audio"


def _run_child_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    command: str,
    engine: _FakeEngine,
) -> object:
    from kokoro_tts.workers import process_worker

    monkeypatch.setattr(process_worker, "create_worker_engine", lambda config, spec: engine)
    commands = _Commands(
        ("request-1", command, {}),
        ("shutdown-1", "shutdown", {}),
    )
    results = _Results()
    spec = EngineWorkerSpec(engine_id="kokoro", factory=_unused_factory)
    _worker_main(None, spec, commands, results, SimpleNamespace(value=0))
    assert results.items[0][:2] == ("request-1", "error")
    return results.items[0][2]


def test_current_child_failure_uses_typed_payload_and_parent_engine_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _future_owner()
    payload = _run_child_failure(
        monkeypatch,
        command="synthesize",
        engine=_FakeEngine(runtime_error=ValueError("detail")),
    )
    assert isinstance(payload, owner.WorkerFailureEnvelope)
    assert (payload.version, payload.code, payload.message) == (
        1,
        ENGINE_RUNTIME_FAILED,
        "ValueError: detail",
    )
    assert "Traceback" not in payload.message

    result = WorkerResult("request-1", "error", payload)
    client = _client_with_result(result)
    with pytest.raises(owner.EngineError) as captured:
        client.request("synthesize", {}, timeout=1.0)

    assert isinstance(captured.value, RuntimeError)
    assert captured.value.code == ENGINE_RUNTIME_FAILED
    assert str(captured.value) == "ValueError: detail"
    assert captured.value.__cause__ is None


def test_current_load_and_runtime_failures_have_separate_stable_codes() -> None:
    from kokoro_tts.workers.process_worker import _initial_worker_failure_code

    assert _initial_worker_failure_code(None) == ENGINE_LOAD_FAILED
    assert _initial_worker_failure_code(object()) == ENGINE_RUNTIME_FAILED
    assert "traceback.format_exc()" not in _source(PROCESS_WORKER_PATH)


def test_current_timeout_remains_a_distinguishable_compatibility_type() -> None:
    assert issubclass(EngineProcessTimeoutError, TimeoutError)
    assert EngineProcessTimeoutError is not TimeoutError
    worker_source = _source(PROCESS_WORKER_PATH)
    assert "self.close(kill=True)" in worker_source
    assert "raise EngineProcessTimeoutError" in worker_source


def test_current_cancellation_is_a_terminal_event_not_a_failure_exception() -> None:
    worker_source = _source(PROCESS_WORKER_PATH)
    assert '"cancelled"' in worker_source
    assert "_soft_cancel_worker" in worker_source
    assert "EngineProcessCancelledError" not in worker_source
    assert '"worker_cancelled"' not in worker_source


def test_current_process_and_protocol_failures_have_stable_engine_error_codes() -> None:
    owner = _future_owner()
    protocol_client = _client_with_result(WorkerResult("request-1", "unknown", None))
    with pytest.raises(owner.EngineError) as captured:
        protocol_client.request("synthesize", {}, timeout=1.0)
    assert captured.value.code == WORKER_PROTOCOL_FAILED

    exited_client = object.__new__(EngineProcessClient)
    exited_client.engine_id = "kokoro"
    exited_client._process = SimpleNamespace(is_alive=lambda: False, exitcode=17)
    exited_client._loaded = True
    exited_client._unhealthy = False
    exited_client._last_exit_reason = ""
    with pytest.raises(owner.EngineError) as captured:
        exited_client._raise_if_worker_exited()
    assert captured.value.code == WORKER_PROCESS_FAILED
    assert "17" in str(captured.value)


def test_current_worker_boundary_never_deserializes_arbitrary_exception_objects() -> None:
    tree = ast.parse(_source(PROCESS_WORKER_PATH), filename=str(PROCESS_WORKER_PATH))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert "pickle" not in imports
    assert "pickle.loads" not in calls
    assert "importlib.import_module" not in calls


def test_current_http_and_websocket_failure_shapes_remain_phase6_owned() -> None:
    http_source = _source(HTTP_ROUTE_PATH)
    ws_source = _source(WS_STREAMING_PATH)
    assert 'detail=f"合成失败，请检查参数（请求 ID: {request_id}）"' in http_source
    assert '{"type": "error", "message": "流式合成失败", "request_id": self.request_id}' in ws_source
    assert "error_code" not in http_source
    assert "error_code" not in ws_source


def test_future_gate_a_minimal_error_owner_is_frozen_and_spawn_safe() -> None:
    owner = _future_owner()
    envelope_type = owner.WorkerFailureEnvelope
    error_type = owner.EngineError

    assert dataclasses.is_dataclass(envelope_type)
    assert envelope_type.__dataclass_params__.frozen is True
    assert tuple(field.name for field in dataclasses.fields(envelope_type)) == (
        "version",
        "code",
        "message",
    )
    envelope = envelope_type(version=1, code=ENGINE_RUNTIME_FAILED, message="runtime detail")
    assert pickle.loads(pickle.dumps(envelope)) == envelope
    assert not hasattr(envelope, "details")
    assert not hasattr(envelope, "retryable")

    error = error_type(code=ENGINE_RUNTIME_FAILED, message="runtime detail")
    assert isinstance(error, RuntimeError)
    assert error.code == ENGINE_RUNTIME_FAILED
    assert error.message == "runtime detail"
    assert str(error) == "runtime detail"

    owner_tree = ast.parse(_source(ERROR_OWNER_PATH), filename=str(ERROR_OWNER_PATH))
    owner_calls = {ast.unparse(node.func) for node in ast.walk(owner_tree) if isinstance(node, ast.Call)}
    assert "pickle.loads" not in owner_calls
    assert "importlib.import_module" not in owner_calls


def test_future_gate_b_child_load_and_runtime_failures_emit_typed_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _owner_or_contract_stub()
    envelope_type = owner.WorkerFailureEnvelope

    load_payload = _run_child_failure(
        monkeypatch,
        command="load",
        engine=_FakeEngine(load_error=ValueError("load detail")),
    )
    assert isinstance(load_payload, envelope_type)
    assert (load_payload.version, load_payload.code, load_payload.message) == (
        1,
        ENGINE_LOAD_FAILED,
        "ValueError: load detail",
    )

    runtime_payload = _run_child_failure(
        monkeypatch,
        command="synthesize",
        engine=_FakeEngine(runtime_error=RuntimeError("runtime detail")),
    )
    assert isinstance(runtime_payload, envelope_type)
    assert (runtime_payload.version, runtime_payload.code, runtime_payload.message) == (
        1,
        ENGINE_RUNTIME_FAILED,
        "RuntimeError: runtime detail",
    )


def test_future_gate_c_parent_mapping_is_stable_and_legacy_deterministic() -> None:
    owner = _owner_or_contract_stub()
    envelope_type = owner.WorkerFailureEnvelope
    error_type = owner.EngineError

    cases = (
        ("load", envelope_type(1, ENGINE_LOAD_FAILED, "load detail"), ENGINE_LOAD_FAILED),
        ("synthesize", envelope_type(1, ENGINE_RUNTIME_FAILED, "runtime detail"), ENGINE_RUNTIME_FAILED),
        ("load", "legacy load detail", ENGINE_LOAD_FAILED),
        ("synthesize", "legacy runtime detail", ENGINE_RUNTIME_FAILED),
    )
    for command, payload, expected_code in cases:
        client = _client_with_result(WorkerResult("request-1", "error", payload))
        with pytest.raises(error_type) as captured:
            client.request(command, {}, timeout=1.0)
        assert captured.value.code == expected_code
        assert captured.value.message

    protocol_client = _client_with_result(WorkerResult("request-1", "unknown", None))
    with pytest.raises(error_type) as captured:
        protocol_client.request("synthesize", {}, timeout=1.0)
    assert captured.value.code == WORKER_PROTOCOL_FAILED

    exited_client = object.__new__(EngineProcessClient)
    exited_client.engine_id = "kokoro"
    exited_client._process = SimpleNamespace(is_alive=lambda: False, exitcode=17)
    exited_client._loaded = True
    exited_client._unhealthy = False
    exited_client._last_exit_reason = ""
    with pytest.raises(error_type) as captured:
        exited_client._raise_if_worker_exited()
    assert captured.value.code == WORKER_PROCESS_FAILED

    assert EngineProcessTimeoutError.code == WORKER_TIMEOUT
    assert issubclass(EngineProcessTimeoutError, TimeoutError)


def test_future_gate_c_protocol_failures_cover_malformed_and_invalid_envelopes() -> None:
    owner = _future_owner()

    malformed_queue: queue.Queue = queue.Queue()
    malformed_queue.put(("request-1", "result"))
    malformed_client = object.__new__(EngineProcessClient)
    malformed_client.engine_id = "kokoro"
    malformed_client._require_result_queue = lambda: malformed_queue
    malformed_client._raise_if_worker_exited = lambda: None
    with pytest.raises(owner.EngineError) as captured:
        malformed_client._wait_for("request-1", timeout=1.0)
    assert captured.value.code == WORKER_PROTOCOL_FAILED

    invalid_envelopes = (
        owner.WorkerFailureEnvelope(2, ENGINE_RUNTIME_FAILED, "future version"),
        owner.WorkerFailureEnvelope(1, "unknown_code", "unknown code"),
    )
    for envelope in invalid_envelopes:
        client = _client_with_result(WorkerResult("request-1", "error", envelope))
        with pytest.raises(owner.EngineError) as captured:
            client.request("synthesize", {}, timeout=1.0)
        assert captured.value.code == WORKER_PROTOCOL_FAILED


def test_future_gate_d_stream_failure_envelope_maps_without_transport_changes() -> None:
    owner = _owner_or_contract_stub()
    envelope = owner.WorkerFailureEnvelope(1, ENGINE_RUNTIME_FAILED, "stream detail")
    result_queue: queue.Queue = queue.Queue()
    result_queue.put(("request-stream", "error", envelope))

    client = object.__new__(EngineProcessClient)
    client.engine_id = "kokoro"
    client.config = SimpleNamespace(
        engine_process_stream_idle_timeout_seconds=5.0,
        engine_process_stream_drain_seconds=0.1,
    )
    client.logger = None
    client._request_lock = threading.RLock()
    client._stream_generation = 0
    client._cancel_flag = SimpleNamespace(value=0)
    client._send = lambda command, payload: "request-stream"
    client._require_result_queue = lambda: result_queue
    client._raise_if_worker_exited = lambda: None
    client._soft_cancel_worker = lambda generation=None: None

    with pytest.raises(owner.EngineError) as captured:
        list(client.stream({}, timeout=1.0))
    assert captured.value.code == ENGINE_RUNTIME_FAILED
    assert captured.value.message == "stream detail"
