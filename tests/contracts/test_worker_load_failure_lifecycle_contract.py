"""Freeze the worker engine load transaction lifecycle.

Current GREEN cases protect successful reuse, construction-failure retry,
load-failure classification, and child-loop survival.  Future RED cases define
the narrow transaction required to stop a partially loaded candidate from
becoming the child-owned engine.  Production intentionally remains unchanged
in this contract-freeze task.
"""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterator

import pytest

from kokoro_tts.contracts.errors import WorkerFailureEnvelope
from kokoro_tts.workers import process_worker
from kokoro_tts.workers.spec import EngineWorkerSpec


pytestmark = pytest.mark.contract


def _unused_factory(config: object, requested_provider: str | None) -> object:
    raise AssertionError("the contract harness replaces the worker factory")


@dataclass
class _FakeEngine:
    label: str
    load_error: BaseException | None = None
    unload_error: BaseException | None = None
    load_calls: int = 0
    unload_calls: int = 0
    metadata_calls: int = 0
    synthesize_calls: int = 0
    loaded: bool = False
    metadata_loaded_observations: list[bool] = field(default_factory=list)
    synthesize_loaded_observations: list[bool] = field(default_factory=list)

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        if self.unload_error is not None:
            raise self.unload_error
        self.loaded = False

    def metadata(self) -> dict[str, object]:
        self.metadata_calls += 1
        self.metadata_loaded_observations.append(self.loaded)
        return {"engine_label": self.label, "loaded": self.loaded}

    def synthesize(self, **_payload) -> dict[str, object]:
        self.synthesize_calls += 1
        self.synthesize_loaded_observations.append(self.loaded)
        return {"engine_label": self.label, "loaded_observed": self.loaded}

    def synthesize_array(self, **_payload) -> list[object]:
        return []

    def synthesize_stream(self, **_payload) -> Iterator[dict[str, object]]:
        yield {"type": "done", "total_audio_chunks": 0}

    def get_voices(self) -> list[str]:
        return []


class _FactorySequence:
    def __init__(self, *items: _FakeEngine | BaseException) -> None:
        self.items = items
        self.calls = 0

    def __call__(self, _config: object, _spec: EngineWorkerSpec) -> _FakeEngine:
        if self.calls >= len(self.items):
            raise AssertionError("worker requested an unexpected extra engine candidate")
        item = self.items[self.calls]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class _WorkerHarness:
    def __init__(self, factory: _FactorySequence) -> None:
        self.factory = factory
        self.commands: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self.results: queue.Queue[tuple[str, str, object]] = queue.Queue()
        self.cancel_flag = SimpleNamespace(value=0)
        self.escaped: list[BaseException] = []
        self._request_number = 0
        self.thread = threading.Thread(
            target=self._run,
            name="worker-load-lifecycle-contract",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            process_worker._worker_main(
                None,
                EngineWorkerSpec("kokoro", _unused_factory),
                self.commands,
                self.results,
                self.cancel_flag,
            )
        except BaseException as exc:  # exercise the child entrypoint boundary
            self.escaped.append(exc)

    def start(self) -> None:
        self.thread.start()

    def request(self, command: str, payload: dict | None = None) -> tuple[str, object]:
        self._request_number += 1
        request_id = f"request-{self._request_number}"
        self.commands.put((request_id, command, payload or {}))
        result_id, kind, result = self.results.get(timeout=3.0)
        assert result_id == request_id
        return kind, result

    @property
    def alive(self) -> bool:
        return self.thread.is_alive()

    def close(self) -> None:
        if not self.thread.is_alive():
            return
        self._request_number += 1
        request_id = f"shutdown-{self._request_number}"
        self.commands.put((request_id, "shutdown", {}))
        result_id, kind, result = self.results.get(timeout=3.0)
        assert (result_id, kind, result) == (request_id, "result", {"ok": True})
        self.thread.join(timeout=3.0)
        assert not self.thread.is_alive()


@contextmanager
def _running_worker(
    monkeypatch: pytest.MonkeyPatch,
    *factory_items: _FakeEngine | BaseException,
) -> Iterator[_WorkerHarness]:
    factory = _FactorySequence(*factory_items)
    monkeypatch.setattr(process_worker, "create_worker_engine", factory)
    harness = _WorkerHarness(factory)
    harness.start()
    try:
        yield harness
    finally:
        harness.close()


def _assert_load_failure(response: tuple[str, object], message: str) -> None:
    kind, payload = response
    assert kind == "error"
    assert isinstance(payload, WorkerFailureEnvelope)
    assert payload.code == "engine_load_failed"
    assert message in payload.message


def test_current_green_successful_engine_is_loaded_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _FakeEngine("candidate-1")
    with _running_worker(monkeypatch, candidate) as worker:
        load_kind, load_result = worker.request("load")
        metadata_kind, metadata_result = worker.request("metadata")
        synthesize_kind, synthesize_result = worker.request("synthesize", {"text": "ok"})

        assert (load_kind, metadata_kind, synthesize_kind) == ("result", "result", "result")
        assert load_result == {"engine_label": "candidate-1", "loaded": True}
        assert metadata_result == {"engine_label": "candidate-1", "loaded": True}
        assert synthesize_result == {"engine_label": "candidate-1", "loaded_observed": True}
        assert worker.factory.calls == 1
        assert candidate.load_calls == 1
        assert candidate.metadata_calls == 2
        assert candidate.synthesize_calls == 1


def test_current_green_construction_failure_does_not_commit_and_retries_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _FakeEngine("candidate-2")
    with _running_worker(monkeypatch, ValueError("construction failed"), candidate) as worker:
        _assert_load_failure(worker.request("load"), "construction failed")
        assert worker.alive
        kind, result = worker.request("load")

        assert kind == "result"
        assert result == {"engine_label": "candidate-2", "loaded": True}
        assert worker.factory.calls == 2
        assert candidate.load_calls == 1


def test_current_green_load_failure_keeps_stable_code_and_primary_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    with _running_worker(monkeypatch, candidate) as worker:
        _assert_load_failure(worker.request("load"), "load failed")


def test_current_green_load_failure_keeps_child_loop_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    with _running_worker(monkeypatch, candidate) as worker:
        _assert_load_failure(worker.request("load"), "load failed")
        assert worker.alive


def test_future_red_failed_candidate_is_not_committed_and_load_uses_fresh_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    recovered = _FakeEngine("candidate-2")
    with _running_worker(monkeypatch, failed, recovered) as worker:
        _assert_load_failure(worker.request("load"), "load failed")
        kind, result = worker.request("load")

        assert kind == "result"
        assert result == {"engine_label": "candidate-2", "loaded": True}
        assert worker.factory.calls == 2
        assert failed.load_calls == 1
        assert recovered.load_calls == 1


@pytest.mark.parametrize("command", ("metadata", "synthesize"))
def test_future_red_runtime_command_only_uses_fresh_loaded_candidate(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    failed = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    recovered = _FakeEngine("candidate-2")
    with _running_worker(monkeypatch, failed, recovered) as worker:
        _assert_load_failure(worker.request("load"), "load failed")
        kind, result = worker.request(command, {"text": "ok"})

        assert kind == "result"
        assert result["engine_label"] == "candidate-2"
        assert result.get("loaded", result.get("loaded_observed")) is True
        assert worker.factory.calls == 2
        assert failed.load_calls == 1
        assert recovered.load_calls == 1
        if command == "metadata":
            assert failed.metadata_calls == 0
            assert recovered.metadata_calls == 1
            assert recovered.metadata_loaded_observations == [True]
        else:
            assert failed.synthesize_calls == 0
            assert recovered.synthesize_calls == 1
            assert recovered.synthesize_loaded_observations == [True]


def test_future_red_failed_candidate_receives_best_effort_cleanup_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    recovered = _FakeEngine("candidate-2")
    with _running_worker(monkeypatch, failed, recovered) as worker:
        _assert_load_failure(worker.request("load"), "load failed")

        assert failed.unload_calls == 1
        assert worker.alive


def test_future_red_cleanup_failure_preserves_primary_load_error_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _FakeEngine(
        "candidate-1",
        load_error=ValueError("primary load failure"),
        unload_error=RuntimeError("cleanup failure"),
    )
    recovered = _FakeEngine("candidate-2")
    with _running_worker(monkeypatch, failed, recovered) as worker:
        response = worker.request("load")
        _assert_load_failure(response, "primary load failure")
        assert "cleanup failure" not in response[1].message
        assert failed.unload_calls == 1
        assert worker.alive

        kind, result = worker.request("load")
        assert kind == "result"
        assert result == {"engine_label": "candidate-2", "loaded": True}


def test_future_red_repeated_load_failures_remain_fresh_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeEngine("candidate-1", load_error=ValueError("load failed one"))
    second = _FakeEngine("candidate-2", load_error=ValueError("load failed two"))
    third = _FakeEngine("candidate-3")
    with _running_worker(monkeypatch, first, second, third) as worker:
        _assert_load_failure(worker.request("load"), "load failed one")
        _assert_load_failure(worker.request("load"), "load failed two")
        kind, result = worker.request("load")

        assert kind == "result"
        assert result == {"engine_label": "candidate-3", "loaded": True}
        assert worker.factory.calls == 3
        assert (first.load_calls, second.load_calls, third.load_calls) == (1, 1, 1)
        assert (first.unload_calls, second.unload_calls) == (1, 1)


def test_future_red_failure_recovery_commits_then_reuses_without_third_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = _FakeEngine("candidate-1", load_error=ValueError("load failed"))
    recovered = _FakeEngine("candidate-2")
    unused = _FakeEngine("candidate-3")
    with _running_worker(monkeypatch, failed, recovered, unused) as worker:
        _assert_load_failure(worker.request("load"), "load failed")
        metadata_kind, metadata_result = worker.request("metadata")
        synthesize_kind, synthesize_result = worker.request("synthesize", {"text": "ok"})

        assert (metadata_kind, synthesize_kind) == ("result", "result")
        assert metadata_result == {"engine_label": "candidate-2", "loaded": True}
        assert synthesize_result == {"engine_label": "candidate-2", "loaded_observed": True}
        assert worker.factory.calls == 2
        assert recovered.load_calls == 1
        assert unused.load_calls == 0
