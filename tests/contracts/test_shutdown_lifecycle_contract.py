"""Application shutdown barrier behavior and ownership contracts.

The tests use only fake engines and FastAPI lifespan orchestration. They do not
start a server, worker process, model runtime, download, or network request.
"""

from __future__ import annotations

import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

from kokoro_tts.config import TTSConfig
from kokoro_tts.engine_manager import EngineManager
from kokoro_tts.engines.adapters.kokoro import KokoroAdapter
from kokoro_tts.server import create_app


pytestmark = pytest.mark.contract

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "kokoro_tts"


class _FakeEngine:
    def __init__(self, *, unload_failures: int = 0) -> None:
        self.is_loaded = False
        self.is_healthy = True
        self.unload_calls: list[bool] = []
        self.soft_cancel_calls = 0
        self._unload_failures = unload_failures

    def load(self):
        self.is_loaded = True
        return self

    def unload(self, *, force: bool = False) -> None:
        self.unload_calls.append(force)
        if self._unload_failures:
            self._unload_failures -= 1
            raise RuntimeError("synthetic unload failure")
        self.is_loaded = False

    def soft_cancel(self) -> None:
        self.soft_cancel_calls += 1

    def metadata(self) -> dict:
        return {"loaded": self.is_loaded, "healthy": self.is_healthy}

    def get_voices(self) -> list[str]:
        return []


class _FakeWorker:
    def __init__(self) -> None:
        self.alive = True
        self.close_calls: list[bool] = []

    def close(self, *, kill: bool = False) -> None:
        self.close_calls.append(kill)
        self.alive = False


class _LegacyUnloadEngine:
    def __init__(self) -> None:
        self.unload_calls = 0

    def unload(self) -> None:
        self.unload_calls += 1


class _InternalTypeErrorEngine:
    def __init__(self) -> None:
        self.unload_calls: list[bool] = []
        self.fail = True

    def unload(self, *, force: bool = False) -> None:
        self.unload_calls.append(force)
        if self.fail:
            raise TypeError("internal defect")


def _config(tmp_path: Path, *, preload: bool = False) -> TTSConfig:
    return TTSConfig(
        model_dir=tmp_path / "models",
        credentials_dir=tmp_path / "credentials",
        api_key="shutdown-contract-key",
        default_model="kokoro",
        startup_preload_enabled=preload,
        startup_preload_model="kokoro",
        update_check_enabled=False,
        model_idle_timeout_seconds=0,
    )


def _definition(tree: ast.AST, name: str):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    )


def _call_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            result.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            result.add(call.func.attr)
    return result


def test_normal_lifespan_shutdown_closes_the_manager_and_initial_engine(tmp_path) -> None:
    engine = _FakeEngine()
    app = create_app(config=_config(tmp_path), engine=engine)

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            assert engine.unload_calls == []

    asyncio.run(run_lifespan())

    manager = app.state.angevoice.model_manager
    assert engine.unload_calls == [False]
    assert manager._closed is True
    assert manager._engines == {}


def test_preload_failure_still_runs_the_shutdown_barrier(monkeypatch, tmp_path) -> None:
    engine = _FakeEngine()
    app = create_app(config=_config(tmp_path, preload=True), engine=engine)
    manager = app.state.angevoice.model_manager

    def fail_preload(_model_id: str):
        raise RuntimeError("synthetic preload failure")

    monkeypatch.setattr(manager, "warm_model", fail_preload)

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("lifespan yielded after preload failure")

    with pytest.raises(RuntimeError, match="synthetic preload failure"):
        asyncio.run(run_lifespan())

    assert engine.unload_calls == [False]
    assert manager._closed is True
    assert manager._engines == {}


def test_close_all_retains_failed_engine_ownership_and_continues_other_cleanup(tmp_path) -> None:
    first = _FakeEngine()
    failing = _FakeEngine(unload_failures=1)
    after_failure = _FakeEngine()
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.0
    manager = EngineManager(cfg, initial_engine=first)
    manager._engines["moss"] = failing
    manager._engines["zipvoice"] = after_failure
    manager._active_counts.update({"kokoro": 0, "moss": 1, "zipvoice": 0})
    manager._pending_rebuild.add("moss")

    assert manager.close_all() is False

    assert first.unload_calls == [False]
    assert failing.unload_calls == [True]
    assert after_failure.unload_calls == [False]
    assert manager._engines == {"moss": failing}
    assert manager._active_counts == {"moss": 1}
    assert manager._pending_rebuild == set()
    assert manager._idle_timer is None

    manager._active_counts["moss"] = 0
    assert manager.close_all() is True
    assert failing.unload_calls == [True, False]
    assert manager._engines == {}
    assert manager._active_counts == {}


def test_close_all_calls_legacy_unload_once_without_force(tmp_path) -> None:
    engine = _LegacyUnloadEngine()
    manager = EngineManager(_config(tmp_path), initial_engine=engine)

    assert manager.close_all() is True

    assert engine.unload_calls == 1
    assert manager._engines == {}


def test_internal_type_error_is_not_retried_as_a_legacy_signature(tmp_path) -> None:
    engine = _InternalTypeErrorEngine()
    manager = EngineManager(_config(tmp_path), initial_engine=engine)

    assert manager.close_all() is False

    assert engine.unload_calls == [False]
    assert manager._engines == {"kokoro": engine}

    engine.fail = False
    assert manager.close_all() is True
    assert engine.unload_calls == [False, False]
    assert manager._engines == {}


@pytest.mark.parametrize(
    ("active_count", "expected_kill"),
    ((0, False), (1, True)),
)
def test_close_all_reaches_real_adapter_worker_close(
    tmp_path, active_count: int, expected_kill: bool
) -> None:
    worker = _FakeWorker()
    adapter = object.__new__(KokoroAdapter)
    adapter._worker = worker
    adapter._engine = None
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.0
    manager = EngineManager(cfg)
    manager._engines["kokoro"] = adapter
    manager._active_counts["kokoro"] = active_count

    assert manager.close_all() is True

    assert worker.close_calls == [expected_kill]
    assert worker.alive is False
    assert manager._engines == {}


def test_active_shutdown_uses_bounded_cancel_then_force_fallback(tmp_path) -> None:
    engine = _FakeEngine()
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.02
    manager = EngineManager(cfg, initial_engine=engine)
    manager._active_counts["kokoro"] = 1

    started = time.monotonic()
    assert manager.close_all() is True
    elapsed = time.monotonic() - started

    assert engine.soft_cancel_calls == 1
    assert engine.unload_calls == [True]
    assert elapsed >= 0.015
    assert elapsed < 0.5


def test_force_close_does_not_resurrect_bookkeeping_after_late_borrow_exit(tmp_path) -> None:
    engine = _FakeEngine()
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.0
    manager = EngineManager(cfg, initial_engine=engine)
    borrowed = threading.Event()
    release = threading.Event()
    borrower_errors: list[BaseException] = []

    def use_engine() -> None:
        try:
            with manager.borrow("kokoro"):
                borrowed.set()
                assert release.wait(timeout=2.0)
        except BaseException as exc:  # noqa: BLE001 - surfaced in the test thread
            borrower_errors.append(exc)

    borrower = threading.Thread(target=use_engine, daemon=True)
    borrower.start()
    assert borrowed.wait(timeout=1.0)
    assert manager._active_counts == {"kokoro": 1}

    assert manager.close_all() is True
    assert engine.unload_calls == [True]
    assert manager._engines == {}
    assert manager._active_counts == {}

    release.set()
    borrower.join(timeout=1.0)

    assert borrower.is_alive() is False
    assert borrower_errors == []
    assert manager._engines == {}
    assert manager._active_counts == {}
    assert "kokoro" not in manager._last_used


def test_failed_force_close_keeps_late_borrow_decrement_for_graceful_retry(tmp_path) -> None:
    engine = _FakeEngine(unload_failures=1)
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.0
    manager = EngineManager(cfg, initial_engine=engine)
    borrowed = threading.Event()
    release = threading.Event()

    def use_engine() -> None:
        with manager.borrow("kokoro"):
            borrowed.set()
            assert release.wait(timeout=2.0)

    borrower = threading.Thread(target=use_engine, daemon=True)
    borrower.start()
    assert borrowed.wait(timeout=1.0)
    assert manager._active_counts == {"kokoro": 1}

    assert manager.close_all() is False
    assert engine.unload_calls == [True]
    assert manager._engines == {"kokoro": engine}
    assert manager._active_counts == {"kokoro": 1}

    release.set()
    borrower.join(timeout=1.0)

    assert borrower.is_alive() is False
    assert manager._engines == {"kokoro": engine}
    assert manager._active_counts == {"kokoro": 0}

    assert manager.close_all() is True
    assert engine.unload_calls == [True, False]
    assert manager._engines == {}
    assert manager._active_counts == {}
    assert manager.close_all() is True
    assert engine.unload_calls == [True, False]


def test_close_all_is_idempotent_and_prevents_new_engine_creation(monkeypatch, tmp_path) -> None:
    engine = _FakeEngine()
    manager = EngineManager(_config(tmp_path), initial_engine=engine)
    monkeypatch.setattr(
        manager,
        "_create_engine",
        lambda *_args, **_kwargs: pytest.fail("closed manager created a new engine"),
    )

    assert manager.close_all() is True
    assert manager.close_all() is True

    assert engine.unload_calls == [False]
    with pytest.raises(RuntimeError, match="EngineManager is closed"):
        manager.get_engine("kokoro", load=False)
    with pytest.raises(RuntimeError, match="EngineManager is closed"):
        manager.switch_model("kokoro", load=False)
    with pytest.raises(RuntimeError, match="EngineManager is closed"):
        with manager.borrow("kokoro"):
            pytest.fail("closed manager accepted a borrow")
    assert manager._engines == {}
    assert manager._active_counts == {}
    assert manager._last_used == {}


def test_close_transition_is_atomic_with_borrow_and_releases_lock_during_drain(tmp_path) -> None:
    engine = _FakeEngine()
    cfg = _config(tmp_path)
    cfg.engine_process_stream_drain_seconds = 0.5
    manager = EngineManager(cfg, initial_engine=engine)
    borrowed = threading.Event()
    release = threading.Event()
    close_result: list[bool] = []

    def use_engine() -> None:
        with manager.borrow("kokoro"):
            borrowed.set()
            assert release.wait(timeout=1.0)

    borrower = threading.Thread(target=use_engine, daemon=True)
    borrower.start()
    assert borrowed.wait(timeout=1.0)

    closer = threading.Thread(
        target=lambda: close_result.append(manager.close_all()),
        daemon=True,
    )
    closer.start()
    deadline = time.monotonic() + 1.0
    while not manager._closed and time.monotonic() < deadline:
        time.sleep(0.005)
    assert manager._closed is True
    with pytest.raises(RuntimeError, match="EngineManager is closed"):
        manager.get_engine("kokoro", load=False)

    release.set()
    borrower.join(timeout=1.0)
    closer.join(timeout=1.0)

    assert borrower.is_alive() is False
    assert closer.is_alive() is False
    assert close_result == [True]
    assert engine.soft_cancel_calls == 1
    assert engine.unload_calls == [False]
    assert manager._engines == {}
    assert manager._active_counts == {}


def test_lifespan_body_failure_still_runs_the_shutdown_barrier(tmp_path) -> None:
    engine = _FakeEngine()
    app = create_app(config=_config(tmp_path), engine=engine)

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            raise RuntimeError("synthetic lifespan body failure")

    with pytest.raises(RuntimeError, match="synthetic lifespan body failure"):
        asyncio.run(run_lifespan())

    assert engine.unload_calls == [False]
    assert app.state.angevoice.model_manager._engines == {}


def test_service_state_initialization_failure_closes_manager_without_masking_error(
    monkeypatch, tmp_path
) -> None:
    from kokoro_tts import server

    engine = _FakeEngine()
    created: list[EngineManager] = []
    real_manager = server.EngineManager

    def manager_factory(config, *, initial_engine=None):
        manager = real_manager(config, initial_engine=initial_engine)
        created.append(manager)
        return manager

    def fail_state(*_args, **_kwargs):
        raise RuntimeError("synthetic service state failure")

    monkeypatch.setattr(server, "EngineManager", manager_factory)
    monkeypatch.setattr(server, "ServiceState", fail_state)

    with pytest.raises(RuntimeError, match="synthetic service state failure"):
        server.create_app(config=_config(tmp_path), engine=engine)

    assert len(created) == 1
    assert created[0]._closed is True
    assert created[0]._engines == {}
    assert engine.unload_calls == [False]


def test_lifespan_and_manager_keep_the_narrow_shutdown_ownership_boundary() -> None:
    server_tree = ast.parse(
        (PACKAGE_ROOT / "server.py").read_text(encoding="utf-8"),
        filename="server.py",
    )
    manager_tree = ast.parse(
        (PACKAGE_ROOT / "engine_manager.py").read_text(encoding="utf-8"),
        filename="engine_manager.py",
    )
    lifespan = _definition(_definition(server_tree, "create_app"), "lifespan")
    close_all = _definition(manager_tree, "close_all")
    begin_close_all = _definition(manager_tree, "_begin_close_all")
    finally_blocks = [
        node.finalbody
        for node in ast.walk(lifespan)
        if isinstance(node, ast.Try) and node.finalbody
    ]

    assert len(finally_blocks) == 1
    assert "close_all" in _call_names(ast.Module(body=finally_blocks[0]))
    assert "close_all" not in {
        node.name
        for node in ast.walk(server_tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "stop_idle_timer" in _call_names(begin_close_all)
    assert {
        "_begin_close_all",
        "_request_shutdown_cancellation",
        "_wait_for_shutdown_drain",
        "_release_shutdown_snapshot",
        "_finish_close_all",
    } <= _call_names(close_all)
    assert not {"get_engine", "_create_engine", "warm_model"} & _call_names(close_all)
    assert any(isinstance(node, ast.Try) for node in ast.walk(close_all))

    manager_init = _definition(_definition(manager_tree, "EngineManager"), "__init__")
    assert "threading.Condition(self._lock)" in ast.unparse(manager_init)
    for method_name, mutation in (
        ("get_engine", "self._engines.get(target_id)"),
        ("switch_model", "self._touch_model(target_id)"),
        ("borrow", "self._active_counts[target_id]"),
    ):
        method = _definition(manager_tree, method_name)
        guarded_blocks = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.With)
            and "self._lock" in ast.unparse(node.items[0].context_expr)
            and "self._ensure_open()" in ast.unparse(node)
        ]
        assert len(guarded_blocks) == 1
        assert mutation in ast.unparse(guarded_blocks[0])
