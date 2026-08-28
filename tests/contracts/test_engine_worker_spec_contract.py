"""WorkerSpec construction ownership and multiprocessing contracts."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import multiprocessing as mp
import pickle
import queue
from types import SimpleNamespace

import pytest

from kokoro_tts.engines.adapters.kokoro import (
    KokoroAdapter,
    _create_kokoro_worker_engine,
)
from kokoro_tts.moss_engine import _create_moss_worker_engine
from kokoro_tts.workers import EngineWorkerSpec
from kokoro_tts.workers import process_worker
from kokoro_tts.zipvoice.engine import _create_zipvoice_worker_engine


pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "kokoro_tts"
_FACTORY_CALLS: list[tuple[object, str | None]] = []


class _ContractEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def metadata(self) -> dict[str, object]:
        return {"loaded": self.loaded, "owner": "contract"}


def _contract_factory(config: object, provider: str | None) -> _ContractEngine:
    _FACTORY_CALLS.append((config, provider))
    return _ContractEngine()


def test_worker_spec_is_frozen_typed_and_minimal() -> None:
    spec = EngineWorkerSpec("contract", _contract_factory, "cpu")
    assert tuple(field.name for field in fields(spec)) == (
        "engine_id",
        "factory",
        "requested_provider",
    )
    with pytest.raises(FrozenInstanceError):
        spec.engine_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        EngineWorkerSpec("", _contract_factory)
    with pytest.raises(TypeError):
        EngineWorkerSpec(123, _contract_factory)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EngineWorkerSpec("contract", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EngineWorkerSpec("contract", _contract_factory, 123)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("engine_id", "factory", "provider"),
    (
        ("kokoro", _create_kokoro_worker_engine, None),
        ("moss", _create_moss_worker_engine, "cpu"),
        ("zipvoice", _create_zipvoice_worker_engine, "cpu"),
    ),
)
def test_production_worker_specs_are_forking_pickle_compatible(
    engine_id: str,
    factory,
    provider: str | None,
) -> None:
    assert "<locals>" not in factory.__qualname__
    assert not factory.__module__.startswith("kokoro_tts.workers")
    spec = EngineWorkerSpec(engine_id, factory, provider)
    restored = pickle.loads(bytes(ForkingPickler.dumps(spec)))
    assert restored == spec
    assert restored.factory is factory


def test_worker_spec_crosses_a_real_spawn_serialization_boundary() -> None:
    context = mp.get_context("spawn")
    output = context.Queue()
    spec = EngineWorkerSpec("kokoro", _create_kokoro_worker_engine)
    process = context.Process(target=output.put, args=(spec,))
    process.start()
    process.join(timeout=15)
    try:
        assert process.exitcode == 0
        restored = output.get(timeout=2)
        assert restored == spec
        assert restored.factory is _create_kokoro_worker_engine
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        output.close()
        output.join_thread()


def test_worker_main_receives_spec_and_invokes_declared_factory() -> None:
    _FACTORY_CALLS.clear()
    config = SimpleNamespace(marker="serialized-config")
    commands: queue.Queue[tuple[str, str, dict]] = queue.Queue()
    results: queue.Queue[tuple[str, str, object]] = queue.Queue()
    commands.put(("load-request", "load", {}))
    commands.put(("shutdown-request", "shutdown", {}))
    spec = EngineWorkerSpec("contract", _contract_factory, "synthetic-provider")

    process_worker._worker_main(
        config,
        spec,
        commands,
        results,
        SimpleNamespace(value=0),
    )

    assert _FACTORY_CALLS == [(config, "synthetic-provider")]
    assert results.get_nowait() == (
        "load-request",
        "result",
        {"loaded": True, "owner": "contract"},
    )
    assert results.get_nowait() == (
        "shutdown-request",
        "result",
        {"ok": True},
    )


def test_worker_infrastructure_has_no_concrete_engine_or_loader_knowledge() -> None:
    forbidden_modules = {
        "kokoro_tts.engine",
        "kokoro_tts.moss_engine",
        "kokoro_tts.zipvoice.engine",
    }
    found: set[str] = set()
    for path in (PACKAGE_ROOT / "workers").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 2:
                    found.add(f"kokoro_tts.{module}")
                else:
                    found.add(module)
        source = path.read_text(encoding="utf-8")
        assert "importlib" not in source
        assert "__import__" not in source
    assert not forbidden_modules & found
    assert not (PACKAGE_ROOT / "workers" / "factories.py").exists()


def test_kokoro_direct_construction_injection_remains_in_process() -> None:
    engine = SimpleNamespace(is_loaded=False, is_healthy=True)
    cfg = SimpleNamespace(kokoro_process_isolation_enabled=True)
    adapter = KokoroAdapter(cfg, engine=engine)
    assert adapter._process_isolated is False
    assert adapter._worker is None
    assert adapter._engine is engine
