"""v2.6.601 worker 生命周期与打包加固回归测试。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException

from kokoro_tts.admin_config_schema import load_runtime_config
from kokoro_tts.config import TTSConfig
from kokoro_tts.engines.parameters import EngineParameter, EngineParameterSchema
from kokoro_tts.moss.process_worker import MossProcessClient
from kokoro_tts.workers import EngineProcessClient, EngineWorkerSpec


ROOT = Path(__file__).resolve().parents[1]


def _test_worker_factory(_config: object, _provider: str | None) -> object:
    return object()


def test_generic_worker_spec_seam_is_extension_oriented():
    spec = EngineWorkerSpec("future-engine", _test_worker_factory)
    client = EngineProcessClient(config=TTSConfig(), spec=spec)
    assert client.spec is spec
    with pytest.raises(ValueError):
        EngineWorkerSpec("", _test_worker_factory)
    with pytest.raises(TypeError):
        EngineWorkerSpec("future-engine", None)  # type: ignore[arg-type]


def test_generic_worker_distinguishes_idle_from_unexpected_loaded_exit():
    class DeadProcess:
        exitcode = 9
        pid = 123
        def is_alive(self):
            return False

    client = EngineProcessClient(
        config=TTSConfig(),
        spec=EngineWorkerSpec("kokoro", _test_worker_factory),
    )
    assert client.is_healthy is True  # 未加载时表示主动空闲，不算异常退出。
    client._process = DeadProcess()
    client._loaded = True
    assert client.is_healthy is False
    with pytest.raises(RuntimeError):
        client._raise_if_worker_exited()
    assert client.is_healthy is False
    assert "退出码：9" in client.last_exit_reason


def test_moss_worker_uses_single_flight_lock_and_fresh_queue_generation():
    client = MossProcessClient(config=TTSConfig(), provider="cpu", engine_id="moss")
    assert isinstance(client._request_lock, type(threading.RLock()))
    assert client._command_queue is None and client._result_queue is None
    source = (ROOT / "src/kokoro_tts/workers/process_worker.py").read_text(encoding="utf-8")
    assert "with self._request_lock:" in source
    assert "self._command_queue = self._ctx.Queue()" in source
    assert "self._result_queue = self._ctx.Queue()" in source
    assert "finally:" in source and 'result_queue.put((request_id, "result", {"ok": True}))' in source


def test_runtime_config_preserves_valid_fields_and_sanitizes_corrupt_field(tmp_path):
    path = tmp_path / "runtime-config.json"
    path.write_text(json.dumps({"values": {
        "cache_max_items": "not-an-int",
        "kokoro_process_isolation_enabled": True,
        "zipvoice_process_isolation_enabled": True,
        "startup_preload_enabled": False,
    }}), encoding="utf-8")
    cfg = TTSConfig(runtime_config_file=path, kokoro_process_isolation_enabled=False, zipvoice_process_isolation_enabled=False, startup_preload_enabled=True)
    loaded = load_runtime_config(cfg)
    assert set(loaded) == {"kokoro_process_isolation_enabled", "zipvoice_process_isolation_enabled", "startup_preload_enabled"}
    assert cfg.kokoro_process_isolation_enabled is True
    assert cfg.zipvoice_process_isolation_enabled is True
    assert cfg.startup_preload_enabled is False
    stored = json.loads(path.read_text(encoding="utf-8"))["values"]
    assert "cache_max_items" not in stored


def test_dynamic_parameter_schema_handles_one_sided_range_without_typeerror():
    schema = EngineParameterSchema()
    schema._schemas["future-engine"] = (EngineParameter("steps", "integer", "步骤", "", minimum=1),)
    with pytest.raises(HTTPException) as exc:
        schema.parse("future-engine", supplied={"steps": 0})
    assert exc.value.status_code == 400
    assert "大于或等于 1" in exc.value.detail


def test_fnos_keeps_verified_profiles_and_versioned_image_contract():
    compose = (ROOT / "packaging/fnos/AngeVoice/app/docker/docker-compose.yaml").read_text(encoding="utf-8")
    wizard = (ROOT / "packaging/fnos/AngeVoice/wizard/install").read_text(encoding="utf-8")
    assert compose.count("profiles:") == 3
    for name in ("cpu", "gpu", "legacy-gpu"):
        assert f'profiles: ["{name}"]' in compose
        assert f"angevoice-{name}:v2.6.616" in compose
    assert "COMPOSE_PROFILES" in wizard
    assert "wizard_run_mode" not in wizard


def test_zipvoice_missing_prompt_asset_falls_back_to_empty_recommendations(monkeypatch):
    from kokoro_tts.routes import zipvoice as routes

    def missing(_self, *args, **kwargs):
        raise FileNotFoundError("asset missing")

    monkeypatch.setattr(routes.Path, "read_text", missing)
    assert routes._recommendations() == []


def test_legacy_zipvoice_delete_invalid_voice_id_returns_validation_error(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kokoro_tts.engine_manager import EngineManager
    from kokoro_tts.routes.zipvoice import create_zipvoice_router
    from kokoro_tts.service_state import ServiceState

    cfg = TTSConfig(
        enabled_models=["kokoro", "moss", "zipvoice"],
        default_model="kokoro",
        zipvoice_profiles_dir=tmp_path / "prompts" / "zipvoice",
        zipvoice_model_root=tmp_path / "models" / "zipvoice",
        zipvoice_distill_dir=tmp_path / "models" / "zipvoice" / "zipvoice_distill",
        zipvoice_vocos_dir=tmp_path / "models" / "zipvoice" / "vocos-mel-24khz",
        model_idle_timeout_seconds=0,
    )
    manager = EngineManager(cfg)
    state = ServiceState(cfg, model_manager=manager)

    async def verify():
        return True

    app = FastAPI()
    app.include_router(create_zipvoice_router(state, verify))
    try:
        with TestClient(app) as client:
            response = client.delete("/v1/zipvoice/profiles/bad!id")
        assert response.status_code == 400
        assert "voice_id" in response.json()["detail"]
    finally:
        manager.stop_idle_timer()


def test_entrypoint_compatibility_override_cannot_be_shadowed_by_cpu_env_defaults():
    entrypoint = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
    gpu_branch = entrypoint.split('gpu)', 1)[1].split(';;', 1)[0]
    assert 'export MOSS_EXECUTION_PROVIDER="cuda"' in gpu_branch
    assert 'export ZIPVOICE_EXECUTION_PROVIDER="cuda"' in gpu_branch
    assert '${MOSS_EXECUTION_PROVIDER:-cuda}' not in gpu_branch
    assert "fnOS 单服务模板" not in entrypoint


def test_zipvoice_native_runtime_can_be_imported_before_engine_registry():
    """Native runtimes must not re-enter eager adapter exports during import."""
    import importlib
    import sys

    for name in [
        "kokoro_tts.zipvoice.engine",
        "kokoro_tts.engines.registry",
        "kokoro_tts.engines.adapters",
        "kokoro_tts.engines",
    ]:
        sys.modules.pop(name, None)
    module = importlib.import_module("kokoro_tts.zipvoice.engine")
    registry = importlib.import_module("kokoro_tts.engines.registry")
    assert module.ZipVoiceEngine.__name__ == "ZipVoiceEngine"
    assert registry.EngineRegistry.public_model_ids == ("kokoro", "moss", "zipvoice")


def test_zipvoice_legacy_adapter_export_remains_lazy_and_compatible():
    from kokoro_tts.engines import adapters
    from kokoro_tts.zipvoice.engine import ZipVoiceEngine

    assert adapters.ZipVoiceEngine is ZipVoiceEngine
