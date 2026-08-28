"""安全与并发回归测试。"""

from __future__ import annotations

import base64
import os
import sys
import time
import types

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from kokoro_tts.config import TTSConfig
from kokoro_tts.config_api_key import effective_api_key
from kokoro_tts.prompt_audio import PROMPT_AUDIO_STALE_SECONDS, save_prompt_audio_bytes
from kokoro_tts.routes.admin_runtime import rotate_api_key
from kokoro_tts.server import create_app
from kokoro_tts.service_state import ServiceState
from kokoro_tts.workers import EngineWorkerSpec


def _basic(username: str = "admin", password: str = "admin123") -> dict[str, str]:
    value = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {value}"}


def _fake_engine() -> MagicMock:
    engine = MagicMock()
    engine.is_loaded = True
    engine.is_healthy = True
    engine.get_voices.return_value = ["zm_010"]
    engine.default_voice = "zm_010"
    engine.metadata.return_value = {"id": "kokoro", "loaded": True, "voice_clone_supported": False}
    return engine


def test_api_key_rotation_uses_durable_file_without_process_env_leak(monkeypatch, tmp_path):
    monkeypatch.setenv("KOKORO_API_KEY", "bootstrap-from-operator")
    cfg = TTSConfig(api_key="bootstrap-from-operator", api_key_file=tmp_path / "credentials" / ".angevoice-api-key")
    rotated = rotate_api_key(cfg)
    assert rotated.startswith("av_")
    assert os.environ["KOKORO_API_KEY"] == "bootstrap-from-operator"
    # 单独的 worker/config 实例应读取持久化后的新密钥。
    other_worker = TTSConfig(api_key="bootstrap-from-operator", api_key_file=cfg.api_key_file)
    assert effective_api_key(other_worker) == rotated


def test_studio_bootstrap_reports_default_admin_warning_without_secrets(tmp_path):
    from kokoro_tts.routes.status import _bootstrap_base

    cfg = TTSConfig(
        admin_enabled=True,
        credentials_dir=tmp_path / "credentials",
        api_key_file=tmp_path / "credentials" / ".angevoice-api-key",
        admin_credentials_file=tmp_path / "credentials" / "admin-credentials.json",
    )
    payload = _bootstrap_base(cfg)

    assert payload["adminDefaultCredentialsActive"] is True
    assert "admin / admin123" in payload["adminSecurityWarning"]
    assert "api_key" not in payload


def test_admin_voice_upload_rejects_symlink_target_and_writes_regular_file(tmp_path):
    model_dir = tmp_path / "models"
    voices_dir = model_dir / "voices"
    voices_dir.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    try:
        (voices_dir / "evil.pt").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"当前平台无法创建符号链接：{exc}")
    cfg = TTSConfig(model_dir=model_dir, admin_enabled=True, voice_upload_enabled=True)
    client = TestClient(create_app(config=cfg, engine=_fake_engine()))
    rejected = client.post("/admin/voices/upload", headers=_basic(), files={"file": ("evil.pt", b"malicious", "application/octet-stream")})
    assert rejected.status_code == 409
    assert outside.read_bytes() == b"outside"
    accepted = client.post("/admin/voices/upload", headers=_basic(), files={"file": ("custom.pt", b"voice", "application/octet-stream")})
    assert accepted.status_code == 200
    assert (voices_dir / "custom.pt").read_bytes() == b"voice"
    assert "path" not in accepted.json()


def test_prompt_audio_cleanup_removes_only_stale_temp_files(monkeypatch, tmp_path):
    import kokoro_tts.prompt_audio as module
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    temp_dir = tmp_path / "angevoice_prompt_audio"
    temp_dir.mkdir()
    stale = temp_dir / "abandoned.wav"
    fresh = temp_dir / "keep.wav"
    stale.write_bytes(b"old")
    fresh.write_bytes(b"new")
    old = time.time() - PROMPT_AUDIO_STALE_SECONDS - 10
    os.utime(stale, (old, old))
    path, _ = save_prompt_audio_bytes(content=b"audio", filename="sample.wav", request_id="req", max_bytes=1024)
    assert path and Path(path).exists()
    assert not stale.exists()
    assert fresh.exists()


def test_websocket_cancellation_uses_threading_event():
    from kokoro_tts.routes.ws import TtsWebSocketSession
    state = MagicMock()
    state.cfg = TTSConfig()
    state.new_request_id.return_value = "req"
    session = TtsWebSocketSession(websocket=MagicMock(), state=state)
    assert hasattr(session, "cancel_event")
    assert not hasattr(session, "cancel_flag")
    session.cancel_event.set()
    assert session.cancel_event.is_set()


def test_client_request_id_is_sanitized_and_reused():
    state = ServiceState(TTSConfig())

    assert state.request_id_from_client("av_stop_123") == "av_stop_123"
    generated = state.request_id_from_client("../bad")
    assert generated != "../bad"
    assert len(generated) == 12


def test_cancel_endpoint_marks_request_without_touching_model_runtime(tmp_path):
    cfg = TTSConfig(model_dir=tmp_path, enabled_models=["kokoro"], default_model="kokoro")
    app = create_app(config=cfg, engine=_fake_engine())
    client = TestClient(app)
    state = app.state.angevoice
    state.mark_request("av_cancel_123", "running", model="kokoro")
    state.model_manager.cancel_model_request = MagicMock()

    response = client.post("/v1/audio/requests/av_cancel_123/cancel")

    assert response.status_code == 200
    assert response.json()["known"] is True
    assert response.json()["status"] == "cancelling"
    assert "runtime_cancel" not in response.json()
    state.model_manager.cancel_model_request.assert_not_called()


def test_websocket_client_cancel_only_marks_request_state():
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig()
    state.new_request_id.return_value = "av_ws_cancel"
    state.request_info.return_value = {"model": "kokoro", "status": "running"}
    state.model_manager.cancel_model_request.return_value = {"soft_cancelled": True}
    websocket = MagicMock()
    websocket.send_json = MagicMock()
    session = TtsWebSocketSession(websocket=websocket, state=state)

    import asyncio
    asyncio.run(session._mark_client_cancelled())

    state.request_cancel.assert_called_once_with("av_ws_cancel")
    state.model_manager.cancel_model_request.assert_not_called()


def test_multipart_prompt_audio_upload_streams_in_chunks_and_enforces_cap(monkeypatch, tmp_path):
    import asyncio
    import kokoro_tts.prompt_audio as module
    from fastapi import HTTPException

    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))

    class Upload:
        def __init__(self, payload: bytes):
            self.payload = payload
            self.offset = 0
            self.calls: list[int] = []

        async def read(self, size: int):
            self.calls.append(size)
            chunk = self.payload[self.offset:self.offset + size]
            self.offset += len(chunk)
            return chunk

    upload = Upload(b"a" * 17)
    path, digest = asyncio.run(module.save_prompt_audio_upload(
        upload=upload, filename="sample.wav", request_id="streamed", max_bytes=32, chunk_bytes=8
    ))
    assert path and Path(path).read_bytes() == b"a" * 17
    assert digest.startswith("sha256:")
    assert len(upload.calls) >= 3

    oversized = Upload(b"b" * 33)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(module.save_prompt_audio_upload(
            upload=oversized, filename="sample.wav", request_id="too-large", max_bytes=32, chunk_bytes=8
        ))
    assert exc_info.value.status_code == 413


def test_websocket_binary_audio_frame_without_data_is_reported_not_crashed():
    import asyncio
    from unittest.mock import AsyncMock
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig(request_timeout_seconds=2)
    state.new_request_id.return_value = "req-frame"
    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    websocket.send_bytes = AsyncMock()
    session = TtsWebSocketSession(websocket=websocket, state=state)

    async def run():
        await session.queue.put({"type": "audio"})
        await session._send_loop(binary=True)

    asyncio.run(run())
    websocket.send_bytes.assert_not_awaited()
    sent = websocket.send_json.await_args.args[0]
    assert sent["type"] == "error"
    assert sent["request_id"] == "req-frame"
    assert session.saw_stream_error is True


def test_websocket_disconnect_after_terminal_frame_keeps_done_status():
    """终止帧后的正常断开不应覆盖已完成状态。"""
    import asyncio
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig(request_timeout_seconds=2)
    state.new_request_id.return_value = "req-terminal-close"
    state.is_cancelled.return_value = False
    session = TtsWebSocketSession(websocket=MagicMock(), state=state)
    session.saw_stream_terminal = True

    asyncio.run(session._mark_client_cancelled())
    session._finish(time.perf_counter())

    state.request_cancel.assert_not_called()
    assert session.cancelled_by_client is False
    assert session.cancel_event.is_set() is False
    state.finish_request.assert_called_once()
    assert state.finish_request.call_args.args[1] == "done"


def test_request_snapshot_returns_recent_copy_and_pruning_works_outside_callers():
    state = ServiceState(TTSConfig(queue_status_enabled=True))
    for idx in range(121):
        state.mark_request(f"req-{idx}", "done", updated_at=float(idx))
    state.finish_request("req-120", "done", updated_at=120.0)
    values = state.request_snapshot(limit=3)
    assert [item["id"] for item in values] == ["req-120", "req-119", "req-118"]
    assert len(state.request_snapshot()) <= 101


def test_websocket_prompt_audio_base64_rejects_oversized_payload_before_decode(monkeypatch):
    import kokoro_tts.prompt_audio as module
    from fastapi import HTTPException

    called = False

    def should_not_decode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("oversized payload must be rejected before base64 allocation")

    monkeypatch.setattr(module.base64, "b64decode", should_not_decode)
    with pytest.raises(HTTPException) as exc_info:
        module.decode_prompt_audio_base64("Y" * 12, max_bytes=4)
    assert exc_info.value.status_code == 413
    assert called is False


def test_vendor_seedtts_reads_test_list_with_context_manager():
    source = (Path(__file__).resolve().parents[1] / "vendor/ZipVoice/zipvoice/eval/wer/seedtts.py").read_text(encoding="utf-8")
    assert 'with open(test_list, encoding="utf-8") as test_file:' in source
    assert "for line in open(test_list).readlines()" not in source


def test_websocket_multiple_error_frames_count_only_one_failed_request():
    import asyncio
    from unittest.mock import AsyncMock
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig(request_timeout_seconds=2)
    state.new_request_id.return_value = "req-double-error"
    websocket = MagicMock()
    websocket.send_json = AsyncMock()
    websocket.send_bytes = AsyncMock()
    session = TtsWebSocketSession(websocket=websocket, state=state)

    async def run():
        await session.queue.put({"type": "segment_error", "message": "first"})
        await session.queue.put({"type": "error", "message": "second"})
        await session.queue.put(session.done_marker)
        await session._send_loop(binary=False)

    asyncio.run(run())
    calls = [call for call in state.inc_stat.call_args_list if call.args and call.args[0] == "requests_error"]
    assert len(calls) == 1
    assert session.saw_stream_error is True


def test_request_history_pruning_never_removes_running_request():
    state = ServiceState(TTSConfig(queue_status_enabled=True))
    state.mark_request("long-running", "running", updated_at=-1.0)
    for idx in range(120):
        state.mark_request(f"done-{idx}", "done", updated_at=float(idx))
    state.finish_request("done-119", "done", updated_at=119.0)
    snapshot = {item["id"]: item for item in state.request_snapshot()}
    assert "long-running" in snapshot
    assert snapshot["long-running"]["status"] == "running"

def test_websocket_cancel_notice_is_not_created_when_event_loop_rejects_scheduling(monkeypatch):
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig(request_timeout_seconds=2)
    state.new_request_id.return_value = "req-loop-closed"
    state.streaming.iter_frames.return_value = []
    session = TtsWebSocketSession(websocket=MagicMock(), state=state)
    session.cancel_event.set()
    session.loop = MagicMock()
    session.loop.is_closed.return_value = False
    session.loop.call_soon_threadsafe.side_effect = RuntimeError("event loop is closed")

    notify = MagicMock()
    monkeypatch.setattr(session, "_notify_cancelled", notify)
    session._producer(MagicMock())

    session.loop.call_soon_threadsafe.assert_called_once_with(session._schedule_cancelled_notice)
    notify.assert_not_called()



def test_websocket_query_token_authenticates_before_payload_and_keeps_legacy_payload_shape():
    engine = _fake_engine()
    engine.synthesize_stream.return_value = iter([{"type": "done", "total_segments": 0, "total_audio_chunks": 0}])
    cfg = TTSConfig(api_key="secret-token")
    app = create_app(config=cfg, engine=engine)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v1/tts?token=secret-token") as ws:
            ws.send_json({"text": "测试", "voice": "zm_010", "format": "pcm_s16le"})
            frame = ws.receive_json()
    assert frame["type"] == "done"


def test_websocket_invalid_query_token_is_rejected_without_processing_request():
    from starlette.websockets import WebSocketDisconnect

    cfg = TTSConfig(api_key="secret-token")
    app = create_app(config=cfg, engine=_fake_engine())
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/ws/v1/tts?token=wrong"):
                pass
    assert exc.value.code == 1008


def test_websocket_preauth_does_not_reject_when_api_key_is_not_configured():
    """开放服务收到查询 token 时不应临时制造鉴权要求。"""
    engine = _fake_engine()
    engine.synthesize_stream.return_value = iter([{"type": "done", "total_segments": 0, "total_audio_chunks": 0}])
    cfg = TTSConfig(api_key=None)
    app = create_app(config=cfg, engine=engine)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v1/tts?token=ignored-when-auth-disabled") as ws:
            ws.send_json({"text": "测试", "voice": "zm_010", "format": "pcm_s16le"})
            frame = ws.receive_json()
    assert frame["type"] == "done"


def test_websocket_preauth_uses_persisted_key_even_when_cfg_api_key_is_none(tmp_path):
    """预认证应读取实际持久化密钥，而不是只看 cfg.api_key。"""
    engine = _fake_engine()
    engine.synthesize_stream.return_value = iter([{"type": "done", "total_segments": 0, "total_audio_chunks": 0}])
    key_file = tmp_path / ".angevoice-api-key"
    key_file.write_text("persisted-token\n", encoding="utf-8")
    cfg = TTSConfig(api_key=None, api_key_file=key_file)
    app = create_app(config=cfg, engine=engine)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v1/tts?token=persisted-token") as ws:
            ws.send_json({"text": "测试", "voice": "zm_010", "format": "pcm_s16le"})
            frame = ws.receive_json()
    assert frame["type"] == "done"


def test_websocket_message_budget_rejects_oversized_json_without_starting_engine():
    from starlette.websockets import WebSocketDisconnect

    engine = _fake_engine()
    cfg = TTSConfig(websocket_max_message_bytes=1024, rate_limit_qps=0, max_queue_length=0)
    app = create_app(config=cfg, engine=engine)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v1/tts") as ws:
            ws.send_json({"text": "超" * 2048, "voice": "zm_010", "format": "pcm_s16le"})
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "过大" in frame["message"]
    engine.synthesize_stream.assert_not_called()


def test_websocket_connection_gate_rejects_excess_idle_sessions():
    from starlette.websockets import WebSocketDisconnect

    cfg = TTSConfig(websocket_max_connections=1, rate_limit_qps=0, max_queue_length=0, request_timeout_seconds=30)
    app = create_app(config=cfg, engine=_fake_engine())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/v1/tts"):
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect("/ws/v1/tts"):
                    pass
            assert exc.value.code == 1013
    assert app.state.angevoice.snapshot_stats()["ws_connections_rejected_total"] == 1


def test_external_bind_without_api_key_warns_but_remains_supported(caplog):
    cfg = TTSConfig(host="0.0.0.0", api_key=None)
    with caplog.at_level("WARNING"):
        cfg.validate_security()
    assert "未启用 API 鉴权" in caplog.text


def test_loopback_bind_without_api_key_does_not_warn(caplog):
    cfg = TTSConfig(host="127.0.0.1", api_key=None)
    with caplog.at_level("WARNING"):
        cfg.validate_security()
    assert "未启用 API 鉴权" not in caplog.text

def test_websocket_producer_drains_after_cancel_without_sending_old_audio(monkeypatch):
    from unittest.mock import MagicMock
    from kokoro_tts.routes.ws import TtsWebSocketSession

    state = MagicMock()
    state.cfg = TTSConfig(request_timeout_seconds=2)
    state.new_request_id.return_value = "req-drain-cancel"
    state.is_cancelled.return_value = False
    consumed = []

    def frames(_request, *, cancel_check=None):
        assert cancel_check is not None
        for idx in range(3):
            consumed.append(idx)
            yield {"type": "audio", "index": idx}

    state.streaming.iter_frames.side_effect = frames
    session = TtsWebSocketSession(websocket=MagicMock(), state=state)
    session.cancel_event.set()
    session.loop = MagicMock()
    session.loop.is_closed.return_value = True
    monkeypatch.setattr(session, "_thread_put", MagicMock(return_value=True))

    session._producer(MagicMock())

    assert consumed == [0, 1, 2]
    session._thread_put.assert_not_called()



def test_status_auth_flags_use_persisted_api_key(tmp_path):
    """状态接口应按实际有效密钥展示鉴权状态。"""
    key_file = tmp_path / ".angevoice-api-key"
    key_file.write_text("persisted-token\n", encoding="utf-8")
    cfg = TTSConfig(api_key=None, api_key_file=key_file, public_status_endpoints=False)
    app = create_app(config=cfg, engine=_fake_engine())
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["auth_required"] is True
    assert health["catalog_protected"] is True
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer persisted-token"}).status_code == 200


def test_studio_api_session_cookie_allows_refresh_without_resubmitting_token(tmp_path):
    cfg = TTSConfig(api_key="persisted-token", public_status_endpoints=False)
    app = create_app(config=cfg, engine=_fake_engine())
    client = TestClient(app)

    assert client.get("/v1/models").status_code == 401
    login = client.post("/v1/auth/session", headers={"Authorization": "Bearer persisted-token"})
    assert login.status_code == 200
    cookie = login.headers.get("set-cookie", "")
    assert "angevoice_api_session=" in cookie
    assert "HttpOnly" in cookie
    assert "persisted-token" not in cookie
    assert client.get("/v1/models").status_code == 200

    clear = client.delete("/v1/auth/session")
    assert clear.status_code == 200
    assert client.get("/v1/models").status_code == 401


def test_health_reports_idle_for_lazy_wakeable_model_without_idle_timer():
    """懒加载且可唤醒的模型不应被健康检查误报为 loading。"""
    engine = _fake_engine()
    engine.is_loaded = False
    engine.metadata.return_value = {"id": "kokoro", "loaded": False, "voice_clone_supported": False}
    cfg = TTSConfig(model_idle_timeout_seconds=0)
    app = create_app(config=cfg, engine=engine)
    client = TestClient(app)

    assert client.get("/health").json()["status"] == "idle"


def test_health_uses_lightweight_snapshot_without_engine_metadata():
    """健康检查不应读取可能阻塞的运行时元数据。"""
    cfg = TTSConfig(enabled_models=["moss"], default_model="moss", moss_cuda_enabled=True, model_idle_timeout_seconds=0)
    app = create_app(config=cfg)
    state = app.state.angevoice
    engine = MagicMock()
    engine.is_loaded = True
    engine.is_healthy = True
    engine.metadata.side_effect = AssertionError("健康检查不应读取 metadata")
    state.model_manager._engines["moss"] = engine
    state.model_manager._current_model_id = "moss"
    client = TestClient(app)

    payload = client.get("/health").json()

    assert payload["status"] in {"ok", "idle"}
    engine.metadata.assert_not_called()


def test_health_lightweight_snapshot_keeps_kokoro_voices(tmp_path):
    """健康检查轻量快照仍应返回 Kokoro 本地音色。"""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "zm_010.pt").write_bytes(b"PK\x03\x04" + b"v" * 700)
    cfg = TTSConfig(enabled_models=["kokoro"], default_model="kokoro", model_dir=tmp_path, model_idle_timeout_seconds=0)
    app = create_app(config=cfg)
    state = app.state.angevoice
    engine = MagicMock()
    engine.is_loaded = True
    engine.is_healthy = True
    engine.metadata.side_effect = AssertionError("健康检查不应读取 metadata")
    state.model_manager._engines["kokoro"] = engine
    state.model_manager._current_model_id = "kokoro"
    client = TestClient(app)

    payload = client.get("/health").json()

    assert payload["voices"] == ["zm_010"]
    assert payload["model"]["default_voice"] == "zm_010"
    engine.metadata.assert_not_called()


def test_stats_uses_lightweight_snapshot_without_engine_metadata():
    """统计轮询不应等待正在合成的 worker 元数据请求。"""
    from kokoro_tts.config_ids import moss_voice_catalog

    cfg = TTSConfig(enabled_models=["moss"], default_model="moss", moss_cuda_enabled=True, model_idle_timeout_seconds=0)
    app = create_app(config=cfg)
    state = app.state.angevoice
    engine = MagicMock()
    engine.is_loaded = True
    engine.is_healthy = True
    engine.metadata.side_effect = AssertionError("统计轮询不应读取 metadata")
    state.model_manager._engines["moss"] = engine
    state.model_manager._current_model_id = "moss"
    client = TestClient(app)

    payload = client.get("/stats").json()

    assert payload["models"]["current"]["voices"] == moss_voice_catalog(cfg.moss_default_voice)
    assert payload["models"]["current"]["default_voice"] == cfg.moss_default_voice
    engine.metadata.assert_not_called()


def test_health_exposes_moss_builtin_voice_catalog_without_loading_metadata():
    """健康检查应返回 MOSS 内置音色目录，避免 Studio 只显示默认音色。"""
    from kokoro_tts.config_ids import moss_voice_catalog

    cfg = TTSConfig(enabled_models=["moss"], default_model="moss", moss_cuda_enabled=True, model_idle_timeout_seconds=0)
    app = create_app(config=cfg)
    state = app.state.angevoice
    engine = MagicMock()
    engine.is_loaded = True
    engine.is_healthy = True
    engine.metadata.side_effect = AssertionError("健康检查不应读取 metadata")
    state.model_manager._engines["moss"] = engine
    state.model_manager._current_model_id = "moss"
    client = TestClient(app)

    payload = client.get("/health").json()

    assert payload["voices"] == moss_voice_catalog(cfg.moss_default_voice)
    assert payload["model"]["default_voice"] == cfg.moss_default_voice
    engine.metadata.assert_not_called()


def test_vram_usage_accepts_torch_total_memory(monkeypatch):
    """显存统计兼容 PyTorch 的 total_memory 属性。"""
    from kokoro_tts.routes import status as status_routes

    props = types.SimpleNamespace(total_memory=8 * 1024 * 1024 * 1024)
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _device: props,
            memory_allocated=lambda _device: 1024,
            memory_reserved=lambda _device: 2048,
            get_device_name=lambda _device: "测试 GPU",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    payload = status_routes._get_vram_usage()

    assert payload["available"] is True
    assert payload["total_bytes"] == props.total_memory
    assert payload["reserved_bytes"] == 2048


def test_openai_speech_rejects_oversized_body_before_synthesis():
    """OpenAI 兼容入口也必须在业务解析前执行请求体大小限制。"""
    engine = _fake_engine()
    cfg = TTSConfig(tts_request_max_bytes=128)
    app = create_app(config=cfg, engine=engine)
    client = TestClient(app)

    response = client.post("/v1/audio/speech", json={"text": "你" * 256, "voice": "zm_010"})
    assert response.status_code == 413
    engine.synthesize.assert_not_called()


def test_engine_process_stream_drain_timeout_can_be_configured(monkeypatch):
    """worker 流取消排空窗口应支持环境变量配置。"""
    from kokoro_tts.config_env import apply_env

    monkeypatch.setenv("ANGEVOICE_ENGINE_PROCESS_STREAM_DRAIN_SECONDS", "12.5")
    cfg = TTSConfig()
    apply_env(cfg)
    assert cfg.engine_process_stream_drain_seconds == 12.5


def test_default_moss_cancel_drain_window_handles_slow_frames():
    """默认 MOSS 取消排空窗口应覆盖较慢首帧，避免频繁 kill+重载。"""
    cfg = TTSConfig()
    assert cfg.engine_process_stream_drain_seconds == 30.0


def test_moss_isolated_cancel_restart_cycles_keep_loaded_client():
    """模拟长文本停止后立刻再次合成，已加载的 MOSS client 不应被反复重载。"""
    from kokoro_tts.moss_engine_streaming import MossStreamingMixin

    class FakeProcessClient:
        is_loaded = True
        alive = True

        def __init__(self):
            self.stream_calls = 0
            self.events_seen: list[int] = []

        def stream(self, payload, *, timeout, cancel_check=None):
            self.stream_calls += 1
            assert cancel_check is not None
            if cancel_check():
                return
            for idx in range(64):
                self.events_seen.append(idx)
                yield {"type": "audio", "index": idx, "data": ""}

    class FakeMoss(MossStreamingMixin):
        engine_id = "moss-nano-cuda"

        def __init__(self):
            self.config = TTSConfig(request_timeout_seconds=30, engine_process_stream_drain_seconds=30.0)
            self._process_client = FakeProcessClient()
            self.load_calls = 0
            self.failures: list[str] = []

        def load(self):
            self.load_calls += 1
            return self

        def _mark_process_failure(self, *, timeout, reason):
            self.failures.append(reason)

    engine = FakeMoss()
    for _ in range(5):
        events = list(engine._synthesize_stream_process_isolated(
            text="这是一段用于模拟长文本的内容。" * 200,
            voice="Junhao",
            speed=1.0,
            fmt="pcm_s16le",
            prompt_audio_path=None,
            cancel_check=lambda: True,
        ))
        assert events == []

    assert engine.load_calls == 0
    assert engine.failures == []
    assert engine._process_client.stream_calls == 5
    assert len(engine._process_client.events_seen) == 0


def test_moss_isolated_stream_reports_missing_protocol_done():
    """MOSS 隔离流缺少协议完成帧时，应明确返回错误帧。"""
    from kokoro_tts.moss_engine_streaming import MossStreamingMixin

    class FakeProcessClient:
        is_loaded = True
        alive = True

        def stream(self, payload, *, timeout, cancel_check=None):
            yield {"type": "started", "segments": 2}
            yield {"type": "audio", "index": 0, "data": "AA==", "sample_rate": 48000, "channels": 2}
            yield {"type": "audio", "index": 1, "data": "AA==", "sample_rate": 48000, "channels": 2}

    class FakeMoss(MossStreamingMixin):
        engine_id = "moss-nano-cuda"

        def __init__(self):
            self.config = TTSConfig(request_timeout_seconds=30, engine_process_stream_idle_timeout_seconds=30)
            self._process_client = FakeProcessClient()

        def load(self):
            return self

        def _mark_process_failure(self, *, timeout, reason):
            raise AssertionError("缺少完成帧不应被记录为 worker 故障")

    frames = list(FakeMoss()._synthesize_stream_process_isolated(
        text="长文本" * 200,
        voice="Junhao",
        speed=1.0,
        fmt="pcm_s16le",
        prompt_audio_path=None,
        cancel_check=lambda: False,
    ))

    assert frames[-1]["type"] == "segment_error"
    assert frames[-1]["index"] == 2
    assert "未收到完成帧" in frames[-1]["message"]


def test_engine_process_cancel_drains_without_killing_worker(monkeypatch):
    """取消命中后应排空到 done，不应杀 worker 或继续向调用方返回旧音频。"""
    import threading
    from kokoro_tts.workers.process_worker import EngineProcessClient

    class Flag:
        value = 0

    class ResultQueue:
        def __init__(self):
            self.items = [
                ("req-current", "event", {"type": "audio", "index": 0}),
                ("req-current", "done", None),
            ]

        def get(self, timeout=None):
            if not self.items:
                raise AssertionError("测试队列不应被继续读取")
            return self.items.pop(0)

    cfg = TTSConfig(engine_process_stream_drain_seconds=30.0, request_timeout_seconds=30)
    client = EngineProcessClient.__new__(EngineProcessClient)
    client.config = cfg
    client.engine_id = "moss"
    client.logger = MagicMock()
    client._request_lock = threading.RLock()
    client._cancel_flag = Flag()
    client._send = MagicMock(return_value="req-current")
    client._require_result_queue = MagicMock(return_value=ResultQueue())
    client._raise_if_worker_exited = MagicMock()
    client.close = MagicMock()

    events = list(client.stream({"text": "长文本" * 200}, timeout=30, cancel_check=lambda: True))

    assert events == []
    assert client._cancel_flag.value == 1
    assert client._require_result_queue.call_count >= 1
    client.close.assert_not_called()


def test_engine_process_stream_drains_late_protocol_done_after_queue_done():
    """队列 done 先到时，父进程会短暂等待迟到的协议完成帧。"""
    import queue
    import threading
    from kokoro_tts.workers.process_worker import EngineProcessClient

    class Flag:
        value = 0

    class ResultQueue:
        def __init__(self):
            self.items = [
                ("req-current", "event", {"type": "audio", "index": 0}),
                ("req-current", "done", None),
                ("req-current", "event", {"type": "done", "total_audio_chunks": 1}),
            ]

        def get(self, timeout=None):
            if not self.items:
                raise queue.Empty()
            return self.items.pop(0)

    client = EngineProcessClient.__new__(EngineProcessClient)
    client.config = TTSConfig(engine_process_stream_drain_seconds=1.0, request_timeout_seconds=30)
    client.engine_id = "moss"
    client.logger = MagicMock()
    client._request_lock = threading.RLock()
    client._cancel_flag = Flag()
    client._send = MagicMock(return_value="req-current")
    client._require_result_queue = MagicMock(return_value=ResultQueue())
    client._raise_if_worker_exited = MagicMock()
    client.close = MagicMock()

    events = list(client.stream({"text": "长文本" * 200}, timeout=30, cancel_check=lambda: False))

    assert events == [
        {"type": "audio", "index": 0},
        {"type": "done", "total_audio_chunks": 1},
    ]
    client.close.assert_not_called()


def test_moss_stream_chunk_seconds_respects_config_bounds():
    """MOSS 长文本流式块不应绕过配置上限放大到超大帧。"""
    from kokoro_tts.moss_engine_streaming import MossStreamingMixin

    class DummyMossStream(MossStreamingMixin):
        pass

    dummy = DummyMossStream()
    dummy.config = TTSConfig(moss_stream_chunk_seconds=0.4)
    assert dummy._stream_chunk_seconds_for_text() == 0.4

    dummy.config = TTSConfig(moss_stream_chunk_seconds=10.0)
    assert dummy._stream_chunk_seconds_for_text() == 2.0

    dummy.config = TTSConfig(moss_stream_chunk_seconds=0.01)
    assert dummy._stream_chunk_seconds_for_text() == 0.05


def test_moss_metadata_uses_cached_vram_without_refresh(monkeypatch):
    """健康检查读取 MOSS 元数据时不应主动刷新 CUDA 显存。"""
    from kokoro_tts.moss_engine import MossNanoEngine

    engine = MossNanoEngine(TTSConfig(moss_vram_guard_enabled=True), execution_provider="cuda", process_isolation=True)
    refresh = MagicMock()
    monkeypatch.setattr(engine, "_refresh_vram_guard", refresh)

    metadata = engine.metadata()

    assert metadata["vram"]["source"] == "not-checked"
    refresh.assert_not_called()



def test_engine_process_generator_close_sets_soft_cancel_flag():
    """调用方提前关闭流生成器时，应设置软取消标志而不是杀 worker。"""
    import threading
    from kokoro_tts.workers.process_worker import EngineProcessClient

    class Flag:
        value = 0

    class ResultQueue:
        def __init__(self):
            self.items = [("req-current", "event", {"type": "audio", "index": 0})]

        def get(self, timeout=None):
            if not self.items:
                raise AssertionError("生成器关闭后不应继续读取队列")
            return self.items.pop(0)

    client = EngineProcessClient.__new__(EngineProcessClient)
    client.config = TTSConfig(request_timeout_seconds=30)
    client.engine_id = "moss"
    client.logger = MagicMock()
    client._request_lock = threading.RLock()
    client._cancel_flag = Flag()
    client._send = MagicMock(return_value="req-current")
    client._require_result_queue = MagicMock(return_value=ResultQueue())
    client._raise_if_worker_exited = MagicMock()
    client.close = MagicMock()

    stream = client.stream({"text": "长文本" * 200}, timeout=30, cancel_check=lambda: False)
    assert next(stream) == {"type": "audio", "index": 0}
    stream.close()

    assert client._cancel_flag.value == 1
    client.close.assert_not_called()


def test_worker_stream_injects_generation_scoped_cancel_check(monkeypatch):
    """子进程收到带代次保护的取消检查，只取消当前流式请求。"""
    import queue
    from kokoro_tts.workers import process_worker

    class Flag:
        value = 1

    class FakeEngine:
        def __init__(self):
            self.saw_cancel_check = False
            self.cancel_check_value = None

        def load(self):
            return self

        def unload(self):
            return None

        def synthesize_stream(self, **kwargs):
            cancel_check = kwargs.get("cancel_check")
            self.saw_cancel_check = callable(cancel_check)
            self.cancel_check_value = bool(cancel_check and cancel_check())
            if self.cancel_check_value:
                return
            yield {"type": "audio", "index": 0}
            yield {"type": "done", "total_audio_chunks": 1}

    engine = FakeEngine()
    monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args, **kwargs: engine)
    command_queue = queue.Queue()
    result_queue = queue.Queue()
    command_queue.put(("req-stream", "synthesize_stream", {"text": "长文本", "_cancel_generation": 2}))
    command_queue.put(("req-stop", "shutdown", {}))

    process_worker._worker_main(
        TTSConfig(),
        EngineWorkerSpec("moss", str, "cuda"),
        command_queue,
        result_queue,
        Flag,
    )

    assert engine.saw_cancel_check is True
    assert engine.cancel_check_value is False
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 0})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "done", "total_audio_chunks": 1})
    assert result_queue.get_nowait() == ("req-stream", "done", None)
    assert result_queue.get_nowait() == ("req-stop", "result", {"ok": True})


def test_worker_stream_skips_cancel_check_for_legacy_engine(monkeypatch):
    """旧式 Kokoro 流式引擎不支持 cancel_check 时不应被额外参数打断。"""
    import queue
    from kokoro_tts.workers import process_worker

    class Flag:
        value = 0

    class FakeEngine:
        def __init__(self):
            self.text = ""

        def load(self):
            return self

        def unload(self):
            return None

        def synthesize_stream(self, text):
            self.text = text
            yield {"type": "audio", "index": 0}
            yield {"type": "done", "total_audio_chunks": 1}

    engine = FakeEngine()
    monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args, **kwargs: engine)
    command_queue = queue.Queue()
    result_queue = queue.Queue()
    command_queue.put(("req-stream", "synthesize_stream", {"text": "你好", "_cancel_generation": 1}))
    command_queue.put(("req-stop", "shutdown", {}))

    process_worker._worker_main(
        TTSConfig(),
        EngineWorkerSpec("kokoro", str, "cuda"),
        command_queue,
        result_queue,
        Flag,
    )

    assert engine.text == "你好"
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 0})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "done", "total_audio_chunks": 1})
    assert result_queue.get_nowait() == ("req-stream", "done", None)
    assert result_queue.get_nowait() == ("req-stop", "result", {"ok": True})


def test_websocket_cancel_wait_keeps_short_safety_grace():
    """WebSocket 收尾只保留短宽限，避免旧长文本长期占住会话。"""
    import asyncio
    from kokoro_tts.routes.ws import TtsWebSocketSession

    async def run_case():
        state = MagicMock()
        state.cfg = TTSConfig(engine_process_stream_drain_seconds=0.2, request_timeout_seconds=10)
        state.new_request_id.return_value = "req-ws-drain-window"
        session = TtsWebSocketSession(websocket=MagicMock(), state=state)
        finished = False

        async def producer():
            nonlocal finished
            await asyncio.sleep(0.3)
            finished = True

        session.producer_task = asyncio.create_task(producer())
        await session._cancel_background_tasks()
        return finished, session.producer_task.cancelled()

    finished, cancelled = asyncio.run(run_case())
    assert finished is True
    assert cancelled is False


def test_websocket_stream_wait_uses_stream_idle_timeout_not_request_timeout():
    """流式连接等待 MOSS 首帧时不应被普通请求超时过早关闭。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from kokoro_tts.routes.ws import TtsWebSocketSession

    async def run_case():
        state = MagicMock()
        state.cfg = TTSConfig(request_timeout_seconds=1, websocket_stream_idle_timeout_seconds=5)
        state.new_request_id.return_value = "req-ws-idle"
        websocket = MagicMock()
        websocket.send_json = AsyncMock()
        session = TtsWebSocketSession(websocket=websocket, state=state)

        async def finish_later():
            await asyncio.sleep(1.2)
            await session.queue.put({"type": "done", "total_segments": 0, "total_audio_chunks": 0})
            await session.queue.put(session.done_marker)

        task = asyncio.create_task(finish_later())
        await session._send_loop(binary=False)
        await task
        return websocket.send_json.await_args_list, session.saw_stream_error

    calls, saw_error = asyncio.run(run_case())
    assert saw_error is False
    assert any(call.args and call.args[0].get("type") == "progress" for call in calls)


def test_engine_process_stream_idle_timeout_has_stable_floor():
    """子进程流式无帧等待窗口应使用独立流式配置，避免慢首帧误杀 worker。"""
    import queue
    import threading
    from unittest.mock import MagicMock
    from kokoro_tts.workers.process_worker import EngineProcessClient

    class Flag:
        value = 0

    class ResultQueue:
        def __init__(self):
            self.timeouts: list[float] = []
            self.items = [("req-current", "done", None)]

        def get(self, timeout=None):
            self.timeouts.append(float(timeout or 0))
            if not self.items:
                raise queue.Empty()
            return self.items.pop(0)

    queue_obj = ResultQueue()
    client = EngineProcessClient.__new__(EngineProcessClient)
    client.config = TTSConfig(request_timeout_seconds=1, engine_process_stream_idle_timeout_seconds=20)
    client.engine_id = "moss"
    client.logger = MagicMock()
    client._request_lock = threading.RLock()
    client._cancel_flag = Flag()
    client._send = MagicMock(return_value="req-current")
    client._require_result_queue = MagicMock(return_value=queue_obj)
    client._raise_if_worker_exited = MagicMock()
    client.close = MagicMock()

    assert list(client.stream({"text": "长文本"}, timeout=1, cancel_check=lambda: False)) == []
    assert queue_obj.timeouts
    assert queue_obj.timeouts[0] <= 0.2
    client.close.assert_not_called()


def test_websocket_done_marker_without_terminal_reports_truncated_stream():
    """生产者没有发送终止帧时，WebSocket 必须把部分音频标记为错误。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from kokoro_tts.routes.ws import TtsWebSocketSession

    async def run_case():
        state = MagicMock()
        state.cfg = TTSConfig(request_timeout_seconds=1, websocket_stream_idle_timeout_seconds=5)
        state.new_request_id.return_value = "req-terminal"
        state.is_cancelled.return_value = False
        websocket = MagicMock()
        websocket.send_json = AsyncMock()
        session = TtsWebSocketSession(websocket=websocket, state=state)
        await session.queue.put({"type": "started", "segments": 23})
        await session.queue.put({"type": "audio", "index": 35, "data": "AA==", "sample_rate": 48000, "channels": 2})
        await session.queue.put(session.done_marker)
        await session._send_loop(binary=False)
        return websocket.send_json.await_args_list, session.saw_stream_error

    calls, saw_error = asyncio.run(run_case())
    frames = [call.args[0] for call in calls]
    assert saw_error is True
    assert frames[-1]["type"] == "error"
    assert "未收到完成帧" in frames[-1]["message"]


def test_streaming_service_reports_missing_terminal_frame():
    """统一流式服务发现底层迭代器提前结束时，应输出错误帧。"""
    from kokoro_tts.services.streaming_service import StreamingService

    class FakeManager:
        current_model_id = "moss"

        def normalize_model_id(self, model_id):
            return model_id or "moss"

        def borrow(self, _model_id):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def synthesize_stream(self, *_args, **_kwargs):
            yield {"type": "started", "segments": 1}
            yield {"type": "audio", "index": 0, "data": "AA==", "sample_rate": 48000, "channels": 2}

    state = MagicMock()
    state.cfg = TTSConfig()
    state.model_manager = FakeManager()
    state.voice_profiles.resolve_condition.return_value = MagicMock()
    state.parameter_schema.parse.return_value = {}
    service = StreamingService(state)
    request = service.build_request(
        text="测试长文本",
        model_id="moss",
        voice="Junhao",
        speed=1.0,
        audio_format="pcm_s16le",
        binary=False,
        request_id="req-stream-missing-done",
    )

    frames = list(service.iter_frames(request, cancel_check=lambda: False))

    assert frames[-1]["type"] == "segment_error"
    assert "未收到完成帧" in frames[-1]["message"]



def test_worker_stream_keeps_queue_done_separate_from_protocol_frames(monkeypatch):
    """worker 会转发协议完成帧，队列级 done 只用于释放父进程等待。"""
    import queue
    from kokoro_tts.workers import process_worker

    class Flag:
        value = 0

    class FakeEngine:
        def load(self):
            return self

        def unload(self):
            return None

        def synthesize_stream(self, **kwargs):
            yield {"type": "audio", "index": 0}
            yield {"type": "audio", "index": 1}
            yield {"type": "done", "total_segments": 2, "total_audio_chunks": 2}

    monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args, **kwargs: FakeEngine())
    command_queue = queue.Queue()
    result_queue = queue.Queue()
    command_queue.put(("req-stream", "synthesize_stream", {"text": "长文本"}))
    command_queue.put(("req-stop", "shutdown", {}))

    process_worker._worker_main(
        TTSConfig(),
        EngineWorkerSpec("moss", str, "cuda"),
        command_queue,
        result_queue,
        Flag,
    )

    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 0})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 1})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "done", "total_segments": 2, "total_audio_chunks": 2})
    assert result_queue.get_nowait() == ("req-stream", "done", None)


def test_worker_stream_marks_missing_terminal_before_protocol_done(monkeypatch):
    """缺失语义终止帧时先报告失败，再发送协议与队列关闭标记。"""
    import queue
    from kokoro_tts.workers import process_worker

    class Flag:
        value = 0

    class FakeEngine:
        def load(self):
            return self

        def unload(self):
            return None

        def synthesize_stream(self, **kwargs):
            yield {"type": "started", "segments": 2}
            yield {"type": "audio", "index": 0}
            yield {"type": "audio", "index": 1}

    monkeypatch.setattr(process_worker, "create_worker_engine", lambda *args, **kwargs: FakeEngine())
    command_queue = queue.Queue()
    result_queue = queue.Queue()
    command_queue.put(("req-stream", "synthesize_stream", {"text": "长文本"}))
    command_queue.put(("req-stop", "shutdown", {}))

    process_worker._worker_main(
        TTSConfig(),
        EngineWorkerSpec("moss", str, "cuda"),
        command_queue,
        result_queue,
        Flag,
    )

    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "started", "segments": 2})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 0})
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "audio", "index": 1})
    missing_terminal = result_queue.get_nowait()
    assert missing_terminal[:2] == ("req-stream", "event")
    assert missing_terminal[2]["type"] == "segment_error"
    assert set(missing_terminal[2]) == {"type", "message"}
    assert isinstance(missing_terminal[2]["message"], str)
    assert missing_terminal[2]["message"]
    assert result_queue.get_nowait() == ("req-stream", "event", {"type": "done", "total_audio_chunks": 2, "total_segments": 2})
    assert result_queue.get_nowait() == ("req-stream", "done", None)


def test_moss_default_prebuffer_keeps_realtime_feel():
    """MOSS 默认预缓冲保持主线实时体验。"""
    cfg = TTSConfig()
    assert cfg.moss_stream_prebuffer_seconds == 0.75


def test_moss_realtime_streaming_decode_defaults_to_realtime_mode():
    """MOSS 默认保持逐帧实时解码，避免回退改动影响原有流式体验。"""
    assert TTSConfig().moss_realtime_streaming_decode is True
