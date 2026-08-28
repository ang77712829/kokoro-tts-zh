"""AngeVoice FastAPI 应用工厂。

历史公开 API 仍保留 ``kokoro_tts.server.create_app`` 与
``kokoro_tts.server.run_server``。具体路由放在 ``kokoro_tts.routes``，
让本文件专注应用装配。
"""

import logging
import os
from pathlib import Path
from typing import Optional

from . import __version__
from .config import TTSConfig, load_config
from .config_env_domain import (
    MOSS_STREAM_BUDGET_ENV_DECLARATIONS,
    UPDATE_CHECK_ENV_DECLARATIONS,
)
from .banner import format_startup_banner
from .engine import TTSEngine
from .engine_manager import EngineManager
from .routes import (
    create_admin_router,
    create_auth_router,
    create_audio_router,
    create_status_router,
    create_ws_router,
    create_zipvoice_router,
)
from .security import make_verify_api_key
from .service_state import ServiceState
from .rate_limit import GlobalQueueMiddleware, RateLimitMiddleware
from .static_assets import StaticAssetManifest

logger = logging.getLogger(__name__)


_WORKER_ENV_EXPORTS = {
    "KOKORO_MODEL_DIR": "model_path",
    "KOKORO_DEVICE": "device",
    "ANGEVOICE_DEPLOYMENT_PROFILE": "deployment_profile",
    "ANGEVOICE_ACCESS_LOG_ENABLED": "access_log_enabled",
    "KOKORO_DEFAULT_VOICE": "default_voice",
    "KOKORO_PREFETCH_VOICES": "kokoro_prefetch_voices",
    "KOKORO_STREAM_FORMAT": "stream_format",
    "KOKORO_STREAM_CHUNK_SECONDS": "stream_chunk_seconds",
    "KOKORO_STREAM_PREBUFFER_SECONDS": "stream_prebuffer_seconds",
    "KOKORO_MP3_BITRATE": "mp3_bitrate",
    "ANGEVOICE_AUDIO_MP3_BITRATE": "mp3_bitrate",
    "ANGEVOICE_AUDIO_OPUS_BITRATE": "audio_opus_bitrate",
    "ANGEVOICE_AUDIO_AAC_BITRATE": "audio_aac_bitrate",
    "ANGEVOICE_FFMPEG_BINARY": "ffmpeg_binary",
    "ANGEVOICE_FFMPEG_TIMEOUT_SECONDS": "ffmpeg_timeout_seconds",
    "KOKORO_MAX_CONCURRENT_REQUESTS": "max_concurrent_requests",
    "KOKORO_MAX_TEXT_LENGTH": "max_text_length",
    "KOKORO_SEGMENT_LENGTH": "segment_length",
    "ANGEVOICE_SINGLE_NEWLINE_POLICY": "text_single_newline_policy",
    "ANGEVOICE_TN_ENGINE": "angevoice_tn_engine",
    "MOSS_SEGMENT_LENGTH": "moss_segment_length",
    "ZIPVOICE_EXECUTION_PROVIDER": "zipvoice_execution_provider",
    "ZIPVOICE_CUDA_ENABLED": "zipvoice_cuda_enabled",
    "ZIPVOICE_AUTO_FALLBACK_CPU": "zipvoice_auto_fallback_cpu",
    "ZIPVOICE_CUDA_DEVICE_INDEX": "zipvoice_cuda_device_index",
    "ZIPVOICE_CUDA_MAX_DURATION": "zipvoice_cuda_max_duration",
    "KOKORO_CACHE_MAX_ITEMS": "cache_max_items",
    "KOKORO_CACHE_MAX_BYTES": "cache_max_bytes",
    "KOKORO_CACHE_SKIP_TEXT_OVER_CHARS": "cache_skip_text_over_chars",
    "KOKORO_CACHE_SKIP_AUDIO_OVER_BYTES": "cache_skip_audio_over_bytes",
    "KOKORO_BATCH_MAX_ITEMS": "batch_max_items",
    "KOKORO_BATCH_CONCURRENCY": "batch_concurrency",
    "KOKORO_DEFAULT_SPEED": "default_speed",
    "KOKORO_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "ANGEVOICE_WEBSOCKET_STREAM_IDLE_TIMEOUT_SECONDS": "websocket_stream_idle_timeout_seconds",
    "KOKORO_STREAM_BINARY_ENABLED": "stream_binary_enabled",
    "KOKORO_CACHE_ENABLED": "cache_enabled",
    "KOKORO_QUEUE_STATUS_ENABLED": "queue_status_enabled",
    "KOKORO_METRICS_ENABLED": "metrics_enabled",
    "KOKORO_BATCH_ENABLED": "batch_enabled",
    "KOKORO_ADMIN_ENABLED": "admin_enabled",
    "KOKORO_VOICE_UPLOAD_ENABLED": "voice_upload_enabled",
    "KOKORO_VOICE_UPLOAD_MAX_BYTES": "voice_upload_max_bytes",
    "KOKORO_TTS_REQUEST_MAX_BYTES": "tts_request_max_bytes",
    "KOKORO_MP3_ENABLED": "mp3_enabled",
    "ANGEVOICE_FFMPEG_ENABLED": "ffmpeg_enabled",
    "ANGEVOICE_ENABLED_MODELS": "enabled_models",
    "ANGEVOICE_DEFAULT_MODEL": "default_model",
    "ANGEVOICE_STARTUP_PRELOAD_MODEL": "startup_preload_model",
    "ANGEVOICE_MODEL_SWITCH_ENABLED": "model_switch_enabled",
    "ANGEVOICE_MODEL_UNLOAD_ON_SWITCH": "model_unload_on_switch",
    "ANGEVOICE_MODEL_SWITCH_TIMEOUT_SECONDS": "model_switch_timeout_seconds",
    "ANGEVOICE_OUTPUT_DIR": "output_dir",
    "ANGEVOICE_SAVE_OUTPUTS": "save_outputs",
    "ANGEVOICE_OUTPUT_MAX_FILES": "output_max_files",
    "ANGEVOICE_RUNTIME_CONFIG_FILE": "runtime_config_file",
    **{
        declaration.env_name: declaration.attr
        for declaration in UPDATE_CHECK_ENV_DECLARATIONS
    },
    "ANGEVOICE_CREDENTIALS_DIR": "credentials_dir",
    "ANGEVOICE_ADMIN_CREDENTIALS_FILE": "admin_credentials_file",
    "ANGEVOICE_MODEL_SOURCE": "model_source",
    "ANGEVOICE_MODEL_SOURCE_DETECT_URL": "model_source_detect_url",
    "ANGEVOICE_MODEL_SOURCE_DETECT_TIMEOUT_SECONDS": "model_source_detect_timeout_seconds",
    "ANGEVOICE_MODEL_SOURCE_PROBE_TIMEOUT_SECONDS": "model_source_probe_timeout_seconds",
    "ANGEVOICE_MODEL_SOURCE_PROBE_HF_URL": "model_source_probe_hf_url",
    "ANGEVOICE_MODEL_SOURCE_PROBE_MODELSCOPE_URL": "model_source_probe_modelscope_url",
    "ANGEVOICE_API_KEY_FILE": "api_key_file",
    "KOKORO_HF_REPO": "kokoro_hf_repo",
    "KOKORO_MODELSCOPE_REPO": "kokoro_modelscope_repo",
    "MOSS_MODELSCOPE_REPO": "moss_modelscope_repo",
    "MOSS_HF_REPO": "moss_hf_repo",
    "MOSS_AUDIO_TOKENIZER_MODELSCOPE_REPO": "moss_audio_tokenizer_modelscope_repo",
    "MOSS_AUDIO_TOKENIZER_HF_REPO": "moss_audio_tokenizer_hf_repo",
    "ANGEVOICE_IDLE_TIMEOUT_SECONDS": "model_idle_timeout_seconds",
    "ANGEVOICE_IDLE_CHECK_INTERVAL": "model_idle_check_interval",
    "ANGEVOICE_IDLE_UNLOAD_CURRENT": "model_idle_unload_current",
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD": "restart_after_idle_unload_enabled",
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_DELAY_SECONDS": "restart_after_idle_unload_delay_seconds",
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_COOLDOWN_SECONDS": "restart_after_idle_unload_cooldown_seconds",
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_EXIT_CODE": "restart_after_idle_unload_exit_code",
    "MOSS_EXECUTION_PROVIDER": "moss_execution_provider",
    "MOSS_CPU_THREADS": "moss_cpu_threads",
    "ZIPVOICE_CPU_THREADS": "zipvoice_cpu_threads",
    "ZIPVOICE_NUM_STEPS": "zipvoice_num_steps",
    "ZIPVOICE_PROMPT_UPLOAD_MAX_BYTES": "zipvoice_prompt_upload_max_bytes",
    "ZIPVOICE_PROMPT_AUDIO_MAX_SECONDS": "zipvoice_prompt_audio_max_seconds",
    "ZIPVOICE_GUIDANCE_SCALE": "zipvoice_guidance_scale",
    "ZIPVOICE_T_SHIFT": "zipvoice_t_shift",
    "ZIPVOICE_TARGET_RMS": "zipvoice_target_rms",
    "ZIPVOICE_FEAT_SCALE": "zipvoice_feat_scale",
    "ZIPVOICE_DOWNLOAD_ENABLED": "zipvoice_download_enabled",
    "ZIPVOICE_REMOVE_LONG_SIL": "zipvoice_remove_long_sil",
    "MOSS_DEFAULT_VOICE": "moss_default_voice",
    "MOSS_PROMPT_UPLOAD_MAX_BYTES": "moss_prompt_upload_max_bytes",
    "MOSS_PROMPT_AUDIO_MAX_SECONDS": "moss_prompt_audio_max_seconds",
    "MOSS_PROMPT_CACHE_MAX_ITEMS": "moss_prompt_cache_max_items",
    "MOSS_MAX_NEW_FRAMES": "moss_max_new_frames",
    "MOSS_VOICE_CLONE_MAX_TEXT_TOKENS": "moss_voice_clone_max_text_tokens",
    "MOSS_SAMPLE_MODE": "moss_sample_mode",
    "MOSS_SEED": "moss_seed",
    "MOSS_CUDA_ENABLED": "moss_cuda_enabled",
    "MOSS_CUDA_MEMORY_LIMIT_MB": "moss_cuda_memory_limit_mb",
    "MOSS_ENABLE_WETEXT_PROCESSING": "moss_enable_wetext_processing",
    "MOSS_ENABLE_NORMALIZE_TTS_TEXT": "moss_enable_normalize_tts_text",
    "MOSS_APPLY_ANGEVOICE_RULES": "moss_apply_angevoice_rules",
    "MOSS_MIXED_ENGLISH_POLICY": "moss_mixed_english_policy",
    "MOSS_REALTIME_STREAMING_DECODE": "moss_realtime_streaming_decode",
    "MOSS_STREAM_CHUNK_SECONDS": "moss_stream_chunk_seconds",
    "MOSS_STREAM_QUEUE_MAX_ITEMS": "moss_stream_queue_max_items",
    "MOSS_STREAM_PREBUFFER_SECONDS": "moss_stream_prebuffer_seconds",
    "MOSS_CUDA_SELF_TEST_ENABLED": "moss_cuda_self_test_enabled",
    "MOSS_AUTO_FALLBACK_CPU": "moss_auto_fallback_cpu",
    "MOSS_QUALITY_GATE_ENABLED": "moss_quality_gate_enabled",
    "MOSS_MAX_CLIP_RATIO": "moss_max_clip_ratio",
    "MOSS_OUTPUT_PEAK_NORMALIZE_ENABLED": "moss_output_peak_normalize_enabled",
    "MOSS_OUTPUT_TARGET_PEAK": "moss_output_target_peak",
    "MOSS_OUTPUT_GAIN": "moss_output_gain",
    "MOSS_PROCESS_ISOLATION_ENABLED": "moss_process_isolation_enabled",
    "KOKORO_PROCESS_ISOLATION_ENABLED": "kokoro_process_isolation_enabled",
    "ZIPVOICE_PROCESS_ISOLATION_ENABLED": "zipvoice_process_isolation_enabled",
    "ANGEVOICE_STARTUP_PRELOAD_ENABLED": "startup_preload_enabled",
    "MOSS_PROCESS_ISOLATION_PROVIDERS": "moss_process_isolation_providers",
    "MOSS_PROCESS_KILL_GRACE_SECONDS": "moss_process_kill_grace_seconds",
    "ANGEVOICE_ENGINE_PROCESS_KILL_GRACE_SECONDS": "engine_process_kill_grace_seconds",
    "ANGEVOICE_ENGINE_PROCESS_STREAM_DRAIN_SECONDS": "engine_process_stream_drain_seconds",
    "ANGEVOICE_ENGINE_PROCESS_STREAM_IDLE_TIMEOUT_SECONDS": "engine_process_stream_idle_timeout_seconds",
    "MOSS_OUTPUT_DECLICK_ENABLED": "moss_output_declick_enabled",
    "MOSS_OUTPUT_EDGE_FADE_MS": "moss_output_edge_fade_ms",
    "MOSS_AUDIO_POLISH_ENABLED": "moss_audio_polish_enabled",
    "MOSS_TRIM_SILENCE_ENABLED": "moss_trim_silence_enabled",
    "MOSS_TRIM_SILENCE_DB": "moss_trim_silence_db",
    "MOSS_MAX_SILENCE_MS": "moss_max_silence_ms",
    "MOSS_CROSSFADE_MS": "moss_crossfade_ms",
    "MOSS_SEGMENT_PAUSE_MS": "moss_segment_pause_ms",
    "MOSS_RUNTIME_PAUSE_MAX_MS": "moss_runtime_pause_max_ms",
    "MOSS_VRAM_GUARD_ENABLED": "moss_vram_guard_enabled",
    "MOSS_VRAM_SAFE_FREE_MB": "moss_vram_safe_free_mb",
    "MOSS_VRAM_CRITICAL_FREE_MB": "moss_vram_critical_free_mb",
    "MOSS_VRAM_SNAPSHOT_TTL_SECONDS": "moss_vram_snapshot_ttl_seconds",
    "MOSS_LOW_VRAM_SEGMENT_LENGTH": "moss_low_vram_segment_length",
    "MOSS_LOW_VRAM_MAX_NEW_FRAMES": "moss_low_vram_max_new_frames",
    "MOSS_LOW_VRAM_TEXT_TOKENS": "moss_low_vram_text_tokens",
    "MOSS_DISABLE_FULL_CODEC_AFTER_OOM": "moss_disable_full_codec_after_oom",
    "MOSS_FULL_CODEC_OOM_COOLDOWN_SECONDS": "moss_full_codec_oom_cooldown_seconds",
    **{
        declaration.env_name: declaration.attr
        for declaration in MOSS_STREAM_BUDGET_ENV_DECLARATIONS
    },
    "KOKORO_RATE_LIMIT_QPS": "rate_limit_qps",
    "KOKORO_RATE_LIMIT_BURST": "rate_limit_burst",
    "KOKORO_MAX_QUEUE_LENGTH": "max_queue_length",
    "KOKORO_TRUST_PROXY_HEADERS": "trust_proxy_headers",
    "KOKORO_PUBLIC_STATUS_ENDPOINTS": "public_status_endpoints",
    "KOKORO_WS_MAX_CONNECTIONS": "websocket_max_connections",
    "KOKORO_WS_MAX_MESSAGE_BYTES": "websocket_max_message_bytes",
}


def _stringify_env_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _export_config_for_workers(cfg: TTSConfig) -> None:
    for env_name, attr in _WORKER_ENV_EXPORTS.items():
        os.environ[env_name] = _stringify_env_value(getattr(cfg, attr))
    os.environ["KOKORO_CORS_ORIGINS"] = ",".join(cfg.cors_origins)
    # API 密钥保留在持久凭证文件或运维人员提供的环境变量中；
    # 绝不将轮换/生成的密钥注入父 worker 环境。
    if cfg.moss_model_dir:
        os.environ["MOSS_MODEL_DIR"] = str(cfg.moss_model_dir)
    if cfg.moss_audio_tokenizer_model_dir:
        os.environ["MOSS_AUDIO_TOKENIZER_MODEL_DIR"] = str(cfg.moss_audio_tokenizer_model_dir)
    if cfg.moss_repo_path:
        os.environ["MOSS_TTS_NANO_PATH"] = str(cfg.moss_repo_path)
    if cfg.moss_prompt_audio_path:
        os.environ["MOSS_PROMPT_AUDIO_PATH"] = str(cfg.moss_prompt_audio_path)
    os.environ["ZIPVOICE_MODEL_ROOT"] = str(cfg.zipvoice_model_root)
    os.environ["ZIPVOICE_DISTILL_DIR"] = str(cfg.zipvoice_distill_dir)
    os.environ["ZIPVOICE_VOCOS_DIR"] = str(cfg.zipvoice_vocos_dir)
    os.environ["ZIPVOICE_PROFILES_DIR"] = str(cfg.zipvoice_profiles_dir)
    if cfg.zipvoice_repo_path:
        os.environ["ZIPVOICE_REPO_PATH"] = str(cfg.zipvoice_repo_path)


def create_app(config: Optional[TTSConfig] = None, engine: Optional[TTSEngine] = None):
    """创建 AngeVoice FastAPI 应用，重量级依赖按需加载。"""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    cfg = config or load_config()
    cfg.validate_security()
    manager = EngineManager(cfg, initial_engine=engine)
    try:
        state = ServiceState(cfg, engine, model_manager=manager)
    except Exception:
        # ServiceState may fail after the manager has started its timer or
        # accepted an initial engine. Preserve the original startup exception.
        try:
            manager.close_all()
        except Exception:
            logger.exception("应用状态初始化失败后的资源清理异常")
        raise
    verify_api_key = make_verify_api_key(cfg)

    @asynccontextmanager
    async def lifespan(app):
        try:
            # 选择 Studio 默认值不应强制将权重加载到 PID 1。
            # 可选的启动预加载始终遵循配置的适配器/worker 路径。
            state.model_manager.switch_model(cfg.default_model, load=False)
            if bool(getattr(cfg, "startup_preload_enabled", False)):
                preload_model = str(getattr(cfg, "startup_preload_model", "") or cfg.default_model)
                available_models = {item.id for item in state.model_manager.list_specs()}
                if preload_model in available_models:
                    state.model_manager.warm_model(preload_model)
                    logger.info("Startup preload completed (model=%s)", preload_model)
                else:
                    logger.warning("Startup preload skipped: model %s is not enabled", preload_model)
            current = state.model_manager.current_snapshot()
            logger.info("AngeVoice service started (selected_model=%s preload=%s)", current.get("id"), bool(getattr(cfg, "startup_preload_enabled", False)))
            yield
        finally:
            if not state.model_manager.close_all():
                logger.error("应用关闭屏障未能释放全部模型资源")

    app = FastAPI(
        title="AngeVoice",
        description="Lightweight local TTS service with selectable model engines",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.angevoice = state

    allow_credentials = "*" not in cfg.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if cfg.rate_limit_qps > 0:
        logger.info("Rate limiting enabled: %.1f QPS, burst=%d", cfg.rate_limit_qps, cfg.rate_limit_burst)
        app.add_middleware(
            RateLimitMiddleware,
            qps=cfg.rate_limit_qps,
            burst=cfg.rate_limit_burst,
            trust_proxy_headers=cfg.trust_proxy_headers,
        )
    if cfg.max_queue_length > 0:
        logger.info("Global queue limit enabled: %d concurrent requests", cfg.max_queue_length)
        app.add_middleware(GlobalQueueMiddleware, max_concurrent=cfg.max_queue_length)

    static_dir = Path(__file__).parent / "static"
    static_assets = None
    if static_dir.exists():
        static_assets = StaticAssetManifest(static_dir)
        app.state.static_assets = static_assets
        try:
            from fastapi.staticfiles import StaticFiles
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        except Exception:
            logger.debug("静态文件服务不可用", exc_info=True)

    templates = None
    if static_assets is not None:
        try:
            from fastapi.templating import Jinja2Templates
            templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
            templates.env.globals.update(
                asset_url=static_assets.url,
                asset_import_map_json=static_assets.import_map_json,
            )
        except Exception:
            logger.debug("Jinja2 templates are unavailable", exc_info=True)

    state.templates = templates
    app.include_router(create_auth_router(cfg))
    app.include_router(create_status_router(state, verify_api_key, templates=templates))
    app.include_router(create_admin_router(state))
    app.include_router(create_audio_router(state, verify_api_key))
    app.include_router(create_zipvoice_router(state, verify_api_key))
    app.include_router(create_ws_router(state))

    from .service_extras import register_extra_routes
    register_extra_routes(app=app, cfg=cfg, eng=state.eng, verify_api_key=verify_api_key, **state.as_service_extras_kwargs())
    return app


def run_server(config: Optional[TTSConfig] = None):
    import uvicorn

    cfg = config or load_config()
    print(format_startup_banner(cfg), flush=True)
    logger.info("Starting AngeVoice service: %s:%s", cfg.host, cfg.port)
    if cfg.workers > 1:
        _export_config_for_workers(cfg)
        uvicorn.run(
            "kokoro_tts.server:create_app",
            factory=True,
            host=cfg.host,
            port=cfg.port,
            workers=cfg.workers,
            ws_max_size=max(1024, int(cfg.websocket_max_message_bytes)),
            access_log=bool(cfg.access_log_enabled),
        )
        return

    app = create_app(config=cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        workers=1,
        ws_max_size=max(1024, int(cfg.websocket_max_message_bytes)),
        access_log=bool(cfg.access_log_enabled),
    )
