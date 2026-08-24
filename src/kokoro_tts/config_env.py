"""AngeVoice 运行时环境变量解析。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

from .config_api_key import AUTO_API_KEY_SENTINELS, load_or_generate_api_key
from .config_env_domain import (
    BATCH_INT_DECLARATIONS,
    CACHE_INT_DECLARATIONS,
    MOSS_STREAM_BUDGET_ENV_DECLARATIONS,
    UPDATE_CHECK_ENV_DECLARATIONS,
    parse_int_env,
)
from .model_source_metadata import MODEL_SOURCE_METADATA
from .rate_limit_config_metadata import RATE_LIMIT_CONFIG_BY_KEY

logger = logging.getLogger(__name__)

_RATE_LIMIT_QPS_METADATA = RATE_LIMIT_CONFIG_BY_KEY["rate_limit_qps"]
_RATE_LIMIT_BURST_METADATA = RATE_LIMIT_CONFIG_BY_KEY["rate_limit_burst"]


class IntEnvSpec(NamedTuple):
    attr: str
    min_value: int | None = None
    max_value: int | None = None


class FloatEnvSpec(NamedTuple):
    attr: str
    min_value: float | None = None
    max_value: float | None = None


def get_env_int(name: str, default: int) -> int:
    return parse_int_env(name, default, warning_sink=logger.warning)


def get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("忽略无效浮点环境变量 %s=%r", name, value)
        return default


def get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def clamp(value, min_value=None, max_value=None):
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


STR_ENV: dict[str, str] = {
    "KOKORO_HOST": "host",
    "KOKORO_DEVICE": "device",
    "ANGEVOICE_DEPLOYMENT_PROFILE": "deployment_profile",
    "KOKORO_DEFAULT_VOICE": "default_voice",
    "KOKORO_STREAM_FORMAT": "stream_format",
    "KOKORO_MP3_BITRATE": "mp3_bitrate",
    "ANGEVOICE_AUDIO_MP3_BITRATE": "mp3_bitrate",
    "ANGEVOICE_AUDIO_OPUS_BITRATE": "audio_opus_bitrate",
    "ANGEVOICE_AUDIO_AAC_BITRATE": "audio_aac_bitrate",
    "ANGEVOICE_FFMPEG_BINARY": "ffmpeg_binary",
    "ANGEVOICE_DEFAULT_MODEL": "default_model",
    "ANGEVOICE_STARTUP_PRELOAD_MODEL": "startup_preload_model",
    "ANGEVOICE_OUTPUT_DIR": "output_dir",
    "ANGEVOICE_RUNTIME_CONFIG_FILE": "runtime_config_file",
    **{
        declaration.env_name: declaration.attr
        for declaration in UPDATE_CHECK_ENV_DECLARATIONS
        if declaration.family == "str"
    },
    "ANGEVOICE_CREDENTIALS_DIR": "credentials_dir",
    "ANGEVOICE_ADMIN_CREDENTIALS_FILE": "admin_credentials_file",
    MODEL_SOURCE_METADATA.canonical_env: MODEL_SOURCE_METADATA.key,
    "ANGEVOICE_MODEL_SOURCE_DETECT_URL": "model_source_detect_url",
    "ANGEVOICE_MODEL_SOURCE_PROBE_HF_URL": "model_source_probe_hf_url",
    "ANGEVOICE_MODEL_SOURCE_PROBE_MODELSCOPE_URL": "model_source_probe_modelscope_url",
    "ANGEVOICE_API_KEY_FILE": "api_key_file",
    "ANGEVOICE_SINGLE_NEWLINE_POLICY": "text_single_newline_policy",
    "ANGEVOICE_TN_ENGINE": "angevoice_tn_engine",
    "KOKORO_HF_REPO": "kokoro_hf_repo",
    "KOKORO_MODELSCOPE_REPO": "kokoro_modelscope_repo",
    "MOSS_MODELSCOPE_REPO": "moss_modelscope_repo",
    "MOSS_HF_REPO": "moss_hf_repo",
    "MOSS_AUDIO_TOKENIZER_MODELSCOPE_REPO": "moss_audio_tokenizer_modelscope_repo",
    "MOSS_AUDIO_TOKENIZER_HF_REPO": "moss_audio_tokenizer_hf_repo",
    "MOSS_EXECUTION_PROVIDER": "moss_execution_provider",
    "ZIPVOICE_EXECUTION_PROVIDER": "zipvoice_execution_provider",
    "MOSS_DEFAULT_VOICE": "moss_default_voice",
    "MOSS_SAMPLE_MODE": "moss_sample_mode",
    "MOSS_PROCESS_ISOLATION_PROVIDERS": "moss_process_isolation_providers",
    "MOSS_APPLY_ANGEVOICE_RULES": "moss_apply_angevoice_rules",
    "MOSS_MIXED_ENGLISH_POLICY": "moss_mixed_english_policy",
}

INT_ENV: dict[str, IntEnvSpec] = {
    "KOKORO_PORT": IntEnvSpec("port", 1),
    "KOKORO_WORKERS": IntEnvSpec("workers", 1),
    "KOKORO_MAX_CONCURRENT_REQUESTS": IntEnvSpec("max_concurrent_requests", 1),
    "KOKORO_MAX_TEXT_LENGTH": IntEnvSpec("max_text_length", 1),
    "KOKORO_SEGMENT_LENGTH": IntEnvSpec("segment_length", 20),
    "MOSS_SEGMENT_LENGTH": IntEnvSpec("moss_segment_length", 20),
    **{
        declaration.env_name: IntEnvSpec(
            declaration.attr,
            declaration.min_value,
            declaration.max_value,
        )
        for declaration in CACHE_INT_DECLARATIONS
    },
    **{
        declaration.env_name: IntEnvSpec(
            declaration.attr,
            declaration.min_value,
            declaration.max_value,
        )
        for declaration in BATCH_INT_DECLARATIONS
    },
    "KOKORO_VOICE_UPLOAD_MAX_BYTES": IntEnvSpec("voice_upload_max_bytes", 1),
    "ANGEVOICE_OUTPUT_MAX_FILES": IntEnvSpec("output_max_files", 0),
    "MOSS_CPU_THREADS": IntEnvSpec("moss_cpu_threads", 1),
    "ZIPVOICE_CPU_THREADS": IntEnvSpec("zipvoice_cpu_threads", 1),
    "ZIPVOICE_CUDA_DEVICE_INDEX": IntEnvSpec("zipvoice_cuda_device_index", 0),
    "ZIPVOICE_NUM_STEPS": IntEnvSpec("zipvoice_num_steps", 1, 32),
    "ZIPVOICE_PROMPT_UPLOAD_MAX_BYTES": IntEnvSpec("zipvoice_prompt_upload_max_bytes", 1),
    "MOSS_PROMPT_UPLOAD_MAX_BYTES": IntEnvSpec("moss_prompt_upload_max_bytes", 1),
    "KOKORO_TTS_REQUEST_MAX_BYTES": IntEnvSpec("tts_request_max_bytes", 1024),
    "MOSS_PROMPT_CACHE_MAX_ITEMS": IntEnvSpec("moss_prompt_cache_max_items", 0),
    "MOSS_MAX_NEW_FRAMES": IntEnvSpec("moss_max_new_frames", 1),
    "MOSS_VOICE_CLONE_MAX_TEXT_TOKENS": IntEnvSpec("moss_voice_clone_max_text_tokens", 1),
    "MOSS_SEED": IntEnvSpec("moss_seed", -1),
    "MOSS_CUDA_MEMORY_LIMIT_MB": IntEnvSpec("moss_cuda_memory_limit_mb", 0),
    "MOSS_STREAM_QUEUE_MAX_ITEMS": IntEnvSpec("moss_stream_queue_max_items", 1, 64),
    "MOSS_VRAM_SAFE_FREE_MB": IntEnvSpec("moss_vram_safe_free_mb", 0),
    "MOSS_VRAM_CRITICAL_FREE_MB": IntEnvSpec("moss_vram_critical_free_mb", 0),
    "MOSS_LOW_VRAM_SEGMENT_LENGTH": IntEnvSpec("moss_low_vram_segment_length", 20),
    "MOSS_LOW_VRAM_MAX_NEW_FRAMES": IntEnvSpec("moss_low_vram_max_new_frames", 1),
    "MOSS_LOW_VRAM_TEXT_TOKENS": IntEnvSpec("moss_low_vram_text_tokens", 1),
    "KOKORO_RATE_LIMIT_BURST": IntEnvSpec(
        _RATE_LIMIT_BURST_METADATA.key,
        _RATE_LIMIT_BURST_METADATA.validation.env_min_value,
        _RATE_LIMIT_BURST_METADATA.validation.env_max_value,
    ),
    "KOKORO_MAX_QUEUE_LENGTH": IntEnvSpec("max_queue_length", 0),
    "KOKORO_WS_MAX_CONNECTIONS": IntEnvSpec("websocket_max_connections", 0),
    "KOKORO_WS_MAX_MESSAGE_BYTES": IntEnvSpec("websocket_max_message_bytes", 1024),
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_EXIT_CODE": IntEnvSpec("restart_after_idle_unload_exit_code", 0, 255),
}

FLOAT_ENV: dict[str, FloatEnvSpec] = {
    "KOKORO_DEFAULT_SPEED": FloatEnvSpec("default_speed"),
    "KOKORO_REQUEST_TIMEOUT_SECONDS": FloatEnvSpec("request_timeout_seconds", 1.0),
    "ANGEVOICE_WEBSOCKET_STREAM_IDLE_TIMEOUT_SECONDS": FloatEnvSpec("websocket_stream_idle_timeout_seconds", 5.0, 3600.0),
    "KOKORO_STREAM_CHUNK_SECONDS": FloatEnvSpec("stream_chunk_seconds", 0.05, 2.0),
    "KOKORO_STREAM_PREBUFFER_SECONDS": FloatEnvSpec("stream_prebuffer_seconds", 0.0, 3.0),
    "ANGEVOICE_MODEL_SWITCH_TIMEOUT_SECONDS": FloatEnvSpec("model_switch_timeout_seconds", 1.0),
    "MOSS_PROMPT_AUDIO_MAX_SECONDS": FloatEnvSpec("moss_prompt_audio_max_seconds", 0.0),
    "ZIPVOICE_PROMPT_AUDIO_MAX_SECONDS": FloatEnvSpec("zipvoice_prompt_audio_max_seconds", 0.1, 30.0),
    "ZIPVOICE_GUIDANCE_SCALE": FloatEnvSpec("zipvoice_guidance_scale", 0.0, 20.0),
    "ZIPVOICE_T_SHIFT": FloatEnvSpec("zipvoice_t_shift", 0.01, 1.0),
    "ZIPVOICE_TARGET_RMS": FloatEnvSpec("zipvoice_target_rms", 0.0, 1.0),
    "ZIPVOICE_FEAT_SCALE": FloatEnvSpec("zipvoice_feat_scale", 0.001, 10.0),
    "ZIPVOICE_CUDA_MAX_DURATION": FloatEnvSpec("zipvoice_cuda_max_duration", 1.0, 120.0),
    "MOSS_STREAM_CHUNK_SECONDS": FloatEnvSpec("moss_stream_chunk_seconds", 0.05, 2.0),
    "MOSS_STREAM_PREBUFFER_SECONDS": FloatEnvSpec("moss_stream_prebuffer_seconds", 0.0, 12.0),
    "MOSS_MAX_CLIP_RATIO": FloatEnvSpec("moss_max_clip_ratio", 0.0, 1.0),
    "MOSS_OUTPUT_TARGET_PEAK": FloatEnvSpec("moss_output_target_peak", 0.1, 1.0),
    "MOSS_OUTPUT_GAIN": FloatEnvSpec("moss_output_gain", 0.1, 2.0),
    "KOKORO_RATE_LIMIT_QPS": FloatEnvSpec(
        _RATE_LIMIT_QPS_METADATA.key,
        _RATE_LIMIT_QPS_METADATA.validation.env_min_value,
        _RATE_LIMIT_QPS_METADATA.validation.env_max_value,
    ),
    **{
        declaration.env_name: FloatEnvSpec(
            declaration.attr,
            declaration.min_value,
            declaration.max_value,
        )
        for declaration in MOSS_STREAM_BUDGET_ENV_DECLARATIONS
    },
    "MOSS_PROCESS_KILL_GRACE_SECONDS": FloatEnvSpec("moss_process_kill_grace_seconds", 0.1, 30.0),
    "ANGEVOICE_ENGINE_PROCESS_KILL_GRACE_SECONDS": FloatEnvSpec("engine_process_kill_grace_seconds", 0.1, 30.0),
    "ANGEVOICE_ENGINE_PROCESS_STREAM_DRAIN_SECONDS": FloatEnvSpec("engine_process_stream_drain_seconds", 0.1, 60.0),
    "ANGEVOICE_ENGINE_PROCESS_STREAM_IDLE_TIMEOUT_SECONDS": FloatEnvSpec("engine_process_stream_idle_timeout_seconds", 5.0, 3600.0),
    "ANGEVOICE_FFMPEG_TIMEOUT_SECONDS": FloatEnvSpec("ffmpeg_timeout_seconds", 1.0, 3600.0),
    "MOSS_OUTPUT_EDGE_FADE_MS": FloatEnvSpec("moss_output_edge_fade_ms", 0.0, 20.0),
    "MOSS_TRIM_SILENCE_DB": FloatEnvSpec("moss_trim_silence_db", -90.0, -10.0),
    "MOSS_MAX_SILENCE_MS": FloatEnvSpec("moss_max_silence_ms", 0.0, 5000.0),
    "MOSS_CROSSFADE_MS": FloatEnvSpec("moss_crossfade_ms", 0.0, 120.0),
    "MOSS_SEGMENT_PAUSE_MS": FloatEnvSpec("moss_segment_pause_ms", 0.0, 2000.0),
    "MOSS_RUNTIME_PAUSE_MAX_MS": FloatEnvSpec("moss_runtime_pause_max_ms", 0.0, 3000.0),
    "MOSS_FULL_CODEC_OOM_COOLDOWN_SECONDS": FloatEnvSpec("moss_full_codec_oom_cooldown_seconds", 0.0, 86400.0),
    "MOSS_VRAM_SNAPSHOT_TTL_SECONDS": FloatEnvSpec("moss_vram_snapshot_ttl_seconds", 0.0, 3600.0),
    "ANGEVOICE_MODEL_SOURCE_DETECT_TIMEOUT_SECONDS": FloatEnvSpec("model_source_detect_timeout_seconds", 0.1, 10.0),
    "ANGEVOICE_MODEL_SOURCE_PROBE_TIMEOUT_SECONDS": FloatEnvSpec("model_source_probe_timeout_seconds", 0.1, 10.0),
    **{
        declaration.env_name: FloatEnvSpec(
            declaration.attr,
            declaration.min_value,
            declaration.max_value,
        )
        for declaration in UPDATE_CHECK_ENV_DECLARATIONS
        if declaration.family == "float"
    },
    "ANGEVOICE_IDLE_TIMEOUT_SECONDS": FloatEnvSpec("model_idle_timeout_seconds", 0.0),
    "ANGEVOICE_IDLE_CHECK_INTERVAL": FloatEnvSpec("model_idle_check_interval", 5.0),
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_DELAY_SECONDS": FloatEnvSpec("restart_after_idle_unload_delay_seconds", 0.0, 3600.0),
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD_COOLDOWN_SECONDS": FloatEnvSpec("restart_after_idle_unload_cooldown_seconds", 0.0, 86400.0),
}

BOOL_ENV: dict[str, str] = {
    "KOKORO_STREAM_BINARY_ENABLED": "stream_binary_enabled",
    "ANGEVOICE_ACCESS_LOG_ENABLED": "access_log_enabled",
    "KOKORO_CACHE_ENABLED": "cache_enabled",
    "KOKORO_QUEUE_STATUS_ENABLED": "queue_status_enabled",
    "KOKORO_METRICS_ENABLED": "metrics_enabled",
    "KOKORO_BATCH_ENABLED": "batch_enabled",
    "KOKORO_ADMIN_ENABLED": "admin_enabled",
    "KOKORO_VOICE_UPLOAD_ENABLED": "voice_upload_enabled",
    "KOKORO_MP3_ENABLED": "mp3_enabled",
    "ANGEVOICE_FFMPEG_ENABLED": "ffmpeg_enabled",
    "ANGEVOICE_MODEL_SWITCH_ENABLED": "model_switch_enabled",
    "ANGEVOICE_MODEL_UNLOAD_ON_SWITCH": "model_unload_on_switch",
    "ANGEVOICE_SAVE_OUTPUTS": "save_outputs",
    "ANGEVOICE_IDLE_UNLOAD_CURRENT": "model_idle_unload_current",
    "KOKORO_PREFETCH_VOICES": "kokoro_prefetch_voices",
    "MOSS_CUDA_ENABLED": "moss_cuda_enabled",
    "MOSS_ENABLE_WETEXT_PROCESSING": "moss_enable_wetext_processing",
    "MOSS_ENABLE_NORMALIZE_TTS_TEXT": "moss_enable_normalize_tts_text",
    "MOSS_REALTIME_STREAMING_DECODE": "moss_realtime_streaming_decode",
    "MOSS_CUDA_SELF_TEST_ENABLED": "moss_cuda_self_test_enabled",
    "MOSS_AUTO_FALLBACK_CPU": "moss_auto_fallback_cpu",
    "MOSS_QUALITY_GATE_ENABLED": "moss_quality_gate_enabled",
    "MOSS_OUTPUT_PEAK_NORMALIZE_ENABLED": "moss_output_peak_normalize_enabled",
    "MOSS_PROCESS_ISOLATION_ENABLED": "moss_process_isolation_enabled",
    "KOKORO_PROCESS_ISOLATION_ENABLED": "kokoro_process_isolation_enabled",
    "ZIPVOICE_PROCESS_ISOLATION_ENABLED": "zipvoice_process_isolation_enabled",
    "ANGEVOICE_STARTUP_PRELOAD_ENABLED": "startup_preload_enabled",
    "MOSS_OUTPUT_DECLICK_ENABLED": "moss_output_declick_enabled",
    "MOSS_AUDIO_POLISH_ENABLED": "moss_audio_polish_enabled",
    "MOSS_TRIM_SILENCE_ENABLED": "moss_trim_silence_enabled",
    "MOSS_VRAM_GUARD_ENABLED": "moss_vram_guard_enabled",
    "MOSS_DISABLE_FULL_CODEC_AFTER_OOM": "moss_disable_full_codec_after_oom",
    "ZIPVOICE_DOWNLOAD_ENABLED": "zipvoice_download_enabled",
    "ZIPVOICE_REMOVE_LONG_SIL": "zipvoice_remove_long_sil",
    "ZIPVOICE_CUDA_ENABLED": "zipvoice_cuda_enabled",
    "ZIPVOICE_AUTO_FALLBACK_CPU": "zipvoice_auto_fallback_cpu",
    "KOKORO_TRUST_PROXY_HEADERS": "trust_proxy_headers",
    "KOKORO_PUBLIC_STATUS_ENDPOINTS": "public_status_endpoints",
    **{
        declaration.env_name: declaration.attr
        for declaration in UPDATE_CHECK_ENV_DECLARATIONS
        if declaration.family == "bool"
    },
    "ANGEVOICE_RESTART_AFTER_IDLE_UNLOAD": "restart_after_idle_unload_enabled",
}


def apply_env(config) -> None:
    """把环境变量覆盖应用到 TTSConfig 风格对象。"""
    for env_name, attr in STR_ENV.items():
        if os.environ.get(env_name) is not None:
            setattr(config, attr, os.environ[env_name])
    if isinstance(config.output_dir, str):
        config.output_dir = Path(config.output_dir).expanduser()
    if isinstance(getattr(config, "credentials_dir", None), str):
        config.credentials_dir = Path(config.credentials_dir).expanduser()
    if isinstance(config.api_key_file, str):
        config.api_key_file = Path(config.api_key_file).expanduser()
    if isinstance(getattr(config, "admin_credentials_file", None), str):
        config.admin_credentials_file = Path(config.admin_credentials_file).expanduser()
    if isinstance(getattr(config, "runtime_config_file", None), str):
        config.runtime_config_file = Path(config.runtime_config_file).expanduser()

    for env_name, spec in INT_ENV.items():
        if os.environ.get(env_name) is not None:
            value = get_env_int(env_name, getattr(config, spec.attr))
            setattr(config, spec.attr, clamp(value, spec.min_value, spec.max_value))

    for env_name, spec in FLOAT_ENV.items():
        if os.environ.get(env_name) is not None:
            value = get_env_float(env_name, getattr(config, spec.attr))
            setattr(config, spec.attr, clamp(value, spec.min_value, spec.max_value))

    for env_name, attr in BOOL_ENV.items():
        if os.environ.get(env_name) is not None:
            setattr(config, attr, get_env_bool(env_name, getattr(config, attr)))

    if os.environ.get("KOKORO_API_KEY") is not None:
        raw_api_key = os.environ.get("KOKORO_API_KEY", "").strip()
        if raw_api_key.lower() in AUTO_API_KEY_SENTINELS:
            config.api_key = load_or_generate_api_key(config)
            config.api_key_auto_generated = True
        else:
            config.api_key = raw_api_key or None
    elif get_env_bool("KOKORO_AUTO_API_KEY", False):
        config.api_key = load_or_generate_api_key(config)
        config.api_key_auto_generated = True
    if os.environ.get("KOKORO_CORS_ORIGINS"):
        config.cors_origins = [o.strip() for o in os.environ["KOKORO_CORS_ORIGINS"].split(",") if o.strip()]
    if os.environ.get("ANGEVOICE_ENABLED_MODELS"):
        config.enabled_models = [item.strip().lower() for item in os.environ["ANGEVOICE_ENABLED_MODELS"].split(",") if item.strip()]
    if os.environ.get("MOSS_MODEL_DIR"):
        config.moss_model_dir = Path(os.environ["MOSS_MODEL_DIR"]).expanduser()
    if os.environ.get("MOSS_AUDIO_TOKENIZER_MODEL_DIR"):
        config.moss_audio_tokenizer_model_dir = Path(os.environ["MOSS_AUDIO_TOKENIZER_MODEL_DIR"]).expanduser()
    if os.environ.get("MOSS_TTS_NANO_PATH"):
        config.moss_repo_path = Path(os.environ["MOSS_TTS_NANO_PATH"]).expanduser()
    if os.environ.get("MOSS_PROMPT_AUDIO_PATH"):
        config.moss_prompt_audio_path = Path(os.environ["MOSS_PROMPT_AUDIO_PATH"]).expanduser()
    if os.environ.get("ZIPVOICE_MODEL_ROOT"):
        config.zipvoice_model_root = Path(os.environ["ZIPVOICE_MODEL_ROOT"]).expanduser()
        config.zipvoice_distill_dir = config.zipvoice_model_root / "zipvoice_distill"
        config.zipvoice_vocos_dir = config.zipvoice_model_root / "vocos-mel-24khz"
    if os.environ.get("ZIPVOICE_DISTILL_DIR"):
        config.zipvoice_distill_dir = Path(os.environ["ZIPVOICE_DISTILL_DIR"]).expanduser()
    if os.environ.get("ZIPVOICE_VOCOS_DIR"):
        config.zipvoice_vocos_dir = Path(os.environ["ZIPVOICE_VOCOS_DIR"]).expanduser()
    if os.environ.get("ZIPVOICE_PROFILES_DIR"):
        config.zipvoice_profiles_dir = Path(os.environ["ZIPVOICE_PROFILES_DIR"]).expanduser()
    if os.environ.get("ZIPVOICE_REPO_PATH"):
        config.zipvoice_repo_path = Path(os.environ["ZIPVOICE_REPO_PATH"]).expanduser()
