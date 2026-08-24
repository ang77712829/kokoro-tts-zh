"""security admin configuration fields."""

from __future__ import annotations

from ...model_source_metadata import MODEL_SOURCE_METADATA
from ...rate_limit_config_metadata import RATE_LIMIT_CONFIG_BY_KEY
from ..fields import AdminConfigField, field_def


_RATE_LIMIT_QPS_METADATA = RATE_LIMIT_CONFIG_BY_KEY["rate_limit_qps"]
_RATE_LIMIT_BURST_METADATA = RATE_LIMIT_CONFIG_BY_KEY["rate_limit_burst"]
_MODEL_SOURCE_CHOICE_LABELS = {
    "auto": "auto 自动",
    "modelscope": "ModelScope",
    "huggingface": "Hugging Face",
    "offline": "offline 离线",
}


FIELDS: tuple[AdminConfigField, ...] = (
    field_def(
        "rate_limit_qps",
        "KOKORO_RATE_LIMIT_QPS",
        "限流 QPS",
        "security",
        "float",
        _RATE_LIMIT_QPS_METADATA.default,
        _RATE_LIMIT_QPS_METADATA.validation.admin_min_value,
        _RATE_LIMIT_QPS_METADATA.validation.admin_max_value,
        _RATE_LIMIT_QPS_METADATA.validation.admin_step,
        restart=True,
    ),
    field_def(
        "rate_limit_burst",
        "KOKORO_RATE_LIMIT_BURST",
        "限流突发",
        "security",
        "int",
        _RATE_LIMIT_BURST_METADATA.default,
        _RATE_LIMIT_BURST_METADATA.validation.admin_min_value,
        _RATE_LIMIT_BURST_METADATA.validation.admin_max_value,
        _RATE_LIMIT_BURST_METADATA.validation.admin_step,
        restart=True,
    ),
    field_def(
        "max_queue_length",
        "KOKORO_MAX_QUEUE_LENGTH",
        "队列上限",
        "security",
        "int",
        50,
        0,
        10000,
        1,
        restart=True,
    ),
    field_def(
        "websocket_max_connections",
        "KOKORO_WS_MAX_CONNECTIONS",
        "WebSocket 连接上限",
        "security",
        "int",
        16,
        0,
        10000,
        1,
        help="同时保持的 WebSocket 会话数量上限；0 表示禁用限制。",
        restart=True,
    ),
    field_def(
        "websocket_max_message_bytes",
        "KOKORO_WS_MAX_MESSAGE_BYTES",
        "WebSocket 单消息上限",
        "security",
        "int",
        33554432,
        1024,
        134217728,
        1024,
        help="前端以 MiB 显示和编辑；限制首包/控制消息大小。32 MiB 可容纳约 20 MiB 参考音频的 base64 JSON。",
        restart=True,
    ),
    field_def(
        "trust_proxy_headers",
        "KOKORO_TRUST_PROXY_HEADERS",
        "信任反代 IP",
        "security",
        "bool",
        False,
        restart=True,
    ),
    field_def(
        "public_status_endpoints",
        "KOKORO_PUBLIC_STATUS_ENDPOINTS",
        "公开模型列表",
        "security",
        "bool",
        True,
    ),
    field_def(
        MODEL_SOURCE_METADATA.key,
        MODEL_SOURCE_METADATA.canonical_env,
        "模型下载源",
        MODEL_SOURCE_METADATA.admin_group,
        "choice",
        MODEL_SOURCE_METADATA.default,
        choices=tuple(
            (value, _MODEL_SOURCE_CHOICE_LABELS[value])
            for value in MODEL_SOURCE_METADATA.admin_choices
        ),
        restart=MODEL_SOURCE_METADATA.admin_restart,
        rebuild_moss=MODEL_SOURCE_METADATA.admin_rebuild_moss,
    ),
)
