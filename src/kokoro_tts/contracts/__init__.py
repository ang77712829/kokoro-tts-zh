"""Stable internal contracts used by AngeVoice adapters and services."""

from .errors import EngineError, WorkerFailureEnvelope
from .runtime import RuntimeResourceStatus
from .streaming import CancellationContext, StreamingRequest, StreamingResult
from .synthesis import GenerationParameters, SynthesisRequest, SynthesisResult, VoiceCondition, VoiceConditionKind

__all__ = [
    "EngineError",
    "RuntimeResourceStatus",
    "CancellationContext",
    "GenerationParameters",
    "SynthesisResult",
    "StreamingRequest",
    "StreamingResult",
    "SynthesisRequest",
    "VoiceCondition",
    "VoiceConditionKind",
    "WorkerFailureEnvelope",
]
