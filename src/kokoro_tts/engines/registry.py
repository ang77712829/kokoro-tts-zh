"""Product-level engine registry and backwards-compatible model resolution."""

from __future__ import annotations

import logging
from typing import Iterable

from fastapi import HTTPException

from ..config import TTSConfig
from ..config_ids import MOSS_CPU_MODEL_IDS, MOSS_CUDA_MODEL_IDS, MOSS_GENERIC_MODEL_IDS
from .base import EngineCapabilities, EngineSpec, ModelResolution
from .parameters import EngineParameterSchema
from .provider_policy import ProviderPolicy

logger = logging.getLogger(__name__)

_KOKORO_ALIASES = {"kokoro", "kokoro-zh", "kokoro-v1.1", "kokoro-v1_1-zh", "tts-1", "tts-1-hd"}


class EngineRegistry:
    """Resolve public model IDs independently from implementation providers."""

    public_model_ids = ("kokoro", "moss", "zipvoice")

    def __init__(self):
        self.parameter_schema = EngineParameterSchema()
        self.provider_policy = ProviderPolicy(self.resolve)

    def resolve(self, model_id: str | None, *, default_id: str = "kokoro") -> ModelResolution:
        raw = str(model_id or default_id or "kokoro").strip().lower()
        if raw in {"", "default", "current"}:
            raw = str(default_id or "kokoro").strip().lower()
        if raw in _KOKORO_ALIASES:
            return ModelResolution(raw, "kokoro", None, raw != "kokoro")
        if raw in MOSS_GENERIC_MODEL_IDS:
            return ModelResolution(raw, "moss", None, raw != "moss")
        if raw in MOSS_CPU_MODEL_IDS:
            return ModelResolution(raw, "moss", "cpu", True)
        if raw in MOSS_CUDA_MODEL_IDS:
            return ModelResolution(raw, "moss", "cuda", True)
        if raw in {"zipvoice", "zipvoice-distill", "zipvoice_distill"}:
            return ModelResolution(raw, "zipvoice", None, raw != "zipvoice")
        return ModelResolution(raw, raw, None, False)

    def _configured_moss_provider(self, cfg: TTSConfig, enabled_models: Iterable[str]) -> str:
        return self.provider_policy.requested_provider("moss", cfg, enabled_models)

    def parameter_schema_for(self, model_id: str) -> list[dict]:
        return self.parameter_schema.schema_for(model_id)

    def list_specs(self, cfg: TTSConfig) -> list[EngineSpec]:
        specs: list[EngineSpec] = []
        seen: set[str] = set()
        moss_provider = self._configured_moss_provider(cfg, cfg.enabled_models)
        zipvoice_provider = self.provider_policy.requested_provider("zipvoice", cfg, cfg.enabled_models)
        for configured_id in cfg.enabled_models:
            resolved = self.resolve(configured_id, default_id=cfg.default_model)
            model_id = resolved.canonical_id
            if model_id in seen:
                continue
            if model_id == "kokoro":
                specs.append(EngineSpec("kokoro", "Kokoro v1.1 Chinese", "kokoro", cfg.device))
                seen.add(model_id)
                continue
            if model_id == "moss":
                if resolved.provider_hint == "cuda" and not bool(getattr(cfg, "moss_cuda_enabled", True)):
                    logger.info("Ignoring disabled MOSS CUDA compatibility alias: %s", configured_id)
                    continue
                specs.append(EngineSpec("moss", "MOSS-TTS-Nano", "moss-tts-nano-onnx", moss_provider, experimental=False))
                seen.add(model_id)
                continue
            if model_id == "zipvoice":
                backend = "zipvoice-distill-pytorch-cuda" if zipvoice_provider == "cuda" else "zipvoice-distill-onnx-int8"
                specs.append(EngineSpec("zipvoice", "ZipVoice", backend, zipvoice_provider, experimental=False))
                seen.add(model_id)
                continue
            logger.warning("忽略未知 AngeVoice 模型 ID：%s", configured_id)
        if not specs:
            specs.append(EngineSpec("kokoro", "Kokoro v1.1 Chinese", "kokoro", cfg.device))
        return specs

    def create_engine(self, model_id: str, cfg: TTSConfig, *, provider_hint: str | None = None, voice_profile_store=None):
        if model_id == "kokoro":
            from .adapters.kokoro import KokoroAdapter

            return KokoroAdapter(cfg)
        if model_id == "moss":
            from .adapters.moss import MossAdapter

            provider = provider_hint or self._configured_moss_provider(cfg, cfg.enabled_models)
            if provider == "cuda" and not bool(getattr(cfg, "moss_cuda_enabled", True)):
                raise HTTPException(status_code=404, detail="MOSS CUDA provider is disabled")
            return MossAdapter(cfg, requested_provider=provider)
        if model_id == "zipvoice":
            # 运行时导入故意延迟：注册表元数据必须保持轻量，
            # 直接导入 zipvoice.engine 不能通过 engines -> registry -> adapters -> zipvoice.engine 递归。
            from ..zipvoice.engine import ZipVoiceEngine

            provider = provider_hint or self.provider_policy.requested_provider("zipvoice", cfg, cfg.enabled_models)
            return ZipVoiceEngine(cfg, profile_store=voice_profile_store, requested_provider=provider)
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")

    @staticmethod
    def capabilities_for(
        spec: EngineSpec | str,
        cfg: TTSConfig,
        *,
        provider: str | None = None,
    ) -> EngineCapabilities:
        # 不能仅为渲染目录就实例化运行时。MOSS 在构造时创建 worker/executor 状态，
        # 每次 /health 或 /v1/models 请求都会泄漏。
        model_id = spec.id if isinstance(spec, EngineSpec) else str(spec or "")
        backend = spec.backend if isinstance(spec, EngineSpec) else ""
        requested_provider = spec.provider if isinstance(spec, EngineSpec) else str(provider or "")
        if model_id == "kokoro":
            return EngineCapabilities(
                modes=("preset_voice",),
                voice_clone_supported=False,
                speed_supported=True,
                text_rules_enabled=True,
                stream_mode="segmented",
                sample_rate=int(getattr(cfg, "sample_rate", 24000)),
                channels=1,
            )
        if model_id == "moss" or backend == "moss-tts-nano-onnx":
            text_rules_mode = str(getattr(cfg, "moss_apply_angevoice_rules", "auto")).strip().lower()
            return EngineCapabilities(
                modes=("preset_voice", "voice_clone"),
                voice_clone_supported=True,
                speed_supported=False,
                text_rules_enabled=text_rules_mode != "false",
                requires_prompt_audio=False,
                requires_prompt_text=False,
                stream_mode="native",
                provider_fallback=True,
                sample_rate=48000,
                channels=2,
            )
        if model_id == "zipvoice" or backend in {"zipvoice-distill-onnx-int8", "zipvoice-distill-pytorch-cuda"}:
            return EngineCapabilities(
                modes=("voice_clone", "saved_voice_profile"),
                voice_clone_supported=True,
                speed_supported=True,
                text_rules_enabled=True,
                requires_prompt_audio=True,
                requires_prompt_text=True,
                supports_saved_voice_profiles=True,
                stream_mode="segmented",
                provider_fallback=requested_provider == "cuda",
                sample_rate=24000,
                channels=1,
            )
        return EngineCapabilities(modes=("preset_voice",), voice_clone_supported=False, speed_supported=False)
