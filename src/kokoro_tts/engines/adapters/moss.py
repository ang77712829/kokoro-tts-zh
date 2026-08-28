"""Product-level MOSS adapter preserving the current MOSS runtime."""

from __future__ import annotations

from typing import Any, Callable

from ...config import TTSConfig
from ...moss_engine import MossNanoEngine
from ..base import EngineCapabilities, ProviderStatus
from ..registry import EngineRegistry


class MossAdapter:
    """Expose one public ``moss`` model while delegating to ``MossNanoEngine``.

    CPU/CUDA are implementation providers rather than separate user-facing
    models. The underlying MOSS engine already owns CUDA self-test and fallback
    behaviour; this adapter only exposes that result consistently.
    """

    public_id = "moss"
    public_name = "MOSS-TTS-Nano"
    backend = "moss-tts-nano-onnx"

    def __init__(self, cfg: TTSConfig, *, requested_provider: str | None = None):
        provider = str(requested_provider or cfg.moss_execution_provider or "cpu").strip().lower()
        if provider == "cuda" and not bool(getattr(cfg, "moss_cuda_enabled", True)):
            provider = "cpu"
        self._cfg = cfg
        self._requested_provider = "cuda" if provider == "cuda" else "cpu"
        self._engine = MossNanoEngine(cfg, execution_provider=self._requested_provider, engine_id=self.public_id)

    @property
    def requested_provider(self) -> str:
        return self._requested_provider

    @property
    def is_loaded(self) -> bool:
        return bool(self._engine.is_loaded)

    @property
    def is_healthy(self) -> bool:
        return bool(getattr(self._engine, "is_healthy", True))

    def load(self):
        self._engine.load()
        return self

    def unload(self, *args, **kwargs) -> None:
        self._engine.unload(*args, **kwargs)

    def synthesize(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        **kwargs,
    ) -> bytes:
        return self._engine.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            prompt_audio_path=prompt_audio_path,
            **kwargs,
        )

    def synthesize_array(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        **kwargs,
    ):
        return self._engine.synthesize_array(
            text=text,
            voice=voice,
            speed=speed,
            prompt_audio_path=prompt_audio_path,
            **kwargs,
        )

    def synthesize_stream(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        fmt: str = "pcm_s16le",
        prompt_audio_path: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
        **kwargs,
    ):
        return self._engine.synthesize_stream(
            text=text,
            voice=voice,
            speed=speed,
            fmt=fmt,
            prompt_audio_path=prompt_audio_path,
            cancel_check=cancel_check,
            **kwargs,
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineRegistry.capabilities_for(self.public_id, self._cfg, provider=self._requested_provider)

    def _provider_status(self, metadata: dict[str, Any]) -> ProviderStatus:
        actual = str(metadata.get("actual_provider") or self._requested_provider).strip().lower()
        fallback = actual != self._requested_provider
        reason = ""
        if fallback:
            self_test = metadata.get("self_test")
            if isinstance(self_test, dict):
                reason = str(self_test.get("reason") or "")
            reason = reason or f"{self._requested_provider} unavailable; using {actual}"
        return ProviderStatus(self._requested_provider, actual, fallback, reason)

    def metadata(self) -> dict[str, Any]:
        value = self._engine.metadata() if callable(getattr(self._engine, "metadata", None)) else {}
        metadata = dict(value) if isinstance(value, dict) else {}
        metadata.update(self.capabilities().as_dict())
        metadata.update(self._provider_status(metadata).as_dict())
        metadata.update({"id": self.public_id, "name": self.public_name, "backend": self.backend})
        return metadata

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"private engine attribute is not exposed: {name}")
        return getattr(self._engine, name)
