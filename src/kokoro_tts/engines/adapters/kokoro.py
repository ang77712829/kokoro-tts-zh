"""Product-level Kokoro adapter with optional killable process isolation."""

from __future__ import annotations

from typing import Any, Callable

from ...config import TTSConfig
from ...engine import TTSEngine
from ...workers import EngineProcessClient, EngineWorkerSpec
from ..base import EngineCapabilities
from ..registry import EngineRegistry


def _create_kokoro_worker_engine(config: object, _requested_provider: str | None) -> TTSEngine:
    """Build the concrete Kokoro runtime inside an isolated worker."""

    return TTSEngine(config)


class KokoroAdapter:
    """Expose Kokoro through an isolated worker in formal deployments.

    Library callers may retain the in-process path for compatibility. Docker and
    fnOS templates enable the worker by default so idle release actually returns
    model RSS/VRAM to the host.
    """

    public_id = "kokoro"
    public_name = "Kokoro v1.1 Chinese"
    backend = "kokoro"

    def __init__(self, cfg: TTSConfig, engine: TTSEngine | None = None):
        self._cfg = cfg
        self._process_isolated = bool(getattr(cfg, "kokoro_process_isolation_enabled", False)) and engine is None
        self._worker = (
            EngineProcessClient(
                config=cfg,
                spec=EngineWorkerSpec(
                    engine_id="kokoro",
                    factory=_create_kokoro_worker_engine,
                ),
            )
            if self._process_isolated
            else None
        )
        self._engine = None if self._process_isolated else (engine or TTSEngine(cfg))

    @property
    def is_loaded(self) -> bool:
        return bool(self._worker.is_loaded if self._worker is not None else self._engine.is_loaded)

    @property
    def is_healthy(self) -> bool:
        if self._worker is not None:
            return self._worker.is_healthy
        return bool(getattr(self._engine, "is_healthy", True))

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._cfg, "sample_rate", 24000))

    @property
    def channels(self) -> int:
        return 1

    @property
    def default_voice(self) -> str:
        return str(getattr(self._cfg, "default_voice", "zm_010"))

    def get_voices(self) -> list[str]:
        return self._cfg.get_voices()

    def load(self):
        if self._worker is not None:
            self._worker.load(timeout=float(getattr(self._cfg, "model_switch_timeout_seconds", 300.0)))
        else:
            self._engine.load()
        return self

    def unload(self, *args, **kwargs) -> None:
        if self._worker is not None:
            self._worker.close(kill=bool(kwargs.get("force", False)))
        else:
            self._engine.unload()

    def soft_cancel(self) -> None:
        if self._worker is not None:
            self._worker.soft_cancel()
            return
        soft_cancel = getattr(self._engine, "soft_cancel", None)
        if callable(soft_cancel):
            soft_cancel()

    def synthesize(self, text: str, voice: str = "zm_010", speed: float = 1.0) -> bytes:
        if self._worker is None:
            return self._engine.synthesize(text, voice, speed)
        if not self.is_loaded:
            self.load()
        return self._worker.request(
            "synthesize", {"text": text, "voice": voice, "speed": speed},
            timeout=float(getattr(self._cfg, "request_timeout_seconds", 300.0)),
        )

    def synthesize_array(self, text: str, voice: str = "zm_010", speed: float = 1.0):
        if self._worker is None:
            return self._engine.synthesize_array(text, voice, speed)
        if not self.is_loaded:
            self.load()
        return self._worker.request(
            "synthesize_array", {"text": text, "voice": voice, "speed": speed},
            timeout=float(getattr(self._cfg, "request_timeout_seconds", 300.0)),
        )

    def synthesize_stream(
        self, text: str, voice: str = "zm_010", speed: float = 1.0,
        fmt: str = "pcm_s16le", *, cancel_check: Callable[[], bool] | None = None,
    ):
        if self._worker is None:
            yield from self._engine.synthesize_stream(text, voice, speed, fmt, cancel_check=cancel_check)
            return
        if not self.is_loaded:
            self.load()
        yield from self._worker.stream(
            {"text": text, "voice": voice, "speed": speed, "fmt": fmt},
            timeout=float(getattr(self._cfg, "request_timeout_seconds", 300.0)),
            cancel_check=cancel_check,
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineRegistry.capabilities_for(self.public_id, self._cfg)

    def metadata(self) -> dict[str, Any]:
        if self._worker is None:
            value = self._engine.metadata() if callable(getattr(self._engine, "metadata", None)) else {}
            metadata = dict(value) if isinstance(value, dict) else {}
        else:
            metadata = dict(self._worker.last_metadata)
            metadata.update({"loaded": self.is_loaded, "device": metadata.get("device") or self._cfg.device})
        metadata.update(self.capabilities().as_dict())
        metadata.update({
            "id": self.public_id, "name": self.public_name, "backend": self.backend,
            "loaded": self.is_loaded, "wakeable": True,
            "process_isolated": self._process_isolated,
            "process_alive": bool(self._worker and self._worker.alive),
            "worker_pid": self._worker.pid if self._worker else None,
            "worker_healthy": self._worker.is_healthy if self._worker else None,
            "worker_last_exit_reason": self._worker.last_exit_reason if self._worker else "",
            "release_guarantee": "worker_exit" if self._process_isolated else "in_process_best_effort",
            "default_voice": self.default_voice, "voices": self.get_voices(),
        })
        return metadata

    def __getattr__(self, name: str):
        if name.startswith("_") or self._engine is None:
            raise AttributeError(f"private or isolated engine attribute is not exposed: {name}")
        return getattr(self._engine, name)
