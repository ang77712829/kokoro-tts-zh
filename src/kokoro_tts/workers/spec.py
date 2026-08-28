"""Spawn-safe construction contract for isolated engine workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias


class WorkerEngine(Protocol):
    """Runtime surface consumed by the child command dispatcher."""

    def load(self): ...

    def unload(self) -> None: ...

    def metadata(self) -> dict: ...

    def synthesize(self, **kwargs): ...

    def synthesize_array(self, **kwargs): ...

    def synthesize_stream(self, **kwargs): ...

    def get_voices(self): ...


WorkerEngineFactory: TypeAlias = Callable[[object, str | None], WorkerEngine]


@dataclass(frozen=True)
class EngineWorkerSpec:
    """Trusted construction input passed across the multiprocessing spawn boundary.

    Product adapters create specs from top-level, spawn-picklable factory callables.
    Worker infrastructure consumes the callable without knowing concrete engine types.
    """

    engine_id: str
    factory: WorkerEngineFactory
    requested_provider: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine_id, str):
            raise TypeError("EngineWorkerSpec.engine_id must be a string")
        engine_id = self.engine_id
        if not engine_id or engine_id != engine_id.strip():
            raise ValueError("EngineWorkerSpec.engine_id must be a non-empty canonical ID")
        if not callable(self.factory):
            raise TypeError("EngineWorkerSpec.factory must be callable")
        if self.requested_provider is not None and not isinstance(
            self.requested_provider,
            str,
        ):
            raise TypeError("EngineWorkerSpec.requested_provider must be a string or None")


def create_worker_engine(config: object, spec: EngineWorkerSpec) -> WorkerEngine:
    """Construct the declared runtime without resolving an engine ID in the worker."""

    return spec.factory(config, spec.requested_provider)
