"""Stable engine and worker failure contracts.

The IPC envelope is deliberately data-only and transport-neutral.  It never
contains exception objects, tracebacks, provider metadata, or HTTP/WebSocket
details.
"""

from __future__ import annotations

from dataclasses import dataclass


WORKER_FAILURE_ENVELOPE_VERSION = 1
ENGINE_ERROR_CODES = frozenset(
    {
        "engine_load_failed",
        "engine_runtime_failed",
        "worker_timeout",
        "worker_process_failed",
        "worker_protocol_failed",
    }
)


@dataclass(frozen=True)
class WorkerFailureEnvelope:
    """Spawn-safe failure data sent from an engine worker to its parent."""

    version: int
    code: str
    message: str


class EngineError(RuntimeError):
    """In-process engine failure with a stable machine-readable code."""

    code: str
    message: str

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


__all__ = [
    "ENGINE_ERROR_CODES",
    "WORKER_FAILURE_ENVELOPE_VERSION",
    "EngineError",
    "WorkerFailureEnvelope",
]
