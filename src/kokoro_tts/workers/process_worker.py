"""可杀死的模型运行时 worker 进程。

API 进程负责路由、持久化配置档案元数据和配置。重量级推理运行时在
生成的子进程中构建，空闲/模型释放时操作系统可可靠回收 CPU RSS 和 GPU 内存。
"""

from __future__ import annotations

from contextlib import suppress
import inspect
import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterator

from ..contracts.errors import (
    ENGINE_ERROR_CODES,
    WORKER_FAILURE_ENVELOPE_VERSION,
    EngineError,
    WorkerFailureEnvelope,
)
from .spec import EngineWorkerSpec, create_worker_engine


@dataclass(frozen=True)
class WorkerResult:
    request_id: str
    kind: str
    payload: object


class EngineProcessTimeoutError(TimeoutError):
    """模型 worker 在请求截止时间前停止产生结果。"""

    code = "worker_timeout"


def _protocol_error(engine_id: str, detail: str) -> EngineError:
    return EngineError(
        code="worker_protocol_failed",
        message=f"{engine_id} worker 协议错误：{detail}",
    )


def _worker_result_from_raw(raw: object, *, engine_id: str) -> WorkerResult:
    try:
        result = raw if isinstance(raw, WorkerResult) else WorkerResult(*raw)
    except (TypeError, ValueError) as exc:
        raise _protocol_error(engine_id, "响应格式非法") from exc
    if not isinstance(result.request_id, str) or not isinstance(result.kind, str):
        raise _protocol_error(engine_id, "响应标识或类型非法")
    return result


def _engine_error_from_payload(
    payload: object,
    *,
    engine_id: str,
    legacy_code: str,
) -> EngineError:
    if isinstance(payload, WorkerFailureEnvelope):
        if payload.version != WORKER_FAILURE_ENVELOPE_VERSION:
            return _protocol_error(engine_id, f"未知错误协议版本：{payload.version}")
        if payload.code not in ENGINE_ERROR_CODES:
            return _protocol_error(engine_id, f"未知错误码：{payload.code}")
        if not isinstance(payload.message, str):
            return _protocol_error(engine_id, "错误消息类型非法")
        return EngineError(code=payload.code, message=payload.message)
    if isinstance(payload, str):
        return EngineError(code=legacy_code, message=payload)
    return _protocol_error(engine_id, f"错误载荷类型非法：{type(payload).__name__}")


def _child_failure(code: str, exc: BaseException) -> WorkerFailureEnvelope:
    return WorkerFailureEnvelope(
        version=WORKER_FAILURE_ENVELOPE_VERSION,
        code=code,
        message=f"{type(exc).__name__}: {exc}",
    )


def _initial_worker_failure_code(engine: object | None) -> str:
    """Classify lazy construction separately from commands on an owned engine."""

    if engine is None:
        return "engine_load_failed"
    return "engine_runtime_failed"


class EngineProcessClient:
    """为推理运行时懒启动的单次执行子进程。"""

    def __init__(self, *, config, spec: EngineWorkerSpec, logger=None):
        if not isinstance(spec, EngineWorkerSpec):
            raise TypeError("EngineProcessClient requires an EngineWorkerSpec")
        self.config = config
        self.spec = spec
        self.engine_id = spec.engine_id
        self.requested_provider = spec.requested_provider
        self.logger = logger
        self._ctx = mp.get_context("spawn")
        self._command_queue = None
        self._result_queue = None
        self._process: mp.Process | None = None
        self._lifecycle_lock = threading.RLock()
        # 已加载的运行时是有意设计为单次执行的。序列化所有队列
        # 消费者可防止同时的 HTTP/WebSocket 调用者互相抢消息
        # 导致伪超时。
        self._request_lock = threading.RLock()
        self._loaded = False
        self._last_metadata: dict = {}
        self._unhealthy = False
        self._last_exit_reason = ""
        # 共享内存中的取消代次：主进程写入要取消的流式请求代次，
        # 子进程只取消匹配代次，避免旧请求的延迟取消误伤新请求。
        self._cancel_flag = self._ctx.Value("i", 0)
        self._stream_generation = 0

    @property
    def alive(self) -> bool:
        process = self._process
        return bool(process is not None and process.is_alive())

    @property
    def pid(self) -> int | None:
        return self._process.pid if self.alive and self._process is not None else None

    @property
    def is_loaded(self) -> bool:
        return bool(self.alive and self._loaded)

    @property
    def last_metadata(self) -> dict:
        return dict(self._last_metadata)

    @property
    def is_healthy(self) -> bool:
        """空闲 worker 是健康的；崩溃的已加载 worker 不健康。"""

        process = self._process
        if self._unhealthy:
            return False
        if self._loaded and process is not None and not process.is_alive():
            return False
        return True

    @property
    def last_exit_reason(self) -> str:
        return self._last_exit_reason

    def soft_cancel(self) -> None:
        """通知当前流式请求在安全检查点停止，不杀进程。"""
        self._soft_cancel_worker(int(getattr(self, "_stream_generation", 0) or 0))

    def _soft_cancel_worker(self, generation: int | None = None) -> None:
        """通知子进程停止指定代次的流式合成。"""
        token = int(generation or 0)
        if token <= 0:
            return
        try:
            self._cancel_flag.value = token
        except Exception:
            if self.logger:
                self.logger.debug("设置 cancel_flag 失败", exc_info=True)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self.alive:
                return
            # 被杀死的 worker 可能残留未消费的命令/结果。新队列
            # 让下次唤醒成为一次干净的运行时代次。
            self._command_queue = self._ctx.Queue()
            self._result_queue = self._ctx.Queue()
            self._loaded = False
            # 启动新 worker 进程时重置取消标志。
            try:
                self._cancel_flag.value = 0
            except Exception:
                if self.logger:
                    self.logger.debug("重置 cancel_flag 失败", exc_info=True)
            self._process = self._ctx.Process(
                target=_worker_main,
                args=(self.config, self.spec,
                      self._command_queue, self._result_queue, self._cancel_flag),
                name=f"angevoice-{self.engine_id}-worker",
                daemon=True,
            )
            self._process.start()

    def close(self, *, kill: bool = False) -> None:
        # 强制释放/取消必须能够杀死卡住的推理，即使请求线程持有 _request_lock。
        # 优雅的空闲释放被序列化，以避免意外中断已接受的请求。
        if kill:
            with self._lifecycle_lock:
                self._close_locked(kill=True)
            return
        with self._request_lock, self._lifecycle_lock:
            self._close_locked(kill=False)

    def _close_locked(self, *, kill: bool) -> None:
        process = self._process
        if process is None:
            self._loaded = False
            self._discard_queues()
            return
        # CUDA 上下文（context）销毁可能需要额外时间；kill 模式给更长的宽限期。
        base_grace = float(getattr(self.config, "engine_process_kill_grace_seconds", 2.0) or 2.0)
        grace = max(base_grace, 5.0) if kill else base_grace
        if process.is_alive() and not kill:
            try:
                if self._command_queue is not None:
                    self._command_queue.put_nowait((uuid.uuid4().hex, "shutdown", {}))
            except Exception:
                pass
            process.join(timeout=grace)
        if process.is_alive():
            process.terminate()
            process.join(timeout=grace)
        if process.is_alive():
            process.kill()
            process.join(timeout=min(1.0, grace))
        self._process = None
        self._loaded = False
        self._unhealthy = False
        self._last_exit_reason = ""
        # 在下次全新启动前重置取消标志。
        try:
            self._cancel_flag.value = 0
        except Exception:
            if self.logger:
                self.logger.debug("重置 cancel_flag 失败", exc_info=True)
        self._discard_queues()

    def load(self, *, timeout: float) -> dict:
        value = self.request("load", {}, timeout=timeout)
        self._loaded = True
        self._unhealthy = False
        self._last_exit_reason = ""
        self._last_metadata = dict(value) if isinstance(value, dict) else {}
        return dict(self._last_metadata)

    def request(self, command: str, payload: dict | None = None, *, timeout: float) -> object:
        with self._request_lock:
            request_id = self._send(command, payload or {})
            result = self._wait_for(request_id, timeout=timeout)
            if result.kind == "error":
                legacy_code = "engine_load_failed" if command == "load" else "engine_runtime_failed"
                raise _engine_error_from_payload(
                    result.payload,
                    engine_id=self.engine_id,
                    legacy_code=legacy_code,
                )
            if result.kind != "result":
                raise _protocol_error(self.engine_id, f"未知消息类型：{result.kind}")
            if command in {"load", "metadata"} and isinstance(result.payload, dict):
                self._last_metadata = dict(result.payload)
                self._loaded = bool(self._last_metadata.get("loaded", command == "load"))
            return result.payload

    def stream(
        self,
        payload: dict,
        *,
        timeout: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[dict]:
        with self._request_lock:
            self._stream_generation = int(getattr(self, "_stream_generation", 0) or 0) + 1
            stream_generation = int(self._stream_generation)
            stream_payload = dict(payload or {})
            stream_payload["_cancel_generation"] = stream_generation
            request_id = self._send("synthesize_stream", stream_payload)
            configured_idle = float(getattr(self.config, "engine_process_stream_idle_timeout_seconds", 120.0) or 120.0)
            idle_timeout = max(5.0, float(timeout), configured_idle)
            drain_timeout = max(0.1, float(getattr(self.config, "engine_process_stream_drain_seconds", 30.0) or 30.0))
            queue_done_grace = min(5.0, max(0.5, drain_timeout))
            queue_done_hard_limit = idle_timeout
            deadline = time.monotonic() + idle_timeout
            drain_deadline: float | None = None
            completed = False
            try:
                while True:
                    if cancel_check is not None:
                        try:
                            if cancel_check():
                                self._soft_cancel_worker(stream_generation)
                                if drain_deadline is None:
                                    drain_deadline = time.monotonic() + drain_timeout
                        except Exception:
                            if self.logger:
                                self.logger.debug("%s worker cancel_check 失败", self.engine_id, exc_info=True)
                    remaining = deadline - time.monotonic()
                    if drain_deadline is not None:
                        drain_remaining = drain_deadline - time.monotonic()
                        if drain_remaining <= 0:
                            # 取消后的排空只用于清理当前请求，超时后返回，
                            # 不杀 worker，避免停止操作导致模型重载。
                            return
                        remaining = min(remaining, drain_remaining)
                    if remaining <= 0:
                        self.close(kill=True)
                        raise EngineProcessTimeoutError(f"{self.engine_id} worker 流式输出空闲超时：{idle_timeout}s")
                    self._raise_if_worker_exited()
                    try:
                        raw = self._require_result_queue().get(timeout=min(0.2, remaining))
                    except queue.Empty:
                        continue
                    result = _worker_result_from_raw(raw, engine_id=self.engine_id)
                    if result.request_id != request_id:
                        deadline = time.monotonic() + idle_timeout
                        if self.logger:
                            self.logger.debug(
                                "%s worker: 丢弃过期流式消息 %s",
                                self.engine_id, result.request_id,
                            )
                        continue
                    deadline = time.monotonic() + idle_timeout
                    if result.kind == "done":
                        completed = True
                        if drain_deadline is not None:
                            return
                        tail_deadline = time.monotonic() + queue_done_grace
                        tail_hard_deadline = time.monotonic() + queue_done_hard_limit
                        while time.monotonic() < tail_deadline and time.monotonic() < tail_hard_deadline:
                            tail_remaining = tail_deadline - time.monotonic()
                            try:
                                raw = self._require_result_queue().get(timeout=min(0.05, max(0.001, tail_remaining)))
                            except queue.Empty:
                                break
                            tail = _worker_result_from_raw(raw, engine_id=self.engine_id)
                            if tail.request_id != request_id:
                                if self.logger:
                                    self.logger.debug(
                                        "%s worker: 丢弃过期尾帧消息 %s",
                                        self.engine_id, tail.request_id,
                                    )
                                continue
                            if tail.kind == "event":
                                if drain_deadline is None:
                                    yield tail.payload
                                if isinstance(tail.payload, dict) and str(tail.payload.get("type") or "") in {
                                    "done",
                                    "cancelled",
                                    "error",
                                    "segment_error",
                                }:
                                    return
                                tail_deadline = min(time.monotonic() + queue_done_grace, tail_hard_deadline)
                                continue
                            if tail.kind == "error":
                                raise _engine_error_from_payload(
                                    tail.payload,
                                    engine_id=self.engine_id,
                                    legacy_code="engine_runtime_failed",
                                )
                            if tail.kind == "done":
                                continue
                            raise _protocol_error(self.engine_id, f"未知流式尾帧类型：{tail.kind}")
                        return
                    if result.kind == "event":
                        if drain_deadline is None:
                            yield result.payload
                        continue
                    if result.kind == "error":
                        completed = True
                        raise _engine_error_from_payload(
                            result.payload,
                            engine_id=self.engine_id,
                            legacy_code="engine_runtime_failed",
                        )
                    completed = True
                    raise _protocol_error(self.engine_id, f"未知流式消息类型：{result.kind}")
            finally:
                if not completed:
                    self._soft_cancel_worker(stream_generation)

    def _send(self, command: str, payload: dict) -> str:
        self.start()
        request_id = uuid.uuid4().hex
        if self._command_queue is None:
            raise RuntimeError(f"{self.engine_id} worker 命令队列不可用")
        self._command_queue.put((request_id, command, payload))
        return request_id

    def _wait_for(self, request_id: str, *, timeout: float) -> WorkerResult:
        deadline = time.monotonic() + max(0.01, float(timeout))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close(kill=True)
                raise EngineProcessTimeoutError(f"{self.engine_id} worker 请求超时：{timeout}s")
            self._raise_if_worker_exited()
            try:
                raw = self._require_result_queue().get(timeout=min(0.2, remaining))
            except queue.Empty:
                continue
            result = _worker_result_from_raw(raw, engine_id=self.engine_id)
            if result.request_id == request_id:
                return result

    def _raise_if_worker_exited(self) -> None:
        if self._process is not None and not self._process.is_alive():
            code = self._process.exitcode
            was_loaded = self._loaded
            self._loaded = False
            if was_loaded:
                self._unhealthy = True
                self._last_exit_reason = f"worker 异常退出，退出码：{code}"
            raise EngineError(
                code="worker_process_failed",
                message=f"{self.engine_id} worker 异常退出，退出码：{code}",
            )

    def _require_result_queue(self):
        if self._result_queue is None:
            raise RuntimeError(f"{self.engine_id} worker 结果队列不可用")
        return self._result_queue

    def _discard_queues(self) -> None:
        result_queue = self._result_queue
        command_queue = self._command_queue

        if result_queue is not None:
            while True:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break

        for managed_queue in (command_queue, result_queue):
            if managed_queue is None:
                continue
            with suppress(Exception):
                managed_queue.close()
            with suppress(Exception):
                managed_queue.join_thread()

        self._command_queue = None
        self._result_queue = None


def _stream_accepts_cancel_check(method) -> bool:
    """判断子进程内真实引擎是否接受取消检查回调。"""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    if "cancel_check" in parameters:
        return True
    return any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def _worker_main(config, spec: EngineWorkerSpec,
                 command_queue, result_queue, cancel_flag) -> None:
    engine_id = spec.engine_id
    engine = None

    def ensure_engine():
        nonlocal engine
        if engine is None:
            engine = create_worker_engine(config, spec)
            engine.load()
        return engine

    while True:
        request_id, command, payload = command_queue.get()
        # 取消信号按流式请求代次匹配。这里不清零共享值，
        # 避免旧请求还在执行时丢失取消；新请求使用新代次，
        # 不会被旧取消信号误伤。
        if command == "shutdown":
            try:
                if engine is not None:
                    engine.unload()
            finally:
                result_queue.put((request_id, "result", {"ok": True}))
            return
        failure_code = _initial_worker_failure_code(engine)
        try:
            current = ensure_engine()
            failure_code = "engine_runtime_failed"
            if command == "load":
                result_queue.put((request_id, "result", current.metadata()))
            elif command == "metadata":
                result_queue.put((request_id, "result", current.metadata()))
            elif command == "synthesize":
                result_queue.put((request_id, "result", current.synthesize(**payload)))
            elif command == "synthesize_array":
                result_queue.put((request_id, "result", current.synthesize_array(**payload)))
            elif command == "synthesize_stream":
                payload.pop("cancel_check", None)
                cancel_generation = int(payload.pop("_cancel_generation", 0) or 0)

                def worker_cancelled() -> bool:
                    try:
                        return bool(cancel_generation and int(cancel_flag.value) == cancel_generation)
                    except Exception:
                        return False

                if _stream_accepts_cancel_check(current.synthesize_stream):
                    payload["cancel_check"] = worker_cancelled
                saw_terminal_event = False
                audio_chunks = 0
                total_segments = None
                cancelled_by_generation = False
                for item in current.synthesize_stream(**payload):
                    # 在每个 yield 的帧之间检查当前请求自己的取消代次。
                    # 旧 WebSocket 的迟到取消不会命中新请求，避免正常长文本被误截断。
                    if worker_cancelled():
                        cancelled_by_generation = True
                        break
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "")
                        if item_type == "started":
                            total_segments = item.get("segments")
                        elif item_type == "audio":
                            audio_chunks += 1
                        elif item_type in {"done", "cancelled", "error", "segment_error"}:
                            saw_terminal_event = True
                    result_queue.put((request_id, "event", item))
                if not saw_terminal_event and not cancelled_by_generation and not worker_cancelled():
                    result_queue.put((
                        request_id,
                        "event",
                        {"type": "segment_error", "message": "流式合成提前结束，未收到语义终止帧"},
                    ))
                    done_payload = {"type": "done", "total_audio_chunks": audio_chunks}
                    if total_segments is not None:
                        done_payload["total_segments"] = total_segments
                    result_queue.put((request_id, "event", done_payload))
                # 始终发送终止消息，以便消费者释放请求锁。
                result_queue.put((request_id, "done", None))
                try:
                    if cancel_generation and int(cancel_flag.value) == cancel_generation:
                        cancel_flag.value = 0
                except Exception:
                    pass  # 子进程内无法用 logger
            elif command == "get_voices":
                result_queue.put((request_id, "result", current.get_voices()))
            else:
                result_queue.put((
                    request_id,
                    "error",
                    WorkerFailureEnvelope(
                        version=WORKER_FAILURE_ENVELOPE_VERSION,
                        code="worker_protocol_failed",
                        message=f"未知 {engine_id} worker 命令：{command}",
                    ),
                ))
        except BaseException as exc:  # noqa: BLE001
            result_queue.put((request_id, "error", _child_failure(failure_code, exc)))
