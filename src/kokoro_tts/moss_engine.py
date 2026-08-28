"""MOSS-TTS-Nano 引擎适配器。

官方推理 runtime 仍由 OpenMOSS 提供；本文件只负责把 runtime 接入
AngeVoice 的引擎接口。纯逻辑已拆到 ``kokoro_tts.moss`` 子包。
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import time
import threading
from collections import OrderedDict

from .audio import write_wav_bytes
from .config import TTSConfig
from .config_ids import moss_voice_catalog
from .moss_engine_streaming import MossStreamingMixin
from .moss import (
    analyze_waveform,
    clean_text as moss_clean_text,
    create_runtime as create_moss_runtime,
    ensure_import_path,
    prepare_prompt_audio,
    get_cuda_vram_snapshot,
    is_memory_allocation_error,
    resolve_prompt_audio_codes_cached,
    segment_text as moss_segment_text,
    temp_output_path,
)
from .moss_runtime.audio import (
    analyze_silence,
    clamp_pause_seconds,
    compress_long_silence,
    concat_waveforms,
    normalize_waveform,
    silence_array,
    trim_silence_edges,
)
from .moss_runtime.prompt import prompt_audio_cache_key
# MOSS 现在统一使用通用 EngineProcessClient，与 Kokoro/ZipVoice 一致。
from .workers import EngineProcessClient, EngineWorkerSpec
from .workers.process_worker import EngineProcessTimeoutError

logger = logging.getLogger(__name__)


def _create_moss_worker_engine(config: object, requested_provider: str | None) -> "MossNanoEngine":
    """Build the concrete MOSS runtime inside an isolated worker."""

    provider = str(requested_provider or "cpu").strip().lower()
    return MossNanoEngine(
        config,
        execution_provider=provider,
        engine_id="moss",
        process_isolation=False,
    )


class MossNanoEngine(MossStreamingMixin):
    """OpenMOSS 官方 ONNX runtime 的 AngeVoice 适配器。"""

    SUPPORTED_STREAM_FORMATS = {"pcm_s16le", "wav"}
    _provider_patch_lock = threading.Lock()

    def __init__(
        self,
        config: TTSConfig,
        execution_provider: str = "cpu",
        engine_id: str = "moss-nano",
        process_isolation: bool | None = None,
    ):
        self.config = config
        self.execution_provider = str(execution_provider or "cpu").lower()
        self.engine_id = engine_id
        self.display_name = "MOSS-TTS-Nano"
        self._process_isolated = self._resolve_process_isolation(process_isolation)
        self._process_client: EngineProcessClient | None = None
        self._cached_sample_rate = 48000
        self._cached_channels = 2
        self._cached_voices: list[str] = moss_voice_catalog(self.config.moss_default_voice)
        self._runtime = None
        self._loaded = False
        self._actual_provider = self.execution_provider
        self._load_error = ""
        self._self_test = None
        self._last_output_quality = None
        self._runtime_lock = threading.Lock()
        self._prompt_cache_lock = threading.Lock()
        self._prompt_audio_code_cache: OrderedDict[str, list[list[int]]] = OrderedDict()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{engine_id}-worker")
        self._unhealthy = False
        self._executor_lock = threading.Lock()
        self._consecutive_timeouts = 0
        self._last_vram_snapshot = None
        self._last_vram_refresh_at = 0.0
        self._low_vram_mode = False
        self._full_decode_disabled_until = 0.0
        self._full_decode_oom_count = 0


    def _resolve_process_isolation(self, explicit: bool | None) -> bool:
        """判断当前 MOSS 引擎是否启用进程级隔离。"""

        if explicit is not None:
            return bool(explicit)
        if not bool(getattr(self.config, "moss_process_isolation_enabled", False)):
            return False
        providers = {
            item.strip().lower()
            for item in str(getattr(self.config, "moss_process_isolation_providers", "cuda")).split(",")
            if item.strip()
        }
        return self.execution_provider in providers

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_healthy(self) -> bool:
        # 加载状态和健康状态是分开的：故意空闲/卸载的
        # 引擎没有被污染，不应在每次状态读取时重建。
        return not self._unhealthy

    @property
    def sample_rate(self) -> int:
        if self._runtime is None:
            return int(self._cached_sample_rate)
        return int(self._runtime.codec_meta["codec_config"]["sample_rate"])

    @property
    def channels(self) -> int:
        if self._runtime is None:
            return int(self._cached_channels)
        return int(self._runtime.codec_meta["codec_config"]["channels"])

    @property
    def device(self) -> str:
        return self._actual_provider

    @property
    def default_voice(self) -> str:
        return self.config.moss_default_voice

    def get_voices(self) -> list[str]:
        if self._process_isolated and self._process_client is not None and self._process_client.alive:
            try:
                # 修复说明：统一使用 "get_voices" 命令（与 _worker_main 中的处理分支对应）。
                # 原代码误用 "voices" 命令，会导致 worker 返回
                # “未知 moss worker 命令：voices” 错误。
                voices = self._process_client.request("get_voices", {}, timeout=10.0)
                self._cached_voices = [str(item) for item in voices] or [self.default_voice]
            except Exception:
                logger.debug("读取 MOSS 隔离进程音色列表失败", exc_info=True)
            return list(self._cached_voices)
        if self._runtime is None:
            return list(self._cached_voices or [self.default_voice])
        try:
            self._cached_voices = [str(item["voice"]) for item in self._runtime.list_builtin_voices()]
            return list(self._cached_voices)
        except Exception:
            logger.debug("读取 MOSS 音色列表失败", exc_info=True)
            return [self.default_voice]

    def metadata(self) -> dict:
        return {
            "id": self.engine_id,
            "name": self.display_name,
            "backend": "moss-tts-nano-onnx",
            "loaded": self.is_loaded,
            "device": self.device,
            "requested_provider": self.execution_provider,
            "actual_provider": self._actual_provider,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "voices": self.get_voices(),
            "default_voice": self.default_voice,
            "modes": ["preset_voice", "voice_clone"],
            "voice_clone_supported": True,
            "voice_clone_enabled": True,
            "prompt_audio_path": str(self.config.moss_prompt_audio_path) if self.config.moss_prompt_audio_path else "",
            "streaming": True,
            "stream_chunk_seconds": self.config.moss_stream_chunk_seconds,
            "stream_queue_max_items": self.config.moss_stream_queue_max_items,
            "speed_supported": False,
            "text_rules_enabled": str(self.config.moss_apply_angevoice_rules).lower() != "false",
            "text_rules_mode": str(self.config.moss_apply_angevoice_rules),
            "experimental": self.execution_provider == "cuda",
            "error": self._load_error,
            "self_test": self._self_test,
            "last_output_quality": self._last_output_quality,
            "prompt_audio_max_seconds": self.config.moss_prompt_audio_max_seconds,
            "output_target_peak": self.config.moss_output_target_peak,
            "sample_mode": self.config.moss_sample_mode,
            "seed": self.config.moss_seed,
            "cuda_memory_limit_mb": self.config.moss_cuda_memory_limit_mb,
            "healthy": self.is_healthy,
            "unhealthy": self._unhealthy,
            "consecutive_timeouts": self._consecutive_timeouts,
            "process_isolated": self._process_isolated,
            "process_alive": bool(self._process_client and self._process_client.alive),
            "worker_pid": self._process_client.pid if self._process_client else None,
            "release_guarantee": "worker_exit" if self._process_isolated else "in_process_best_effort",
            "vram_guard_enabled": bool(getattr(self.config, "moss_vram_guard_enabled", True)),
            "vram": self._vram_status(),
            "low_vram_mode": self._low_vram_mode,
            "full_decode_disabled": self._full_decode_disabled_until > time.monotonic(),
            "full_decode_disabled_until": round(self._full_decode_disabled_until, 3) if self._full_decode_disabled_until else 0,
            "full_decode_oom_count": self._full_decode_oom_count,
        }

    def load(self) -> "MossNanoEngine":
        if self._loaded:
            return self

        if self._process_isolated:
            return self._load_process_isolated()

        self._ensure_import_path()
        try:
            self._runtime = self._create_runtime(self.execution_provider)
            self._actual_provider = self.execution_provider
            self._run_self_test_if_needed()
        except Exception as exc:
            if self.execution_provider != "cuda" or not self.config.moss_auto_fallback_cpu:
                self._load_error = str(exc)
                raise
            logger.warning("MOSS CUDA load/self-test failed, falling back to CPU: %s", exc)
            fallback_reason = str(exc)
            self.unload()
            self._runtime = self._create_runtime("cpu")
            self._actual_provider = "cpu"
            if self.config.moss_quality_gate_enabled:
                self._run_cpu_fallback_quality_gate(fallback_reason)
            else:
                self._self_test = {"ok": True, "fallback_from": "cuda", "reason": fallback_reason}

        self._loaded = True
        logger.info("MOSS-TTS-Nano loaded (requested=%s actual=%s)", self.execution_provider, self._actual_provider)
        return self


    def _load_process_isolated(self) -> "MossNanoEngine":
        """在独立子进程中加载 MOSS runtime。

        使用统一的 EngineProcessClient（与 Kokoro/ZipVoice 相同的基础设施）。
        具体 runtime factory 由 MOSS owner 通过 EngineWorkerSpec 提供。
        """
        self._process_client = EngineProcessClient(
            config=self.config,
            spec=EngineWorkerSpec(
                engine_id="moss",
                factory=_create_moss_worker_engine,
                requested_provider=self.execution_provider,
            ),
            logger=logger,
        )
        try:
            metadata = self._process_client.load(timeout=float(self.config.request_timeout_seconds))
        except Exception as exc:
            self._load_error = str(exc)
            self._process_client.close(kill=True)
            self._process_client = None
            raise
        self._runtime = None
        self._loaded = True
        self._unhealthy = False
        self._actual_provider = (
            str(metadata.get("actual_provider") or self.execution_provider)
            if isinstance(metadata, dict)
            else self.execution_provider
        )
        self._cached_sample_rate = int(metadata.get("sample_rate") or 48000) if isinstance(metadata, dict) else 48000
        self._cached_channels = int(metadata.get("channels") or 2) if isinstance(metadata, dict) else 2
        voices = metadata.get("voices") if isinstance(metadata, dict) else None
        if voices:
            self._cached_voices = [str(item) for item in voices]
        self._self_test = metadata.get("self_test") if isinstance(metadata, dict) else None
        self._last_output_quality = metadata.get("last_output_quality") if isinstance(metadata, dict) else None
        logger.info("MOSS-TTS-Nano loaded in isolated process (requested=%s actual=%s)", self.execution_provider, self._actual_provider)
        return self

    def _rebuild_executor(self) -> None:
        """重建单 worker 线程池，隔离已经超时或可能卡死的推理任务。"""
        with self._executor_lock:
            old = self._executor
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self.engine_id}-worker",
            )
        try:
            old.shutdown(wait=False)
        except Exception:
            logger.debug("Failed to shut down old executor", exc_info=True)

    def unload(self, *, force: bool = False) -> None:
        if self._process_client is not None:
            self._process_client.close(kill=bool(force))
            self._process_client = None
        # 非隔离的 ONNX 运行时一旦 worker 开始执行就无法中断。
        # 超时后，绝不阻塞服务等待旧 worker 的锁：EngineManager 丢弃
        # 此受损实例并构建全新的引擎对象。正式的 Docker/fnOS 配置
        # 启用进程隔离，以便超时的 worker 可被干净地杀死。
        lock_acquired = False
        if force or self._unhealthy:
            lock_acquired = self._runtime_lock.acquire(timeout=0.25)
        else:
            self._runtime_lock.acquire()
            lock_acquired = True
        if not lock_acquired:
            logger.warning("Abandoning timed-out non-isolated MOSS runtime without waiting for its stuck worker")
            self._loaded = False
            self._unhealthy = True
            self._rebuild_executor()
            return
        try:
            self._runtime = None
            self._loaded = False
            self._unhealthy = False
            self._consecutive_timeouts = 0
            self._low_vram_mode = False
            self._last_vram_snapshot = None
            self._last_vram_refresh_at = 0.0
            with self._prompt_cache_lock:
                self._prompt_audio_code_cache.clear()
        finally:
            self._runtime_lock.release()
        self._rebuild_executor()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    logger.debug("MOSS CUDA IPC cleanup skipped", exc_info=True)
        except Exception:
            logger.debug("MOSS CUDA cache cleanup skipped", exc_info=True)

    def soft_cancel(self) -> None:
        if self._process_client is not None:
            self._process_client.soft_cancel()

    def synthesize(self, text: str, voice: str = "", speed: float = 1.0, prompt_audio_path: str | None = None) -> bytes:
        waveform = self.synthesize_array(text=text, voice=voice, speed=speed, prompt_audio_path=prompt_audio_path)
        return write_wav_bytes(waveform, self.sample_rate)

    def synthesize_array(self, text: str, voice: str = "", speed: float = 1.0, prompt_audio_path: str | None = None):
        """非流式合成。

        进程隔离模式下超时可终止子进程；非隔离模式下 cancel 仅在推理
        帧间隙生效，无法中断正在进行的 ONNX/CUDA 单帧推理。
        """
        self._validate_request(text=text, voice=voice, speed=speed)
        prepared_text = self._clean_text(text)
        segments = self._segment_text(prepared_text)
        prompt_audio = prompt_audio_path or (
            str(self.config.moss_prompt_audio_path) if self.config.moss_prompt_audio_path else None
        )
        timeout = self.config.request_timeout_seconds

        if self._process_isolated:
            return self._synthesize_array_process_isolated(
                text=text,
                voice=voice,
                speed=speed,
                prompt_audio_path=prompt_audio,
                timeout=timeout,
            )

        logger.info("MOSS synthesize: segments=%d, voice=%s", len(segments), voice or self.default_voice)

        def _run():
            with self._runtime_lock:
                return self._concat_waveforms(
                    list(self._iter_waveforms(segments=segments, voice=voice, prompt_audio_path=prompt_audio))
                )

        future = self._executor.submit(_run)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            self._consecutive_timeouts += 1
            self._unhealthy = True
            self._loaded = False
            self._rebuild_executor()
            logger.warning(
                "MOSS 推理超时（%.0fs），已重建执行器并标记引擎不健康（连续次数=%d）",
                timeout, self._consecutive_timeouts,
            )
            raise RuntimeError(
                f"MOSS 推理超时（{timeout}s）。"
                "引擎已标记为不健康并隔离；下次请求会创建新的引擎实例。"
                "如果持续出现，请切换到 CPU 或重启容器。"
            )


    def _synthesize_array_process_isolated(self, *, text: str, voice: str, speed: float, prompt_audio_path: str | None, timeout: float):
        """通过隔离子进程执行一次非流式 MOSS 推理。"""

        # 修复说明：同时检查 is_loaded，处理进程崩溃后 client 存在但未加载的场景。
        if self._process_client is None or not self._process_client.is_loaded:
            self.load()
        try:
            return self._process_client.request(
                "synthesize_array",
                {"text": text, "voice": voice, "speed": speed, "prompt_audio_path": prompt_audio_path},
                timeout=float(timeout),
            )
        except EngineProcessTimeoutError as exc:
            self._mark_process_failure(timeout=timeout, reason="timeout")
            raise RuntimeError(
                f"MOSS 隔离 worker 推理超时（{timeout}s）。worker 进程已被终止，下次请求会重新创建。"
            ) from exc
        except Exception:
            self._mark_process_failure(timeout=timeout, reason="worker_error")
            raise

    def _mark_process_failure(self, *, timeout: float, reason: str) -> None:
        """记录隔离进程失败，标记引擎不健康。

        修复说明（原实现有两个问题）：
        1. 原代码设 ``_process_client = None``，导致 EngineManager 无法通过
           ``engine.is_healthy`` 感知到不健康状态，错误地跳过 drop+rebuild 流程。
        2. 原代码设 ``_loaded = False``，但 EngineManager 通过 ``is_healthy`` 判断
           是否需要重建——应由 EngineManager 统一管理，而非在此强制修改加载状态。

        现在只设 ``_unhealthy = True`` 并 kill 进程，让 EngineManager 在下次
        borrow() 时检测到 is_healthy == False，自动 drop + rebuild。
        """
        self._consecutive_timeouts += 1
        self._unhealthy = True
        # 保持 _process_client 引用：EngineManager 通过 is_healthy 感知状态，
        # drop_model 时会调用 unload()，unload() 会正确地 close + null client。
        if self._process_client is not None:
            self._process_client.close(kill=True)
        logger.warning(
            "MOSS isolated worker failed (%s, %.0fs); process killed (consecutive=%d)",
            reason,
            timeout,
            self._consecutive_timeouts,
        )

    def _iter_waveforms(self, *, segments: list[str], voice: str = "", prompt_audio_path: str | None = None):
        prompt_audio_codes = self._resolve_prompt_audio_codes_cached(voice=voice, prompt_audio_path=prompt_audio_path)
        self._configure_runtime_generation()
        total_segments = len([item for item in segments if item.strip()])
        emitted = 0
        for seg in segments:
            if not seg.strip():
                continue
            emitted += 1
            t0 = time.monotonic()
            for waveform in self._iter_runtime_chunks(seg, voice=voice, prompt_audio_codes=prompt_audio_codes):
                processed = self._postprocess_waveform(waveform, trim_silence=bool(getattr(self.config, "moss_audio_polish_enabled", True)))
                logger.info(
                    "MOSS segment %d/%d chunk done (%.1fs, %d samples)",
                    emitted,
                    total_segments,
                    time.monotonic() - t0,
                    processed.shape[0],
                )
                yield processed
            if emitted < total_segments:
                silence = self._silence_array(self._segment_pause_seconds())
                if silence.size:
                    yield silence

    def _iter_runtime_chunks(self, text: str, *, voice: str = "", prompt_audio_codes: list[list[int]]):
        import numpy as np

        text_chunks = self._prepare_runtime_text_chunks(text, voice=voice)
        if not text_chunks:
            return
        for chunk_index, chunk_text in enumerate(text_chunks):
            try:
                if self._should_use_incremental_codec() and self._runtime_supports_frame_streaming():
                    waveform = self._synthesize_single_chunk_incremental(chunk_text, prompt_audio_codes)
                else:
                    chunk_result = self._runtime.synthesize_single_chunk(
                        text=chunk_text,
                        prompt_audio_codes=prompt_audio_codes,
                        streaming=bool(self.config.moss_realtime_streaming_decode),
                    )
                    waveform = np.asarray(chunk_result["waveform"], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                if not is_memory_allocation_error(exc) or not self._runtime_supports_frame_streaming():
                    raise
                self._record_full_decode_oom(exc)
                waveform = self._synthesize_single_chunk_incremental(chunk_text, prompt_audio_codes)
            yield np.asarray(waveform, dtype=np.float32)
            if chunk_index < len(text_chunks) - 1:
                pause_seconds = self._runtime_pause_seconds(chunk_text)
                silence = self._silence_array(pause_seconds)
                if silence.size:
                    yield silence

    def _prepare_runtime_text_chunks(self, text: str, *, voice: str = "") -> list[str]:
        prepared_texts = self._runtime.prepare_synthesis_text(
            text=text,
            voice=voice or self.default_voice,
            enable_wetext=bool(self.config.moss_enable_wetext_processing),
            enable_normalize_tts_text=bool(self.config.moss_enable_normalize_tts_text),
        )
        prepared_text = str(prepared_texts["text"])
        return list(
            self._runtime.split_voice_clone_text(
                prepared_text,
                max_tokens=int(self._effective_text_tokens()),
            )
        )

    def _configure_runtime_generation(self) -> None:
        generation = self._runtime.manifest["generation_defaults"]
        generation["max_new_frames"] = int(self._effective_max_new_frames())
        generation["sample_mode"] = self.config.moss_sample_mode
        generation["do_sample"] = self.config.moss_sample_mode != "greedy"
        seed = int(self.config.moss_seed)
        if seed >= 0:
            try:
                import numpy as np

                self._runtime.rng = np.random.default_rng(seed)
            except Exception:
                logger.debug("Failed to reset MOSS RNG seed", exc_info=True)

    def _concat_waveforms(self, waveforms: list):
        if bool(getattr(self.config, "moss_audio_polish_enabled", True)):
            audio = concat_waveforms(
                waveforms,
                crossfade_ms=float(getattr(self.config, "moss_crossfade_ms", 35.0)),
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            audio, silence_metrics = compress_long_silence(
                audio,
                sample_rate=self.sample_rate,
                channels=self.channels,
                threshold_db=float(getattr(self.config, "moss_trim_silence_db", -45.0)),
                max_silence_ms=float(getattr(self.config, "moss_max_silence_ms", 900.0)),
            )
            quality = analyze_silence(
                audio,
                sample_rate=self.sample_rate,
                channels=self.channels,
                threshold_db=float(getattr(self.config, "moss_trim_silence_db", -45.0)),
            )
            quality.update(
                {
                    "long_silence_count_before": silence_metrics.get("long_silence_count", 0),
                    "max_silence_ms_before": silence_metrics.get("max_silence_ms", 0.0),
                    "silence_ratio_before": silence_metrics.get("silence_ratio", 0.0),
                }
            )
            self._last_output_quality = {**(self._last_output_quality or {}), **quality}
            return audio
        return concat_waveforms(waveforms)

    def _runtime_pause_seconds(self, chunk_text: str) -> float:
        try:
            estimated = float(self._runtime.estimate_voice_clone_inter_chunk_pause_seconds(chunk_text))
        except Exception:
            logger.debug("MOSS runtime pause estimate failed", exc_info=True)
            estimated = 0.0
        return clamp_pause_seconds(
            estimated,
            max_ms=float(getattr(self.config, "moss_runtime_pause_max_ms", 350.0)),
        )

    def _segment_pause_seconds(self) -> float:
        return max(0.0, float(getattr(self.config, "moss_segment_pause_ms", 120.0)) / 1000.0)

    def _silence_array(self, seconds: float):
        return silence_array(seconds, sample_rate=self.sample_rate, channels=self.channels)

    def _resolve_prompt_audio_codes_cached(self, *, voice: str = "", prompt_audio_path: str | None = None) -> list[list[int]]:
        return resolve_prompt_audio_codes_cached(
            runtime=self._runtime,
            cache=self._prompt_audio_code_cache,
            cache_lock=self._prompt_cache_lock,
            voice=voice,
            default_voice=self.default_voice,
            prompt_audio_path=prompt_audio_path,
            max_items=int(self.config.moss_prompt_cache_max_items),
            max_seconds=float(self.config.moss_prompt_audio_max_seconds),
            sample_rate=self.sample_rate,
            channels=self.channels,
            logger=logger,
        )

    def _prompt_audio_cache_key(self, *, voice: str = "", prompt_audio_path: str | None = None) -> str:
        return prompt_audio_cache_key(
            voice=voice,
            default_voice=self.default_voice,
            prompt_audio_path=prompt_audio_path,
            max_seconds=float(self.config.moss_prompt_audio_max_seconds),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def _prepare_prompt_audio(self, prompt_audio_path: str | None) -> tuple[str | None, str | None]:
        return prepare_prompt_audio(
            prompt_audio_path,
            max_seconds=float(self.config.moss_prompt_audio_max_seconds),
            sample_rate=self.sample_rate,
            channels=self.channels,
            logger=logger,
        )

    def _postprocess_waveform(
        self,
        waveform,
        *,
        update_quality: bool = True,
        edge_fade: bool = True,
        trim_silence: bool = False,
        compress_silence: bool = False,
    ):
        processed, quality = normalize_waveform(
            waveform,
            channels=self.channels,
            gain=float(self.config.moss_output_gain),
            target_peak=float(self.config.moss_output_target_peak),
            peak_normalize_enabled=bool(self.config.moss_output_peak_normalize_enabled),
            declick_enabled=bool(getattr(self.config, "moss_output_declick_enabled", True)),
            edge_fade_samples=(
                int(self.sample_rate * float(getattr(self.config, "moss_output_edge_fade_ms", 1.5)) / 1000.0)
                if bool(edge_fade)
                else 0
            ),
        )
        quality_dict = quality.as_dict()
        if compress_silence and bool(getattr(self.config, "moss_audio_polish_enabled", True)):
            processed, silence_metrics = compress_long_silence(
                processed,
                sample_rate=self.sample_rate,
                channels=self.channels,
                threshold_db=float(getattr(self.config, "moss_trim_silence_db", -45.0)),
                max_silence_ms=float(getattr(self.config, "moss_max_silence_ms", 550.0)),
            )
            quality_dict.update({
                "stream_long_silence_count_before": silence_metrics.get("long_silence_count", 0),
                "stream_max_silence_ms_before": silence_metrics.get("max_silence_ms", 0.0),
                "stream_silence_ratio_before": silence_metrics.get("silence_ratio", 0.0),
            })
        if trim_silence and bool(getattr(self.config, "moss_trim_silence_enabled", True)):
            processed, trimmed_start_ms, trimmed_end_ms = trim_silence_edges(
                processed,
                sample_rate=self.sample_rate,
                channels=self.channels,
                threshold_db=float(getattr(self.config, "moss_trim_silence_db", -45.0)),
            )
            quality_dict.update({
                "trimmed_start_ms": round(float(trimmed_start_ms), 3),
                "trimmed_end_ms": round(float(trimmed_end_ms), 3),
            })
        if update_quality:
            quality_dict.update(
                analyze_silence(
                    processed,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    threshold_db=float(getattr(self.config, "moss_trim_silence_db", -45.0)),
                )
            )
            self._last_output_quality = quality_dict
        return processed

    def _ensure_import_path(self) -> None:
        ensure_import_path(self.config.moss_repo_path)

    def _create_runtime(self, provider: str):
        return create_moss_runtime(
            config=self.config,
            provider=provider,
            provider_patch_lock=self._provider_patch_lock,
            logger=logger,
        )

    def _run_self_test_if_needed(self) -> None:
        if self._runtime is None:
            return
        if self.execution_provider != "cuda" and not self.config.moss_quality_gate_enabled:
            return
        if self.execution_provider == "cuda" and self.config.moss_cuda_self_test_enabled:
            self._runtime.warmup()
        if not self.config.moss_quality_gate_enabled:
            self._self_test = {"ok": True, "mode": "warmup"}
            return
        result = self._runtime.synthesize(
            text="你好，AngeVoice 正在进行模型自检。",
            voice=self.default_voice,
            prompt_audio_path=str(self.config.moss_prompt_audio_path) if self.config.moss_prompt_audio_path else None,
            output_audio_path=self._temp_output_path(),
            sample_mode="greedy",
            do_sample=False,
            streaming=True,
            max_new_frames=min(96, int(self.config.moss_max_new_frames)),
            voice_clone_max_text_tokens=min(40, int(self.config.moss_voice_clone_max_text_tokens)),
            enable_wetext=False,
            enable_normalize_tts_text=True,
        )
        self._self_test = self._analyze_waveform(result["waveform"], int(result["sample_rate"]))
        if not self._self_test["ok"]:
            raise RuntimeError("MOSS quality gate failed: " + self._self_test["reason"])

    def _run_cpu_fallback_quality_gate(self, fallback_reason: str) -> None:
        result = self._runtime.synthesize(
            text="你好，AngeVoice 正在验证 CPU 回退模型。",
            voice=self.default_voice,
            prompt_audio_path=str(self.config.moss_prompt_audio_path) if self.config.moss_prompt_audio_path else None,
            output_audio_path=self._temp_output_path(),
            sample_mode="greedy",
            do_sample=False,
            streaming=True,
            max_new_frames=min(96, int(self.config.moss_max_new_frames)),
            voice_clone_max_text_tokens=min(40, int(self.config.moss_voice_clone_max_text_tokens)),
            enable_wetext=False,
            enable_normalize_tts_text=True,
        )
        self._self_test = self._analyze_waveform(result["waveform"], int(result["sample_rate"]))
        self._self_test.update({"fallback_from": "cuda", "fallback_reason": fallback_reason})
        if not self._self_test["ok"]:
            raise RuntimeError("MOSS CPU fallback quality gate failed: " + self._self_test["reason"])

    def _analyze_waveform(self, waveform, sample_rate: int) -> dict:
        return analyze_waveform(
            waveform,
            sample_rate,
            max_clip_ratio=float(self.config.moss_max_clip_ratio),
        )

    def _validate_request(self, text: str, voice: str, speed: float) -> None:
        if not self._loaded or (self._runtime is None and not self._process_isolated):
            raise RuntimeError("MOSS 引擎未加载，请先加载模型")
        if not text or not str(text).strip():
            raise ValueError("文本不能为空")
        if len(text) > self.config.max_text_length:
            raise ValueError(f"文本过长 ({len(text)} 字符)，上限 {self.config.max_text_length}")
        try:
            value = float(speed)
        except (TypeError, ValueError):
            raise ValueError("speed 必须是数字") from None
        if abs(value - 1.0) > 1e-6:
            raise ValueError("MOSS-TTS-Nano 暂不支持语速调节，请使用 speed=1.0")

    def _clean_text(self, text: str) -> str:
        return moss_clean_text(
            text,
            apply_angevoice_rules=self.config.moss_apply_angevoice_rules,
            mixed_english_policy=getattr(self.config, "moss_mixed_english_policy", "translate"),
            model=self.engine_id,
        )

    def _segment_text(self, text: str) -> list[str]:
        return moss_segment_text(
            text,
            max_text_length=int(self.config.max_text_length),
            segment_length=int(self._effective_segment_length()),
            single_newline_policy=str(getattr(self.config, "text_single_newline_policy", "auto")),
        )

    def _vram_status(self, *, refresh: bool = False) -> dict:
        if refresh and self.execution_provider == "cuda" and bool(getattr(self.config, "moss_vram_guard_enabled", True)):
            self._refresh_vram_guard(force=False)
        snapshot = self._last_vram_snapshot
        data = snapshot.as_dict() if snapshot is not None else {"available": False, "source": "not-checked"}
        data.update({
            "low_vram_mode": self._low_vram_mode,
            "safe_free_mb": int(getattr(self.config, "moss_vram_safe_free_mb", 1200)),
            "critical_free_mb": int(getattr(self.config, "moss_vram_critical_free_mb", 600)),
        })
        return data

    def _refresh_vram_guard(self, *, force: bool = False) -> None:
        if self.execution_provider != "cuda" or not bool(getattr(self.config, "moss_vram_guard_enabled", True)):
            self._low_vram_mode = False
            return
        now = time.monotonic()
        ttl = max(0.0, float(getattr(self.config, "moss_vram_snapshot_ttl_seconds", 10.0) or 0.0))
        if not force and self._last_vram_snapshot is not None and ttl > 0 and now - self._last_vram_refresh_at < ttl:
            return
        snapshot = get_cuda_vram_snapshot()
        # 探测失败时不清除保护状态，也不缓存——等下次 TTL 过期后重试
        if not snapshot.available or snapshot.free_mb is None:
            self._last_vram_refresh_at = now
            return
        self._last_vram_snapshot = snapshot
        self._last_vram_refresh_at = now
        safe = int(getattr(self.config, "moss_vram_safe_free_mb", 1200) or 0)
        critical = int(getattr(self.config, "moss_vram_critical_free_mb", 600) or 0)
        was_low = self._low_vram_mode
        self._low_vram_mode = snapshot.free_mb < safe if safe > 0 else False
        if snapshot.free_mb < critical and critical > 0:
            self._disable_full_codec_decode(reason=f"critical free VRAM {snapshot.free_mb}MB < {critical}MB")
        if self._low_vram_mode and not was_low:
            logger.warning(
                "MOSS VRAM guard: low free VRAM %s/%s MB, using conservative limits",
                snapshot.free_mb,
                snapshot.total_mb,
            )

    def _effective_segment_length(self) -> int:
        self._refresh_vram_guard()
        configured = int(getattr(self.config, "moss_segment_length", self.config.segment_length))
        if self._low_vram_mode:
            return min(configured, int(getattr(self.config, "moss_low_vram_segment_length", 160)))
        return configured

    def _effective_max_new_frames(self) -> int:
        self._refresh_vram_guard()
        configured = int(getattr(self.config, "moss_max_new_frames", 320))
        if self._low_vram_mode:
            return min(configured, int(getattr(self.config, "moss_low_vram_max_new_frames", 300)))
        return configured

    def _effective_text_tokens(self) -> int:
        self._refresh_vram_guard()
        configured = int(getattr(self.config, "moss_voice_clone_max_text_tokens", 64))
        if self._low_vram_mode:
            return min(configured, int(getattr(self.config, "moss_low_vram_text_tokens", 56)))
        return configured

    def _should_use_incremental_codec(self) -> bool:
        if self._full_decode_disabled_until > time.monotonic():
            return True
        self._refresh_vram_guard()
        return bool(self._low_vram_mode)

    def _disable_full_codec_decode(self, *, reason: str) -> None:
        if not bool(getattr(self.config, "moss_disable_full_codec_after_oom", True)):
            return
        cooldown = float(getattr(self.config, "moss_full_codec_oom_cooldown_seconds", 600.0) or 0.0)
        self._full_decode_disabled_until = max(self._full_decode_disabled_until, time.monotonic() + cooldown)
        logger.warning("MOSS full codec decode disabled for %.0fs: %s", cooldown, reason)

    def _record_full_decode_oom(self, exc: BaseException) -> None:
        self._full_decode_oom_count += 1
        self._low_vram_mode = True
        # OOM 刚发生，不立即刷新 VRAM 探测——探测可能显示瞬时空闲但压力仍在，
        # 清除 _low_vram_mode 会导致后续 chunk 重复 OOM。
        # 保留 _low_vram_mode=True，等自然 TTL 过期后再探测。
        self._last_vram_refresh_at = time.monotonic()
        self._disable_full_codec_decode(reason=f"decode OOM: {exc}")

    def _synthesize_single_chunk_incremental(self, chunk_text: str, prompt_audio_codes: list[list[int]]):
        import numpy as np

        if not self._runtime_supports_frame_streaming():
            raise RuntimeError("MOSS 运行时不支持增量编解码流式传输")
        pending_decode_frames: list[list[int]] = []
        waveforms: list = []
        self._runtime.codec_streaming_session.reset()

        def decode_pending(force: bool) -> None:
            pending_count = len(pending_decode_frames)
            if pending_count <= 0:
                return
            # 保持较小的帧批次，避免完整 codec 路径瞬时占用过高。
            budget = 12 if self._low_vram_mode else 24
            if not force and pending_count < budget:
                return
            frame_budget = pending_count if force else min(pending_count, budget)
            frame_chunk = pending_decode_frames[:frame_budget]
            del pending_decode_frames[:frame_budget]
            decoded = self._runtime.codec_streaming_session.run_frames(frame_chunk)
            if decoded is None:
                return
            audio, audio_length = decoded
            if int(audio_length) <= 0:
                return
            waveforms.append(self._merge_codec_audio(audio, int(audio_length)))

        def on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
            pending_decode_frames.append(list(frame))
            decode_pending(False)

        try:
            text_token_ids = self._runtime.encode_text(chunk_text)
            request_rows = self._runtime.build_voice_clone_request_rows(prompt_audio_codes, text_token_ids)
            self._runtime.generate_audio_frames(request_rows, on_frame=on_frame)
            decode_pending(True)
        finally:
            self._runtime.codec_streaming_session.reset()
        if not waveforms:
            return np.zeros((0, self.channels), dtype=np.float32)
        return concat_waveforms(waveforms, channels=self.channels)

    def _temp_output_path(self) -> str:
        return temp_output_path()
