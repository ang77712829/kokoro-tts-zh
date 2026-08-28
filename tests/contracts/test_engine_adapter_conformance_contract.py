"""Behavior contracts for the three product engine adapter surfaces."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, dataclass
import inspect
from pathlib import Path
import threading
from typing import Any

from fastapi import HTTPException
import pytest

from kokoro_tts.engine_manager import EngineManager
from kokoro_tts.engines.adapters.kokoro import KokoroAdapter
from kokoro_tts.engines.adapters.moss import MossAdapter
from kokoro_tts.engines.adapters.zipvoice import ZipVoiceEngine as ExportedZipVoiceEngine
from kokoro_tts.engines.base import EngineAdapter, EngineCapabilities, EngineSpec
from kokoro_tts.engines.registry import EngineRegistry
from kokoro_tts.services.synthesis_service import SynthesisService
from kokoro_tts.workers.process_worker import EngineProcessClient
from kokoro_tts.zipvoice.engine import ZipVoiceEngine


pytestmark = pytest.mark.contract


class _FakeAssets:
    def status(self) -> dict[str, Any]:
        return {"ready": True, "status_file": "fake-status"}


class _FakeProfiles:
    def list(self) -> list[dict[str, str]]:
        return []


class _LazyStream:
    def __init__(self) -> None:
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(())


class _FakeRuntime:
    def __init__(self, *, sample_rate: int = 24000) -> None:
        self.is_loaded = False
        self.loaded = False
        self.is_healthy = True
        self.sample_rate = sample_rate
        self.assets = _FakeAssets()
        self.last_metrics: dict[str, Any] = {}
        self.load_calls = 0
        self.unload_calls = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.array_result = object()
        self.stream_result = _LazyStream()
        self.failure_method: str | None = None
        self.failure: BaseException | None = None

    def _raise_when_requested(self, method: str) -> None:
        if self.failure_method == method and self.failure is not None:
            raise self.failure

    def load(self):
        self.load_calls += 1
        self.is_loaded = True
        self.loaded = True
        return self

    def unload(self, *args, **kwargs) -> None:
        self.unload_calls += 1
        self.is_loaded = False
        self.loaded = False

    def metadata(self) -> dict[str, Any]:
        return {"loaded": self.is_loaded}

    def synthesize(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        **kwargs,
    ) -> bytes:
        self._raise_when_requested("synthesize")
        self.calls.append(
            (
                "synthesize",
                {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "prompt_audio_path": prompt_audio_path,
                    **kwargs,
                },
            )
        )
        return b"fake-wav"

    def synthesize_array(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        **kwargs,
    ):
        self._raise_when_requested("synthesize_array")
        self.calls.append(
            (
                "synthesize_array",
                {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "prompt_audio_path": prompt_audio_path,
                    **kwargs,
                },
            )
        )
        return self.array_result

    def synthesize_stream(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        fmt: str = "pcm_s16le",
        prompt_audio_path: str | None = None,
        cancel_check=None,
        **kwargs,
    ):
        self._raise_when_requested("synthesize_stream")
        self.calls.append(
            (
                "synthesize_stream",
                {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                    "fmt": fmt,
                    "prompt_audio_path": prompt_audio_path,
                    "cancel_check": cancel_check,
                    **kwargs,
                },
            )
        )
        return self.stream_result

    def moss_runtime_extension(self, value: str) -> tuple[str, str]:
        return ("moss-extension", value)


class _FakeConfig:
    kokoro_process_isolation_enabled = False
    moss_execution_provider = "cpu"
    moss_cuda_enabled = True
    moss_apply_angevoice_rules = "auto"
    zipvoice_execution_provider = "cpu"
    zipvoice_process_isolation_enabled = False
    zipvoice_auto_fallback_cpu = False
    sample_rate = 24000
    default_voice = "fake-voice"
    device = "cpu"
    request_timeout_seconds = 30.0
    model_switch_timeout_seconds = 30.0

    @staticmethod
    def get_voices() -> list[str]:
        return ["fake-voice"]


@dataclass(frozen=True)
class _ProductCase:
    product_id: str
    adapter: object
    runtime: _FakeRuntime


def _build_zipvoice(runtime: _FakeRuntime, cfg: _FakeConfig) -> ZipVoiceEngine:
    engine = ZipVoiceEngine.__new__(ZipVoiceEngine)
    engine.cfg = cfg
    engine.requested_provider = "cpu"
    engine._process_isolated = False
    engine.profiles = _FakeProfiles()
    engine.assets = runtime.assets
    engine._worker = None
    engine._cpu_runtime = runtime
    engine._cuda_runtime = None
    engine.runtime = runtime
    engine._unhealthy = False
    engine._actual_provider = None
    engine._fallback = False
    engine._fallback_reason = ""
    engine._state_lock = threading.RLock()
    return engine


def _build_product_cases(monkeypatch) -> dict[str, _ProductCase]:
    cfg = _FakeConfig()
    kokoro_runtime = _FakeRuntime()
    moss_runtime = _FakeRuntime(sample_rate=48000)
    zipvoice_runtime = _FakeRuntime()
    monkeypatch.setattr(
        "kokoro_tts.engines.adapters.moss.MossNanoEngine",
        lambda *args, **kwargs: moss_runtime,
    )
    return {
        "kokoro": _ProductCase("kokoro", KokoroAdapter(cfg, engine=kokoro_runtime), kokoro_runtime),
        "moss": _ProductCase("moss", MossAdapter(cfg), moss_runtime),
        "zipvoice": _ProductCase("zipvoice", _build_zipvoice(zipvoice_runtime, cfg), zipvoice_runtime),
    }


@pytest.fixture
def product_cases(monkeypatch) -> dict[str, _ProductCase]:
    return _build_product_cases(monkeypatch)


@pytest.mark.parametrize("product_id", ("kokoro", "moss", "zipvoice"))
def test_real_product_satisfies_runtime_engine_adapter_protocol(
    product_id: str,
    product_cases: dict[str, _ProductCase],
) -> None:
    adapter = product_cases[product_id].adapter
    assert isinstance(adapter, EngineAdapter)
    assert EngineAdapter not in type(adapter).__mro__


def test_moss_protocol_methods_are_statically_discoverable() -> None:
    for method_name in ("synthesize", "synthesize_array", "synthesize_stream"):
        method = inspect.getattr_static(MossAdapter, method_name)
        assert callable(method)
        assert method is MossAdapter.__dict__[method_name]


def test_moss_explicit_method_signatures_preserve_runtime_extensions() -> None:
    synthesize = inspect.signature(MossAdapter.synthesize).parameters
    synthesize_array = inspect.signature(MossAdapter.synthesize_array).parameters
    synthesize_stream = inspect.signature(MossAdapter.synthesize_stream).parameters
    assert tuple(synthesize) == ("self", "text", "voice", "speed", "prompt_audio_path", "kwargs")
    assert tuple(synthesize_array) == ("self", "text", "voice", "speed", "prompt_audio_path", "kwargs")
    assert tuple(synthesize_stream) == (
        "self",
        "text",
        "voice",
        "speed",
        "fmt",
        "prompt_audio_path",
        "cancel_check",
        "kwargs",
    )
    assert synthesize["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert synthesize_stream["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_moss_explicit_methods_delegate_without_transforming_results(product_cases) -> None:
    case = product_cases["moss"]
    cancel_check = lambda: False
    wav = case.adapter.synthesize(
        "text",
        "voice",
        1.0,
        prompt_audio_path="reference.wav",
        runtime_option="kept",
    )
    array = case.adapter.synthesize_array("text", "voice", 1.0, prompt_audio_path="reference.wav")
    stream = case.adapter.synthesize_stream(
        "text",
        "voice",
        1.0,
        "wav",
        prompt_audio_path="reference.wav",
        cancel_check=cancel_check,
    )
    assert wav == b"fake-wav"
    assert array is case.runtime.array_result
    assert stream is case.runtime.stream_result
    assert stream.iterations == 0
    assert case.runtime.calls == [
        (
            "synthesize",
            {
                "text": "text",
                "voice": "voice",
                "speed": 1.0,
                "prompt_audio_path": "reference.wav",
                "runtime_option": "kept",
            },
        ),
        (
            "synthesize_array",
            {
                "text": "text",
                "voice": "voice",
                "speed": 1.0,
                "prompt_audio_path": "reference.wav",
            },
        ),
        (
            "synthesize_stream",
            {
                "text": "text",
                "voice": "voice",
                "speed": 1.0,
                "fmt": "wav",
                "prompt_audio_path": "reference.wav",
                "cancel_check": cancel_check,
            },
        ),
    ]


@pytest.mark.parametrize("method_name", ("synthesize", "synthesize_array", "synthesize_stream"))
def test_moss_explicit_methods_preserve_exception_identity(method_name, product_cases) -> None:
    case = product_cases["moss"]
    sentinel = RuntimeError(f"sentinel-{method_name}")
    case.runtime.failure_method = method_name
    case.runtime.failure = sentinel
    with pytest.raises(RuntimeError) as raised:
        getattr(case.adapter, method_name)("text", "voice", 1.0)
    assert raised.value is sentinel


def test_moss_getattr_preserves_non_protocol_runtime_extensions(product_cases) -> None:
    assert product_cases["moss"].adapter.moss_runtime_extension("value") == (
        "moss-extension",
        "value",
    )


def test_registry_owns_the_three_public_product_ids_and_unknown_semantics() -> None:
    registry = EngineRegistry()
    assert registry.public_model_ids == ("kokoro", "moss", "zipvoice")
    assert tuple(registry.resolve(model_id).canonical_id for model_id in registry.public_model_ids) == (
        "kokoro",
        "moss",
        "zipvoice",
    )
    unknown = registry.resolve("not-a-product")
    assert unknown.canonical_id == "not-a-product"
    with pytest.raises(HTTPException) as raised:
        registry.create_engine("not-a-product", _FakeConfig())
    assert raised.value.status_code == 404


def test_zipvoice_adapter_export_is_the_native_product_owner() -> None:
    assert ExportedZipVoiceEngine is ZipVoiceEngine


def test_capability_differences_are_explicit_and_registry_consistent(product_cases) -> None:
    cfg = _FakeConfig()
    registry = EngineRegistry()
    specs = {
        "kokoro": EngineSpec("kokoro", "Kokoro", "kokoro", "cpu"),
        "moss": EngineSpec("moss", "MOSS", "moss-tts-nano-onnx", "cpu"),
        "zipvoice": EngineSpec("zipvoice", "ZipVoice", "zipvoice-distill-onnx-int8", "cpu"),
    }
    capabilities = {
        product_id: product_cases[product_id].adapter.capabilities()
        for product_id in registry.public_model_ids
    }
    assert capabilities == {
        "kokoro": EngineCapabilities(
            modes=("preset_voice",),
            voice_clone_supported=False,
            speed_supported=True,
            stream_mode="segmented",
            sample_rate=24000,
            channels=1,
        ),
        "moss": EngineCapabilities(
            modes=("preset_voice", "voice_clone"),
            voice_clone_supported=True,
            speed_supported=False,
            stream_mode="native",
            provider_fallback=True,
            sample_rate=48000,
            channels=2,
        ),
        "zipvoice": EngineCapabilities(
            modes=("voice_clone", "saved_voice_profile"),
            voice_clone_supported=True,
            speed_supported=True,
            requires_prompt_audio=True,
            requires_prompt_text=True,
            supports_saved_voice_profiles=True,
            stream_mode="segmented",
            sample_rate=24000,
            channels=1,
        ),
    }
    for product_id, capability in capabilities.items():
        assert registry.capabilities_for(specs[product_id], cfg) == capability
    with pytest.raises(FrozenInstanceError):
        capabilities["kokoro"].sample_rate = 48000


def test_registry_is_the_only_static_product_capability_value_owner() -> None:
    root = Path(__file__).resolve().parents[2]
    owner_path = root / "src/kokoro_tts/engines/registry.py"
    adapter_paths = (
        root / "src/kokoro_tts/engines/adapters/kokoro.py",
        root / "src/kokoro_tts/engines/adapters/moss.py",
        root / "src/kokoro_tts/zipvoice/engine.py",
    )

    def capability_constructors(path: Path) -> list[ast.Call]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "EngineCapabilities"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "EngineCapabilities"
            )
        ]

    assert capability_constructors(owner_path)
    assert {
        path.relative_to(root).as_posix(): len(capability_constructors(path))
        for path in adapter_paths
        if capability_constructors(path)
    } == {}

    for path in adapter_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        capability_methods = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "capabilities"
        ]
        assert len(capability_methods) == 1
        assert any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "capabilities_for"
            for node in ast.walk(capability_methods[0])
        )


def _synthesize_common_input(case: _ProductCase) -> bytes:
    if case.product_id == "zipvoice":
        return case.adapter.synthesize(
            "shared text",
            "shared voice",
            1.25,
            prompt_audio_path="reference.wav",
            prompt_text="reference text",
            text_prepared=True,
            prompt_text_prepared=True,
        )
    return case.adapter.synthesize("shared text", "shared voice", 1.25)


@pytest.mark.parametrize("product_id", ("kokoro", "moss", "zipvoice"))
def test_nonstream_common_input_returns_bytes_without_nonempty_contract(
    product_id,
    product_cases,
) -> None:
    case = product_cases[product_id]
    result = _synthesize_common_input(case)
    assert isinstance(result, bytes)
    _, call = case.runtime.calls[-1]
    assert (call["text"], call["speed"]) == ("shared text", 1.25)
    if product_id == "zipvoice":
        # ZipVoice consumes the public voice at its adapter/condition boundary;
        # the native runtime receives the explicit reference context instead.
        assert call["voice"] == ""
        assert call["prompt_audio_path"] == "reference.wav"
        assert call["prompt_text"] == "reference text"
    else:
        assert call["voice"] == "shared voice"


@pytest.mark.parametrize("product_id", ("kokoro", "moss", "zipvoice"))
def test_nonstream_exception_identity_is_preserved(product_id, product_cases) -> None:
    case = product_cases[product_id]
    sentinel = RuntimeError(f"sentinel-{product_id}")
    case.runtime.failure_method = "synthesize"
    case.runtime.failure = sentinel
    with pytest.raises(RuntimeError) as raised:
        _synthesize_common_input(case)
    assert raised.value is sentinel


@pytest.mark.parametrize("product_id", ("kokoro", "moss", "zipvoice"))
def test_minimum_lifecycle_delegates_and_keeps_instance_state_isolated(
    product_id,
    product_cases,
) -> None:
    case = product_cases[product_id]
    assert case.adapter.is_loaded is False
    assert case.adapter.load() is case.adapter
    assert case.adapter.load() is case.adapter
    assert case.adapter.is_loaded is True
    case.adapter.unload()
    case.adapter.unload()
    assert case.adapter.is_loaded is False
    assert case.runtime.load_calls == 2
    assert case.runtime.unload_calls == 2
    other_runtimes = [item.runtime for key, item in product_cases.items() if key != product_id]
    assert all(runtime.load_calls == 0 and runtime.unload_calls == 0 for runtime in other_runtimes)


def test_engine_specific_extensions_do_not_expand_the_shared_protocol() -> None:
    protocol_members = vars(EngineAdapter)
    assert "prompt_audio_path" not in protocol_members
    assert "prompt_text" not in protocol_members
    assert "zipvoice_num_steps" not in protocol_members
    assert "moss_runtime_extension" not in protocol_members
    assert "prompt_audio_path" not in inspect.signature(KokoroAdapter.synthesize).parameters
    assert "prompt_audio_path" in inspect.signature(MossAdapter.synthesize).parameters
    assert "prompt_audio_path" in inspect.signature(ZipVoiceEngine.synthesize).parameters
    assert "prompt_text" in inspect.signature(ZipVoiceEngine.synthesize).parameters


def test_manager_worker_and_service_are_not_adapter_owners() -> None:
    required_methods = {
        "load",
        "unload",
        "capabilities",
        "metadata",
        "synthesize",
        "synthesize_array",
        "synthesize_stream",
    }
    for owner in (EngineManager, EngineProcessClient, SynthesisService):
        declared = {name for cls in owner.__mro__ for name in vars(cls)}
        assert not required_methods.issubset(declared)


def test_output_transcoding_remains_service_owned_not_engine_capability() -> None:
    response_source = inspect.getsource(SynthesisService.response_result)
    assert "transcode_wav_bytes" in response_source
    assert "encode_audio_segment" in response_source
    for invented_capability in ("supports_mp3", "supports_ogg", "supports_m4a"):
        assert invented_capability not in EngineCapabilities.__dataclass_fields__
