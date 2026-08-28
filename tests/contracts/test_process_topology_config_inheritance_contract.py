"""P2G-A process-topology and configuration-inheritance contracts.

The suite is hermetic: Uvicorn, FastAPI assembly, multiprocessing primitives,
engine factories, and lifecycle resources are replaced with spies/fakes.  It
freezes current ownership and propagation boundaries without starting a
server, child process, model, network request, or runtime-config reader.

Contract classifications:

* launcher topology, per-worker ownership, spawn, and explicit close:
  BEHAVIOR/OWNERSHIP CONTRACT;
* export inventory and child config consumption: STATIC OWNERSHIP CONTRACT;
* ambient environment inheritance: CURRENT-BEHAVIOR CHARACTERIZATION
  (not design endorsement).
"""

from __future__ import annotations

import ast
import hashlib
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from kokoro_tts import server
from kokoro_tts.config import TTSConfig
from kokoro_tts.engine_manager import EngineManager
from kokoro_tts.moss import process_worker as moss_process_worker
from kokoro_tts.service_state import ServiceState
from kokoro_tts.workers import EngineWorkerSpec, process_worker


pytestmark = pytest.mark.contract

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "kokoro_tts"
EXPECTED_EXPORT_HASH = (
    "13af027c56ae45ffbc4765c11d8f173adbd1359f0197d4c1700aaf57d6595945"
)
MODEL_SOURCE_EFFECTIVE_FIELDS = {
    "model_source_effective",
    "model_source_country",
    "model_source_hf_reachable",
    "model_source_modelscope_reachable",
}


def _contract_worker_factory(_config: object, _provider: str | None) -> object:
    """Top-level spawn-safe factory used by hermetic process-client contracts."""

    return object()


def _contract_worker_spec(
    engine_id: str = "kokoro",
    provider: str | None = None,
) -> EngineWorkerSpec:
    return EngineWorkerSpec(
        engine_id=engine_id,
        factory=_contract_worker_factory,
        requested_provider=provider,
    )


def _module_tree(relative: str) -> ast.Module:
    return ast.parse(
        (PACKAGE_ROOT / relative).read_text(encoding="utf-8"),
        filename=relative,
    )


def _definition(
    tree: ast.AST, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    )


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            names.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _declaration_export_mapping(
    declaration_name: str,
    constructor_name: str,
) -> dict[str, str]:
    tree = _module_tree("config_env_domain.py")
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == declaration_name
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, (ast.Tuple, ast.List))

    mapping: dict[str, str] = {}
    for declaration in assignment.value.elts:
        assert isinstance(declaration, ast.Call)
        assert isinstance(declaration.func, ast.Name)
        assert declaration.func.id == constructor_name
        env_name = ast.literal_eval(declaration.args[0])
        attr = ast.literal_eval(declaration.args[1])
        mapping[env_name] = attr
    return mapping


def _expanded_worker_export_mapping() -> tuple[dict[str, str], int, int]:
    """Expand only the two explicitly owned declaration projections."""

    tree = _module_tree("server.py")
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_WORKER_ENV_EXPORTS"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)

    mapping: dict[str, str] = {}
    literal_count = 0
    expansion_count = 0
    projection_mappings = {
        "MOSS_STREAM_BUDGET_ENV_DECLARATIONS": _declaration_export_mapping(
            "MOSS_STREAM_BUDGET_ENV_DECLARATIONS",
            "EnvFloatDeclaration",
        ),
        "UPDATE_CHECK_ENV_DECLARATIONS": _declaration_export_mapping(
            "UPDATE_CHECK_ENV_DECLARATIONS",
            "UpdateCheckEnvDeclaration",
        ),
    }
    seen_projections: set[str] = set()
    for key, value in zip(assignment.value.keys, assignment.value.values):
        if key is None:
            assert isinstance(value, ast.DictComp)
            assert len(value.generators) == 1
            generator = value.generators[0]
            assert isinstance(generator.target, ast.Name)
            assert generator.target.id == "declaration"
            assert generator.ifs == []
            assert generator.is_async == 0
            iterator = generator.iter
            assert isinstance(iterator, ast.Name)
            assert iterator.id in projection_mappings
            assert iterator.id not in seen_projections
            assert isinstance(value.key, ast.Attribute)
            assert isinstance(value.key.value, ast.Name)
            assert (value.key.value.id, value.key.attr) == (
                "declaration",
                "env_name",
            )
            assert isinstance(value.value, ast.Attribute)
            assert isinstance(value.value.value, ast.Name)
            assert (value.value.value.id, value.value.attr) == (
                "declaration",
                "attr",
            )
            declared_mapping = projection_mappings[iterator.id]
            mapping.update(declared_mapping)
            expansion_count += len(declared_mapping)
            seen_projections.add(iterator.id)
            continue
        mapping[ast.literal_eval(key)] = ast.literal_eval(value)
        literal_count += 1

    assert seen_projections == set(projection_mappings)
    assert mapping == server._WORKER_ENV_EXPORTS
    return mapping, literal_count, expansion_count


def _synthetic_config(tmp_path: Path, *, workers: int = 1) -> TTSConfig:
    """Build a config without invoking environment-backed default model lookup."""

    return TTSConfig(
        model_dir=tmp_path / f"models-{workers}",
        workers=workers,
        host="127.0.0.27",
        port=18267 + workers,
    )


class _FakeQueue:
    def __init__(self, events: list[object] | None = None, label: str = "queue"):
        self.events = events if events is not None else []
        self.label = label
        self.items: list[object] = []

    def put(self, item) -> None:
        self.items.append(item)
        self.events.append((self.label, "put", item))

    def put_nowait(self, item) -> None:
        self.items.append(item)
        self.events.append((self.label, "put_nowait", item))

    def get_nowait(self):
        self.events.append((self.label, "get_nowait"))
        raise queue.Empty

    def close(self) -> None:
        self.events.append((self.label, "close"))

    def join_thread(self) -> None:
        self.events.append((self.label, "join_thread"))


class _FakeValue:
    def __init__(self, value: int):
        self.value = value


class _FakeProcess:
    def __init__(self, *, events: list[object] | None = None, **kwargs):
        self.events = events if events is not None else []
        self.kwargs = kwargs
        self.started = False
        self.killed = False
        self.exitcode = 17
        self.pid = 267

    def start(self) -> None:
        self.started = True
        self.events.append("start")

    def is_alive(self) -> bool:
        return bool(self.started and not self.killed)

    def join(self, *, timeout: float) -> None:
        self.events.append(("join", timeout))

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.killed = True


class _FakeMultiprocessingContext:
    def __init__(self):
        self.events: list[object] = []
        self.queues: list[_FakeQueue] = []
        self.processes: list[_FakeProcess] = []

    def Value(self, typecode: str, value: int) -> _FakeValue:
        assert (typecode, value) == ("i", 0)
        return _FakeValue(value)

    def Queue(self) -> _FakeQueue:
        item = _FakeQueue(self.events, f"queue-{len(self.queues) + 1}")
        self.queues.append(item)
        return item

    def Process(self, **kwargs) -> _FakeProcess:
        item = _FakeProcess(events=self.events, **kwargs)
        self.processes.append(item)
        return item


class TestLauncherWorkerTopology:
    def test_single_worker_reuses_loaded_config_and_passes_app_object(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR CONTRACT: one launcher config object owns the one app."""

        cfg = _synthetic_config(tmp_path, workers=1)
        load_calls: list[object] = []
        app_calls: list[object] = []
        uvicorn_calls: list[tuple[object, dict[str, object]]] = []
        app = object()

        monkeypatch.setattr(
            server,
            "load_config",
            lambda: load_calls.append("load_config") or cfg,
        )
        monkeypatch.setattr(
            server,
            "create_app",
            lambda *, config: app_calls.append(config) or app,
        )
        monkeypatch.setattr(
            server,
            "_export_config_for_workers",
            lambda _cfg: pytest.fail("single-worker launch must not export"),
        )
        monkeypatch.setattr(server, "format_startup_banner", lambda _cfg: "banner")

        import uvicorn

        monkeypatch.setattr(
            uvicorn,
            "run",
            lambda target, **kwargs: uvicorn_calls.append((target, kwargs)),
        )

        server.run_server()

        assert load_calls == ["load_config"]
        assert app_calls == [cfg]
        assert app_calls[0] is cfg
        assert uvicorn_calls == [
            (
                app,
                {
                    "host": cfg.host,
                    "port": cfg.port,
                    "workers": 1,
                    "ws_max_size": cfg.websocket_max_message_bytes,
                    "access_log": cfg.access_log_enabled,
                },
            )
        ]

    def test_multiworker_exports_then_uses_canonical_factory(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR CONTRACT: Uvicorn workers receive no direct config object."""

        cfg = _synthetic_config(tmp_path, workers=3)
        events: list[object] = []

        monkeypatch.setattr(
            server,
            "load_config",
            lambda: events.append("load_config") or cfg,
        )
        monkeypatch.setattr(
            server,
            "_export_config_for_workers",
            lambda exported: events.append(("export", exported)),
        )
        monkeypatch.setattr(
            server,
            "create_app",
            lambda **_kwargs: pytest.fail("launcher must not build multiworker app"),
        )
        monkeypatch.setattr(server, "format_startup_banner", lambda _cfg: "banner")

        import uvicorn

        def fake_run(target, **kwargs):
            events.append(("uvicorn", target, kwargs))

        monkeypatch.setattr(uvicorn, "run", fake_run)

        server.run_server()

        assert events[0] == "load_config"
        assert events[1][0] == "export"
        assert events[1][1] is cfg
        assert events[2] == (
            "uvicorn",
            "kokoro_tts.server:create_app",
            {
                "factory": True,
                "host": cfg.host,
                "port": cfg.port,
                "workers": 3,
                "ws_max_size": cfg.websocket_max_message_bytes,
                "access_log": cfg.access_log_enabled,
            },
        )


class TestPerWebWorkerOwnership:
    def test_factory_calls_build_independent_config_manager_state_and_lifespan(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR CONTRACT using constructor spies; no lifespan is entered."""

        configs = [
            _synthetic_config(tmp_path / "first", workers=2),
            _synthetic_config(tmp_path / "second", workers=2),
        ]
        config_iterator = iter(configs)
        managers: list[object] = []
        states: list[object] = []

        class FakeManager:
            def __init__(self, config, *, initial_engine=None):
                self.config = config
                self.initial_engine = initial_engine
                managers.append(self)

        class FakeState:
            def __init__(self, config, engine, *, model_manager):
                self.cfg = config
                self.eng = engine
                self.model_manager = model_manager
                self.templates = None
                states.append(self)

            def as_service_extras_kwargs(self):
                return {}

        class FakePath:
            @property
            def parent(self):
                return self

            def __truediv__(self, _other):
                return self

            def exists(self):
                return False

        class FakeApp:
            def __init__(self, **kwargs):
                self.state = SimpleNamespace()
                self.lifespan = kwargs["lifespan"]

            def add_middleware(self, *_args, **_kwargs):
                return None

            def include_router(self, _router):
                return None

        import fastapi
        from kokoro_tts import service_extras

        monkeypatch.setattr(server, "load_config", lambda: next(config_iterator))
        monkeypatch.setattr(TTSConfig, "validate_security", lambda _self: None)
        monkeypatch.setattr(server, "EngineManager", FakeManager)
        monkeypatch.setattr(server, "ServiceState", FakeState)
        monkeypatch.setattr(server, "Path", lambda *_args: FakePath())
        monkeypatch.setattr(server, "make_verify_api_key", lambda _cfg: object())
        monkeypatch.setattr(fastapi, "FastAPI", FakeApp)
        monkeypatch.setattr(service_extras, "register_extra_routes", lambda **_kwargs: None)
        for name in (
            "create_auth_router",
            "create_status_router",
            "create_admin_router",
            "create_audio_router",
            "create_zipvoice_router",
            "create_ws_router",
        ):
            monkeypatch.setattr(server, name, lambda *_args, **_kwargs: object())

        first_app = server.create_app()
        second_app = server.create_app()
        first_state = first_app.state.angevoice
        second_state = second_app.state.angevoice

        assert first_state.cfg is configs[0]
        assert second_state.cfg is configs[1]
        assert first_state.cfg is not second_state.cfg
        assert first_state.model_manager is managers[0]
        assert second_state.model_manager is managers[1]
        assert managers[0] is not managers[1]
        assert first_state is states[0]
        assert second_state is states[1]
        assert first_state is not second_state
        assert first_app.lifespan is not second_app.lifespan

        first_state.cfg.default_speed = 1.67
        assert second_state.cfg.default_speed != 1.67

    def test_service_and_manager_are_instance_owners_not_module_singletons(self):
        """STATIC OWNERSHIP CONTRACT for the actual EngineManager owner."""

        tree = _module_tree("server.py")
        module_level_values: list[ast.AST] = []
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                module_level_values.append(statement.value)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                module_level_values.append(statement.value)
            elif isinstance(statement, ast.Expr) and not isinstance(
                statement.value, ast.Constant
            ):
                module_level_values.append(statement.value)
        module_level_calls = {
            node.func.id
            for value in module_level_values
            for node in ast.walk(value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "EngineManager" not in module_level_calls
        assert "ServiceState" not in module_level_calls
        assert EngineManager.__module__ == "kokoro_tts.engine_manager"
        assert ServiceState.__module__ == "kokoro_tts.service_state"


class TestWorkerEnvironmentExportInventory:
    def test_canonical_export_inventory_count_prefixes_hash_and_entries(self):
        """STATIC OWNERSHIP CONTRACT: structured AST expansion, not literal_eval."""

        mapping, literal_count, expansion_count = _expanded_worker_export_mapping()
        assert (literal_count, expansion_count, len(mapping)) == (149, 8, 157)
        assert {
            prefix: sum(name.startswith(f"{prefix}_") for name in mapping)
            for prefix in ("ANGEVOICE", "KOKORO", "MOSS", "ZIPVOICE")
        } == {
            "ANGEVOICE": 45,
            "KOKORO": 39,
            "MOSS": 57,
            "ZIPVOICE": 16,
        }
        canonical = "\n".join(
            f"{name}={mapping[name]}" for name in sorted(mapping)
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == EXPECTED_EXPORT_HASH
        assert {
            "ANGEVOICE_MODEL_SOURCE": "model_source",
            "ANGEVOICE_RUNTIME_CONFIG_FILE": "runtime_config_file",
            "ANGEVOICE_STARTUP_PRELOAD_ENABLED": "startup_preload_enabled",
            "MOSS_PROCESS_ISOLATION_ENABLED": "moss_process_isolation_enabled",
            "KOKORO_PROCESS_ISOLATION_ENABLED": "kokoro_process_isolation_enabled",
            "ZIPVOICE_PROCESS_ISOLATION_ENABLED": "zipvoice_process_isolation_enabled",
            "ANGEVOICE_UPDATE_CHECK_ENABLED": "update_check_enabled",
            "ANGEVOICE_UPDATE_REPOSITORY": "update_repository",
            "ANGEVOICE_UPDATE_CHECK_TIMEOUT_SECONDS": (
                "update_check_timeout_seconds"
            ),
            "ANGEVOICE_UPDATE_CHECK_CACHE_SECONDS": "update_check_cache_seconds",
        }.items() <= mapping.items()

    def test_process_fields_and_effective_model_source_cache_are_not_explicit_exports(
        self,
    ):
        """STATIC OWNERSHIP + CURRENT-BEHAVIOR CHARACTERIZATION."""

        mapping, _, _ = _expanded_worker_export_mapping()
        assert "model_source" in mapping.values()
        assert not {"host", "port", "workers"} & set(mapping.values())
        assert not MODEL_SOURCE_EFFECTIVE_FIELDS & set(mapping.values())

    def test_direct_projection_inventory_and_secret_material_exclusions(self):
        """STATIC OWNERSHIP CONTRACT for assignments outside the 157-map."""

        function = _definition(_module_tree("server.py"), "_export_config_for_workers")
        direct_keys = {
            ast.literal_eval(node.slice)
            for node in ast.walk(function)
            if isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
            and isinstance(node.slice, ast.Constant)
        }
        assert direct_keys == {
            "KOKORO_CORS_ORIGINS",
            "MOSS_MODEL_DIR",
            "MOSS_AUDIO_TOKENIZER_MODEL_DIR",
            "MOSS_TTS_NANO_PATH",
            "MOSS_PROMPT_AUDIO_PATH",
            "ZIPVOICE_MODEL_ROOT",
            "ZIPVOICE_DISTILL_DIR",
            "ZIPVOICE_VOCOS_DIR",
            "ZIPVOICE_PROFILES_DIR",
            "ZIPVOICE_REPO_PATH",
        }
        assert "KOKORO_API_KEY" not in direct_keys
        assert "KOKORO_AUTO_API_KEY" not in direct_keys
        assert server._WORKER_ENV_EXPORTS["ANGEVOICE_CREDENTIALS_DIR"] == (
            "credentials_dir"
        )


class TestWorkerEnvironmentProjection:
    def test_projection_serializes_declared_values_and_preserves_ambient_environment(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR CONTRACT plus non-scrubbed environment characterization."""

        cfg = _synthetic_config(tmp_path)
        cfg.cache_enabled = False
        cfg.batch_max_items = 7
        cfg.default_speed = 1.25
        cfg.enabled_models = ["kokoro", "moss"]
        cfg.credentials_dir = tmp_path / "credentials"
        cfg.cors_origins = ["https://one.invalid", "https://two.invalid"]
        cfg.moss_model_dir = None
        cfg.moss_audio_tokenizer_model_dir = tmp_path / "moss-tokenizer"
        cfg.moss_repo_path = tmp_path / "moss-repo"
        cfg.moss_prompt_audio_path = None
        cfg.zipvoice_model_root = tmp_path / "zipvoice"
        cfg.zipvoice_distill_dir = tmp_path / "zipvoice-distill"
        cfg.zipvoice_vocos_dir = tmp_path / "zipvoice-vocos"
        cfg.zipvoice_profiles_dir = tmp_path / "zipvoice-profiles"
        cfg.zipvoice_repo_path = None

        environment = {
            "UNDECLARED_AMBIENT": "preserved",
            "MOSS_MODEL_DIR": "operator-owned-existing-value",
            "MOSS_PROMPT_AUDIO_PATH": "operator-owned-prompt",
            "ZIPVOICE_REPO_PATH": "operator-owned-repository",
        }
        monkeypatch.setattr(server.os, "environ", environment)

        server._export_config_for_workers(cfg)

        assert environment["KOKORO_CACHE_ENABLED"] == "false"
        assert environment["KOKORO_BATCH_MAX_ITEMS"] == "7"
        assert environment["KOKORO_DEFAULT_SPEED"] == "1.25"
        assert environment["ANGEVOICE_ENABLED_MODELS"] == "kokoro,moss"
        assert environment["KOKORO_MODEL_DIR"] == str(cfg.model_dir)
        assert environment["ANGEVOICE_CREDENTIALS_DIR"] == str(cfg.credentials_dir)
        assert environment["KOKORO_CORS_ORIGINS"] == (
            "https://one.invalid,https://two.invalid"
        )
        assert environment["MOSS_AUDIO_TOKENIZER_MODEL_DIR"] == str(
            cfg.moss_audio_tokenizer_model_dir
        )
        assert environment["MOSS_TTS_NANO_PATH"] == str(cfg.moss_repo_path)
        assert environment["ZIPVOICE_MODEL_ROOT"] == str(cfg.zipvoice_model_root)
        assert environment["ZIPVOICE_DISTILL_DIR"] == str(cfg.zipvoice_distill_dir)
        assert environment["ZIPVOICE_VOCOS_DIR"] == str(cfg.zipvoice_vocos_dir)
        assert environment["ZIPVOICE_PROFILES_DIR"] == str(
            cfg.zipvoice_profiles_dir
        )

        # Optional None values are skipped; pre-existing ambient values are not
        # removed.  This characterizes current behavior rather than endorsing it.
        assert environment["MOSS_MODEL_DIR"] == "operator-owned-existing-value"
        assert environment["MOSS_PROMPT_AUDIO_PATH"] == "operator-owned-prompt"
        assert environment["ZIPVOICE_REPO_PATH"] == "operator-owned-repository"
        assert environment["UNDECLARED_AMBIENT"] == "preserved"
        assert "KOKORO_API_KEY" not in environment
        assert "KOKORO_AUTO_API_KEY" not in environment


class TestEngineChildSpawnBoundary:
    def test_spawn_passes_full_config_and_current_process_arguments(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR/OWNERSHIP CONTRACT without starting a real process."""

        context = _FakeMultiprocessingContext()
        requested_contexts: list[str] = []

        def fake_get_context(method: str):
            requested_contexts.append(method)
            return context

        monkeypatch.setattr(process_worker.mp, "get_context", fake_get_context)
        cfg = _synthetic_config(tmp_path)
        cfg.model_source = "modelscope"
        cfg.model_source_effective = "huggingface"
        cfg.model_source_country = "US"
        client = process_worker.EngineProcessClient(
            config=cfg,
            spec=_contract_worker_spec("kokoro", "cpu"),
        )

        client.start()

        assert requested_contexts == ["spawn"]
        assert len(context.processes) == 1
        process = context.processes[0]
        assert process.started is True
        assert process.kwargs["target"] is process_worker._worker_main
        assert process.kwargs["name"] == "angevoice-kokoro-worker"
        assert process.kwargs["daemon"] is True
        assert set(process.kwargs) == {"target", "args", "name", "daemon"}
        args = process.kwargs["args"]
        assert args == (
            cfg,
            client.spec,
            context.queues[0],
            context.queues[1],
            client._cancel_flag,
        )
        assert args[0] is cfg
        assert args[0].model_source == "modelscope"
        assert args[0].model_source_effective == "huggingface"
        assert args[0].model_source_country == "US"

        # An already-live child is not recreated after a parent mutation.  The
        # real spawn boundary serializes args at Process.start; the fake only
        # freezes project ownership and intentionally makes no pickle identity claim.
        cfg.model_source = "offline"
        client.start()
        assert len(context.processes) == 1

    def test_start_has_no_config_reload_runtime_reader_or_scrubbed_env(self):
        """STATIC OWNERSHIP + CURRENT-BEHAVIOR CHARACTERIZATION."""

        start = _definition(
            _module_tree("workers/process_worker.py"),
            "start",
        )
        names = _call_names(start)
        assert not {
            "load_config",
            "load_runtime_config",
            "read_runtime_config",
            "apply_env",
            "TTSConfig",
        } & names
        process_call = next(
            node
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Process"
        )
        assert {keyword.arg for keyword in process_call.keywords} == {
            "target",
            "args",
            "name",
            "daemon",
        }
        assert "env" not in {keyword.arg for keyword in process_call.keywords}


class TestWorkerMainConfigConsumption:
    def test_child_entry_consumes_serialized_config_without_reassembly(self):
        """STATIC OWNERSHIP CONTRACT for the engine-child entry point."""

        worker_main = _definition(
            _module_tree("workers/process_worker.py"),
            "_worker_main",
        )
        names = _call_names(worker_main)
        assert not {
            "load_config",
            "load_runtime_config",
            "read_runtime_config",
            "apply_env",
            "TTSConfig",
            "validate_security",
            "normalize",
        } & names
        factory_call = next(
            node
            for node in ast.walk(worker_main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_worker_engine"
        )
        assert [
            argument.id if isinstance(argument, ast.Name) else None
            for argument in factory_call.args
        ] == ["config", "spec"]

    def test_requested_and_effective_model_source_have_distinct_process_boundaries(
        self,
    ):
        """STATIC OWNERSHIP + CURRENT-BEHAVIOR CHARACTERIZATION."""

        mapping, _, _ = _expanded_worker_export_mapping()
        assert mapping["ANGEVOICE_MODEL_SOURCE"] == "model_source"
        assert not MODEL_SOURCE_EFFECTIVE_FIELDS & set(mapping.values())

        start = _definition(
            _module_tree("workers/process_worker.py"),
            "start",
        )
        process_call = next(
            node
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Process"
        )
        args_keyword = next(
            keyword for keyword in process_call.keywords if keyword.arg == "args"
        )
        assert isinstance(args_keyword.value, ast.Tuple)
        assert isinstance(args_keyword.value.elts[0], ast.Attribute)
        assert args_keyword.value.elts[0].attr == "config"


class TestEngineWorkerSpecOwnership:
    def test_worker_constructs_only_from_the_passed_spec(self):
        """BEHAVIOR CONTRACT without importing or constructing real engines."""

        config = object()
        calls: list[tuple[object, str | None]] = []

        def factory(cfg, provider):
            calls.append((cfg, provider))
            return "synthetic-engine"

        spec = EngineWorkerSpec("synthetic", factory, "synthetic-provider")
        assert process_worker.create_worker_engine(config, spec) == "synthetic-engine"
        assert calls == [(config, "synthetic-provider")]

    def test_worker_modules_have_no_concrete_engine_or_dynamic_loader_owner(self):
        """STATIC OWNERSHIP CONTRACT."""

        worker_root = PACKAGE_ROOT / "workers"
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(worker_root.glob("*.py"))
        )
        for forbidden in (
            "from ..engine import",
            "from ..moss_engine import",
            "from ..zipvoice.engine import",
            "importlib",
            "__import__",
        ):
            assert forbidden not in sources
        assert not (worker_root / "factories.py").exists()


class TestMossProcessCompatibilityFacade:
    def test_moss_process_client_is_only_a_deprecated_compatibility_subclass(self):
        """STATIC OWNERSHIP CONTRACT: there is one canonical process client."""

        tree = _module_tree("moss/process_worker.py")
        moss_class = _definition(tree, "MossProcessClient")
        assert isinstance(moss_class, ast.ClassDef)
        assert len(moss_class.bases) == 1
        assert isinstance(moss_class.bases[0], ast.Name)
        assert moss_class.bases[0].id == "_EngineProcessClient"
        assert issubclass(
            moss_process_worker.MossProcessClient,
            process_worker.EngineProcessClient,
        )
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_worker_main"
            for node in tree.body
        )
        assert "Process" not in _call_names(tree)
        assert "warnings.warn" in (PACKAGE_ROOT / "moss/process_worker.py").read_text(
            encoding="utf-8"
        )


class TestEngineChildLifecycleOwnership:
    def test_explicit_close_owns_shutdown_terminate_kill_and_queue_cleanup(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR CONTRACT with an always-stuck fake until kill."""

        context = _FakeMultiprocessingContext()
        monkeypatch.setattr(
            process_worker.mp,
            "get_context",
            lambda method: context if method == "spawn" else pytest.fail(method),
        )
        cfg = _synthetic_config(tmp_path)
        cfg.engine_process_kill_grace_seconds = 0.25
        client = process_worker.EngineProcessClient(
            config=cfg,
            spec=_contract_worker_spec(),
        )
        command_queue = _FakeQueue(context.events, "command")
        result_queue = _FakeQueue(context.events, "result")
        stuck = _FakeProcess(events=context.events)
        stuck.started = True
        client._command_queue = command_queue
        client._result_queue = result_queue
        client._process = stuck
        client._loaded = True

        client.close()

        assert context.events[0][0:2] == ("command", "put_nowait")
        assert context.events[0][2][1] == "shutdown"
        assert context.events[1:] == [
            ("join", 0.25),
            "terminate",
            ("join", 0.25),
            "kill",
            ("join", 0.25),
            ("result", "get_nowait"),
            ("command", "close"),
            ("command", "join_thread"),
            ("result", "close"),
            ("result", "join_thread"),
        ]
        assert client._process is None
        assert client._command_queue is None
        assert client._result_queue is None
        assert client._loaded is False

    def test_timeout_and_dead_child_health_are_owned_by_process_client(
        self, monkeypatch, tmp_path
    ):
        """BEHAVIOR/STATIC OWNERSHIP CONTRACT; no retry or backoff promise."""

        wait_for = _definition(
            _module_tree("workers/process_worker.py"),
            "_wait_for",
        )
        close_calls = [
            node
            for node in ast.walk(wait_for)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
        ]
        assert any(
            any(
                keyword.arg == "kill"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in call.keywords
            )
            for call in close_calls
        )
        assert "EngineProcessTimeoutError" in _call_names(wait_for)

        context = _FakeMultiprocessingContext()
        monkeypatch.setattr(process_worker.mp, "get_context", lambda _method: context)
        client = process_worker.EngineProcessClient(
            config=_synthetic_config(tmp_path),
            spec=_contract_worker_spec(),
        )
        dead = _FakeProcess()
        dead.started = False
        dead.exitcode = 23
        client._process = dead
        client._loaded = True

        with pytest.raises(RuntimeError, match="退出码：23"):
            client._raise_if_worker_exited()
        assert client.is_healthy is False
        assert client.last_exit_reason == "worker 异常退出，退出码：23"


class TestAppLifespanShutdownBoundary:
    def test_lifespan_delegates_to_the_close_all_child_barrier(self):
        """STATIC OWNERSHIP CONTRACT: lifespan orchestrates one manager primitive."""

        create_app = _definition(_module_tree("server.py"), "create_app")
        lifespan = _definition(create_app, "lifespan")
        calls = _call_names(lifespan)
        assert "close_all" in calls
        assert not {
            "close",
            "shutdown",
            "terminate",
            "kill",
            "unload",
            "drop_model",
        } & calls
