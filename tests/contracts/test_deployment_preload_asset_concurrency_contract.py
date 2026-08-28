"""P2G-C deployment, preload, and asset-concurrency contracts.

Deployment inputs are parsed from checked-in formal files only.  No Docker,
fnOS, Uvicorn, process, model, network, credential, cache, or volume is used.
Preload behavior uses constructor spies and invokes only the captured lifespan.

Classifications include DEPLOYMENT CONTRACT, BEHAVIOR/OWNERSHIP CONTRACT, and
CURRENT-BEHAVIOR/TOPOLOGY CHARACTERIZATION (not design endorsement).
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from kokoro_tts import server
from kokoro_tts.config import TTSConfig


pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "kokoro_tts"
ALLOWED_ENV_KEYS = {
    "KOKORO_HOST",
    "KOKORO_PORT",
    "KOKORO_WORKERS",
    "ANGEVOICE_STARTUP_PRELOAD_ENABLED",
    "ANGEVOICE_STARTUP_PRELOAD_MODEL",
    "ANGEVOICE_MODEL_SOURCE",
    "ANGEVOICE_RUNTIME_CONFIG_FILE",
    "KOKORO_PROCESS_ISOLATION_ENABLED",
    "MOSS_PROCESS_ISOLATION_ENABLED",
    "ZIPVOICE_PROCESS_ISOLATION_ENABLED",
    "KOKORO_MODEL_DIR",
    "MOSS_MODEL_DIR",
    "MOSS_AUDIO_TOKENIZER_MODEL_DIR",
    "ZIPVOICE_MODEL_ROOT",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "MODELSCOPE_CACHE",
}


def _module_tree(relative: str) -> ast.Module:
    return ast.parse(
        (PACKAGE_ROOT / relative).read_text(encoding="utf-8"),
        filename=relative,
    )


def _definition(tree: ast.AST, name: str):
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name == name
    )


def _call_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
        if isinstance(call.func, ast.Name):
            result.add(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            result.add(call.func.attr)
    return result


def _allowlisted_env(relative: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (REPO_ROOT / relative).read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key in ALLOWED_ENV_KEYS:
            values[key] = value
    return values


def _service_block(relative: str, service: str) -> list[str]:
    """Return one exact two-space-indented Compose service block."""

    lines = (REPO_ROOT / relative).read_text(encoding="utf-8").splitlines()
    marker = f"  {service}:"
    start = lines.index(marker)
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
            break
        block.append(line)
    return block


def _mapping_value(block: list[str], key: str, *, indent: int = 4) -> str | None:
    prefix = " " * indent + key + ":"
    matches = [line[len(prefix) :].strip().strip('"') for line in block if line.startswith(prefix)]
    assert len(matches) <= 1
    return matches[0] if matches else None


def _section_lines(block: list[str], section: str, *, indent: int = 4) -> list[str]:
    prefix = " " * indent + section + ":"
    start = next(index for index, line in enumerate(block) if line.startswith(prefix))
    result: list[str] = []
    for line in block[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        result.append(line)
    return result


def _environment(block: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _section_lines(block, "environment"):
        match = re.match(r"^\s{6}([A-Z0-9_]+):\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"')
    return values


def _list_values(block: list[str], section: str) -> list[str]:
    inline = _mapping_value(block, section)
    if inline:
        parsed = ast.literal_eval(inline)
        assert isinstance(parsed, list)
        assert all(isinstance(value, str) for value in parsed)
        return parsed
    return [
        match.group(1).strip().strip('"')
        for line in _section_lines(block, section)
        if (match := re.match(r"^\s{6}-\s*(.+)$", line))
    ]


def _synthetic_config(tmp_path: Path, *, preload: bool) -> TTSConfig:
    return TTSConfig(
        model_dir=tmp_path / "models",
        moss_model_dir=tmp_path / "moss",
        moss_audio_tokenizer_model_dir=tmp_path / "moss-tokenizer",
        zipvoice_model_root=tmp_path / "zipvoice",
        startup_preload_enabled=preload,
        startup_preload_model="kokoro",
        default_model="kokoro",
    )


def _create_spied_app(monkeypatch, cfg, events: list[object], *, warm_error=None):
    managers: list[object] = []

    class FakeManager:
        def __init__(self, config, *, initial_engine=None):
            self.config = config
            self.initial_engine = initial_engine
            managers.append(self)

        def switch_model(self, model, *, load):
            events.append(("switch", model, load))

        def list_specs(self):
            events.append(("list_specs",))
            return [SimpleNamespace(id="kokoro")]

        def warm_model(self, model):
            events.append(("warm", model))
            if warm_error is not None:
                raise warm_error

        def current_snapshot(self):
            events.append(("snapshot",))
            return {"id": "kokoro"}

        def close_all(self):
            events.append(("close_all",))
            return True

    class FakeState:
        def __init__(self, config, engine, *, model_manager):
            self.cfg = config
            self.eng = engine
            self.model_manager = model_manager
            self.templates = None

        def as_service_extras_kwargs(self):
            return {}

    class MissingStaticPath:
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

    monkeypatch.setattr(server, "load_config", lambda: cfg)
    monkeypatch.setattr(TTSConfig, "validate_security", lambda _self: None)
    monkeypatch.setattr(server, "EngineManager", FakeManager)
    monkeypatch.setattr(server, "ServiceState", FakeState)
    monkeypatch.setattr(server, "Path", lambda *_args: MissingStaticPath())
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
    return server.create_app(), managers[0]


class TestDirectCliDeploymentDefaults:
    def test_console_scripts_and_application_defaults(self, tmp_path):
        """DEPLOYMENT CONTRACT without invoking CLI parsing."""

        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'angevoice = "kokoro_tts.cli:main"' in pyproject
        assert 'kokoro-tts = "kokoro_tts.cli:main"' in pyproject
        cfg = _synthetic_config(tmp_path, preload=False)
        assert (cfg.host, cfg.port, cfg.workers) == ("0.0.0.0", 8000, 1)
        assert cfg.startup_preload_enabled is False
        assert cfg.model_source == "auto"
        assert cfg.runtime_config_file == Path("/app/config/runtime-config.json")

    def test_cli_serve_exposes_worker_override_without_prefork_owner(self):
        """STATIC OWNERSHIP CONTRACT."""

        source = (PACKAGE_ROOT / "cli.py").read_text(encoding="utf-8")
        assert 'sub.add_parser("serve"' in source
        assert 'serve_p.add_argument("--workers", type=int, default=None' in source
        assert "workers=args.workers" in source
        assert "gunicorn" not in source.lower()
        assert "prefork" not in source.lower()
        assert "KOKORO_WORKERS" in (
            PACKAGE_ROOT / "config_env.py"
        ).read_text(encoding="utf-8")


class TestDockerEntrypointAndFormalDefaults:
    def test_entrypoint_execs_application_without_worker_or_restart_loop(self):
        """DEPLOYMENT CONTRACT."""

        source = (REPO_ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
        executable = [
            line.strip()
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert executable[-1] == "exec angevoice serve"
        assert "--workers" not in source
        assert "gunicorn" not in source.lower()
        assert "supervisor" not in source.lower()
        assert not any(re.match(r"^(while|until)\b", line) for line in executable)
        for cache_key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "MODELSCOPE_CACHE"):
            assert f"export {cache_key}=" in source

    def test_image_dockerfiles_delegate_to_the_common_entrypoint(self):
        for relative in (
            "docker/cpu/Dockerfile",
            "docker/gpu/Dockerfile",
            "docker/legacy-gpu/Dockerfile",
        ):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            assert 'ENTRYPOINT ["/entrypoint.sh"]' in source
            assert "gunicorn" not in source.lower()
            assert "supervisor" not in source.lower()

    def test_formal_env_defaults_are_allowlisted_and_exact(self):
        values = _allowlisted_env("docker/angevoice.env")
        assert values == {
            "KOKORO_HOST": "0.0.0.0",
            "KOKORO_PORT": "8000",
            "KOKORO_WORKERS": "1",
            "ANGEVOICE_RUNTIME_CONFIG_FILE": "/app/config/runtime-config.json",
            "KOKORO_MODEL_DIR": "/app/models/models--hexgrad--Kokoro-82M-v1.1-zh",
            "MOSS_MODEL_DIR": "/app/models/MOSS-TTS-Nano-100M-ONNX",
            "MOSS_AUDIO_TOKENIZER_MODEL_DIR": (
                "/app/models/MOSS-Audio-Tokenizer-Nano-ONNX"
            ),
            "HUGGINGFACE_HUB_CACHE": "/app/models",
            "HF_HOME": "/app/models/.hf",
            "MODELSCOPE_CACHE": "/app/models/modelscope-cache",
            "ANGEVOICE_MODEL_SOURCE": "auto",
            "MOSS_PROCESS_ISOLATION_ENABLED": "true",
            "KOKORO_PROCESS_ISOLATION_ENABLED": "true",
            "ZIPVOICE_PROCESS_ISOLATION_ENABLED": "true",
            "ANGEVOICE_STARTUP_PRELOAD_ENABLED": "false",
            "ANGEVOICE_STARTUP_PRELOAD_MODEL": "kokoro",
            "ZIPVOICE_MODEL_ROOT": "/app/models/zipvoice",
        }


class TestComposeDeploymentProfiles:
    @pytest.mark.parametrize(
        ("relative", "service", "port"),
        [
            ("docker/cpu/docker-compose.yml", "angevoice-cpu", "8100:8000"),
            ("docker/gpu/docker-compose.yml", "angevoice-gpu", "8101:8000"),
            (
                "docker/legacy-gpu/docker-compose.yml",
                "angevoice-legacy-gpu",
                "8102:8000",
            ),
        ],
    )
    def test_formal_compose_ports_restart_volumes_and_no_replica_command(
        self, relative, service, port
    ):
        """DEPLOYMENT CONTRACT using an indentation-aware service parser."""

        block = _service_block(relative, service)
        assert port in _list_values(block, "ports")
        assert _mapping_value(block, "restart") == "always"
        assert _mapping_value(block, "command") is None
        assert not any(re.match(r"^\s+replicas:", line) for line in block)
        environment = _environment(block)
        assert "KOKORO_WORKERS" not in environment
        assert environment["ANGEVOICE_STARTUP_PRELOAD_ENABLED"] == "false"
        assert environment["KOKORO_PROCESS_ISOLATION_ENABLED"] == "true"
        assert environment["ZIPVOICE_PROCESS_ISOLATION_ENABLED"] == "true"
        volumes = _list_values(block, "volumes")
        for destination in (
            ":/app/models",
            ":/app/prompts",
            ":/app/outputs",
            ":/app/credentials",
            ":/app/config",
            ":/app/logs",
        ):
            assert any(destination in volume for volume in volumes)

    def test_legacy_moss_cuda_file_is_an_override_not_full_service(self):
        block = _service_block(
            "docker/legacy-gpu/docker-compose.moss-cuda.yml",
            "angevoice-legacy-gpu",
        )
        assert _environment(block)["MOSS_EXECUTION_PROVIDER"] == "cuda"
        assert _mapping_value(block, "image") is None
        assert _mapping_value(block, "ports") is None
        assert _mapping_value(block, "restart") is None


class TestFnosDeploymentDefaults:
    @pytest.mark.parametrize(
        ("service", "profile"),
        [
            ("angevoice-cpu", "cpu"),
            ("angevoice-gpu", "gpu"),
            ("angevoice-legacy-gpu", "legacy-gpu"),
        ],
    )
    def test_fnos_profiles_override_env_defaults_and_persist_paths(
        self, service, profile
    ):
        """DEPLOYMENT CONTRACT."""

        block = _service_block(
            "packaging/fnos/AngeVoice/app/docker/docker-compose.yaml", service
        )
        assert _list_values(block, "profiles") == [profile]
        assert "${wizard_http_port:-8101}:8000" in _list_values(block, "ports")
        assert _mapping_value(block, "restart") == "unless-stopped"
        environment = _environment(block)
        assert environment["ANGEVOICE_MODEL_SOURCE"] == (
            "${wizard_model_source:-modelscope}"
        )
        assert environment["ANGEVOICE_STARTUP_PRELOAD_ENABLED"] == "false"
        assert environment["KOKORO_PROCESS_ISOLATION_ENABLED"] == "true"
        assert environment["MOSS_PROCESS_ISOLATION_ENABLED"] == "true"
        assert environment["ZIPVOICE_PROCESS_ISOLATION_ENABLED"] == "true"
        assert environment["ANGEVOICE_RUNTIME_CONFIG_FILE"] == (
            "/app/config/runtime-config.json"
        )
        volumes = _list_values(block, "volumes")
        for destination in (
            ":/app/models",
            ":/app/prompts",
            ":/app/outputs",
            ":/app/credentials",
            ":/app/config",
            ":/app/logs",
        ):
            assert any(destination in volume for volume in volumes)
        assert not any(re.match(r"^\s+replicas:", line) for line in block)

    def test_fnos_wizard_default_is_modelscope_over_general_env_auto(self):
        formal_env = _allowlisted_env(
            "packaging/fnos/AngeVoice/app/docker/angevoice.env"
        )
        assert formal_env["ANGEVOICE_MODEL_SOURCE"] == "auto"
        for relative in (
            "packaging/fnos/AngeVoice/wizard/config",
            "packaging/fnos/AngeVoice/wizard/install",
        ):
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            marker = source.index('"field": "wizard_model_source"')
            section = source[marker : marker + 220]
            assert '"initValue": "modelscope"' in section

    def test_fnos_cmd_owners_are_container_actions_not_app_reload(self):
        main = (REPO_ROOT / "packaging/fnos/AngeVoice/cmd/main").read_text(
            encoding="utf-8"
        )
        callback = (
            REPO_ROOT / "packaging/fnos/AngeVoice/cmd/config_callback"
        ).read_text(encoding="utf-8")
        assert "docker inspect" in main
        assert "start|stop) exit 0" in main
        assert "COMPOSE_PROFILES" in callback
        assert "load_config" not in main + callback
        assert "load_runtime_config" not in main + callback


class TestStartupPreloadLifecycle:
    def test_preload_is_synchronous_before_yield_and_stops_after_exit(
        self, monkeypatch, tmp_path
    ):
        events: list[object] = []
        app, _manager = _create_spied_app(
            monkeypatch, _synthetic_config(tmp_path, preload=True), events
        )

        async def run_lifespan():
            async with app.lifespan(app):
                events.append(("yielded",))

        asyncio.run(run_lifespan())
        assert events == [
            ("switch", "kokoro", False),
            ("list_specs",),
            ("warm", "kokoro"),
            ("snapshot",),
            ("yielded",),
            ("close_all",),
        ]

    def test_preload_failure_prevents_this_worker_yield(self, monkeypatch, tmp_path):
        events: list[object] = []
        app, _manager = _create_spied_app(
            monkeypatch,
            _synthetic_config(tmp_path, preload=True),
            events,
            warm_error=RuntimeError("synthetic preload failure"),
        )

        async def run_lifespan():
            async with app.lifespan(app):
                events.append(("yielded",))

        with pytest.raises(RuntimeError, match="synthetic preload failure"):
            asyncio.run(run_lifespan())
        assert ("yielded",) not in events
        assert events[-1] == ("close_all",)

    def test_preload_runs_for_each_independent_app_factory_invocation(
        self, monkeypatch, tmp_path
    ):
        first_events: list[object] = []
        first_app, first_manager = _create_spied_app(
            monkeypatch,
            _synthetic_config(tmp_path, preload=True),
            first_events,
        )
        second_events: list[object] = []
        second_app, second_manager = _create_spied_app(
            monkeypatch,
            _synthetic_config(tmp_path, preload=True),
            second_events,
        )

        async def run_lifespan(app):
            async with app.lifespan(app):
                pass

        asyncio.run(run_lifespan(first_app))
        asyncio.run(run_lifespan(second_app))
        assert first_manager is not second_manager
        assert first_events.count(("warm", "kokoro")) == 1
        assert second_events.count(("warm", "kokoro")) == 1

    def test_preload_disabled_yields_without_warm_model(self, monkeypatch, tmp_path):
        events: list[object] = []
        app, _manager = _create_spied_app(
            monkeypatch, _synthetic_config(tmp_path, preload=False), events
        )

        async def run_lifespan():
            async with app.lifespan(app):
                events.append(("yielded",))

        asyncio.run(run_lifespan())
        assert not any(event[0] == "warm" for event in events)
        assert events.index(("yielded",)) < events.index(("close_all",))

    def test_first_request_model_use_is_owned_by_the_current_manager(self):
        """STATIC OWNERSHIP CONTRACT for the preload-disabled request path."""

        for relative, owner in (
            ("services/synthesis_service.py", "response_result"),
            ("services/streaming_service.py", "iter_frames"),
        ):
            method = _definition(_module_tree(relative), owner)
            assert "self.state.model_manager.borrow" in ast.unparse(method)

    def test_lifespan_has_no_background_preload_task_or_exception_swallow(self):
        create_app = _definition(_module_tree("server.py"), "create_app")
        lifespan = _definition(create_app, "lifespan")
        assert "warm_model" in _call_names(lifespan)
        assert not {"create_task", "run_in_executor", "to_thread"} & _call_names(
            lifespan
        )
        warm = next(
            node
            for node in ast.walk(lifespan)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warm_model"
        )
        yield_node = next(
            node for node in ast.walk(lifespan) if isinstance(node, ast.Yield)
        )
        assert warm.lineno < yield_node.lineno


class TestSameAndCrossWorkerCoordination:
    def test_engine_manager_uses_process_local_lock_and_single_instance_map(self):
        """STATIC OWNERSHIP CONTRACT."""

        tree = _module_tree("engine_manager.py")
        init = _definition(tree, "__init__")
        get_engine = _definition(tree, "get_engine")
        init_source = ast.unparse(init)
        get_source = ast.unparse(get_engine)
        assert "threading.RLock()" in init_source
        assert "self._engines" in init_source
        assert "with self._lock" in get_source
        assert "self._engines.get(target_id)" in get_source
        assert "self._engines[target_id] = engine" in get_source

    def test_independent_worker_configs_can_share_all_asset_roots(self, tmp_path):
        """STATIC/TOPOLOGY CHARACTERIZATION: reachability, not a race."""

        first = _synthetic_config(tmp_path, preload=True)
        second = _synthetic_config(tmp_path, preload=True)
        assert first is not second
        assert first.model_dir == second.model_dir
        assert first.moss_model_dir == second.moss_model_dir
        assert first.moss_audio_tokenizer_model_dir == (
            second.moss_audio_tokenizer_model_dir
        )
        assert first.zipvoice_model_root == second.zipvoice_model_root

    def test_reachability_matrix_distinguishes_same_worker_barrier(self):
        matrix = {
            ("same-worker preload", "same-worker first request"): (
                False,
                "lifespan barrier",
            ),
            ("worker preload", "worker preload"): (True, "per-worker only"),
            ("worker preload", "other-worker first request"): (
                True,
                "per-worker only",
            ),
            ("worker first request", "other-worker first request"): (
                True,
                "per-worker only",
            ),
            ("Admin repair/drop/load", "other-worker load"): (
                True,
                "current manager only",
            ),
            ("MOSS child ensure", "other worker/child ensure"): (
                True,
                "child/request local",
            ),
            ("container A", "container B"): (True, "operator/platform conditional"),
        }
        assert matrix[("same-worker preload", "same-worker first request")][0] is False
        assert all(
            reachable
            for pair, (reachable, _owner) in matrix.items()
            if pair != ("same-worker preload", "same-worker first request")
        )


class TestSharedVolumeAndP2FHandoff:
    def test_p2f_contract_remains_the_asset_integrity_destination_owner(self):
        path = (
            REPO_ROOT
            / "tests/contracts/test_model_asset_integrity_partial_destination_contract.py"
        )
        assert path.is_file()
        assert "test_deployment_preload_asset_concurrency_contract.py" not in (
            path.name
        )

    def test_asset_modules_have_no_service_global_acquisition_registry(self):
        """STATIC OWNERSHIP CHARACTERIZATION bound to concrete caller modules."""

        for relative in (
            "model_assets.py",
            "kokoro_assets.py",
            "engine.py",
            "moss/runtime.py",
            "zipvoice/assets.py",
            "zipvoice/engine.py",
        ):
            tree = _module_tree(relative)
            top_level_names = {
                target.id
                for node in tree.body
                if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            assert not {
                "ASSET_ACQUISITION_REGISTRY",
                "DESTINATION_LOCK_REGISTRY",
                "ACQUISITION_GENERATION",
                "ALL_WORKERS_ACQUISITION_STATE",
            } & top_level_names


class TestIdleRestartAndLifecycleDelegation:
    def test_run_server_delegates_worker_lifecycle_to_uvicorn(self):
        """RUNTIME-DELEGATED CHARACTERIZATION, not a respawn guarantee."""

        run_server = _definition(_module_tree("server.py"), "run_server")
        source = ast.unparse(run_server)
        assert "uvicorn.run" in source
        assert "'kokoro_tts.server:create_app'" in source
        assert "factory=True" in source
        assert "workers=cfg.workers" in source
        assert "workers=1" in source
        assert "signal.signal" not in source

    def test_idle_restart_calls_injected_current_process_exit_only(self):
        """STATIC OWNERSHIP / CURRENT-BEHAVIOR CHARACTERIZATION."""

        state_tree = _module_tree("service_state.py")
        init = _definition(state_tree, "__init__")
        assert "self._process_exit = os._exit" in ast.unparse(init)
        resource_tree = _module_tree("services/state_parts/resource_state.py")
        perform = _definition(resource_tree, "_perform_idle_restart")
        assert "_process_exit" in _call_names(perform)
        assert not {
            "docker",
            "compose",
            "broadcast",
            "create_app",
            "run_server",
        } & _call_names(perform)

    def test_restart_policy_is_external_and_profile_specific(self):
        for relative, service in (
            ("docker/cpu/docker-compose.yml", "angevoice-cpu"),
            ("docker/gpu/docker-compose.yml", "angevoice-gpu"),
            ("docker/legacy-gpu/docker-compose.yml", "angevoice-legacy-gpu"),
        ):
            assert _mapping_value(_service_block(relative, service), "restart") == (
                "always"
            )
        for service in ("angevoice-cpu", "angevoice-gpu", "angevoice-legacy-gpu"):
            assert _mapping_value(
                _service_block(
                    "packaging/fnos/AngeVoice/app/docker/docker-compose.yaml",
                    service,
                ),
                "restart",
            ) == "unless-stopped"
