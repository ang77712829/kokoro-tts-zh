"""Phase 0 architecture ratchets.

These checks do not claim the current boundaries are clean. They prevent known
debt from growing while the implementation is split behind characterization
tests. A baseline entry must be removed when its function is split or reduced
to the shared threshold.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "kokoro_tts"
MAX_NEW_FUNCTION_COMPLEXITY = 15
pytestmark = pytest.mark.quality

# AST McCabe approximation for the current first-party hotspots. Existing
# functions may decrease, but must not exceed these values. No new unlisted
# function may exceed MAX_NEW_FUNCTION_COMPLEXITY.
COMPLEXITY_BASELINE = {
    "kokoro_tts.admin_config.schema.load_runtime_config": 16,
    "kokoro_tts.config.TTSConfig._validate_auth_and_admin_security": 19,
    "kokoro_tts.config.load_config": 18,
    "kokoro_tts.config_env.apply_env": 33,
    "kokoro_tts.engine_manager.EngineManager.get_engine": 25,
    "kokoro_tts.engines.parameters.EngineParameterSchema.parse": 23,
    "kokoro_tts.engines.provider_policy.ProviderPolicy.requested_provider": 20,
    "kokoro_tts.model_sources.ensure_kokoro_model_dir": 18,
    "kokoro_tts.moss_engine.MossNanoEngine._refresh_vram_guard": 17,
    "kokoro_tts.moss_engine_streaming.MossStreamingMixin._synthesize_stream_process_isolated": 18,
    "kokoro_tts.routes.status_parts.models.model_capabilities": 16,
    "kokoro_tts.routes.status_parts.models.model_catalog_snapshot": 19,
    "kokoro_tts.validation._looks_like_non_natural_text": 20,
    "kokoro_tts.workers.process_worker.EngineProcessClient.stream": 35,
    "kokoro_tts.workers.process_worker._worker_main": 27,
    "kokoro_tts.ws.session.TtsWebSocketSession._parse_and_validate_first_message": 21,
    "kokoro_tts.ws.streaming.StreamingLoopMixin._producer": 19,
    "kokoro_tts.ws.streaming.StreamingLoopMixin._send_loop": 26,
    "kokoro_tts.zipvoice.assets.ZipVoiceAssetManager.ensure": 21,
    "kokoro_tts.zipvoice.assets.ZipVoiceAssetManager.status": 17,
    "kokoro_tts.zipvoice.engine.ZipVoiceEngine.synthesize_stream": 20,
}


class _DecisionCounter(ast.NodeVisitor):
    """Measure one function without charging it for nested definitions."""

    def __init__(self) -> None:
        self.score = 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_If(self, node: ast.If) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.score += 1
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.score += max(0, len(node.cases) - 1)
        self.generic_visit(node)


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["kokoro_tts", *parts])


def _walk_definitions(body: list[ast.stmt], prefix: str = ""):
    for node in body:
        if isinstance(node, ast.ClassDef):
            yield from _walk_definitions(node.body, f"{prefix}{node.name}.")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}{node.name}"
            yield name, node
            yield from _walk_definitions(node.body, f"{name}.<locals>.")


def _complexities() -> dict[str, int]:
    result: dict[str, int] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(path)
        for name, node in _walk_definitions(tree.body):
            counter = _DecisionCounter()
            for statement in node.body:
                counter.visit(statement)
            result[f"{module}.{name}"] = counter.score
    return result


def _complexity_ratchet_deltas(
    current: dict[str, int], baseline: dict[str, int]
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Return (regressions, improvements) without silently moving a ceiling."""
    regressions: dict[str, tuple[int, int]] = {}
    improvements: dict[str, tuple[int, int]] = {}
    for name, score in current.items():
        ceiling = baseline.get(name, MAX_NEW_FUNCTION_COMPLEXITY)
        if score > ceiling:
            regressions[name] = (score, ceiling)
        elif name in baseline and score < ceiling:
            improvements[name] = (score, ceiling)
    return regressions, improvements


@pytest.mark.parametrize(
    ("current_score", "baseline_score", "expected"),
    [
        (36, 35, "regression"),
        (20, 35, "improvement"),
        (35, 35, "equal"),
    ],
)
def test_complexity_ratchet_requires_an_explicit_baseline_update(
    current_score: int, baseline_score: int, expected: str
) -> None:
    regressions, improvements = _complexity_ratchet_deltas(
        {"example.function": current_score}, {"example.function": baseline_score}
    )
    if expected == "regression":
        assert regressions == {"example.function": (36, 35)}
        assert not improvements
    elif expected == "improvement":
        assert improvements == {"example.function": (20, 35)}
        assert not regressions
    else:
        assert not regressions
        assert not improvements


def test_first_party_function_complexity_only_moves_down() -> None:
    current = _complexities()
    stale = sorted(set(COMPLEXITY_BASELINE) - set(current))
    assert not stale, f"Remove stale complexity baseline entries after splitting: {stale}"

    regressions, improvements = _complexity_ratchet_deltas(current, COMPLEXITY_BASELINE)
    assert not regressions, f"Complexity regression: {regressions}"
    assert not improvements, (
        "Complexity improved; tighten COMPLEXITY_BASELINE before merging: "
        f"{improvements}"
    )


class _ModuleImportCollector(ast.NodeVisitor):
    """Collect imports reachable at module load time, excluding definitions."""

    def __init__(self) -> None:
        self.nodes: list[ast.Import | ast.ImportFrom] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.nodes.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.nodes.append(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_If(self, node: ast.If) -> None:
        # TYPE_CHECKING imports are intentionally unavailable at runtime and
        # must not be reported as module-load cycles.
        is_type_checking = isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        is_typing_attribute = (
            isinstance(node.test, ast.Attribute)
            and isinstance(node.test.value, ast.Name)
            and node.test.value.id == "typing"
            and node.test.attr == "TYPE_CHECKING"
        )
        if is_type_checking or is_typing_attribute:
            return None
        self.generic_visit(node)


def _resolve_from(current_module: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = current_module if is_package else current_module.rpartition(".")[0]
    relative = "." * node.level + (node.module or "")
    return importlib.util.resolve_name(relative, package)


def _module_level_graph() -> dict[str, set[str]]:
    paths = list(PACKAGE_ROOT.rglob("*.py"))
    modules = {_module_name(path): path for path in paths}
    graph = {module: set() for module in modules}
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _ModuleImportCollector()
        collector.visit(tree)
        for node in collector.nodes:
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            else:
                base = _resolve_from(module, path.name == "__init__.py", node)
                candidates = [base, *(f"{base}.{alias.name}" for alias in node.names)]
            for candidate in candidates:
                matches = [known for known in modules if candidate == known or candidate.startswith(f"{known}.")]
                if matches:
                    target = max(matches, key=len)
                    if target != module:
                        graph[module].add(target)
    return graph


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph[node]:
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.add(target)
                if target == node:
                    break
            if len(component) > 1:
                result.append(component)

    for node in graph:
        if node not in indices:
            visit(node)
    return result


def test_first_party_modules_have_no_module_load_cycles() -> None:
    assert not _strongly_connected_components(_module_level_graph())


def test_worker_infrastructure_has_no_concrete_engine_imports() -> None:
    concrete_targets = {
        "kokoro_tts.engine",
        "kokoro_tts.moss_engine",
        "kokoro_tts.zipvoice.engine",
    }
    references: dict[str, set[str]] = {}
    workers_root = PACKAGE_ROOT / "workers"
    for path in workers_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(path)
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_from(module, path.name == "__init__.py", node)
                targets.add(base)
                targets.update(f"{base}.{alias.name}" for alias in node.names)
        matched = {target for target in targets if target in concrete_targets}
        if matched:
            references[path.relative_to(PACKAGE_ROOT).as_posix()] = matched

    assert references == {}, "WorkerSpec factories belong to concrete product owners"
