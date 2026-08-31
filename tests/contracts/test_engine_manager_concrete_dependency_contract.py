"""Contract for EngineManager's shared engine dependency boundary."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

from kokoro_tts.config import TTSConfig
from kokoro_tts.engine_manager import EngineManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENGINE_MANAGER_PATH = REPOSITORY_ROOT / "src" / "kokoro_tts" / "engine_manager.py"
CONCRETE_IMPORTS = {
    ("engine", "TTSEngine"),
    ("moss_engine", "MossNanoEngine"),
    ("zipvoice.engine", "ZipVoiceEngine"),
}


def _module_tree() -> ast.Module:
    return ast.parse(ENGINE_MANAGER_PATH.read_text(encoding="utf-8"))


def _concrete_imports(tree: ast.Module) -> set[tuple[str, str]]:
    return {
        (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if (node.module or "", alias.name) in CONCRETE_IMPORTS
    }


def _borrow(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "borrow":
            return node
    raise AssertionError("EngineManager.borrow() is missing")


def test_engine_manager_has_no_concrete_engine_imports_and_uses_shared_protocol() -> None:
    tree = _module_tree()

    assert _concrete_imports(tree) == set()
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "engines.base"
        and any(alias.name == "EngineAdapter" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert ast.unparse(_borrow(tree).returns) == "Iterator[EngineAdapter]"


def test_importing_engine_manager_does_not_load_concrete_engine_modules() -> None:
    probe = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
import kokoro_tts.engine_manager
print(json.dumps({name: name in sys.modules for name in (
    'kokoro_tts.engine',
    'kokoro_tts.moss_engine',
    'kokoro_tts.zipvoice.engine',
)}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "kokoro_tts.engine": False,
        "kokoro_tts.moss_engine": False,
        "kokoro_tts.zipvoice.engine": False,
    }


class _DuckTypedEngine:
    is_loaded = True
    is_healthy = True

    def unload(self, *, force: bool = False) -> None:
        return None


def test_initial_engine_remains_duck_typed_without_runtime_protocol_enforcement() -> None:
    tree = _module_tree()
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and any(isinstance(argument, ast.Name) and argument.id == "EngineAdapter" for argument in node.args)
        for node in ast.walk(tree)
    )

    fake = _DuckTypedEngine()
    manager = EngineManager(
        TTSConfig(enabled_models=["kokoro"], default_model="kokoro"),
        initial_engine=fake,
    )
    with manager.borrow("kokoro") as borrowed:
        assert borrowed is fake
