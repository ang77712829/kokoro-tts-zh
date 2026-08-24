"""Compatibility contract for the ModelSource metadata consumer pilot."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
from pathlib import Path

import pytest

from kokoro_tts import admin_config, config as config_module, config_env
from kokoro_tts.config import TTSConfig
from kokoro_tts.model_source_metadata import MODEL_SOURCE_METADATA


pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "kokoro_tts"
METADATA_PATH = PACKAGE_ROOT / "model_source_metadata.py"
CONFIG_PATH = PACKAGE_ROOT / "config.py"
ENV_SNAPSHOT_PATH = Path(__file__).with_name("data") / "config_env_surface_2_6_7.json"
P2A1_SNAPSHOT_PATHS = (
    Path(__file__).with_name("data") / "admin_config_surface_2_6_7.json",
    ENV_SNAPSHOT_PATH,
    Path(__file__).with_name("data") / "ttsconfig_shape_2_6_7.json",
)


@dataclasses.dataclass(frozen=True)
class _NormalizationMetadata:
    default: str
    accepted_values: frozenset[str]


def _ttsconfig_default(name: str):
    field = next(field for field in dataclasses.fields(TTSConfig) if field.name == name)
    assert field.default is not dataclasses.MISSING
    return field.default


def _metadata_tree() -> ast.Module:
    return ast.parse(METADATA_PATH.read_text(encoding="utf-8"))


def _config_tree() -> tuple[ast.Module, ast.ClassDef]:
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TTSConfig"
    )
    return tree, owner


def _normalization_owner_tree() -> ast.FunctionDef:
    _, owner = _config_tree()
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_model_source"
    )


def _metadata_importers() -> list[str]:
    imported_by = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        if path == METADATA_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.endswith("model_source_metadata"):
                    imported_by.append(path.relative_to(PACKAGE_ROOT).as_posix())
            elif isinstance(node, ast.Import):
                if any(alias.name.endswith("model_source_metadata") for alias in node.names):
                    imported_by.append(path.relative_to(PACKAGE_ROOT).as_posix())
    return imported_by


def test_metadata_is_frozen_with_immutable_nested_values() -> None:
    assert dataclasses.is_dataclass(MODEL_SOURCE_METADATA)
    assert type(MODEL_SOURCE_METADATA).__dataclass_params__.frozen is True
    assert isinstance(MODEL_SOURCE_METADATA.accepted_values, frozenset)
    assert isinstance(MODEL_SOURCE_METADATA.country_env_aliases, tuple)
    assert isinstance(MODEL_SOURCE_METADATA.admin_choices, tuple)
    assert isinstance(MODEL_SOURCE_METADATA.engine_scope, frozenset)
    with pytest.raises(dataclasses.FrozenInstanceError):
        MODEL_SOURCE_METADATA.default = "offline"
    with pytest.raises(AttributeError):
        MODEL_SOURCE_METADATA.accepted_values.add("unexpected")
    with pytest.raises(TypeError):
        MODEL_SOURCE_METADATA.country_env_aliases[0] = "unexpected"


def test_ttsconfig_default_consumes_metadata_without_changing_shape() -> None:
    assert MODEL_SOURCE_METADATA.key == "model_source"
    assert MODEL_SOURCE_METADATA.default == _ttsconfig_default("model_source")
    fields = list(dataclasses.fields(TTSConfig))
    model_source = next(field for field in fields if field.name == "model_source")
    assert len(fields) == 179
    assert model_source.type is str
    assert model_source.default == "auto"

    _, owner = _config_tree()
    declaration = next(
        node
        for node in owner.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "model_source"
    )
    assert isinstance(declaration.value, ast.Attribute)
    assert isinstance(declaration.value.value, ast.Name)
    assert declaration.value.value.id == "MODEL_SOURCE_METADATA"
    assert declaration.value.attr == "default"


def test_metadata_matches_canonical_env_and_p2a1_country_direct_readers() -> None:
    assert config_env.STR_ENV[MODEL_SOURCE_METADATA.canonical_env] == MODEL_SOURCE_METADATA.key
    snapshot = json.loads(ENV_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    country_aliases = tuple(
        item["env"]
        for item in snapshot["direct_readers"]
        if item["reader"] == "kokoro_tts.model_sources._detect_country"
    )
    assert MODEL_SOURCE_METADATA.country_env_aliases == country_aliases


def test_metadata_matches_admin_surface_without_owning_admin_behavior() -> None:
    field = admin_config.ADMIN_CONFIG_FIELDS[MODEL_SOURCE_METADATA.key]
    assert field.key == MODEL_SOURCE_METADATA.key
    assert field.env == MODEL_SOURCE_METADATA.canonical_env
    assert field.group == MODEL_SOURCE_METADATA.admin_group
    assert field.type == "choice"
    assert field.default == MODEL_SOURCE_METADATA.default
    assert tuple(value for value, _label in field.choices) == MODEL_SOURCE_METADATA.admin_choices
    assert field.restart is MODEL_SOURCE_METADATA.admin_restart
    assert field.rebuild_moss is MODEL_SOURCE_METADATA.admin_rebuild_moss


def test_engine_scope_and_zipvoice_exclusion_are_explicit() -> None:
    assert MODEL_SOURCE_METADATA.engine_scope == frozenset(
        {"kokoro", "moss", "moss_audio_tokenizer"}
    )
    assert MODEL_SOURCE_METADATA.excluded_engine_scope == "zipvoice"
    assert MODEL_SOURCE_METADATA.excluded_engine_scope not in MODEL_SOURCE_METADATA.engine_scope
    assert MODEL_SOURCE_METADATA.resolver_owner == "model_sources.resolve_model_source"


def test_metadata_module_is_declaration_only() -> None:
    tree = _metadata_tree()
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1
    imported = imports[0]
    assert isinstance(imported, ast.ImportFrom)
    assert imported.module == "dataclasses"
    assert [(alias.name, alias.asname) for alias in imported.names] == [("dataclass", None)]
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert all(isinstance(node.func, ast.Name) for node in calls)
    assert {node.func.id for node in calls} == {"dataclass", "frozenset", "ModelSourceMetadata"}


def test_metadata_production_consumers_stay_within_frozen_projection_boundary() -> None:
    actual = set(_metadata_importers())
    required_base = {"config.py"}
    allowed = {
        "config.py",
        "config_env.py",
        "admin_config/groups/security.py",
    }

    assert required_base <= actual
    assert actual <= allowed, f"unexpected metadata consumers: {sorted(actual - allowed)}"


def test_normalization_consumes_metadata_without_a_duplicate_value_set(monkeypatch) -> None:
    tree = _normalization_owner_tree()
    metadata_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "MODEL_SOURCE_METADATA"
    }
    assert {"default", "accepted_values"} <= metadata_attributes
    duplicate_sets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Set)
        and {
            item.value
            for item in node.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        == {"auto", "huggingface", "modelscope", "offline"}
    ]
    assert duplicate_sets == []

    replacement = _NormalizationMetadata(
        default="fallback",
        accepted_values=frozenset({"fallback", "accepted"}),
    )
    monkeypatch.setattr(config_module, "MODEL_SOURCE_METADATA", replacement)
    fallback = TTSConfig(model_source="")
    fallback._normalize_model_source()
    assert fallback.model_source == "fallback"
    accepted = TTSConfig(model_source="  ACCEPTED  ")
    accepted._normalize_model_source()
    assert accepted.model_source == "accepted"


def test_model_source_normalization_behavior_and_error_contract_are_unchanged() -> None:
    for value in ("auto", "huggingface", "modelscope", "offline"):
        config = TTSConfig(model_source=value)
        config._normalize_model_source()
        assert config.model_source == value
    for value in ("", None):
        config = TTSConfig(model_source=value)
        config._normalize_model_source()
        assert config.model_source == "auto"
    config = TTSConfig(model_source="  ModelScope  ")
    config._normalize_model_source()
    assert config.model_source == "modelscope"

    invalid = TTSConfig(model_source="  INVALID-SOURCE  ")
    with pytest.raises(
        ValueError,
        match=r"^ANGEVOICE_MODEL_SOURCE must be auto, huggingface, modelscope, or offline$",
    ):
        invalid._normalize_model_source()
    assert invalid.model_source == "invalid-source"


def test_p2a1_snapshots_remain_unmodified() -> None:
    assert all(path.is_file() for path in P2A1_SNAPSHOT_PATHS)
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *(str(path.relative_to(ROOT)) for path in P2A1_SNAPSHOT_PATHS),
        ],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode == 0
