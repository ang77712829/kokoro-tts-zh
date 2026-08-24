"""Freeze the Model Source ENV/Admin metadata projection seam.

Current-state tests preserve the public ENV and Admin surfaces.  The two
future gates stay independently red until each projection module both imports
and consumes the existing declaration-only metadata owner.  Resolver policy,
runtime fallback, and source probing are deliberately outside this contract.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from kokoro_tts import config_env
from kokoro_tts.admin_config import ADMIN_CONFIG_FIELDS
from kokoro_tts.config import TTSConfig
from kokoro_tts.model_source_metadata import MODEL_SOURCE_METADATA


pytestmark = pytest.mark.contract


PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "kokoro_tts"
CONFIG_ENV_PATH = PACKAGE_ROOT / "config_env.py"
ADMIN_SECURITY_PATH = PACKAGE_ROOT / "admin_config" / "groups" / "security.py"
CANONICAL_OWNER_MODULE = "model_source_metadata"
CANONICAL_OWNER_SYMBOL = "MODEL_SOURCE_METADATA"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignment_value(tree: ast.Module, name: str) -> ast.AST:
    matches: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            matches.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            matches.append(node.value)
    assert len(matches) == 1, f"expected one assignment for {name}"
    return matches[0]


def _top_level_definitions(tree: ast.Module) -> dict[str, ast.AST]:
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            definitions[node.target.id] = node.value
    return definitions


def _owner_import_bindings(tree: ast.Module) -> set[str]:
    bindings: set[str] = set()
    for node in tree.body:
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == CANONICAL_OWNER_MODULE
        ):
            continue
        bindings.update(
            alias.asname or alias.name
            for alias in node.names
            if alias.name == CANONICAL_OWNER_SYMBOL
        )
    return bindings


def _depends_on_owner_attribute(
    node: ast.AST,
    attribute: str,
    *,
    definitions: dict[str, ast.AST],
    resolving: frozenset[str] = frozenset(),
) -> bool:
    if (
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == CANONICAL_OWNER_SYMBOL
    ):
        return True
    if isinstance(node, ast.Name) and node.id in definitions and node.id not in resolving:
        return _depends_on_owner_attribute(
            definitions[node.id],
            attribute,
            definitions=definitions,
            resolving=resolving | {node.id},
        )
    return any(
        _depends_on_owner_attribute(
            child,
            attribute,
            definitions=definitions,
            resolving=resolving,
        )
        for child in ast.iter_child_nodes(node)
    )


def _model_source_env_row(tree: ast.Module) -> tuple[ast.AST, ast.AST]:
    mapping = _assignment_value(tree, "STR_ENV")
    assert isinstance(mapping, ast.Dict)
    matches = [
        (key, value)
        for key, value in zip(mapping.keys, mapping.values, strict=True)
        if key is not None
        and (
            (isinstance(key, ast.Constant) and key.value == "ANGEVOICE_MODEL_SOURCE")
            or (isinstance(value, ast.Constant) and value.value == "model_source")
            or (
                CANONICAL_OWNER_SYMBOL in ast.unparse(key)
                and ".canonical_env" in ast.unparse(key)
            )
            or (
                CANONICAL_OWNER_SYMBOL in ast.unparse(value)
                and ".key" in ast.unparse(value)
            )
        )
    ]
    assert len(matches) == 1, "expected one Model Source STR_ENV projection"
    return matches[0]


def _model_source_admin_field(tree: ast.Module) -> ast.Call:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "field_def"
        and node.args
        and (
            (isinstance(node.args[0], ast.Constant) and node.args[0].value == "model_source")
            or (
                CANONICAL_OWNER_SYMBOL in ast.unparse(node.args[0])
                and ".key" in ast.unparse(node.args[0])
            )
        )
    ]
    assert len(matches) == 1, "expected one Model Source Admin field"
    return matches[0]


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(matches) <= 1
    return matches[0] if matches else None


def test_current_model_source_owner_identity_is_exact_and_immutable() -> None:
    metadata_type = type(MODEL_SOURCE_METADATA)
    assert dataclasses.is_dataclass(metadata_type)
    assert metadata_type.__dataclass_params__.frozen is True
    assert MODEL_SOURCE_METADATA.key == "model_source"
    assert MODEL_SOURCE_METADATA.canonical_env == "ANGEVOICE_MODEL_SOURCE"
    assert MODEL_SOURCE_METADATA.default == "auto"
    assert MODEL_SOURCE_METADATA.admin_group == "security"
    assert MODEL_SOURCE_METADATA.admin_choices == (
        "auto",
        "modelscope",
        "huggingface",
        "offline",
    )
    assert MODEL_SOURCE_METADATA.admin_restart is False
    assert MODEL_SOURCE_METADATA.admin_rebuild_moss is False
    assert isinstance(MODEL_SOURCE_METADATA.accepted_values, frozenset)
    assert isinstance(MODEL_SOURCE_METADATA.admin_choices, tuple)


def test_current_model_source_env_admin_and_ttsconfig_surfaces_are_unchanged() -> None:
    metadata = MODEL_SOURCE_METADATA
    field = ADMIN_CONFIG_FIELDS[metadata.key]

    assert config_env.STR_ENV[metadata.canonical_env] == metadata.key
    assert TTSConfig.__dataclass_fields__[metadata.key].default == metadata.default
    assert (
        field.key,
        field.env,
        field.group,
        field.type,
        field.default,
        tuple(value for value, _label in field.choices),
        field.restart,
        field.rebuild_moss,
    ) == (
        metadata.key,
        metadata.canonical_env,
        metadata.admin_group,
        "choice",
        metadata.default,
        metadata.admin_choices,
        metadata.admin_restart,
        metadata.admin_rebuild_moss,
    )
    assert field.label == "模型下载源"
    assert tuple(label for _value, label in field.choices) == (
        "auto 自动",
        "ModelScope",
        "Hugging Face",
        "offline 离线",
    )


def test_future_model_source_env_admin_owner_import_gate_a() -> None:
    actual = {
        "config_env": _owner_import_bindings(_tree(CONFIG_ENV_PATH)),
        "admin_security": _owner_import_bindings(_tree(ADMIN_SECURITY_PATH)),
    }
    assert actual == {
        "config_env": {CANONICAL_OWNER_SYMBOL},
        "admin_security": {CANONICAL_OWNER_SYMBOL},
    }, (
        "Model Source ENV and Admin projection modules must import the "
        "canonical MODEL_SOURCE_METADATA owner directly"
    )


def test_future_model_source_env_admin_owner_projection_gate_b() -> None:
    env_tree = _tree(CONFIG_ENV_PATH)
    env_definitions = _top_level_definitions(env_tree)
    env_key, env_value = _model_source_env_row(env_tree)

    admin_tree = _tree(ADMIN_SECURITY_PATH)
    admin_definitions = _top_level_definitions(admin_tree)
    admin_call = _model_source_admin_field(admin_tree)
    assert len(admin_call.args) >= 6

    admin_projection: dict[str, ast.AST | None] = {
        "key": admin_call.args[0],
        "canonical_env": admin_call.args[1],
        "admin_group": admin_call.args[3],
        "default": admin_call.args[5],
        "admin_choices": _keyword_value(admin_call, "choices"),
        "admin_restart": _keyword_value(admin_call, "restart"),
        "admin_rebuild_moss": _keyword_value(admin_call, "rebuild_moss"),
    }
    admin_actual = {
        attribute: node is not None
        and _depends_on_owner_attribute(
            node,
            attribute,
            definitions=admin_definitions,
        )
        for attribute, node in admin_projection.items()
    }

    assert {
        "env": {
            "canonical_env": _depends_on_owner_attribute(
                env_key,
                "canonical_env",
                definitions=env_definitions,
            ),
            "key": _depends_on_owner_attribute(
                env_value,
                "key",
                definitions=env_definitions,
            ),
        },
        "admin": admin_actual,
    } == {
        "env": {"canonical_env": True, "key": True},
        "admin": {
            "key": True,
            "canonical_env": True,
            "admin_group": True,
            "default": True,
            "admin_choices": True,
            "admin_restart": True,
            "admin_rebuild_moss": True,
        },
    }, (
        "Model Source ENV/Admin structural values must project from the "
        "canonical owner; local user-facing choice labels remain local"
    )
