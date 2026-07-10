# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version  2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Guard tests that prevent drift between the Python runtime exports and the
hand-maintained ``.pyi`` stub files.

If a symbol is exported at runtime but missing from the stub (or vice-versa),
IDE autocomplete and static type checkers (mypy / pyright) silently break.
These tests catch that class of regression.

The checks work by parsing the ``.pyi`` files with the ``ast`` module — no
third-party typing-tool dependency is required.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import zvec
import zvec.typing as zvec_typing

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent.parent  # python/zvec/
_PYI_TOP = _REPO_ROOT / "zvec" / "__init__.pyi"
_PYI_TYPING = _REPO_ROOT / "zvec" / "typing" / "__init__.pyi"


def _ast_names_in_all(node: ast.Assign) -> list[str]:
    """Extract string names from an ``__all__ = [...]`` assignment node."""
    if not isinstance(node.value, ast.List):
        raise TypeError(f"__all__ is not a list literal (got {type(node.value)})")
    names: list[str] = []
    for elt in node.value.elts:
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            names.append(elt.value)
        else:
            raise TypeError(f"Unsupported __all__ element: {ast.dump(elt)}")
    return names


def _ast_class_names(module: ast.Module) -> set[str]:
    """Return the set of top-level class names in *module*."""
    return {node.name for node in module.body if isinstance(node, ast.ClassDef)}


def _ast_function_names(module: ast.Module) -> set[str]:
    """Return the set of top-level function names in *module*."""
    return {node.name for node in module.body if isinstance(node, ast.FunctionDef)}


def _ast_method_names(module: ast.Module, class_name: str) -> set[str]:
    """Return method names declared inside *class_name* in *module*."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _load_stub(path: Path) -> ast.Module:
    """Parse a ``.pyi`` file into an AST module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _stub_all_names(module: ast.Module) -> list[str]:
    """Return the ``__all__`` list from a stub AST, or raise."""
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _ast_names_in_all(node)
    raise AssertionError("__all__ not found in stub")


def _stub_enum_members(module: ast.Module, class_name: str) -> set[str]:
    """Extract enum member names from ``ClassVar`` annotations in a stub class."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            members: set[str] = set()
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(
                    child.target, ast.Name
                ):
                    if not child.target.id.startswith("_"):
                        members.add(child.target.id)
            return members
    return set()


# --------------------------------------------------------------------------- #
# Pre-parsed stubs (parsed once, shared by all tests)
# --------------------------------------------------------------------------- #

_stub_top = _load_stub(_PYI_TOP)
_stub_typing = _load_stub(_PYI_TYPING)

_stub_top_all = set(_stub_all_names(_stub_top))
_stub_typing_all = set(_stub_all_names(_stub_typing))
_stub_typing_classes = _ast_class_names(_stub_typing)
_stub_top_functions = _ast_function_names(_stub_top)


# --------------------------------------------------------------------------- #
# Stub __all__ → runtime: every stub name must be reachable at runtime
# --------------------------------------------------------------------------- #


def test_stub_all_names_exist_at_runtime():
    """Every name declared in ``__init__.pyi`` ``__all__`` must be
    accessible via ``getattr(zvec, ...)`` at runtime."""
    missing = [name for name in _stub_top_all if not hasattr(zvec, name)]
    assert not missing, (
        f"Names in __init__.pyi __all__ but not accessible at runtime: "
        f"{sorted(missing)}"
    )


# --------------------------------------------------------------------------- #
# zvec.typing.__all__ ↔ typing/__init__.pyi __all__  (bidirectional)
# --------------------------------------------------------------------------- #


def test_typing_all_matches_stub():
    """Every name in ``zvec.typing.__all__`` (runtime) must appear in the
    ``typing/__init__.pyi`` ``__all__``, and vice-versa."""
    runtime_all = set(zvec_typing.__all__)
    missing_in_stub = runtime_all - _stub_typing_all
    missing_in_runtime = _stub_typing_all - runtime_all
    assert not missing_in_stub, (
        f"Runtime typing exports missing from typing/__init__.pyi __all__: "
        f"{sorted(missing_in_stub)}"
    )
    assert not missing_in_runtime, (
        f"Stub typing __all__ names not exported at runtime: "
        f"{sorted(missing_in_runtime)}"
    )


# --------------------------------------------------------------------------- #
# Enum members: runtime __members__ vs stub ClassVar declarations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "enum_name, runtime_enum",
    [
        ("DataType", zvec.DataType),
        ("IndexType", zvec.IndexType),
        ("IOBackendType", zvec.IOBackendType),
        ("MetricType", zvec.MetricType),
        ("QuantizeType", zvec.QuantizeType),
        ("StatusCode", zvec.StatusCode),
    ],
)
def test_enum_members_match_stub(enum_name, runtime_enum):
    """Every runtime enum member must be declared as a ``ClassVar`` in the
    ``typing/__init__.pyi`` stub, and vice-versa."""
    runtime_members = set(runtime_enum.__members__)
    stub_members = _stub_enum_members(_stub_typing, enum_name)
    missing_in_stub = runtime_members - stub_members
    missing_in_runtime = stub_members - runtime_members
    assert not missing_in_stub, (
        f"Runtime {enum_name} members missing from stub: {sorted(missing_in_stub)}"
    )
    assert not missing_in_runtime, (
        f"Stub {enum_name} members not in runtime: {sorted(missing_in_runtime)}"
    )


# --------------------------------------------------------------------------- #
# _Collection debug methods: runtime vs stub
# --------------------------------------------------------------------------- #


def test_collection_debug_methods_in_stub():
    """All ``_debug_*`` methods on the runtime ``_Collection`` must be
    declared in the ``_Collection`` stub class in ``__init__.pyi``."""
    stub_methods = _ast_method_names(_stub_top, "_Collection")
    from zvec._zvec import _Collection as _RuntimeCollection

    runtime_debug_methods = {
        name
        for name in dir(_RuntimeCollection)
        if name.startswith("_debug") and callable(getattr(_RuntimeCollection, name))
    }
    missing = runtime_debug_methods - stub_methods
    assert not missing, (
        f"Runtime _Collection debug methods missing from stub: {sorted(missing)}"
    )


# --------------------------------------------------------------------------- #
# Top-level functions: io_backend_type, io_backend_description
# --------------------------------------------------------------------------- #


def test_io_backend_functions_in_stub():
    """``io_backend_type`` and ``io_backend_description`` must be declared as
    top-level functions in ``__init__.pyi`` and listed in ``__all__``."""
    expected = {"io_backend_type", "io_backend_description"}
    missing_funcs = expected - _stub_top_functions
    assert not missing_funcs, (
        f"Functions missing from __init__.pyi: {sorted(missing_funcs)}"
    )
    missing_all = expected - _stub_top_all
    assert not missing_all, (
        f"Functions missing from __init__.pyi __all__: {sorted(missing_all)}"
    )


def test_io_backend_type_enum_in_typing_stub():
    """``IOBackendType`` must be declared as a class in ``typing/__init__.pyi``
    and listed in its ``__all__``."""
    assert "IOBackendType" in _stub_typing_classes, (
        "IOBackendType class missing from typing/__init__.pyi"
    )
    assert "IOBackendType" in _stub_typing_all, (
        "IOBackendType missing from typing/__init__.pyi __all__"
    )
