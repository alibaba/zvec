#!/usr/bin/env python3
"""Generate the link-time export whitelist for the ``libzvec`` shared library.

``libzvec`` whole-archives every zvec component plus rocksdb, arrow, antlr and
protobuf, so without a filter it would export the entire engine.  The existing
``src/exported_symbols_cpp.txt`` already hides third-party symbols, but its
granularity is ``__ZN4zvec*`` -- the whole ``zvec`` namespace -- which measured
**10784** exported symbols on macOS, of which only ~2600 are declared by a
public header.  Everything else (``zvec::sqlengine``, ``zvec::fts``,
``zvec::SegmentImpl``, ``zvec::CollectionImpl``, ``zvec::core`` internals ...)
is implementation detail a consumer can accidentally link against.

This script derives a whitelist from *what the public headers actually
declare* instead of a hand-maintained list:

1. Collect the public header set (the stable closure from
   ``cmake/zvec_public_headers.cmake``, plus ``zvec/core``/``zvec/ailego``/
   ``zvec/turbo`` when the engine surface is published).
2. Ask clang for the declarations in those headers
   (``-Xclang -ast-dump=json -Xclang -ast-dump-filter zvec``).
3. Turn every declared class / function / namespace-scope variable into
   mangled-name *glob* patterns, one per Itanium encoding variant
   (``_ZN``, ``_ZNK``, ``_ZTV``, ``_ZTI``, ``_ZTS``, ``_ZTT``, ``_ZGV``,
   ``_ZZ``, thunks ...).

Globs rather than literal symbol names, for two reasons: an Itanium nested name
is length prefixed, so ``_ZN4zvec3Doc*`` can only ever match members of
``zvec::Doc`` (``zvec::Document`` would be ``8Document``); and lld's default
``--no-undefined-version`` rejects a version-script entry that matches no
symbol, which a literal list hits the moment a signature changes.

Measured effect on ``libzvec.dylib`` (macOS/arm64, Release):

    engine surface published (default)   10784 -> 2602 symbols  (-75.9%)
    stable surface only                  10784 ->  366 symbols  (-96.6%)

Run from the repository root::

    python3 scripts/generate_export_whitelist.py            # check for drift
    python3 scripts/generate_export_whitelist.py --write    # regenerate
    python3 scripts/generate_export_whitelist.py --surface stable --write

``--surface engine`` (the default) covers every public header under
``src/include``.  That is deliberately *wider* than the default package, which
ships the 18 header stable closure because ``ZVEC_INSTALL_ENGINE_HEADERS`` is
off: a source-tree consumer -- ``add_subdirectory(zvec)``, or the in-tree
``examples/c++`` -- reads the engine headers straight from ``src/include`` and
links ``zvec_shared``, so narrowing the list to the installed subset would make
the engine API unlinkable for exactly the consumers who can still see its
declarations.  The extra exports cost nothing at runtime and are unreachable
from a prebuilt package, which carries no header that declares them.

``--surface stable`` (366 symbols, measured) is the tighter option if the engine
surface is ever withdrawn from source-tree consumers too; switching to it means
repointing ``examples/c++`` and ``subproject-integration`` at
``zvec_core_shared`` / ``zvec_ailego_shared``, which keep the namespace-wide
list.

Exit status is 0 when the checked-in lists are up to date and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

INCLUDE_ROOT = Path("src/include")
STABLE_MANIFEST = Path("cmake/zvec_public_headers.cmake")
ENGINE_DIRS = ("zvec/core", "zvec/ailego", "zvec/turbo")
ENGINE_SUFFIXES = (".h", ".hpp", ".hxx")

OUTPUTS = {
    "macos": Path("src/exported_symbols_zvec.txt"),
    "linux": Path("src/exported_symbols_zvec.lds"),
}

# Everything that can sit between "_Z" and a nested-name component list.  A
# class prefix also covers its nested classes and members, so free functions
# only need the plain function encodings.
VARIANTS = (
    "N",  # function or method
    "NK",  # const method
    "TVN",  # vtable
    "TIN",  # typeinfo
    "TSN",  # typeinfo name
    "TTN",  # VTT
    "GVN",  # guard variable
    "ZN",  # entity local to a function (lambda, static local)
    "ZNK",  # ... inside a const method
    "GVZN",  # guard variable for a function-local entity
    "GVZNK",
)
THUNK_VARIANTS = ("Thn", "Tv")

# Declaration kinds worth an export pattern.  Fields, enum constants, aliases
# and namespaces emit no symbol of their own; methods are already covered by
# their class prefix.
PATTERN_KINDS = frozenset({"class", "function", "variable"})

DECL_KINDS = {
    "NamespaceDecl": "namespace",
    "CXXRecordDecl": "class",
    "RecordDecl": "class",
    "ClassTemplateDecl": "class",
    "ClassTemplateSpecializationDecl": "class",
    "ClassTemplatePartialSpecializationDecl": "class",
    "FunctionDecl": "function",
    "CXXMethodDecl": "method",
    "CXXConstructorDecl": "method",
    "CXXDestructorDecl": "method",
    "CXXConversionDecl": "method",
    "FunctionTemplateDecl": "function",
    "EnumDecl": "enum",
    "EnumConstantDecl": "enum-constant",
    "VarDecl": "variable",
    "TypeAliasDecl": "alias",
    "TypedefDecl": "alias",
    "TypeAliasTemplateDecl": "alias",
    "FieldDecl": "field",
    "IndirectFieldDecl": "field",
}

# A *TemplateDecl owns the templated declaration as its first child and that
# child repeats the same name.  Pushing both would invent entities such as
# ``zvec::Foo::Foo`` and -- after prefix minimisation -- shadow the real
# ``zvec::Foo``, hiding the whole class.
TEMPLATE_DECL_KINDS = {
    "ClassTemplateDecl": "CXXRecordDecl",
    "FunctionTemplateDecl": "FunctionDecl",
    "TypeAliasTemplateDecl": "TypeAliasDecl",
    "VarTemplateDecl": "VarDecl",
}

# Itanium ABI operator encodings.  Only namespace-scope operators need them: a
# member operator already falls under its class prefix.
OPERATOR_MANGLING = {
    "operator<<": "ls",
    "operator>>": "rs",
    "operator<<=": "lS",
    "operator>>=": "rS",
    "operator<": "lt",
    "operator>": "gt",
    "operator<=": "le",
    "operator>=": "ge",
    "operator==": "eq",
    "operator!=": "ne",
    "operator!": "nt",
    "operator&&": "aa",
    "operator||": "oo",
    "operator~": "co",
    "operator+": "pl",
    "operator-": "mi",
    "operator*": "ml",
    "operator/": "dv",
    "operator%": "rm",
    "operator^": "er",
    "operator&": "an",
    "operator|": "or",
    "operator=": "aS",
    "operator+=": "pL",
    "operator-=": "mI",
    "operator*=": "mL",
    "operator/=": "dV",
    "operator%=": "rM",
    "operator^=": "eO",
    "operator&=": "aN",
    "operator|=": "oR",
    "operator++": "pp",
    "operator--": "mm",
    "operator()": "cl",
    "operator[]": "ix",
    "operator->": "pm",
    "operator->*": "pt",
    "operator new": "nw",
    "operator delete": "dl",
}

HEADER = """\
# GENERATED -- do not edit by hand.
#
#   python3 scripts/generate_export_whitelist.py --write
#
# Link-time export whitelist for the all-in-one libzvec shared library: every
# symbol a public header under src/include declares, and nothing else.  Derived
# from the clang AST of that header set, so adding a public class or method
# needs a regeneration (the default --check mode fails otherwise).
#
# This is deliberately wider than the installed header set: the default package
# ships only the stable closure (ZVEC_INSTALL_ENGINE_HEADERS=OFF), but a
# source-tree consumer reads the engine headers from src/include and links
# libzvec.  See the module docstring for the trade-off.
#
# Patterns are mangled-name globs.  An Itanium nested name is length prefixed,
# so "_ZN4zvec3Doc*" matches members of zvec::Doc only -- zvec::Document would
# be encoded as "8Document".  The variants after "_Z" cover const methods (NK),
# vtables/typeinfo/VTT (TVN/TIN/TSN/TTN), guard variables (GVN), function-local
# entities such as lambdas and static locals (ZN/ZNK/GVZN/GVZNK) and thunks.
#
# Surface: {surface} -- {header_count} public headers, {prefix_count} entity
# prefixes, {pattern_count} patterns.
#
# Keep this list globs-only: lld's default --no-undefined-version rejects a
# version-script entry that matches no symbol, so literal names would break the
# Android build whenever a signature changes.
"""


def _out(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def stable_headers() -> list[str]:
    """The stable closure CMake always installs, from the checked-in manifest."""
    block = re.search(
        r"set\(ZVEC_STABLE_PUBLIC_HEADERS(.*?)\)",
        _read_text(STABLE_MANIFEST),
        re.DOTALL,
    )
    if block is None:
        msg = f"cannot parse ZVEC_STABLE_PUBLIC_HEADERS from {STABLE_MANIFEST}"
        raise SystemExit(msg)
    return [
        line.strip()
        for line in block.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def engine_headers() -> list[str]:
    """Headers the ZVEC_INSTALL_ENGINE_HEADERS=ON install rule publishes."""
    found = []
    for directory in ENGINE_DIRS:
        root = INCLUDE_ROOT / directory
        for path in sorted(root.rglob("*")):
            if path.suffix in ENGINE_SUFFIXES and path.is_file():
                found.append(path.relative_to(INCLUDE_ROOT).as_posix())
    return found


def public_headers(surface: str) -> list[str]:
    headers = set(stable_headers())
    if surface == "engine":
        headers.update(engine_headers())
    missing = [h for h in sorted(headers) if not (INCLUDE_ROOT / h).is_file()]
    if missing:
        msg = f"public headers missing from {INCLUDE_ROOT}: {missing[:5]}"
        raise SystemExit(msg)
    return sorted(headers)


def find_clang() -> str | None:
    """A clang capable of -Xclang -ast-dump=json (gcc has no equivalent)."""
    override = os.environ.get("CXX")
    candidates = [override] if override else []
    candidates += ["clang++", "clang++-18", "clang++-17", "clang++-16"]
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved and "clang" in Path(resolved).name:
            return resolved
    return None


def dump_ast(headers: list[str], compiler: str, workdir: Path) -> Path:
    """Run clang's filtered AST dump over one TU that includes every header."""
    probe = workdir / "export_whitelist_probe.cc"
    lines = ["// Generated probe TU: one include per public header."]
    lines += [f"#include <{header}>" for header in headers]
    lines.append("int main() { return 0; }")
    probe.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ast_path = workdir / "ast.json"
    with ast_path.open("wb") as handle:
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-I",
                str(INCLUDE_ROOT),
                "-Xclang",
                "-ast-dump=json",
                "-Xclang",
                "-ast-dump-filter",
                "-Xclang",
                "zvec",
                "-fsyntax-only",
                str(probe),
            ],
            stdout=handle,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )
    errors = [
        line
        for line in result.stderr.decode("utf-8", "replace").splitlines()
        if "error:" in line
    ]
    if result.returncode != 0 or errors:
        detail = errors[0] if errors else f"clang exited with {result.returncode}"
        msg = f"the public headers do not compile as one TU: {detail}"
        raise SystemExit(msg)
    return ast_path


def iter_json_documents(text: str):
    """The filtered dump emits one JSON document per matching root decl."""
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " \n\r\t":
            index += 1
        if index >= length:
            break
        document, index = decoder.raw_decode(text, index)
        yield document


def clean_name(name: str) -> str:
    """Drop template argument lists, but never inside an operator spelling."""
    if name.startswith("operator") and " " not in name:
        return name
    return name.split("<", 1)[0].strip()


def walk(
    node: dict,
    stack: list[str],
    entities: list[tuple[str, str]],
    skip_push: bool = False,
) -> None:
    kind = node.get("kind", "")
    raw_name = node.get("name")
    name = clean_name(raw_name) if raw_name else None
    pushes = bool(name) and kind in DECL_KINDS and not skip_push
    qualified = "::".join([*stack, name] if pushes else stack)
    if kind in DECL_KINDS and name and not node.get("isImplicit"):
        entities.append((DECL_KINDS[kind], qualified))
    templated_child = TEMPLATE_DECL_KINDS.get(kind)
    child_stack = [*stack, name] if pushes else stack
    for position, child in enumerate(node.get("inner") or []):
        duplicated = (
            position == 0
            and templated_child is not None
            and child.get("kind") == templated_child
            and clean_name(child.get("name") or "") == name
        )
        walk(child, child_stack, entities, duplicated)


def collect_entities(ast_path: Path) -> list[tuple[str, str]]:
    text = _read_text(ast_path)
    entities: list[tuple[str, str]] = []
    for document in iter_json_documents(text):
        walk(document, [], entities)
    return entities


def mangled_components(qualified: str) -> str | None:
    """Itanium nested-name components for a qualified name (no leading N/E)."""
    parts = qualified.split("::")
    if not parts or not all(parts):
        return None
    encoded = []
    last = len(parts) - 1
    for index, raw in enumerate(parts):
        component = raw
        if "<" in component and not component.startswith("operator"):
            component = component.split("<", 1)[0]
        if not component:
            return None
        operator = OPERATOR_MANGLING.get(component)
        if operator and index == last:
            encoded.append(operator)
        else:
            encoded.append(f"{len(component.encode())}{component}")
    return "".join(encoded)


def minimal_prefixes(entities: list[tuple[str, str]]) -> list[str]:
    prefixes = set()
    for kind, qualified in entities:
        if kind not in PATTERN_KINDS or not qualified.startswith("zvec::"):
            continue
        components = mangled_components(qualified)
        if components:
            prefixes.add(components)
    ordered = sorted(prefixes)
    # A shorter prefix already covers everything a longer one would match.
    return [
        prefix
        for prefix in ordered
        if not any(other != prefix and prefix.startswith(other) for other in ordered)
    ]


def patterns_for(prefix: str) -> list[str]:
    built = [f"_Z{variant}{prefix}*" for variant in VARIANTS]
    built += [f"_Z{thunk}*_N{prefix}*" for thunk in THUNK_VARIANTS]
    return built


def render_macos(prefixes: list[str]) -> str:
    body = sorted(
        {f"_{pattern}" for prefix in prefixes for pattern in patterns_for(prefix)}
    )
    return "\n".join([*body, "_zvec_*"]) + "\n"


def render_linux(prefixes: list[str], surface: str, header_count: int) -> str:
    """GNU/lld version script.  Patterns match mangled names (no extern "C++").

    Quoted entries would mean a literal match, and lld (Android NDK) fails with
    --no-undefined-version when a literal entry matches no symbol.
    """
    body = sorted({p for prefix in prefixes for p in patterns_for(prefix)})
    lines = [
        "/* GENERATED -- do not edit by hand.",
        "   python3 scripts/generate_export_whitelist.py --write",
        f"   Surface: {surface} -- {header_count} public headers,",
        f"   {len(prefixes)} entity prefixes, {len(body)} patterns.",
        "   Rationale and variant legend: src/exported_symbols_zvec.txt. */",
        "{",
        "  global:",
    ]
    lines += [f"    {pattern};" for pattern in body]
    lines += ["    zvec_*;", "  local:", "    *;", "};"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the libzvec link-time export whitelist."
    )
    parser.add_argument(
        "--surface",
        choices=("engine", "stable"),
        default="engine",
        help="public header set to derive the whitelist from "
        "(engine = ZVEC_INSTALL_ENGINE_HEADERS=ON, the release default)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the checked-in lists instead of checking for drift",
    )
    parser.add_argument(
        "--require-clang",
        action="store_true",
        help="fail instead of skipping when no clang++ is available (CI)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    required = [INCLUDE_ROOT, STABLE_MANIFEST]
    if not args.write:
        required += list(OUTPUTS.values())
    for path in required:
        if not path.exists():
            _out(f"error: {path} not found; run from the repository root")
            return 1

    compiler = find_clang()
    if compiler is None:
        if args.require_clang:
            _out(
                "error: no clang++ found; the AST dump needs clang "
                "(set CXX to a clang++ binary)"
            )
            return 1
        _out("skipped: no clang++ found (the AST dump needs clang, not gcc)")
        return 0

    headers = public_headers(args.surface)
    _out(f"public headers ({args.surface} surface): {len(headers)}")

    with tempfile.TemporaryDirectory() as tmp:
        ast_path = dump_ast(headers, compiler, Path(tmp))
        entities = collect_entities(ast_path)

    prefixes = minimal_prefixes(entities)
    _out(f"declared entities: {len(entities)}")
    _out(f"export prefixes: {len(prefixes)} (after minimisation)")

    macos_body = render_macos(prefixes)
    pattern_count = len(macos_body.splitlines()) - 1
    _out(f"patterns: {pattern_count}")
    note = HEADER.format(
        surface=args.surface,
        header_count=len(headers),
        prefix_count=len(prefixes),
        pattern_count=pattern_count,
    )
    rendered = {
        "macos": note + macos_body,
        "linux": render_linux(prefixes, args.surface, len(headers)),
    }

    if args.write:
        for name, path in OUTPUTS.items():
            path.write_text(rendered[name], encoding="utf-8")
            _out(f"wrote {path} ({name})")
        return 0

    stale = [
        str(path)
        for name, path in OUTPUTS.items()
        if _read_text(path) != rendered[name]
    ]
    if stale:
        _out(f"DRIFT: {' and '.join(stale)} do not match the public headers")
        _out("  regenerate with: python3 scripts/generate_export_whitelist.py --write")
        return 1
    _out("export whitelist matches the public headers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
