#!/usr/bin/env python3
"""Header-hygiene guard for the public C++ API under ``src/include/zvec``.

The stable C++ API is ``zvec/db/*.h`` (plus ``zvec/c_api.h`` for C).  The rest
of what ships from ``src/include`` -- ``zvec/core/**``, ``zvec/ailego/**``,
``zvec/turbo/**`` -- is an advanced, unstable surface.  This script keeps the
stable surface from silently absorbing engine internals, the way
``ARROW_INSTALL_ALL_HEADERS`` does for Apache Arrow and the explicit
``install(FILES ...)`` whitelist does for LevelDB.

The stable closure is engine-free: the three headers it used to need from
``zvec/core`` and ``zvec/ailego`` now live under ``zvec/detail/``, a neutral
home both the stable API and the engine may include without inverting the layer
order.  Their old paths remain as shims so engine code keeps compiling, but the
shims are not in the closure, so a default package installs no ``zvec/core/**``,
``zvec/ailego/**`` or ``zvec/turbo/**`` at all.  Rule 6 fails if that regresses.

Checks
------
1. ``zvec/db/*.h`` must not include an internal header (``*_internal.h``) nor
   anything from the engine directories (``zvec/core/**``, ``zvec/ailego/**``,
   ``zvec/turbo/**``).
2. Every non-``zvec/db`` include reached from ``zvec/db/*.h`` must appear in
   ``ALLOWED_STABLE_DEPS``, so new cross-layer dependencies need a reviewed
   allowlist change.
3. Every shipped header must carry the project license header and an include
   guard (``#pragma once`` or ``#ifndef``).
4. The transitive include budget of the entry headers must stay within
   ``BUDGETS``, turning physical-design regressions into a number in CI.
5. With ``--compile-check``: every ``zvec/db/*.h`` must compile on its own with
   only ``src/include`` on the include path, i.e. it is self-contained and needs
   no internal header directory.  Skipped gracefully when no C++ compiler is
   available.
6. The stable API's transitive include closure must be fully listed in
   ``ALLOWED_STABLE_DEPS`` / ``INDIRECT_STABLE_DEPS``, and must contain no
   engine header.  That closure *is* the minimal installable SDK header set, so
   an unlisted member means a trimmed install would ship headers that do not
   compile.  The measured footprint is reported on every run.
7. ``cmake/zvec_public_headers.cmake`` -- the manifest CMake actually installs --
   must list exactly that closure, so the install rules cannot drift from the
   include graph.

Run from the repository root::

    python3 scripts/check_public_headers.py [--verbose] [--compile-check]

``--write-install-manifest`` rewrites ``cmake/zvec_public_headers.cmake`` from the
computed closure; use it after changing a public header, then review the diff.

Exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

INCLUDE_ROOT = Path("src/include")
STABLE_PREFIX = "zvec/db/"
# Entry points of the published API.  ``zvec/db/**`` is the stable C++ surface;
# ``c_api.h`` is the stable C surface; ``export.h`` carries the visibility macro
# both of them use.
STABLE_ENTRIES: tuple[str, ...] = ("zvec/c_api.h", "zvec/export.h")

# CMake install manifest; must equal the stable closure (rule 7).
INSTALL_MANIFEST = Path("cmake/zvec_public_headers.cmake")
MANIFEST_VAR = "ZVEC_STABLE_PUBLIC_HEADERS"

# Non-zvec/db includes the stable API may depend on, with the reason each one is
# currently unavoidable.  Keep this list short.  Every entry is either a
# top-level zvec header or lives under zvec/detail/ -- none of them is an engine
# header, so a default package carries no zvec/core, zvec/ailego or zvec/turbo.
ALLOWED_STABLE_DEPS: dict[str, str] = {
    "zvec/export.h": "ZVEC_API visibility macro",
    "zvec/detail/index_defaults.h": (
        "leaf header (<cstdint> only) providing default index knobs"
    ),
    "zvec/detail/expected.hpp": "backs Result<T> = tl::expected<T, Status>",
    "zvec/detail/float_helper.h": (
        "zvec::ailego::Float16 in Doc::Value; the namespace is part of the ABI, "
        "only the header path moved"
    ),
}

# Headers the allowlisted ones drag in transitively.  Users never include these
# directly, but a trimmed SDK install must still ship them or the stable headers
# stop compiling.  Kept separate so a *direct* db -> engine edge stays reviewable.
#
# Currently empty: every allowlisted header is a leaf.  ailego/internal/platform.h
# and the four ailego string headers used to live here until the Status message
# formatting, the GlobalConfig singleton and the Float16 storage guard were made
# self-contained (see the header-governance doc, section 6.6).
INDIRECT_STABLE_DEPS: dict[str, str] = {}

# Transitive-include budget for the headers a typical user includes, as
# (max headers, max total lines).  Values leave a little headroom over the
# measured baseline; tighten them after each cleanup.
BUDGETS: dict[str, tuple[int, int]] = {
    "zvec/db/status.h": (4, 3200),
    "zvec/db/index_params.h": (7, 4500),
    "zvec/db/doc.h": (10, 5600),
    "zvec/db/query.h": (13, 6200),
    "zvec/db/collection.h": (17, 6500),
}

# Upper bound for the whole installable stable surface (rule 6).  Measured
# baseline is 18 headers / ~10.8k lines; the headroom absorbs small additions.
INSTALL_FOOTPRINT_BUDGET: tuple[int, int] = (19, 11200)

# Vendored third-party headers shipped as-is; they keep their upstream license.
THIRD_PARTY_HEADERS: frozenset[str] = frozenset(
    {
        "zvec/ailego/buffer/concurrentqueue.h",  # moodycamel (BSD-2)
        "zvec/detail/expected.hpp",  # tl::expected (Boost SL 1.0)
    }
)

LICENSE_MARKER = "Licensed under the Apache License, Version 2.0"
INCLUDE_RE = re.compile(r"""^\s*#\s*include\s*[<"]([^">]+)[">]""", re.MULTILINE)
GUARD_RE = re.compile(r"^\s*#\s*ifndef\s+\S+", re.MULTILINE)
# Everything under these directories is the advanced engine surface.  The stable
# API may not reach any of it, directly or transitively (rules 1 and 6).
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "zvec/core/",
    "zvec/ailego/",
    "zvec/turbo/",
)


def _out(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _scan_headers() -> list[str]:
    """Include-root-relative paths of every shipped header, sorted."""
    headers = [
        path.relative_to(INCLUDE_ROOT).as_posix()
        for path in INCLUDE_ROOT.rglob("*")
        if path.suffix in {".h", ".hpp"} and path.is_file()
    ]
    return sorted(headers)


def _resolve(base: str, include: str, known: frozenset[str]) -> str | None:
    """Map an ``#include`` spelling onto an include-root-relative path."""
    if include in known:
        return include
    candidate = (Path(base).parent / include).as_posix()
    normalized = Path(candidate).as_posix()
    return normalized if normalized in known else None


def _direct_includes(rel: str, known: frozenset[str]) -> list[str]:
    text = _read_text(INCLUDE_ROOT / rel)
    return [
        resolved
        for resolved in (_resolve(rel, raw, known) for raw in INCLUDE_RE.findall(text))
        if resolved is not None
    ]


def check_stable_deps(
    headers: Sequence[str], known: frozenset[str], verbose: bool
) -> list[str]:
    """Rule 1 + 2: what the stable API is allowed to pull in."""
    errors: list[str] = []
    for rel in headers:
        if not rel.startswith(STABLE_PREFIX):
            continue
        for dep in _direct_includes(rel, known):
            if dep.startswith(STABLE_PREFIX):
                continue
            if dep.endswith("_internal.h"):
                errors.append(
                    f"{rel}: stable public header includes internal header {dep}"
                )
            elif dep.startswith(FORBIDDEN_PREFIXES):
                errors.append(
                    f"{rel}: stable public header includes engine header {dep}; "
                    f"hide it behind an opaque member or an internal header, or "
                    f"move it to zvec/detail/ if the stable API genuinely needs it"
                )
            elif dep not in ALLOWED_STABLE_DEPS:
                errors.append(
                    f"{rel}: unlisted dependency {dep}; add it to "
                    f"ALLOWED_STABLE_DEPS with a justification, or remove it"
                )
            elif verbose:
                _out(f"  allow {rel} -> {dep} ({ALLOWED_STABLE_DEPS[dep]})")
    return errors


def check_boilerplate(headers: Iterable[str]) -> list[str]:
    """Rule 3: license header and include guard on every shipped header."""
    errors: list[str] = []
    for rel in headers:
        if rel in THIRD_PARTY_HEADERS:
            continue
        text = _read_text(INCLUDE_ROOT / rel)
        if LICENSE_MARKER not in text[:2000]:
            errors.append(f"{rel}: missing Apache-2.0 license header")
        if "#pragma once" not in text and GUARD_RE.search(text) is None:
            errors.append(f"{rel}: missing '#pragma once' or an include guard")
    return errors


def _closure(start: str, graph: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph[current])
    return seen


def check_budgets(headers: Sequence[str]) -> tuple[list[str], list[str]]:
    """Rule 4: transitive include budget of the documented entry headers."""
    known = frozenset(headers)
    graph = {rel: set(_direct_includes(rel, known)) for rel in headers}

    errors: list[str] = []
    report: list[str] = []
    for rel, (max_headers, max_lines) in sorted(BUDGETS.items()):
        if rel not in known:
            errors.append(f"{rel}: budget entry no longer exists")
            continue
        closure = _closure(rel, graph)
        total_lines = sum(
            _read_text(INCLUDE_ROOT / member).count("\n") + 1 for member in closure
        )
        report.append(
            f"  {rel}: {len(closure)} headers, {total_lines} lines "
            f"(budget {max_headers}/{max_lines})"
        )
        if len(closure) > max_headers or total_lines > max_lines:
            errors.append(
                f"{rel}: transitive include budget exceeded: {len(closure)} "
                f"headers / {total_lines} lines (max {max_headers} / "
                f"{max_lines})"
            )
    return errors, report


def _stable_closure(headers: Sequence[str]) -> set[str]:
    """Every ``src/include`` header the published API transitively needs."""
    known = frozenset(headers)
    graph = {rel: set(_direct_includes(rel, known)) for rel in headers}
    closure: set[str] = set()
    for entry in headers:
        if entry.startswith(STABLE_PREFIX) or entry in STABLE_ENTRIES:
            closure |= _closure(entry, graph)
    return closure


def check_install_closure(
    headers: Sequence[str], verbose: bool
) -> tuple[list[str], list[str]]:
    """Rule 6: the stable API's include closure is the minimal SDK header set.

    A trimmed ``install(DIRECTORY ...)`` (see the P1 item in the header-governance
    doc) can only ship this closure.  Any member that is neither under
    ``zvec/db/`` nor allowlisted would silently disappear from the package and
    break compilation for SDK users, so it has to be listed with a reason.
    """
    closure = _stable_closure(headers)

    listed = set(ALLOWED_STABLE_DEPS) | set(INDIRECT_STABLE_DEPS) | set(STABLE_ENTRIES)
    errors: list[str] = []
    for member in sorted(closure):
        if member.startswith(STABLE_PREFIX) or member in listed:
            continue
        errors.append(
            f"{member}: reached from the stable API but not allowlisted; a "
            f"trimmed SDK install would omit it and break compilation. Add it "
            f"to ALLOWED_STABLE_DEPS or INDIRECT_STABLE_DEPS with a reason, or "
            f"remove the dependency"
        )
    for stale in sorted(listed - closure):
        if stale in headers:
            errors.append(
                f"{stale}: allowlisted but no longer reachable from the stable "
                f"API; drop the stale entry"
            )

    total_lines = sum(
        _read_text(INCLUDE_ROOT / member).count("\n") + 1 for member in closure
    )
    max_headers, max_lines = INSTALL_FOOTPRINT_BUDGET
    non_db = sorted(m for m in closure if not m.startswith(STABLE_PREFIX))
    engine = [m for m in non_db if m.startswith(FORBIDDEN_PREFIXES)]
    if engine:
        errors.append(
            "the stable closure reaches engine headers "
            f"({', '.join(engine)}); a default package must install nothing "
            "under zvec/core, zvec/ailego or zvec/turbo, so move whatever the "
            "stable API needs to zvec/detail/ instead"
        )
    report = [
        f"  installable stable surface: {len(closure)} headers, "
        f"{total_lines} lines (budget {max_headers}/{max_lines})",
        f"  of which outside zvec/db: {len(non_db)} "
        f"(engine headers: {len(engine)}; shipped headers total {len(headers)})",
    ]
    if verbose:
        for member in sorted(closure):
            if not member.startswith(STABLE_PREFIX):
                reason = (
                    ALLOWED_STABLE_DEPS.get(member)
                    or INDIRECT_STABLE_DEPS.get(member)
                    or "stable API entry point"
                )
                report.append(f"    {member}: {reason}")
    if len(closure) > max_headers or total_lines > max_lines:
        errors.append(
            f"installable stable surface exceeded budget: {len(closure)} headers "
            f"/ {total_lines} lines (max {max_headers} / {max_lines})"
        )
    return errors, report


MANIFEST_BLOCK_RE = re.compile(rf"set\(\s*{MANIFEST_VAR}(?P<body>.*?)\)", re.DOTALL)


def _parse_manifest() -> list[str] | None:
    """Headers listed in ``ZVEC_STABLE_PUBLIC_HEADERS``, or None if unreadable."""
    if not INSTALL_MANIFEST.is_file():
        return None
    match = MANIFEST_BLOCK_RE.search(_read_text(INSTALL_MANIFEST))
    if match is None:
        return None
    entries = []
    for raw_line in match.group("body").splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    return entries


MANIFEST_TEMPLATE = """\
# Stable public API header manifest.  GENERATED -- do not edit by hand.
#
#   python3 scripts/check_public_headers.py --write-install-manifest
#
# This is the *transitive include closure* of the published C++/C API
# (zvec/db/**, zvec/c_api.h, zvec/export.h): the complete set of headers an SDK
# consumer needs to compile against libzvec and libzvec_c_api, and therefore the
# set CMake always installs.  scripts/check_public_headers.py recomputes the
# closure from the sources and fails when this list drifts.
#
# Paths are relative to src/include/.  The zvec/core and zvec/ailego entries are
# unavoidable implementation detail reached from the stable headers; they install
# regardless of ZVEC_INSTALL_ENGINE_HEADERS, which only controls the *remaining*
# advanced engine headers.
set(ZVEC_STABLE_PUBLIC_HEADERS
{entries}
)
"""


def _manifest_entries(closure: Iterable[str]) -> str:
    """Group the closure by top-level directory for a readable manifest."""
    order = {"top": 0, "db": 1, "core": 2, "ailego": 3, "turbo": 4}

    def group(header: str) -> str:
        parts = header.split("/")
        return parts[1] if len(parts) > 2 else "top"

    lines: list[str] = []
    previous: str | None = None
    for header in sorted(closure, key=lambda h: (order.get(group(h), 9), h)):
        current = group(header)
        if previous is not None and current != previous:
            lines.append("")
        lines.append(f"    {header}")
        previous = current
    return "\n".join(lines)


def write_install_manifest(headers: Sequence[str]) -> int:
    """Regenerate cmake/zvec_public_headers.cmake from the computed closure."""
    closure = _stable_closure(headers)
    INSTALL_MANIFEST.write_text(
        MANIFEST_TEMPLATE.format(entries=_manifest_entries(closure)),
        encoding="utf-8",
    )
    _out(f"wrote {INSTALL_MANIFEST}: {len(closure)} headers")
    return 0


def check_install_manifest(headers: Sequence[str]) -> tuple[list[str], list[str]]:
    """Rule 7: the CMake install manifest equals the computed closure."""
    listed = _parse_manifest()
    if listed is None:
        return (
            [
                f"{INSTALL_MANIFEST}: missing or has no set({MANIFEST_VAR} ...) "
                f"block; CMake would install an uncontrolled header set"
            ],
            [],
        )

    closure = _stable_closure(headers)
    manifest = set(listed)
    errors: list[str] = []
    for missing in sorted(closure - manifest):
        errors.append(
            f"{INSTALL_MANIFEST}: {missing} is required by the stable API but "
            f"not listed; the installed SDK would not compile"
        )
    for extra in sorted(manifest - closure):
        errors.append(
            f"{INSTALL_MANIFEST}: {extra} is listed but not reachable from the "
            f"stable API; drop it or move it behind an install option"
        )
    if len(listed) != len(manifest):
        errors.append(f"{INSTALL_MANIFEST}: {MANIFEST_VAR} contains duplicates")

    report = [f"  install manifest: {len(listed)} headers (matches closure)"]
    return errors, report


def _find_compiler() -> str | None:
    """C++ compiler for the optional self-containment check."""
    override = os.environ.get("CXX")
    if override and shutil.which(override):
        return override
    for candidate in ("c++", "g++", "clang++"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def check_self_contained(headers: Iterable[str]) -> list[str]:
    """Rule 5: each stable header compiles alone with only src/include."""
    compiler = _find_compiler()
    if compiler is None:
        _out("  self-containment compile check skipped: no C++ compiler found")
        return []

    errors: list[str] = []
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "header_self_containment.cc"
        for rel in headers:
            if not rel.startswith(STABLE_PREFIX):
                continue
            checked += 1
            source.write_text(f"#include <{rel}>\nint main() {{ return 0; }}\n")
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsyntax-only",
                    "-I",
                    str(INCLUDE_ROOT),
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                first_error = next(
                    (line for line in result.stderr.splitlines() if "error:" in line),
                    result.stderr.strip()[:400],
                )
                errors.append(f"{rel}: not self-contained: {first_error}")
    _out(f"  self-containment compile check: {checked} headers via {compiler}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check public C++ header hygiene under src/include/zvec."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print allowlisted dependencies and per-header budgets",
    )
    parser.add_argument(
        "--compile-check",
        action="store_true",
        help="also verify each stable header compiles on its own",
    )
    parser.add_argument(
        "--write-install-manifest",
        action="store_true",
        help=f"rewrite {INSTALL_MANIFEST} from the computed closure and exit",
    )
    args = parser.parse_args(argv)

    if not INCLUDE_ROOT.is_dir():
        sys.stderr.write(
            f"error: {INCLUDE_ROOT} not found; run from the repository root\n"
        )
        return 2

    headers = _scan_headers()
    if args.write_install_manifest:
        return write_install_manifest(headers)
    known = frozenset(headers)
    _out(f"public headers scanned: {len(headers)}")

    errors = check_stable_deps(headers, known, args.verbose)
    errors += check_boilerplate(headers)
    budget_errors, budget_report = check_budgets(headers)
    errors += budget_errors
    closure_errors, closure_report = check_install_closure(headers, args.verbose)
    errors += closure_errors
    manifest_errors, manifest_report = check_install_manifest(headers)
    errors += manifest_errors
    if args.compile_check:
        errors += check_self_contained(headers)

    _out("transitive include budget:")
    for line in budget_report:
        _out(line)
    _out("stable API install closure:")
    for line in closure_report:
        _out(line)
    for line in manifest_report:
        _out(line)

    if errors:
        _out(f"\nFAILED ({len(errors)} problem(s)):")
        for error in errors:
            _out(f"  - {error}")
        return 1

    _out("OK: public header hygiene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
