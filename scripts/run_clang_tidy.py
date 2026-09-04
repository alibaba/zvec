#!/usr/bin/env python3
"""Run clang-tidy on staged C/C++ files, for use as a pre-commit hook.

This mirrors the CI clang-tidy check (``.github/workflows/clang_tidy.yml``):
only files that appear in the CMake compilation database are analysed, using
the repository ``.clang-tidy`` config (auto-discovered by clang-tidy) and the
same ``--warnings-as-errors="*"`` policy. Keeping the two in lockstep means a
green local hook is a good predictor of a green CI run.

The hook degrades gracefully. If clang-tidy or the compilation database is not
available it prints how to enable the check and exits 0, so contributors who
have not configured a build are never blocked from committing -- CI stays the
hard gate. Set ``ZVEC_CLANG_TIDY_STRICT=1`` to turn those skips into failures.

Configuration via environment variables:
  CLANG_TIDY   clang-tidy executable to use (default: first of ``clang-tidy``,
               ``clang-tidy-18`` found on PATH).
  BUILD_DIR    directory holding ``compile_commands.json`` (default: ``build``).
  ZVEC_CLANG_TIDY_STRICT
               when set to ``1``/``true``/``yes``, missing tooling is an error,
               not a skip.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# clang-tidy binaries to look for, in order, when CLANG_TIDY is unset.
CLANG_TIDY_CANDIDATES = ("clang-tidy", "clang-tidy-18")

PREFIX = "clang-tidy pre-commit"


def _log(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _strict() -> bool:
    return os.environ.get("ZVEC_CLANG_TIDY_STRICT", "").lower() in ("1", "true", "yes")


def _skip(message: str) -> int:
    """Report a reason the check could not run.

    Returns 0 (skip) normally, or 1 when strict mode is enabled.
    """
    if _strict():
        _log(f"{PREFIX}: {message}")
        return 1
    _log(f"{PREFIX}: {message} -- skipping.")
    return 0


def _find_clang_tidy() -> str | None:
    override = os.environ.get("CLANG_TIDY")
    if override:
        return override if shutil.which(override) else None
    for candidate in CLANG_TIDY_CANDIDATES:
        if shutil.which(candidate):
            return candidate
    return None


def _normalize(path: Path) -> str:
    """Absolute, symlink-resolved, case-normalized path for reliable matching."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return os.path.normcase(str(resolved))


def _compile_db_files(build_dir: Path) -> set[str]:
    """Normalized set of every source file in ``compile_commands.json``.

    Each entry's ``file`` may be absolute or relative to its ``directory``;
    both forms are resolved so lookups are exact.
    """
    db_path = build_dir / "compile_commands.json"
    with db_path.open(encoding="utf-8") as fh:
        entries = json.load(fh)

    files: set[str] = set()
    for entry in entries:
        file_field = entry.get("file")
        if not file_field:
            continue
        candidate = Path(file_field)
        if not candidate.is_absolute():
            candidate = Path(entry.get("directory", "")) / candidate
        files.add(_normalize(candidate))
    return files


def main(argv: list[str]) -> int:
    files = [Path(arg) for arg in argv[1:]]
    if not files:
        return 0

    clang_tidy = _find_clang_tidy()
    if clang_tidy is None:
        hint = os.environ.get("CLANG_TIDY", " or ".join(CLANG_TIDY_CANDIDATES))
        return _skip(f"could not find clang-tidy ({hint}) on PATH")

    build_dir = Path(os.environ.get("BUILD_DIR", "build"))
    db_path = build_dir / "compile_commands.json"
    if not db_path.is_file():
        return _skip(
            f"no compilation database at {db_path}. Configure a build with "
            "`cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`"
        )

    try:
        db_files = _compile_db_files(build_dir)
    except (OSError, ValueError) as exc:
        return _skip(f"could not read {db_path}: {exc}")

    selected: list[str] = []
    skipped: list[str] = []
    for path in files:
        if _normalize(path) in db_files:
            selected.append(str(path))
        else:
            skipped.append(str(path))

    if skipped:
        # Headers and files not yet in the build are analysed transitively (via
        # the .clang-tidy HeaderFilterRegex) or not at all, exactly as in CI.
        joined = "\n  ".join(sorted(skipped))
        _log(f"{PREFIX}: not in compilation database, skipping:\n  {joined}")

    if not selected:
        return 0

    cmd = [
        clang_tidy,
        "-p",
        str(build_dir),
        "--quiet",
        "--warnings-as-errors=*",
        *selected,
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
