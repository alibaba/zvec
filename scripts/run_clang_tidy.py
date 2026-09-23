#!/usr/bin/env python3
"""Run clang-tidy on changed C/C++ files, for use as a pre-push hook.

This mirrors the CI clang-tidy check (``.github/workflows/clang_tidy.yml``):
only files that appear in the CMake compilation database are analysed, using
the repository ``.clang-tidy`` config (auto-discovered by clang-tidy) and the
same ``--warnings-as-errors="*"`` policy. Keeping the two in lockstep means a
green local hook is a good predictor of a green CI run.

The hook degrades gracefully. If clang-tidy or the compilation database is not
available it prints how to enable the check and exits 0, so contributors who
have not configured a build are never blocked from pushing -- CI stays the
hard gate.

The compilation database is located flexibly (see ``_find_build_dir``): the
repo root wins outright, otherwise the configured/well-known build directories
are tried, falling back to a shallow scan of the working tree.

Configuration via environment variables:
  BUILD_DIR    preferred directory holding ``compile_commands.json``. When
               unset the well-known ``build`` / ``build.release`` /
               ``build.debug`` directories are tried instead.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Well-known build directories to probe when BUILD_DIR is unset.
BUILD_DIR_CANDIDATES = ("build", "build.release", "build.debug")

PREFIX = "clang-tidy pre-push"


def _log(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _skip(message: str) -> int:
    """Report a reason the check could not run, then let the push proceed.

    CI remains the hard gate, so a missing local build never blocks a push.
    """
    _log(f"{PREFIX}: {message} -- skipping.")
    return 0


def _normalize(path: Path) -> str:
    """Absolute, symlink-resolved, case-normalized path for reliable matching."""
    return os.path.normcase(str(path.resolve()))


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


def _find_build_dir() -> Path | None:
    """Locate the directory that holds ``compile_commands.json``.

    Discovery order:
      1. the repo root -- if it holds the database, use it and stop.
      2. ``$BUILD_DIR`` when set, else the well-known build directories.
      3. a shallow glob of the working tree, up to two levels deep.
    """
    if Path("compile_commands.json").is_file():
        return Path()

    override = os.environ.get("BUILD_DIR")
    candidates = (
        [Path(override)] if override else [Path(c) for c in BUILD_DIR_CANDIDATES]
    )
    for candidate in candidates:
        if (candidate / "compile_commands.json").is_file():
            return candidate

    for pattern in ("*/compile_commands.json", "*/*/compile_commands.json"):
        for hit in sorted(Path().glob(pattern)):
            # Skip dot-directories (.git, .cache, ...); pathlib glob matches them.
            if not any(part.startswith(".") for part in hit.parent.parts):
                return hit.parent
    return None


def main(argv: list[str]) -> int:
    files = [Path(arg) for arg in argv[1:]]
    if not files:
        return 0

    clang_tidy = shutil.which("clang-tidy")
    if clang_tidy is None:
        return _skip("could not find clang-tidy on PATH")

    build_dir = _find_build_dir()
    if build_dir is None:
        return _skip(
            "no compilation database (compile_commands.json) found. Configure a "
            "build with `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`"
        )
    db_path = build_dir / "compile_commands.json"

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
