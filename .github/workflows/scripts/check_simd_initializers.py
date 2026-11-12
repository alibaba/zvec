"""Reject load-time constructors in MSVC objects compiled for AVX.

Run after the Windows build. Runtime dispatch cannot protect a CPU from an
initializer registered in .CRT$XC* or .CRT$XI*: the loader calls it first.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def check_initializers(build_dir: Path) -> None:
    entries = json.loads((build_dir / "compile_commands.json").read_text())
    checked = set()
    failures = []
    for entry in entries:
        if not re.search(r"/arch:AVX", entry["command"], re.IGNORECASE):
            continue
        obj = build_dir / entry["output"]
        # The compilation database also lists targets not requested by pip.
        if obj in checked or not obj.is_file():
            continue
        checked.add(obj)
        headers = subprocess.run(
            ["dumpbin", "/nologo", "/headers", str(obj)],
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        if re.search(r"\.CRT\$X[CI]", headers):
            failures.append(str(obj))
    if not checked:
        raise RuntimeError("No built AVX objects found; initializer audit did not run")
    if failures:
        raise RuntimeError(
            "Load-time initializers in AVX objects:\n" + "\n".join(failures)
        )
    print(f"Checked {len(checked)} AVX objects: no load-time initializers")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    check_initializers(parser.parse_args().build_dir.resolve())
