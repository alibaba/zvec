"""Process isolation and provenance helpers for scalar-filter public API tools."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def library(directory):
    name = "libzvec.dylib" if sys.platform == "darwin" else "libzvec.so"
    return (Path(directory).resolve(strict=True) / name).resolve(strict=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(lib, jit, dependencies=(), legacy_direct=False):
    env = os.environ.copy()
    dirs = [
        str(Path(lib).parent),
        *(str(Path(p).resolve(strict=True)) for p in dependencies),
    ]
    for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        old = env.get(variable)
        env[variable] = os.pathsep.join(dirs + ([old] if old else []))
    env["ZVEC_SCALAR_JIT"] = "1" if jit else "0"
    # Optional old hardcoded comparator uses its original feature switches.
    # No profiling API is called by these executables or runners.
    env["ZVEC_ISSUE868_FORWARD_QUERY_LOCK"] = "1" if legacy_direct else "0"
    env["ZVEC_ISSUE868_BOUND_INT32"] = "1" if legacy_direct else "0"
    env.pop("ZVEC_ISSUE868_BITMAP_DUMP", None)
    env.pop("ZVEC_ISSUE868_FORWARD_TRACE", None)
    return env


def run(command, env, stdout_path, stderr_path):
    with Path(stdout_path).open("w") as stdout, Path(stderr_path).open("w") as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    require(code == 0, f"Command exited {code}; inspect {stderr_path}")


def records(path):
    result = []
    for line in Path(path).read_text().splitlines():
        # Runtime logs may share stdout; result dumps are parsed strictly.
        with suppress(json.JSONDecodeError):
            result.append(json.loads(line))
    return result


def new_output(path):
    path = Path(path).resolve()
    require(
        not path.exists() or not any(path.iterdir()),
        f"Output directory must be absent or empty: {path}",
    )
    path.mkdir(parents=True, exist_ok=True)
    return path
