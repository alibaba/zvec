"""Require actual JIT execution through the installed C++ shared SDK.

The benchmark executable uses only installed public headers and libzvec.so.
Its scalar-field and result checks also run with JIT disabled, and optional
QEMU runs must produce the same ordered IDs, scores and returned fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--library-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--qemu", type=Path)
    parser.add_argument("--cpu", default="Haswell-v4")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary, collection = args.binary.resolve(), args.collection.resolve()
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(args.library_directory.resolve())

    def execute(name: str, options: list[str], prefix: list[str], enabled: bool):
        log = output / f"{name}.log"
        result = output / f"{name}.json"
        env = dict(environment, ZVEC_SCALAR_JIT="1" if enabled else "0")
        command = prefix + [str(binary), "--path", str(collection),
                            "--output", str(result)] + options
        with log.open("wb") as stream:
            subprocess.run(command, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        return json.loads(result.read_text()), log.read_text(errors="replace")

    if not collection.exists():
        execute("build", ["--mode", "build", "--rows", "2048",
                          "--workers", "1", "--build-workers", "1"], [], True)

    common = ["--mode", "dump", "--strategy", "forward", "--count", "16",
              "--workers", "1", "--debug", "1"]
    runs = [("native-jit", [], True), ("native-arrow", [], False)]
    if args.qemu:
        prefix = [str(args.qemu.resolve()), "-cpu", args.cpu]
        runs += [("haswell-jit", prefix, True), ("haswell-arrow", prefix, False)]
    reference = None
    checks = []
    for name, prefix, enabled in runs:
        result, log = execute(name, common, prefix, enabled)
        queries = result["queries"]
        assert len(queries) == 16, (name, len(queries))
        if reference is None:
            reference = queries
        assert queries == reference, f"{name}: IDs/scores/fields differ"
        ready = log.count("Scalar JIT forward filter ready")
        if enabled:
            assert ready >= len(queries), (name, ready)
            assert "Detected CPU Name" in log, name
            assert "Scalar JIT unavailable" not in log, name
            if prefix:
                assert "+avx2" in log and "-avx512f" in log, name
                assert "+avx512f" not in log, name
        else:
            assert ready == 0, (name, ready)
        checks.append({"name": name, "queries": len(queries),
                       "jit_bindings": ready,
                       "cpu_names": re.findall(r"Detected CPU Name : (\S+)", log)})
    library = args.library_directory / "libzvec.so"
    summary = {"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
               "checks": checks, "all_results_equal": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
