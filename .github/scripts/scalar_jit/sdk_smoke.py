# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Require actual JIT execution through the installed C++ shared SDK.

The smoke executable uses only installed public headers and libzvec.so.
Its scalar-field and result checks also run with JIT disabled, and optional
QEMU runs must produce the same ordered IDs, scores and returned fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--library-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--qemu", type=Path)
    parser.add_argument("--cpu", action="append", choices=("Haswell-v4", "Nehalem"))
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915 - sequential acceptance checks and failure report
    args = parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary, collection = args.binary.resolve(), args.collection.resolve()
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(args.library_directory.resolve())

    def execute(name: str, options: list[str], prefix: list[str], enabled: bool):
        log = output / f"{name}.log"
        result = output / f"{name}.json"
        env = dict(
            environment,
            ZVEC_SCALAR_JIT="1" if enabled else "0",
            ZVEC_SCALAR_JIT_PROFILE="1",
        )
        command = [
            *prefix,
            str(binary),
            "--path",
            str(collection),
            "--output",
            str(result),
            *options,
        ]
        with log.open("wb") as stream:
            subprocess.run(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        return json.loads(result.read_text()), log.read_text(errors="replace")

    common = ["--mode", "query"]
    runs = [("native-jit", [], True), ("native-arrow", [], False)]
    if args.qemu:
        for cpu in args.cpu or ["Haswell-v4", "Nehalem"]:
            prefix = [str(args.qemu.resolve()), "-cpu", cpu]
            runs += [(f"{cpu}-jit", prefix, True), (f"{cpu}-arrow", prefix, False)]
    reference = None
    checks = []
    libraries = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.library_directory.glob("*.so*")
        if p.is_file()
    }
    summary = {"library_sha256": libraries, "checks": checks, "passed": False}
    try:
        assert libraries, "No SDK shared libraries found"
        if not collection.exists():
            execute("build", ["--mode", "build"], [], True)
        for name, prefix, enabled in runs:
            evidence = {"name": name, "library_sha256": libraries, "passed": False}
            checks.append(evidence)
            result, log = execute(name, common, prefix, enabled)
            queries = result["queries"]
            assert len(queries) == 16, (name, len(queries))
            if reference is None:
                reference = queries
            for actual, expected in zip(queries, reference, strict=True):
                assert actual["threshold"] == expected["threshold"]
                for doc, ref in zip(actual["docs"], expected["docs"], strict=True):
                    assert {k: v for k, v in doc.items() if k != "score"} == {
                        k: v for k, v in ref.items() if k != "score"
                    }
                    assert math.isclose(
                        doc["score"], ref["score"], rel_tol=2e-5, abs_tol=2e-5
                    )
            ready = log.count("Scalar JIT forward filter ready")
            samples = {
                backend: [
                    int(t)
                    for t in re.findall(
                        rf"Scalar JIT execution sample backend={backend} ns=(\d+)", log
                    )
                ]
                for backend in ("jit", "arrow")
            }
            cpu = re.findall(
                r"Scalar JIT CPU sse42=(\d) avx=(\d) avx2=(\d) avx512f=(\d)", log
            )
            assert cpu, name
            assert len(set(cpu)) == 1, (name, cpu)
            evidence["diagnostics"] = {
                "preparation_selections": ready,
                "execution_samples": {
                    k: {"count": len(v), "nanoseconds": sum(v)}
                    for k, v in samples.items()
                },
                "cpu_features": dict(
                    zip(
                        ("sse42", "avx", "avx2", "avx512f"),
                        map(int, cpu[0]),
                        strict=True,
                    )
                ),
                "cpu_names": re.findall(r"Detected CPU Name : (\S+)", log),
            }
            if prefix:
                assert cpu[0] == (
                    ("1", "1", "1", "0")
                    if prefix[-1] == "Haswell-v4"
                    else ("1", "0", "0", "0")
                ), (name, cpu)
            if enabled:
                # Four LIKE/IN queries cannot enter the callback under the
                # unchanged planner. Native differential tests cover helpers.
                assert ready >= len(queries) - 4, (name, ready)
                assert samples["jit"], name
                evidence["non_callback_cases"] = ["LIKE", "IN"]
                assert "Detected CPU Name" in log, name
                assert "Scalar JIT unavailable" not in log, name
            else:
                assert ready == 0, name
                assert not samples["jit"], name
                assert samples["arrow"], name
                assert "Detected CPU Name" not in log, name
            evidence["passed"] = True
            evidence["queries"] = len(queries)
        summary["passed"] = True
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        if checks and not checks[-1]["passed"]:
            checks[-1]["error"] = summary["error"]
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
