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
"""Run reproducible issue-868 comparisons; preserve raw logs and commands."""

# Command-line benchmark progress is intentionally written to stdout.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():  # noqa: PLR0915 - Keep the ordered benchmark phases together.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=10_000_000)
    parser.add_argument("--stage", choices=["all", "build", "query"], default="all")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--query-cpus", default="")
    parser.add_argument("--sign-macos", action="store_true")
    parser.add_argument("--python-driver", type=Path)
    parser.add_argument("--scalar-range-path", type=Path)
    parser.add_argument("--scalar-bsi-path", type=Path)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    if bool(args.scalar_range_path) != bool(args.scalar_bsi_path):
        parser.error("Provide both scalar index paths when reusing prepared data")
    scalar_paths = {
        "TERM_RANGE": args.scalar_range_path,
        "TERM_BSI": args.scalar_bsi_path,
    }
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    def metadata():
        return {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "rows": args.rows,
            "binary": str(args.binary),
            "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
            "scenario": "https://github.com/alibaba/zvec/issues/868#issuecomment-6593377888",
            "seed": 868,
            "timestamp": "splitmix64(id + seed) & INT32_MAX",
            "vectors": "62 independent deterministic uniform float32 components in [-1, 1)",
            "pid": "id % 1024",
            "hnsw": {
                "metric": "COSINE",
                "quantize": "INT8",
                "m": 16,
                "ef_construction": 200,
                "ef_search": 300,
            },
            "filter": "dataTime > 536870911",
            "topk": 21,
            "query_pool_threads": 8,
            "build_threads": 8,
            "memory_gib": 8,
            "warmup": 64,
            "query_vectors": 256,
            "query_cpus": args.query_cpus,
            "seconds_per_round": args.seconds,
            "passes": args.passes,
            "cache": "warm; correctness scan and warm-up precede timing; no global cache eviction",
            "sign_macos": args.sign_macos,
            "python_driver": str(args.python_driver) if args.python_driver else None,
            "reused_scalar_paths": {
                key: str(path) for key, path in scalar_paths.items() if path
            },
        }

    (root / "metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")

    def run(name, mode, layout, path, *extra, timed=False):
        marker = logs / (name + ".done.json")
        command = [
            str(args.binary),
            f"--mode={mode}",
            f"--layout={layout}",
            f"--path={path}",
            f"--rows={args.rows}",
            *extra,
        ]
        if args.python_driver:
            command = [sys.executable, str(args.python_driver.resolve()), *command[1:]]
        if timed and args.query_cpus:
            command = ["taskset", "-c", args.query_cpus, *command]
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["command"] != command:
                raise RuntimeError(f"Existing run used different arguments: {name}")
            print(f"Already completed: {name}", flush=True)
            return
        if args.sign_macos:
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(args.binary)], check=True
            )
        print(f"Starting {name}", flush=True)
        start = time.time()
        with (logs / (name + ".log")).open("w") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        record = {
            "command": command,
            "started_at_unix": start,
            "elapsed_s": time.time() - start,
        }
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Completed {name}: {record['elapsed_s']:.1f}s", flush=True)

    if args.stage in ("all", "build"):
        for layout in ("TERM_RANGE", "TERM_BSI"):
            if scalar_paths[layout]:
                if not scalar_paths[layout].is_dir():
                    raise RuntimeError(
                        f"Missing prepared scalar index: {scalar_paths[layout]}"
                    )
            else:
                run(
                    f"build_scalar_{layout}",
                    "scalar_build",
                    layout,
                    root / f"scalar_{layout}",
                )
        run(
            "build_hybrid_TERM_RANGE",
            "hybrid_build",
            "TERM_RANGE",
            root / "hybrid_TERM_RANGE",
        )
        source, destination = root / "hybrid_TERM_RANGE", root / "hybrid_TERM_BSI"
        clone_marker = logs / "clone.done.json"
        if not clone_marker.exists():
            if destination.exists():
                raise RuntimeError(
                    "Unfinished clone exists; preserve it and use a fresh work directory"
                )
            copy_command = (
                ["/bin/cp", "-cR"]
                if platform.system() == "Darwin"
                else ["cp", "--reflink=auto", "-a"]
            )
            subprocess.run([*copy_command, str(source), str(destination)], check=True)
            clone_marker.write_text(
                json.dumps({"source": str(source), "destination": str(destination)})
                + "\n"
            )
        run("switch_hybrid_TERM_BSI", "switch", "TERM_BSI", destination)

    if args.stage in ("all", "query"):
        reference = root / "ann_reference.txt"
        for mode in ("scalar", "hybrid"):
            for clients in (1, 8):
                for repetition in range(args.passes):
                    layouts = (
                        ("TERM_RANGE", "TERM_BSI")
                        if repetition % 2 == 0
                        else ("TERM_BSI", "TERM_RANGE")
                    )
                    for layout in layouts:
                        extra = [
                            f"--threads={clients}",
                            "--rounds=1",
                            f"--seconds={args.seconds}",
                        ]
                        if mode == "hybrid":
                            extra.append(f"--reference={reference}")
                        run(
                            f"query_{mode}_{layout}_t{clients}_p{repetition}",
                            f"{mode}_query",
                            layout,
                            (scalar_paths[layout] or root / f"{mode}_{layout}")
                            if mode == "scalar"
                            else root / f"{mode}_{layout}",
                            *extra,
                            timed=True,
                        )
        for clients in (1, 8):
            run(
                f"query_unfiltered_t{clients}",
                "hybrid_query",
                "TERM_RANGE",
                root / "hybrid_TERM_RANGE",
                "--unfiltered",
                f"--threads={clients}",
                "--rounds=1",
                f"--seconds={args.seconds}",
                timed=True,
            )

    results = []
    for log in sorted(logs.glob("*.log")):
        for line in log.read_text().splitlines():
            if line.startswith('{"event":'):
                results.append({"run": log.stem, **json.loads(line)})
    (root / "results.json").write_text(
        json.dumps({"metadata": metadata(), "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
