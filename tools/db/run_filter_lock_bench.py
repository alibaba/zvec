# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare per-candidate locked fetch directly with owning views using Arrow.

Example (an optimized INT8 collection must already exist):
  python3 tools/db/run_filter_lock_bench.py --cmake /path/to/cmake-3.31.12 \
    --build-dir build31 --index /path/to/collection --out /path/to/new-results

The output directory must not exist. No source or input dataset is modified.
With --delete-stride=N (N >= 2), a fresh copy under the output directory gets
one tombstone per N primary keys, without rebuilding/compacting the index.
Validation checks predicate/result count and excludes tombstoned keys; it is
not an exact ANN recall measurement. Debug preflights verify access paths.
Both builds share query lifecycle plumbing; the locked control is upstream-style,
not a pristine upstream binary. Deletion-enabled runs retain deletion-path locks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROW = re.compile(
    r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s*$",
    re.M,
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def capture(args):
    return subprocess.check_output([str(x) for x in args], cwd=ROOT, text=True)


def source_state():
    files = capture(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "src",
            "tools/db",
            "tests/db",
            "cmake",
            "CMakeLists.txt",
        ]
    )
    return {
        name: digest(ROOT / name)
        for name in sorted(set(files.split("\0")))
        if name and (ROOT / name).is_file()
    }


def main():  # noqa: PLR0915 -- Keep provenance checks and measurements in execution order.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--docs", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delete-stride", type=int, default=0)
    args = parser.parse_args()
    if args.delete_stride < 0 or args.delete_stride == 1:
        parser.error("delete-stride must be 0 or at least 2")
    if min(args.docs, args.queries, args.rounds) <= 0 or args.warmup < 0:
        parser.error("counts must be positive (warmup may be zero)")
    cmake, build, index, out = (
        p.resolve() for p in (args.cmake, args.build_dir, args.index, args.out)
    )
    version = capture([cmake, "--version"])
    if version.splitlines()[0] != "cmake version 3.31.12":
        raise RuntimeError("This experiment requires the requested CMake 3.31.12")
    if not index.is_dir():
        raise RuntimeError("Build the INT8 synthetic collection first")
    if out == index or index in out.parents:
        raise RuntimeError("Results must be outside the input collection")
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(command, name):
        command = [str(x) for x in command]
        print(name, flush=True)  # noqa: T201 -- CLI progress output.
        started = time.time()
        with (out / name).open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
        commands.append(
            {
                "command": command,
                "log": name,
                "started": started,
                "duration": time.time() - started,
                "returncode": result.returncode,
            }
        )
        (out / "commands.json").write_text(json.dumps(commands, indent=2))
        result.check_returncode()
        return (out / name).read_text()

    state = source_state()
    (out / "source.patch").write_text(capture(["git", "diff", "--binary", "HEAD"]))
    untracked = capture(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "src",
            "tools/db",
            "tests/db",
        ]
    ).splitlines()
    (out / "new-sources.json").write_text(
        json.dumps({name: (ROOT / name).read_text() for name in untracked}, indent=2)
    )
    run(
        [cmake, "-S", ROOT, "-B", build, "-DZVEC_FORWARD_FILTER_SANITIZERS=OFF"],
        "configure.log",
    )
    targets = {"locked": "filter_lock_bench_locked", "view": "filter_view_bench"}
    run(
        [cmake, "--build", build, "--target", *targets.values(), "-j", "8"], "build.log"
    )
    if source_state() != state:
        raise RuntimeError("Source changed during build; discard this experiment")
    binaries = {mode: build / "bin" / target for mode, target in targets.items()}
    modes = tuple(binaries)
    hashes = {mode: digest(binary) for mode, binary in binaries.items()}
    cache = (build / "CMakeCache.txt").read_text()
    (out / "CMakeCache.txt").write_text(cache)
    compiler_entry = re.search(
        r"^CMAKE_CXX_COMPILER:(?:FILEPATH|STRING)=(.+)$", cache, re.M
    )
    if not compiler_entry:
        raise RuntimeError("Cannot identify the configured C++ compiler")
    compiler = compiler_entry.group(1)
    manifest = {
        "head": capture(["git", "rev-parse", "HEAD"]).strip(),
        "sources": state,
        "binary_sha256": hashes,
        "cmake": version,
        "compiler": capture([compiler, "--version"]),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "index": str(index),
        "docs": args.docs,
        "workload": "synthetic INT8 HNSW, not an exact PR #889 reproduction",
        "data_seed": 42,
        "query_seed": 7,
        "query_pool": 1024,
        "dim": 62,
        "topk": 21,
        "ef": 300,
        "m": 16,
        "ef_construction": 200,
        "query_threads": 12,
        "queries_per_client": args.queries,
        "warmup_per_client": args.warmup,
        "rounds": args.rounds,
        "validation": True,
        "experiment": "locked-versus-view",
        "baseline": "upstream-style per-candidate locking with shared query plumbing",
        "evaluator": "Arrow scalar expression",
        "modes": modes,
    }

    # Fingerprint the input and the actual measured dataset outside timed work.
    def dataset_state(path):
        return {
            str(p.relative_to(path)): digest(p)
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }

    input_index = index
    input_state = dataset_state(input_index)
    if args.delete_stride:
        index = out / "deleted-data"
        run(
            [
                binaries["view"],
                "-cmd=clone_deleted",
                f"-source_index={input_index}",
                f"-index={index}",
                f"-docs={args.docs}",
                f"-delete_stride={args.delete_stride}",
            ],
            "clone-deleted.log",
        )
    manifest.update(
        index=str(index),
        input_index=str(input_index),
        input_dataset_sha256=input_state,
        dataset_sha256=dataset_state(index),
        delete_stride=args.delete_stride,
        validation_scope="result count, predicate, key range, tombstone exclusion",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    common = [
        f"-index={index}",
        f"-docs={args.docs}",
        "-dim=62",
        "-topk=21",
        f"-delete_stride={args.delete_stride}",
        "-ef=300",
        "-m=16",
        "-efc=200",
        "-quantization=int8",
        "-pool=1024",
        "-query_threads=12",
        "-ratio=0",
        "-invert=true",
        "-verify=true",
    ]
    for mode, binary in binaries.items():
        text = run(
            [
                binary,
                "-cmd=run",
                *common,
                "-threads=1",
                "-queries=1",
                "-warmup=0",
                "-rounds=1",
                "-debug_log=true",
            ],
            f"preflight-{mode}.log",
        )
        locked = "Forward filter locked fetch:" in text
        views = "Forward filter view:" in text
        path_ok = views and not locked if mode == "view" else locked and not views
        banner = "owning-view-arrow" if mode == "view" else "per-candidate-locked-arrow"
        if (
            "All invert conds moved to forward cond" not in text
            or not path_ok
            or f"build_mode={banner}" not in text
            or f"deletion_stride={args.delete_stride} " not in text
            or (mode == "view" and f"rows[{args.docs}]" not in text)
        ):
            raise RuntimeError(f"Actual {mode} forward-filter path was not verified")
    points = []
    for filtered in (True, False):
        label = "filtered" if filtered else "unfiltered"
        for round_id in range(args.rounds):
            order = modes if round_id % 2 == 0 else modes[::-1]
            for mode in order:
                text = run(
                    [
                        binaries[mode],
                        "-cmd=run",
                        *common,
                        "-threads=1,2,4,8",
                        f"-queries={args.queries}",
                        f"-warmup={args.warmup}",
                        "-rounds=1",
                        f"-selectivity={0.75 if filtered else 0}",
                        "-debug_log=false",
                    ],
                    f"{label}-{round_id}-{mode}.log",
                )
                rows = ROW.findall(text)
                if [int(row[0]) for row in rows] != [1, 2, 4, 8]:
                    raise RuntimeError("Missing measured concurrency point")
                for clients, _, qps, p50, p99, seconds, queries, bad in rows:
                    if int(bad) or int(queries) != int(clients) * args.queries:
                        raise RuntimeError("Correctness/accounting validation failed")
                    points.append(
                        {
                            "workload": label,
                            "round": round_id,
                            "mode": mode,
                            "clients": int(clients),
                            "qps": float(qps),
                            "p50_us": float(p50),
                            "p99_us": float(p99),
                            "seconds": float(seconds),
                            "queries": int(queries),
                            "failures": int(bad),
                        }
                    )
                (out / "points.json").write_text(json.dumps(points, indent=2))
    run(
        [binaries["locked"], "-cmd=lockmicro", "-threads=1,2,4,8", "-micro_secs=3"],
        "lockmicro.log",
    )
    run(
        [
            binaries["view"],
            "-cmd=snapshotmicro",
            f"-index={out / 'snapshot-data'}",
            f"-docs={args.docs}",
            "-queries=16",
        ],
        "snapshotmicro.log",
    )
    if source_state() != state or hashes != {
        mode: digest(path) for mode, path in binaries.items()
    }:
        raise RuntimeError("Source/binary provenance changed during measurements")
    if (
        dataset_state(input_index) != input_state
        or dataset_state(index) != manifest["dataset_sha256"]
    ):
        raise RuntimeError("Dataset changed during measurements")
    summary = []
    for workload in ("filtered", "unfiltered"):
        medians = {
            (mode, clients): statistics.median(
                p["qps"]
                for p in points
                if p["workload"] == workload
                and p["mode"] == mode
                and p["clients"] == clients
            )
            for mode in binaries
            for clients in (1, 2, 4, 8)
        }
        for clients in (1, 2, 4, 8):
            baseline, candidate = modes
            a, b = (medians[(mode, clients)] for mode in modes)
            summary.append(
                {
                    "workload": workload,
                    "clients": clients,
                    "delete_stride": args.delete_stride,
                    f"{baseline}_qps": a,
                    f"{candidate}_qps": b,
                    f"{candidate}_over_{baseline}": b / a,
                    f"{baseline}_scaling": a / medians[(baseline, 1)],
                    f"{candidate}_scaling": b / medians[(candidate, 1)],
                }
            )
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)  # noqa: T201 -- CLI results.


if __name__ == "__main__":
    main()
