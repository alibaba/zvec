#!/usr/bin/env python3
"""Compare complete public-API results with ZVEC_SCALAR_JIT=0/1."""

# CLI progress and the final JSON summary intentionally use stdout.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from runner_common import environment, library, new_output, require, run, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--dependency-lib-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    lib = library(args.lib_dir)
    out = new_output(args.output_dir)
    fixtures = out / "fixtures"
    run(
        [str(binary), "create", str(fixtures)],
        environment(lib, False, args.dependency_lib_dir),
        out / "create.stdout.log",
        out / "create.stderr.log",
    )
    comparisons = []
    for name, rows, deleted, writable in (
        ("single", 384, False, False),
        ("multi", 2304, False, False),
        ("deleted", 384, True, False),
        ("writable_mixed", 384, False, True),
    ):
        dumps = []
        invocations = []
        for enabled in (False, True):
            case = f"{name}_jit{int(enabled)}"
            dataset = fixtures / name
            if writable:
                # Separate copies preserve identical persisted graphs while
                # each process inserts its own live, unflushed writing rows.
                dataset = out / f"{case}_dataset"
                shutil.copytree(fixtures / "single", dataset)
            dump = out / f"{case}.jsonl"
            command = [
                str(binary),
                "query",
                str(dataset),
                "--rows",
                str(rows),
                "--deleted",
                str(int(deleted)),
                "--writable",
                str(int(writable)),
                "--dump",
                str(dump),
            ]
            print(case, flush=True)
            run(
                command,
                environment(lib, enabled, args.dependency_lib_dir),
                out / f"{case}.stdout.log",
                out / f"{case}.stderr.log",
            )
            data = [json.loads(line) for line in dump.read_text().splitlines()]
            require(
                len(data) == 128, f"{case}: expected 15x8 queries and 8 group queries"
            )
            require(
                sum(row["kind"] == "query" for row in data) == 120,
                f"{case}: missing ordinary query records",
            )
            require(
                sum(row["kind"] == "group_by" for row in data) == 8,
                f"{case}: missing group records",
            )
            dumps.append(data)
            invocations.append({"command": command, "jit": enabled, "dump": str(dump)})
        require(
            dumps[0] == dumps[1],
            f"{name}: JIT changed IDs/order/scores/scalar values/groups; inspect JSONL dumps",
        )
        comparisons.append(
            {
                "fixture": name,
                "records": len(dumps[0]),
                "invert_to_forward_scan_ratio": 0.0 if writable else 0.5,
                "exact_results_equal": True,
                "invocations": invocations,
            }
        )
    summary = {
        "library": str(lib),
        "library_sha256": sha256(lib),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "comparisons": comparisons,
        "scope": "Public result equivalence; does not prove internal fast-path activation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "fixtures": len(comparisons),
                "records_per_flag": sum(item["records"] for item in comparisons),
            }
        )
    )


if __name__ == "__main__":
    main()
