#!/usr/bin/env python3
"""Run the publishable DiskANN storage matrix, one fresh process per point."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKLOADS = ("uniform_unique", "semantic80", "zipf1", "exact90")
MIB = 1024 * 1024


def parse_events(output: str) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = value.get("event")
        if event:
            events[event] = value
    return events


def build_points(args: argparse.Namespace) -> list[tuple[str, int, str, Path]]:
    points: list[tuple[str, int, str, Path]] = []
    for workload in args.workloads:
        points.append(("mmap", args.mmap_memory_mb, workload, args.mmap_collection))
        for budget in args.buffer_memory_mb:
            points.append(("buffer_pool", budget, workload, args.buffer_collection))
    return points


def command_for(
    args: argparse.Namespace,
    mode: str,
    memory_mb: int,
    workload: str,
    collection: Path,
) -> list[str]:
    return [
        args.python,
        str(Path(__file__).with_name("benchmark.py")),
        "query",
        "--mode",
        mode,
        "--collection",
        str(collection),
        "--query-file",
        str(args.query_file),
        "--memory-mb",
        str(memory_mb),
        "--threads",
        str(args.engine_threads),
        "--client-threads",
        str(args.client_threads),
        "--list-size",
        str(args.list_size),
        "--topk",
        str(args.topk),
        "--repetitions",
        str(args.repetitions),
        "--workload",
        workload,
        "--workload-seed",
        str(args.workload_seed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run mmap/direct-I/O and Buffer Pool points serially. Every point "
            "uses a fresh process so cache and RSS state cannot leak across runs."
        )
    )
    parser.add_argument("--mmap-collection", type=Path, required=True)
    parser.add_argument("--buffer-collection", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--buffer-memory-mb", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--mmap-memory-mb", type=int, default=4096)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--engine-threads", type=int, default=4)
    parser.add_argument("--client-threads", type=int, default=4)
    parser.add_argument("--list-size", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workload-seed", type=int, default=20260825)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0 or args.repetitions <= 0:
        parser.error("--runs and --repetitions must be positive")
    if args.engine_threads <= 0 or args.client_threads <= 0:
        parser.error("thread counts must be positive")
    if any(value <= 0 for value in args.buffer_memory_mb):
        parser.error("Buffer memory budgets must be positive")

    points = build_points(args)
    if args.dry_run:
        for point in points:
            for _ in range(args.runs):
                print(shlex.join(command_for(args, *point)))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for mode, memory_mb, workload, collection in points:
        for run in range(1, args.runs + 1):
            command = command_for(args, mode, memory_mb, workload, collection)
            label = f"{workload}_{mode}_{memory_mb}mb_run{run}"
            print(f"running {label}", flush=True)
            completed = subprocess.run(command, text=True, capture_output=True)
            log_path = args.output_dir / f"{label}.log"
            log_path.write_text(completed.stdout + completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{label} failed; see {log_path}")

            events = parse_events(completed.stdout)
            ready = events["process_ready"]
            opened = events["open_done"]
            first = events["first_pass_done"]
            steady = events["steady_pass_done"]
            done = events["query_done"]
            metrics = steady["metrics"]
            fingerprint = steady["result_fingerprint"]
            previous = fingerprints.setdefault(workload, fingerprint)
            if previous != fingerprint:
                raise RuntimeError(
                    f"result mismatch for {workload}: {previous} != {fingerprint}"
                )
            peak_rss = max(
                opened["peak_rss"],
                first["peak_rss"],
                steady["peak_rss"],
                done["memory"].get("VmHWM", 0),
            )
            qps_per_gib = metrics["qps"] / (peak_rss / (1024**3))
            rows.append(
                {
                    "run": run,
                    "workload": workload,
                    "mode": mode,
                    "memory_mb": memory_mb,
                    "engine_threads": ready["engine_threads"],
                    "client_threads": ready["client_threads"],
                    "list_size": args.list_size,
                    "qps": metrics["qps"],
                    "p50_ms": metrics["p50_ms"],
                    "p95_ms": metrics["p95_ms"],
                    "p99_ms": metrics["p99_ms"],
                    "peak_rss_mib": peak_rss / MIB,
                    "qps_per_gib": qps_per_gib,
                    "read_mib": (
                        done["memory"].get("read_bytes", 0)
                        - first["memory"].get("read_bytes", 0)
                    )
                    / MIB,
                    "self_hit_at_k": metrics["self_recall_at_k"],
                    "unique_queries": metrics["unique_queries"],
                    "query_top20_share": metrics["query_top20_share"],
                    "configured_hot_share": metrics["configured_hot_share"],
                    "open_ms": opened["elapsed_seconds"] * 1000,
                    "fingerprint": fingerprint,
                    "log": log_path.name,
                }
            )

    csv_path = args.output_dir / "matrix.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, Any]] = []
    numeric = (
        "qps",
        "p99_ms",
        "peak_rss_mib",
        "qps_per_gib",
        "read_mib",
        "self_hit_at_k",
        "query_top20_share",
        "configured_hot_share",
    )
    for workload in args.workloads:
        for mode, memory_mb in [("mmap", args.mmap_memory_mb), *[
            ("buffer_pool", value) for value in args.buffer_memory_mb
        ]]:
            group = [
                row
                for row in rows
                if row["workload"] == workload
                and row["mode"] == mode
                and row["memory_mb"] == memory_mb
            ]
            summary: dict[str, Any] = {
                "workload": workload,
                "mode": mode,
                "memory_mb": memory_mb,
                "runs": len(group),
                "qps_stdev": statistics.stdev(row["qps"] for row in group)
                if len(group) > 1
                else 0.0,
            }
            summary.update(
                {
                    f"median_{field}": statistics.median(
                        row[field] for row in group
                    )
                    for field in numeric
                }
            )
            summary_rows.append(summary)

    mmap_by_workload = {
        row["workload"]: row
        for row in summary_rows
        if row["mode"] == "mmap"
    }
    for row in summary_rows:
        baseline = mmap_by_workload[row["workload"]]
        row["qps_vs_mmap"] = row["median_qps"] / baseline["median_qps"]
        row["rss_vs_mmap"] = (
            row["median_peak_rss_mib"] / baseline["median_peak_rss_mib"]
        )
        row["qps_per_gib_vs_mmap"] = (
            row["median_qps_per_gib"] / baseline["median_qps_per_gib"]
        )
        row["physical_read_vs_mmap"] = (
            row["median_read_mib"] / baseline["median_read_mib"]
            if baseline["median_read_mib"]
            else 0.0
        )
    summary_path = args.output_dir / "matrix_summary.csv"
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
