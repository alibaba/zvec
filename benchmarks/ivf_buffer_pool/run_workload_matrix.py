#!/usr/bin/env python3
"""Run the Cohere IVF storage matrix serially and write one reproducible CSV."""

from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
from pathlib import Path


WORKLOADS = {
    "uniform_unique": ("cyclic", 1000),
    "centroid_80_20": ("centroid80", 1000),
    "zipf_alpha_1": ("zipf1", 1000),
    "exact_repeat_90_10": ("hot90", 5),
}


def parse_result(output: str) -> dict[str, str]:
    line = next(
        (value for value in output.splitlines() if value.startswith("result ")),
        None,
    )
    if line is None:
        raise RuntimeError("benchmark emitted no result line")
    result: dict[str, str] = {}
    for field in line.split()[1:]:
        key, separator, value = field.partition("=")
        if separator:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run each point in a fresh process and never run points in parallel. "
            "This avoids the memory exhaustion caused by overlapping mmap pools."
        )
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool-mb", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nprobe", type=int, default=4)
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument("--measure-seconds", type=int, default=15)
    parser.add_argument("--active-queries", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.threads <= 0 or args.nprobe <= 0:
        parser.error("--threads and --nprobe must be positive")
    if args.warmup_seconds <= 0 or args.measure_seconds <= 0:
        parser.error("phase durations must be positive")
    if any(value <= 0 for value in args.pool_mb):
        parser.error("Buffer pool budgets must be positive")

    points: list[tuple[str, int, str, str, int]] = []
    for workload, (pattern, hot_queries) in WORKLOADS.items():
        points.append(("mmap", 0, workload, pattern, hot_queries))
        points.extend(
            ("buffer", budget, workload, pattern, hot_queries)
            for budget in args.pool_mb
        )

    commands: list[tuple[str, list[str]]] = []
    for storage, pool_mb, workload, pattern, hot_queries in points:
        label = f"{workload}_{storage}_{pool_mb}mb"
        command = [
            str(args.binary),
            str(args.index),
            str(args.queries),
            storage,
            str(pool_mb),
            str(args.threads),
            str(args.nprobe),
            str(args.warmup_seconds),
            str(args.measure_seconds),
            "none",
            str(args.active_queries),
            "0",
            pattern,
            str(hot_queries),
        ]
        commands.append((label, command))

    if args.dry_run:
        for _, command in commands:
            for _ in range(args.runs):
                print(shlex.join(command))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for label, command in commands:
        for run in range(1, args.runs + 1):
            run_label = f"{label}_run{run}"
            print(f"running {run_label}", flush=True)
            completed = subprocess.run(command, text=True, capture_output=True)
            log_path = args.output_dir / f"{run_label}.log"
            log_path.write_text(completed.stdout + completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"{run_label} failed; see {log_path}")
            result = parse_result(completed.stdout)
            result["run"] = str(run)
            result["workload"] = label.rsplit("_", 2)[0]
            result["log"] = log_path.name
            rows.append(result)

    csv_path = args.output_dir / "matrix.csv"
    fieldnames = ["workload", *rows[0].keys()]
    fieldnames = list(dict.fromkeys(fieldnames))
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    groups = sorted({(row["workload"], row["storage"], row["pool_mb"]) for row in rows})
    for workload, storage, pool_mb in groups:
        group = [
            row
            for row in rows
            if row["workload"] == workload
            and row["storage"] == storage
            and row["pool_mb"] == pool_mb
        ]
        qps_values = [float(row["qps"]) for row in group]
        summary_rows.append(
            {
                "workload": workload,
                "storage": storage,
                "pool_mb": pool_mb,
                "runs": len(group),
                "median_qps": statistics.median(qps_values),
                "qps_stdev": statistics.stdev(qps_values)
                if len(qps_values) > 1
                else 0.0,
                "median_p99_ms": statistics.median(
                    float(row["p99_ms"]) for row in group
                ),
                "median_rss_peak_mib": statistics.median(
                    float(row["rss_peak_bytes"]) / (1024 * 1024)
                    for row in group
                ),
                "median_centroid_top20_share": statistics.median(
                    float(row["centroid_top20_share"]) for row in group
                ),
            }
        )

    mmap_by_workload = {
        row["workload"]: row
        for row in summary_rows
        if row["storage"] == "mmap"
    }
    for row in summary_rows:
        baseline = mmap_by_workload[row["workload"]]
        rss_gib = float(row["median_rss_peak_mib"]) / 1024
        baseline_rss_gib = float(baseline["median_rss_peak_mib"]) / 1024
        row["qps_per_gib"] = float(row["median_qps"]) / rss_gib
        baseline_density = float(baseline["median_qps"]) / baseline_rss_gib
        row["qps_vs_mmap"] = (
            float(row["median_qps"]) / float(baseline["median_qps"])
        )
        row["rss_vs_mmap"] = (
            float(row["median_rss_peak_mib"])
            / float(baseline["median_rss_peak_mib"])
        )
        row["qps_per_gib_vs_mmap"] = row["qps_per_gib"] / baseline_density
    summary_path = args.output_dir / "matrix_summary.csv"
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
