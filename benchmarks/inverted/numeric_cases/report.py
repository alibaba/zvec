# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Markdown reports for numeric inverted-index benchmarks."""

from __future__ import annotations

import datetime

BUILD_COLUMNS = (
    ("Insert (s)", "insert_seconds"),
    ("Optimize (s)", "optimize_seconds"),
    ("Total (s)", "total_seconds"),
    ("Index (MiB)", "index_file_mib"),
    ("Peak RSS (MiB)", "process_memory.peak_rss_mib"),
    ("Final RSS (MiB)", "process_memory.final_rss_mib"),
)
QUERY_COLUMNS = (("Query time (µs)", "query_time_us"),)


def _number(value):
    return "N/A" if value is None else f"{value:,.2f}"


def _table(headers, rows):
    lines = []
    for row in (headers, ["---"] * len(headers), *rows):
        cells = [str(cell).replace("|", r"\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _query_table(profile):
    inputs = {
        case["name"]: len(case["inputs"]) for case in profile["dataset"]["queries"]
    }
    rows = []
    for case in profile["cases"]:
        counts = [str(detail["match_counts"]) for detail in case["details"]]
        # Keep one list when repetitions agree; otherwise show each repetition.
        matches = (
            counts[0]
            if all(value == counts[0] for value in counts)
            else "; ".join(f"run {i}: {value}" for i, value in enumerate(counts, 1))
        )
        rows.append(
            [case["name"], inputs[case["name"]]]
            + [_number(case["metrics"][key]) for _, key in QUERY_COLUMNS]
            + [matches]
        )
    return _table(
        ["Case", "Inputs"] + [label for label, _ in QUERY_COLUMNS] + ["Matches/input"],
        rows,
    )


def render_report(run, summary):
    """Render completed workloads without reading or modifying their raw results."""
    profiles = summary["profiles"]
    elapsed = (
        datetime.datetime.fromisoformat(run["finished_at"])
        - datetime.datetime.fromisoformat(run["started_at"])
    ).total_seconds()
    lines = [
        "# Numeric inverted-index benchmark report",
        "",
        f"- Run: `{run['run_id']}`",
        f"- Status: {run['status']}",
        f"- Started: {run['started_at']}",
        f"- Total wall time: {elapsed:.2f} s",
        f"- Repetitions per workload: {summary['repetitions']}",
        "",
        "Metrics are arithmetic means across repetitions. "
        "`N/A` means a measurement was unavailable.",
        "",
        "## Workloads",
        "",
    ]
    workload_rows = []
    for index, profile in enumerate(profiles, 1):
        configuration = profile["dataset"]["profile"]
        workload_rows.append(
            [
                index,
                profile["type"],
                profile["profile"],
                f"{configuration['num_docs']:,}",
                configuration["value_range"],
                configuration["value_order"],
                f"{configuration['null_ratio'] * 100:.2f}%",
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "Workload",
                    "Data type",
                    "Distribution",
                    "Documents",
                    "Value range",
                    "Order",
                    "NULL",
                ],
                workload_rows,
            ),
            "",
            "## Build",
            "",
            _table(
                ["Workload"] + [label for label, _ in BUILD_COLUMNS],
                [
                    [index]
                    + [_number(profile["build"][key]) for _, key in BUILD_COLUMNS]
                    for index, profile in enumerate(profiles, 1)
                ],
            ),
            "",
            "## Queries",
            "",
            "Matches/input lists the document count for each query input in dataset order.",
            "",
        ]
    )
    for index, profile in enumerate(profiles, 1):
        configuration = profile["dataset"]["profile"]
        lines.extend(
            [
                f"### Workload {index}: {profile['type']} / {profile['profile']}",
                "",
                f"{configuration['value_range']} / {configuration['value_order']} / "
                f"NULL {configuration['null_ratio'] * 100:.2f}%",
                "",
                _query_table(profile),
                "",
            ]
        )
    lines.extend(
        [
            "## Measurement notes",
            "",
            "- Query time is the mean time per query in microseconds. "
            "Each input runs 100 times per repetition, without warmup. Query timing "
            "includes search, result counting, and release; index opening, input "
            "preparation, and document-ID verification are excluded.",
            "- Build total is insert plus optimize time. Insert includes dataset reads; "
            "optimize includes sealing, flushing, and compaction. Index size excludes "
            "dataset files and reports.",
            "- Build RSS measures the whole process. Build runs in a fresh process "
            "after dataset preparation.",
            "- The benchmark calls the index on one thread; RocksDB can use background "
            "threads. OS caches are not cleared between repetitions.",
            "- Floating-point values use two decimal places. "
            "Independently rounded values may not sum exactly.",
            "",
        ]
    )
    return "\n".join(lines)
