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

from statistics import mean

BUILD_COLUMNS = (
    ("Insert (s)", "insert_seconds"),
    ("Optimize (s)", "optimize_seconds"),
    ("Total (s)", "total_seconds"),
    ("Index (MiB)", "index_file_mib"),
    ("Peak RSS (MiB)", "process_memory.peak_rss_mib"),
    ("Final RSS (MiB)", "process_memory.final_rss_mib"),
)
QUERY_COLUMNS = (("Mean query time (µs)", "query_time_us"),)


def _number(value):
    return "N/A" if value is None else f"{value:,.2f}"


def _table(headers, rows):
    lines = []
    for row in (headers, ["---"] * len(headers), *rows):
        cells = [str(cell).replace("|", r"\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _query_table(profile):
    rows = []
    for case in profile["cases"]:
        matches_per_query = mean(
            count for detail in case["details"] for count in detail["match_counts"]
        )
        rows.append(
            [case["name"]]
            + [_number(case["metrics"][key]) for _, key in QUERY_COLUMNS]
            + [_number(matches_per_query)]
        )
    return _table(
        ["Case"] + [label for label, _ in QUERY_COLUMNS] + ["Matches per query"],
        rows,
    )


def render_report(run, summary):
    """Render completed workloads without reading or modifying their raw results."""
    profiles = summary["profiles"]
    environment = profiles[0]["environment"]
    lines = [
        "# Numeric inverted-index benchmark report",
        "",
        f"- Run: `{run['run_id']}`",
        f"- Status: {run['status']}",
        f"- Started: {run['started_at']}",
        "",
        "## Environment",
        "",
        _table(
            [
                "Platform",
                "Target architecture",
                "Compiler",
                "`__cplusplus`",
                "Hardware threads",
            ],
            [
                [
                    environment["platform"],
                    environment["architecture"],
                    environment.get("compiler", "N/A"),
                    environment["cplusplus"],
                    environment["hardware_threads"] or "N/A",
                ]
            ],
        ),
        "",
        "## Workloads",
        "",
        f"Repetitions per workload: {summary['repetitions']}",
        "",
    ]
    workload_rows = []
    stats_rows = []
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
        stats = profile["dataset"]["stats"]
        docs_per_value = stats["documents_per_value"]
        stats_rows.append(
            [
                index,
                f"{stats['null_count']:,}",
                f"{stats['distinct_value_count']:,}",
                f"{stats['min_value']:,}",
                f"{stats['max_value']:,}",
                " / ".join(
                    f"{docs_per_value[q]:,}" for q in ("min", "p50", "p95", "max")
                ),
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
            "### Observed dataset statistics",
            "",
            _table(
                [
                    "Workload",
                    "NULL documents",
                    "Distinct non-NULL values",
                    "Min value",
                    "Max value",
                    "Documents per non-NULL value (min / p50 / p95 / max)",
                ],
                stats_rows,
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
    return "\n".join(lines)
