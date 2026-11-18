#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Run numeric inverted-index benchmarks and summarize the results."""

from __future__ import annotations

import argparse
import datetime
import sys
import uuid
from itertools import product
from pathlib import Path

from interactive import (
    interactive_args,
    phase_progress,
    print_error,
    print_report_path,
    print_workload,
)
from report import BUILD_COLUMNS, QUERY_COLUMNS, render_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from benchmark_helpers import (
    binary_record,
    fail,
    number,
    report_details,
    run_worker,
    safe_name,
    summarize_metrics,
    timestamp,
    validate_metrics,
    write_json,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = "inverted_numeric_bench"
PROFILE_CHOICES = {
    "data_type": ("int32", "int64"),
    "distribution": ("uniform", "hotspot_moderate", "hotspot_extreme"),
    "value_range": ("0_to_30000", "0_to_type_max"),
    "value_order": ("ascending", "shuffled"),
    "null_ratio": (0, 0.1, 0.5),
}
DOCUMENT_COUNTS = (100000, 10000000)
DEFAULTS = {
    "distribution": "all",
    "value_range": "0_to_30000",
    "value_order": "ascending",
    "null_ratio": "0",
    "repetitions": 3,
    "bin_dir": ROOT / "build/bin",
}
BUILD_METRICS = tuple(key for _, key in BUILD_COLUMNS)
QUERY_METRICS = tuple(key for _, key in QUERY_COLUMNS)


def select_binary(args):
    executable = BENCHMARK + (".exe" if sys.platform == "win32" else "")
    binary = args.bin_dir.expanduser().resolve() / executable
    if not binary.is_file():
        fail(f"Missing numeric benchmark: {binary}; build target {BENCHMARK}")
    return binary


def select_profiles(args):
    """Expand selected values into every combination, in profile field order."""
    selections = []
    for name, choices in PROFILE_CHOICES.items():
        selected = getattr(args, name)
        selections.append(
            [
                (name, value)
                for value in choices
                if selected == "all" or str(value) == selected
            ]
        )
    return [dict(items) for items in product(*selections)]


def validate_dataset(dataset, profile, args):
    configuration = dataset.get("profile")
    if not isinstance(configuration, dict):
        fail("Dataset must contain a profile")
    expected = {
        **profile,
        "name": profile["distribution"],
        "num_docs": args.num_docs,
    }
    expected["type"] = expected.pop("data_type")
    for key, value in expected.items():
        if configuration.get(key) != value:
            fail(f"Dataset profile mismatch: {key}")
    entries = dataset.get("queries")
    if not isinstance(entries, list) or not entries:
        fail("Dataset must contain query cases")
    cases = {}
    for case in entries:
        if not isinstance(case, dict):
            fail("Invalid dataset query case")
        name = safe_name(case.get("name"))
        if name in cases:
            fail(f"Dataset contains duplicate query case: {name}")
        if not isinstance(case.get("inputs"), list) or not case["inputs"]:
            fail(f"Missing query inputs: {name}")
        cases[name] = case
    return cases


def validate_build(build):
    validate_metrics(build, BUILD_METRICS)
    for key in (
        "insert_seconds",
        "optimize_seconds",
        "total_seconds",
        "index_file_mib",
    ):
        # Valid measurements can round to zero in reports.
        number(build[key], key)


def validate_query(query, case, num_docs):
    name = case["name"]
    if query.get("name") != name:
        fail(f"Unexpected query case: {name}")
    counts = query.get("match_counts")
    if not isinstance(counts, list) or len(counts) != len(case["inputs"]):
        fail(f"Incomplete query hit counts: {name}")
    if any(type(count) is not int or not 0 <= count <= num_docs for count in counts):
        fail(f"Invalid query hit count: {name}")
    number(query["query_time_us"], "query_time_us")
    validate_metrics(query, QUERY_METRICS)


def validate_numeric_queries(report, cases, num_docs):
    entries = report.get("cases")
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("name"), str)
        for entry in entries
    ):
        fail("Numeric query report must contain query case results")
    queries = {entry["name"]: entry for entry in entries}
    if len(queries) != len(entries) or set(queries) != set(cases):
        fail("Numeric query report must cover exactly all workload cases")
    for name, query in queries.items():
        validate_query(query, cases[name], num_docs)
    return queries


def run_profile(binary, output, profile, args, workload):
    profile_dir = (
        output
        / profile["data_type"]
        / profile["distribution"]
        / profile["value_range"]
        / profile["value_order"]
        / f"null_{profile['null_ratio'] * 100:.0f}_percent"
    )
    profile_dir.mkdir(parents=True)
    builds, queries = [], {}
    expected_dataset = None
    for repetition in range(1, args.repetitions + 1):
        directory = profile_dir / f"repeat-{repetition:03d}"
        directory.mkdir()
        logs = directory / "logs"
        prefix = f"Workload {workload}, repetition {repetition}/{args.repetitions}"
        with (
            phase_progress(f"{prefix} prepare"),
            run_worker(
                binary,
                "prepare",
                logs / "prepare.log",
                report=directory / "dataset.json",
                run_dir=directory,
                num_docs=args.num_docs,
                **profile,
            ) as dataset,
        ):
            cases = validate_dataset(dataset, profile, args)
            if expected_dataset is None:
                expected_dataset = dataset
                queries = {name: [] for name in cases}
            elif dataset != expected_dataset:
                fail(f"Data or query inputs changed across repetitions: {prefix}")
        with (
            phase_progress(f"{prefix} build"),
            run_worker(
                binary,
                "build",
                logs / "build.log",
                report=directory / "build.json",
                run_dir=directory,
            ) as build,
        ):
            validate_build(build)
        with (
            phase_progress(f"{prefix} verify"),
            run_worker(
                binary,
                "verify",
                logs / "verify.log",
                run_dir=directory,
            ),
        ):
            pass
        with (
            phase_progress(f"{prefix} query (all cases)"),
            run_worker(
                binary,
                "query",
                logs / "query.log",
                report=directory / "query.json",
                run_dir=directory,
            ) as report,
        ):
            query_results = validate_numeric_queries(report, cases, args.num_docs)
        builds.append(build)
        for name in cases:
            queries[name].append(query_results[name])
    return {
        "suite": "numeric",
        "type": profile["data_type"],
        "profile": profile["distribution"],
        "dataset": expected_dataset,
        "environment": builds[0]["environment"],
        "directory": str(profile_dir.relative_to(output)),
        "build": summarize_metrics(builds, BUILD_METRICS),
        "build_details": [report_details(sample, BUILD_METRICS) for sample in builds],
        "cases": [
            {
                "name": name,
                "metrics": summarize_metrics(samples, QUERY_METRICS),
                "details": [
                    report_details(sample, QUERY_METRICS) for sample in samples
                ],
            }
            for name, samples in queries.items()
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--data_type",
        required=True,
        choices=(*PROFILE_CHOICES["data_type"], "all"),
        help="Numeric data type, or all",
    )
    parser.add_argument(
        "--num_docs",
        type=int,
        choices=DOCUMENT_COUNTS,
        required=True,
        help="Document count per workload",
    )
    parser.add_argument(
        "--distribution",
        default=DEFAULTS["distribution"],
        choices=(*PROFILE_CHOICES["distribution"], "all"),
        help="Value distribution (default: all)",
    )
    parser.add_argument(
        "--value_range",
        choices=(*PROFILE_CHOICES["value_range"], "all"),
        default=DEFAULTS["value_range"],
        help="Value domain: [0, 30000) (default), [0, type maximum), or all",
    )
    parser.add_argument(
        "--value_order",
        choices=(*PROFILE_CHOICES["value_order"], "all"),
        default=DEFAULTS["value_order"],
        help="Order of non-NULL values across document IDs (default: ascending)",
    )
    parser.add_argument(
        "--null_ratio",
        choices=(*(str(value) for value in PROFILE_CHOICES["null_ratio"]), "all"),
        default=DEFAULTS["null_ratio"],
        help="NULL document fraction, or all (default: 0)",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Output directory for data, indexes, and reports; must not already exist "
        "(default: build/results/inverted/numeric/<run-id>)",
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULTS["repetitions"])
    parser.add_argument(
        "--bin_dir",
        type=Path,
        default=DEFAULTS["bin_dir"],
        help="Directory containing inverted_numeric_bench (default: build/bin)",
    )
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main(argv=None):
    output, metadata = None, None
    try:
        argv = sys.argv[1:] if argv is None else argv
        run_id = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        default_run_dir = ROOT / "build/results/inverted/numeric" / run_id
        if not argv and sys.stdin.isatty() and sys.stdout.isatty():
            argv = interactive_args(
                PROFILE_CHOICES,
                DOCUMENT_COUNTS,
                {**DEFAULTS, "run_dir": default_run_dir},
            )
            if argv is None:
                return 0
        args = parse_args(argv)
        profiles = select_profiles(args)
        binary = select_binary(args)
        output = (args.run_dir or default_run_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        metadata = {
            "run_id": run_id,
            "status": "running",
            "started_at": timestamp(),
            "binaries": {"numeric": binary_record(binary)},
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected": [{"suite": "numeric", **profile} for profile in profiles],
        }
        write_json(output / "run.json", metadata)
        summaries = []
        for index, profile in enumerate(profiles, 1):
            print_workload(index, len(profiles), profile)
            summaries.append(run_profile(binary, output, profile, args, index))
        summary = {
            "run_id": run_id,
            "repetitions": args.repetitions,
            "profiles": summaries,
        }
        for index, result in enumerate(summaries):
            metadata["selected"][index]["cases"] = [
                case["name"] for case in result["cases"]
            ]
        metadata.update(status="passed", finished_at=timestamp())
        report_path = output / "report.md"
        temporary_report = report_path.with_suffix(".md.tmp")
        temporary_report.write_text(render_report(metadata, summary), encoding="utf-8")
        write_json(output / "run.json", metadata)
        # Publish only after both the report and the final run status are saved.
        temporary_report.replace(report_path)
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (KeyboardInterrupt, EOFError))
        message = (
            "Interrupted"
            if isinstance(error, EOFError)
            else str(error) or "Interrupted"
        )
        if metadata is not None:
            metadata.update(
                status="failed",
                finished_at=timestamp(),
                error=str(error) or "Interrupted",
            )
            try:
                write_json(output / "run.json", metadata)
            except OSError as status_error:
                message += f"\n  Could not save run status: {status_error}"
        print_error(message)
        return 130 if interrupted else 1

    print_report_path(report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
