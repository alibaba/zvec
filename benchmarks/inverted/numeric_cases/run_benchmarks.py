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

# This command-line tool intentionally prints its workload preview and run progress.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import datetime
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from benchmark_helpers import (
    binary_record,
    fail,
    names,
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
DATA_TYPES = ("int32", "int64")
DISTRIBUTIONS = ("uniform", "hotspot_moderate", "hotspot_extreme")
BUILD_METRICS = (
    "insert_seconds",
    "optimize_seconds",
    "total_seconds",
    "index_file_mib",
    "process_memory.peak_rss_mib",
    "process_memory.final_rss_mib",
)
QUERY_METRICS = (
    "mean_ns_per_query",
    "qps",
    "memory_before.rss_mib",
    "memory_after.rss_mib",
    "memory_after.peak_rss_mib",
)


def select_binary(args):
    binary = args.bin_dir.expanduser().resolve() / BENCHMARK
    if not binary.is_file():
        fail(f"Missing numeric benchmark: {binary}; build target {BENCHMARK}")
    return binary


def select_profiles(args):
    requested_types = set() if args.type in (None, "all") else names(args.type)
    requested_profiles = names(args.profile)
    unknown = requested_profiles - set(DISTRIBUTIONS)
    if unknown:
        fail(f"Unknown profiles: {', '.join(sorted(unknown))}")
    return [
        {"type": data_type, "name": distribution}
        for data_type in DATA_TYPES
        for distribution in DISTRIBUTIONS
        if (not requested_types or data_type in requested_types)
        and (not requested_profiles or distribution in requested_profiles)
    ]


def configuration(args):
    return {
        name: getattr(args, name)
        for name in ("value_range", "value_order", "null_ratio")
        if getattr(args, name) is not None
    }


def workload_flags(profile, overrides):
    return {
        "data_type": profile["type"],
        "distribution": profile["name"],
        **overrides,
    }


def validate_dataset(dataset, profile, args, overrides):
    configuration = dataset.get("profile")
    if not isinstance(configuration, dict):
        fail("Dataset must contain a profile")
    expected = {
        "type": profile["type"],
        "name": profile["name"],
        "distribution": profile["name"],
        "num_docs": args.num_docs,
        **overrides,
    }
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
    if "completed_queries" in query:
        if type(query["completed_queries"]) is not int:
            fail(f"Invalid completed query count: {name}")
        number(query["completed_queries"], "completed_queries", positive=True)
    for key in ("elapsed_seconds", "mean_ns_per_query", "qps"):
        if key in query:
            number(query[key], key)
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


def run_profile(binary, output, profile, args, overrides):
    profile_dir = output / profile["type"] / profile["name"]
    profile_dir.mkdir(parents=True)
    builds, queries = [], {}
    expected_dataset = None
    for repetition in range(1, args.repetitions + 1):
        directory = profile_dir / f"repeat-{repetition:03d}"
        logs = directory / "logs"
        logs.mkdir(parents=True)
        prefix = (
            f"{profile['type']}/{profile['name']} [{repetition}/{args.repetitions}]"
        )
        print(f"{prefix} prepare", flush=True)
        dataset = run_worker(
            binary,
            "prepare",
            logs / "prepare.log",
            report=directory / "dataset.json",
            run_dir=directory,
            num_docs=args.num_docs,
            **workload_flags(profile, overrides),
        )
        cases = validate_dataset(dataset, profile, args, overrides)
        if expected_dataset is None:
            expected_dataset = dataset
            queries = {name: [] for name in cases}
        elif dataset != expected_dataset:
            fail(f"Data or query inputs changed across repetitions: {prefix}")
        print(f"{prefix} build", flush=True)
        build = run_worker(
            binary,
            "build",
            logs / "build.log",
            report=directory / "build.json",
            run_dir=directory,
        )
        validate_build(build)
        print(f"{prefix} verify", flush=True)
        run_worker(
            binary,
            "verify",
            logs / "verify.log",
            run_dir=directory,
        )
        print(f"{prefix} query (all cases)", flush=True)
        report = run_worker(
            binary,
            "query",
            logs / "query.log",
            report=directory / "query.json",
            run_dir=directory,
        )
        query_results = validate_numeric_queries(report, cases, args.num_docs)
        builds.append(build)
        for name in cases:
            queries[name].append(query_results[name])
    return {
        "suite": "numeric",
        "type": profile["type"],
        "profile": profile["name"],
        "dataset": expected_dataset,
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
        "--type",
        help="Comma-separated data types: int32, int64, or all; required to run",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        choices=(100000, 10000000),
        help="Document count; required to run",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Comma-separated distributions; default: all three distributions",
    )
    parser.add_argument(
        "--value-range",
        choices=("0_to_30000", "0_to_type_max"),
        help="Value domain: [0, 30000) or [0, type maximum)",
    )
    parser.add_argument(
        "--value-order",
        choices=("ascending", "shuffled"),
        help="Order of non-NULL values across document IDs",
    )
    parser.add_argument(
        "--null-ratio",
        type=float,
        choices=(0, 0.1, 0.5),
        help="NULL document fraction",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "build/bin",
        help="Directory containing inverted_numeric_bench",
    )
    parser.add_argument(
        "--output", type=Path, help="New result directory; must not already exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Preview selected workloads without running",
    )
    args = parser.parse_args(argv)
    if not args.list:
        if args.type is None:
            parser.error("--type is required unless --list is used")
        if args.num_docs is None:
            parser.error("--num-docs is required unless --list is used")
    if args.type not in (None, "all"):
        try:
            requested_types = names(args.type)
        except RuntimeError as error:
            parser.error(str(error))
        if not requested_types or requested_types - set(DATA_TYPES):
            parser.error("--type must select int32, int64, or all")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output, metadata = None, None
    try:
        profiles = select_profiles(args)
        overrides = configuration(args)
        if args.list:
            for profile in profiles:
                print(f"numeric/{profile['type']}/{profile['name']}")
            if args.num_docs is not None:
                print(f"Document count: {args.num_docs}")
            if overrides:
                print(
                    "Options: "
                    + " ".join(
                        f"--{name.replace('_', '-')}={value}"
                        for name, value in overrides.items()
                    )
                )
            return 0
        binary = select_binary(args)
        run_id = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        output = (
            (args.output or ROOT / "build/results/inverted/numeric" / run_id)
            .expanduser()
            .resolve()
        )
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
            "selected": [
                {
                    "suite": "numeric",
                    "type": p["type"],
                    "profile": p["name"],
                }
                for p in profiles
            ],
            "configuration_overrides": overrides,
        }
        write_json(output / "run.json", metadata)
        summaries = [
            run_profile(binary, output, profile, args, overrides)
            for profile in profiles
        ]
        summary = {
            "run_id": run_id,
            "repetitions": args.repetitions,
            "profiles": summaries,
        }
        write_json(output / "summary.json", summary)
        for index, result in enumerate(summaries):
            metadata["selected"][index]["cases"] = [
                case["name"] for case in result["cases"]
            ]
        metadata.update(status="passed", finished_at=timestamp())
        write_json(output / "run.json", metadata)
        print(f"Results: {output}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if metadata is not None:
            metadata.update(
                status="failed",
                finished_at=timestamp(),
                error=str(error) or "Interrupted",
            )
            write_json(output / "run.json", metadata)
        print(f"error: {str(error) or 'Interrupted'}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
