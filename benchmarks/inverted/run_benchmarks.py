#!/usr/bin/env python3
# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Run isolated inverted-index benchmarks and aggregate their JSON reports."""

# This command-line tool intentionally prints its catalog and run progress.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
BENCHMARKS = {
    "numeric": "inverted_numeric_bench",
    "array": "inverted_array_bench",
}
TYPE_SUITES = {"int32": "numeric", "int64": "numeric", "array_int32": "array"}
NUMERIC_SEED = 20261115
NUMERIC_WARMUP_SECONDS = 0.2
NUMERIC_MEASURE_SECONDS = 1.0


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fail(message):
    raise RuntimeError(message)


def load_json(text):
    def invalid_constant(value):
        fail(f"Invalid JSON numeric constant: {value}")

    result = json.loads(text, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        fail("Expected a JSON object")
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def names(value):
    if not value:
        return set()
    result = {part.strip() for part in value.split(",")}
    if "" in result:
        fail("Filters must contain nonempty comma-separated names")
    return result


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        fail(f"Invalid name in benchmark catalog: {value!r}")
    return value


def select_binaries(args):
    if args.suite != "all" and args.suite not in BENCHMARKS:
        fail(f"Unknown suite: {args.suite}")
    requested_types = set() if args.type in (None, "all") else names(args.type)
    if args.type not in (None, "all") and not requested_types:
        fail("Select at least one data type, or all")
    if requested_types - TYPE_SUITES.keys():
        fail(
            f"Unknown types: {', '.join(sorted(requested_types - TYPE_SUITES.keys()))}"
        )
    suites = (
        {TYPE_SUITES[data_type] for data_type in requested_types}
        if requested_types
        else set(BENCHMARKS)
    )
    if args.suite != "all":
        if requested_types and suites != {args.suite}:
            fail("Requested data types do not belong to the selected suite")
        suites = {args.suite}
    binaries = {}
    directory = args.bin_dir.expanduser().resolve()
    for suite, target in BENCHMARKS.items():
        if suite in suites:
            binary = directory / target
            if not binary.is_file():
                fail(f"Missing {suite} benchmark: {binary}; build target {target}")
            binaries[suite] = binary
    return binaries


def data_type_flags(profile):
    return {"data_type": profile["type"]} if profile["suite"] == "numeric" else {}


def binary_record(binary):
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(binary), "sha256": digest.hexdigest()}


def select_profiles(catalog, args):
    if catalog.get("schema_version") != SCHEMA_VERSION:
        fail("Unsupported benchmark catalog version")
    suites = {suite["name"] for suite in catalog["suites"]}
    if args.suite != "all" and args.suite not in suites:
        fail(f"Unknown suite: {args.suite}")
    requested_types = set() if args.type in (None, "all") else names(args.type)
    if args.type not in (None, "all") and not requested_types:
        fail("Select at least one data type, or all")
    available_types = {
        profile["type"] for suite in catalog["suites"] for profile in suite["profiles"]
    }
    if requested_types - available_types:
        fail(f"Unknown types: {', '.join(sorted(requested_types - available_types))}")
    requested_profiles = names(args.profile)
    candidates = []
    for suite in catalog["suites"]:
        suite_name = safe_name(suite["name"])
        if args.suite not in ("all", suite_name):
            continue
        for profile in suite["profiles"]:
            safe_name(profile["name"])
            safe_name(profile["type"])
            if requested_types and profile["type"] not in requested_types:
                continue
            if requested_profiles and profile["name"] not in requested_profiles:
                continue
            if not requested_profiles and profile["supplemental"] and not args.list:
                continue
            for case in profile["cases"]:
                safe_name(case)
            candidates.append({**profile, "suite": suite_name})
    missing_types = requested_types - {p["type"] for p in candidates}
    if missing_types:
        fail(f"No profiles for selected types: {', '.join(sorted(missing_types))}")
    missing_profiles = requested_profiles - {p["name"] for p in candidates}
    if missing_profiles:
        fail(f"Unknown profiles for selection: {', '.join(sorted(missing_profiles))}")
    if not candidates:
        fail("No benchmark profiles selected")
    return candidates


def select_cases(candidates, case_filter):
    requested_cases = names(case_filter)
    available_cases = {case for p in candidates for case in p["cases"]}
    if requested_cases - available_cases:
        fail(
            f"Unknown cases for selected profiles: {', '.join(sorted(requested_cases - available_cases))}"
        )
    selected = []
    for profile in candidates:
        cases = [
            case
            for case in profile["cases"]
            if not requested_cases or case in requested_cases
        ]
        if cases:
            selected.append((profile, cases))
    if not selected:
        fail("No benchmark cases selected")
    return selected


def read_catalog(binary, **flags):
    command = [str(binary), "--mode=list"]
    command.extend(f"--{key}={value}" for key, value in flags.items())
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        fail(f"Cannot read benchmark catalog: {completed.stderr.strip()}")
    return load_json(completed.stdout)


def read_catalogs(binaries):
    suites = []
    for name, binary in binaries.items():
        catalog = read_catalog(binary)
        entries = catalog.get("suites", [])
        if (
            catalog.get("schema_version") != SCHEMA_VERSION
            or len(entries) != 1
            or entries[0]["name"] != name
        ):
            fail(f"Unexpected catalog from {binary}")
        if any(
            TYPE_SUITES.get(profile["type"]) != name
            for profile in entries[0]["profiles"]
        ):
            fail(f"Catalog data types do not belong to {name}")
        suites.extend(entries)
    return {"schema_version": SCHEMA_VERSION, "suites": suites}


def configuration(args, profiles):
    common = ("null_ratio",)
    numeric = (
        "value_range",
        "value_order",
        "hot_value_count",
        "hot_doc_ratio",
    )
    array = ("cardinality", "array_length", "empty_ratio", "duplicate_ratio")
    overrides = {
        name: getattr(args, name)
        for name in (*common, *numeric, *array)
        if getattr(args, name) is not None
    }
    for suite, fields in (("numeric", numeric), ("array", array)):
        supplied = set(fields) & overrides.keys()
        if supplied and any(profile["suite"] != suite for profile in profiles):
            fail(
                f"{', '.join('--' + name.replace('_', '-') for name in sorted(supplied))} "
                f"requires selecting only the {suite} suite"
            )
    return overrides


def resolve_profiles(binaries, profiles, overrides):
    if not overrides:
        return profiles
    resolved = []
    encoded = json.dumps(overrides, allow_nan=False)
    for profile in profiles:
        catalog = read_catalog(
            binaries[profile["suite"]],
            profile=profile["name"],
            config=encoded,
            **data_type_flags(profile),
        )
        entries = [
            {**entry, "suite": suite["name"]}
            for suite in catalog["suites"]
            for entry in suite["profiles"]
        ]
        if len(entries) != 1 or any(
            entries[0][key] != profile[key] for key in ("suite", "type", "name")
        ):
            fail("Resolved catalog did not identify exactly the requested profile")
        resolved.extend(entries)
    return resolved


def run_worker(binary, phase, report, log, *, explicit_output=True, **flags):
    command = [str(binary), f"--mode={phase}"]
    if explicit_output:
        command.append(f"--output={report}")
    command.extend(f"--{key}={value}" for key, value in flags.items())
    with log.open("x") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode:
        fail(f"{phase} exited with {completed.returncode}; see {log}")
    if not report.is_file():
        fail(f"{phase} produced no report; see {log}")
    result = load_json(report.read_text())
    if result.get("schema_version") != SCHEMA_VERSION or result.get("phase") != phase:
        fail(f"Unexpected phase in {report}")
    return result


def number(value, label, positive=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        fail(f"Invalid metric {label}: {value!r}")
    if value < 0 or (positive and value == 0):
        fail(f"Invalid metric {label}: {value!r}")
    return value


def validate_build(build, profile, args):
    dataset = build["dataset"]
    expected = {
        key: value for key, value in profile.items() if key not in ("cases", "suite")
    }
    seed = NUMERIC_SEED if profile["suite"] == "numeric" else args.seed
    expected.update(num_docs=args.num_docs, seed=seed)
    for key, value in expected.items():
        if dataset.get(key) != value:
            fail(f"Build dataset mismatch: {key}")
    if (
        not isinstance(dataset.get("data_fingerprint"), str)
        or not dataset["data_fingerprint"]
    ):
        fail("Build is missing its data fingerprint")
    validate_metrics(build)
    for key in (
        "insert_seconds",
        "seal_seconds",
        "build_seconds",
        "docs_per_second",
        "index_file_bytes",
        "bytes_per_doc",
    ):
        if key in build:
            number(build[key], key, positive=True)
    cases = {case["name"]: case for case in build["cases"]}
    if len(cases) != len(build["cases"]) or set(cases) != set(profile["cases"]):
        fail("Build cases disagree with the catalog")
    return cases


def validate_verification(verification, build, cases, selected):
    if verification.get("passed") is not True:
        fail("Correctness verification did not pass")
    if verification.get("data_fingerprint") != build["dataset"]["data_fingerprint"]:
        fail("Build and verification data fingerprints differ")
    verified = {case["name"]: case for case in verification["cases"]}
    if len(verified) != len(verification["cases"]) or set(verified) != set(selected):
        fail("Verification did not cover exactly the selected cases")
    for name in selected:
        case, result = cases[name], verified[name]
        if result["query_fingerprint"] != case["query_fingerprint"]:
            fail(f"Verification query fingerprint mismatch: {name}")
        if not isinstance(case["inputs"], list) or not case["inputs"]:
            fail(f"Missing query inputs: {name}")
        for field in ("counts", "id_fingerprints"):
            if field in result and (
                not isinstance(result[field], list)
                or len(result[field]) != len(case["inputs"])
            ):
                fail(f"Incomplete verification {field}: {name}")
        for count in result.get("counts", []):
            if type(count) is not int or not 0 <= count <= build["dataset"]["num_docs"]:
                fail(f"Invalid verification hit count: {name}")
    return verified


def validate_query(query, name, build, verified):
    if query.get("case") != name:
        fail(f"Unexpected query case: {name}")
    if query.get("data_fingerprint") != build["dataset"]["data_fingerprint"]:
        fail(f"Query data fingerprint mismatch: {name}")
    if query["query_fingerprint"] != verified["query_fingerprint"]:
        fail(f"Query input fingerprint mismatch: {name}")
    if "match_counts" in query and query["match_counts"] != verified.get("counts"):
        fail(f"Query hit counts differ from verification: {name}")
    if "completed_queries" in query:
        if type(query["completed_queries"]) is not int:
            fail(f"Invalid completed query count: {name}")
        number(query["completed_queries"], "completed_queries", positive=True)
    for key in ("elapsed_seconds", "mean_ns_per_query", "qps"):
        if key in query:
            number(query[key], key, positive=True)
    validate_metrics(query)


def metric_value(report, key):
    value = report
    for component in key.split("."):
        if not isinstance(value, dict) or component not in value:
            fail(f"Missing declared metric: {key}")
        value = value[component]
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value)
    ):
        fail(f"Invalid declared metric {key}: {value!r}")
    return value


def validate_metrics(report):
    keys = report.get("metric_names")
    if (
        not isinstance(keys, list)
        or not keys
        or any(not isinstance(key, str) or not all(key.split(".")) for key in keys)
    ):
        fail("Report must declare nonempty metric_names")
    if len(set(keys)) != len(keys):
        fail("Report declares duplicate metric names")
    for key in keys:
        metric_value(report, key)
    return keys


def summarize_metrics(samples):
    keys = validate_metrics(samples[0])
    if any(set(validate_metrics(sample)) != set(keys) for sample in samples[1:]):
        fail("Metric declarations changed across repetitions")
    result = {}
    for key in keys:
        values = [metric_value(sample, key) for sample in samples]
        result[key] = (
            {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            if all(value is not None for value in values)
            else None
        )
    return result


def report_details(report):
    details = copy.deepcopy(report)
    for key in (
        "schema_version",
        "phase",
        "metric_names",
        "case",
        "name",
        "dataset",
        "cases",
        "index_path",
        "environment",
        "data_fingerprint",
        "query_fingerprint",
    ):
        details.pop(key, None)
    for key in report.get("metric_names", []):
        components = key.split(".")
        parent = details
        for component in components[:-1]:
            parent = parent.get(component, {})
        parent.pop(components[-1], None)
    return {key: value for key, value in details.items() if value != {}}


def run_profile(binary, output, profile, selected, args, overrides):
    numeric = profile["suite"] == "numeric"
    profile_dir = output / profile["type"] / profile["name"]
    profile_dir.mkdir(parents=True)
    builds, queries = [], {name: [] for name in selected}
    verifications = {name: [] for name in selected}
    expected_fingerprint = None
    expected_queries = None
    for repetition in range(1, args.repetitions + 1):
        directory = profile_dir / f"repeat-{repetition:03d}"
        logs = directory / "logs"
        logs.mkdir(parents=True)
        index = directory / "index"
        build_path, verify_path = directory / "build.json", directory / "verify.json"
        prefix = (
            f"{profile['type']}/{profile['name']} [{repetition}/{args.repetitions}]"
        )
        build_flags = (
            {"run_dir": directory, "data_type": profile["type"]}
            if numeric
            else {"index": index, "seed": args.seed}
        )
        print(f"{prefix} build", flush=True)
        build = run_worker(
            binary,
            "build",
            build_path,
            logs / "build.log",
            explicit_output=not numeric,
            **build_flags,
            profile=profile["name"],
            num_docs=args.num_docs,
            config=json.dumps(overrides, allow_nan=False),
        )
        cases = validate_build(build, profile, args)
        fingerprint = build["dataset"]["data_fingerprint"]
        query_fingerprints = {
            name: cases[name]["query_fingerprint"] for name in selected
        }
        if expected_fingerprint is not None and (
            fingerprint != expected_fingerprint
            or query_fingerprints != expected_queries
        ):
            fail(f"Data or query inputs changed across repetitions: {prefix}")
        expected_fingerprint, expected_queries = fingerprint, query_fingerprints
        print(f"{prefix} verify", flush=True)
        verify_flags = (
            {"run_dir": directory}
            if numeric
            else {"index": index, "manifest": build_path}
        )
        verification = run_worker(
            binary,
            "verify",
            verify_path,
            logs / "verify.log",
            explicit_output=not numeric,
            **verify_flags,
            case=",".join(selected),
        )
        verified = validate_verification(verification, build, cases, selected)
        builds.append(build)
        for name in selected:
            verifications[name].append(report_details(verified[name]))
            print(f"{prefix} {name}", flush=True)
            query_flags = (
                {"run_dir": directory}
                if numeric
                else {
                    "index": index,
                    "manifest": build_path,
                    "verification": verify_path,
                    "warmup_seconds": args.warmup_seconds,
                    "measure_seconds": args.measure_seconds,
                }
            )
            query = run_worker(
                binary,
                "query",
                directory / f"query-{name}.json",
                logs / f"query-{name}.log",
                explicit_output=not numeric,
                **query_flags,
                case=name,
            )
            validate_query(query, name, build, verified[name])
            queries[name].append(query)
    return {
        "suite": profile["suite"],
        "type": profile["type"],
        "profile": profile["name"],
        "dataset": builds[0]["dataset"],
        "directory": str(profile_dir.relative_to(output)),
        "build": summarize_metrics(builds),
        "build_details": [report_details(sample) for sample in builds],
        "cases": [
            {
                "name": name,
                "query_fingerprint": expected_queries[name],
                "metrics": summarize_metrics(samples),
                "details": [report_details(sample) for sample in samples],
                "verification": verifications[name],
            }
            for name, samples in queries.items()
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=ROOT / "build/bin",
        help="Directory containing the selected benchmark executables",
    )
    parser.add_argument("--suite", default="all", help="Suite name, or all (default)")
    parser.add_argument(
        "--type",
        help="Comma-separated data types, or all; required for numeric runs",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        help="Document count: numeric accepts 100000 or 10000000; array accepts any positive uint32",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Comma-separated profile names; default: all primary profiles",
    )
    parser.add_argument(
        "--case",
        default="",
        help="Comma-separated case names; default: all matching cases",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=20261115,
        help="Array: random seed; numeric uses fixed seed 20261115",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.2,
        help="Array: warmup duration; numeric uses fixed 0.2 seconds",
    )
    parser.add_argument(
        "--measure-seconds",
        type=float,
        default=1.0,
        help="Array: measurement duration; numeric uses fixed 1.0 seconds",
    )
    parser.add_argument("--cardinality", type=int, help="Array: value domain size")
    parser.add_argument(
        "--null-ratio",
        type=float,
        help="NULL document ratio; numeric accepts 0, 0.1, or 0.5",
    )
    parser.add_argument(
        "--value-range",
        choices=("zero_to_type_max", "zero_to_30000"),
        help="Numeric: [0, type maximum] or [0, 30000], including both bounds",
    )
    parser.add_argument(
        "--value-order",
        choices=("shuffled", "ascending"),
        help="Numeric: order of non-NULL values across document IDs",
    )
    parser.add_argument(
        "--hot-value-count", type=int, help="Numeric: number of hot values"
    )
    parser.add_argument(
        "--hot-doc-ratio",
        type=float,
        help="Numeric: fraction of non-NULL documents on hot keys",
    )
    parser.add_argument(
        "--array-length", type=int, help="Array: raw nonempty input length"
    )
    parser.add_argument("--empty-ratio", type=float, help="Array: empty document ratio")
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        help="Array: duplicate ratio among nonempty documents",
    )
    parser.add_argument(
        "--output", type=Path, help="New result directory; must not already exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching profiles and cases without running",
    )
    args = parser.parse_args(argv)
    if not args.list and args.num_docs is None:
        parser.error("--num-docs is required unless --list is used")
    for name in ("num_docs", "cardinality", "array_length", "hot_value_count"):
        value = getattr(args, name)
        if value is not None and not 0 < value < 2**32:
            parser.error(f"--{name.replace('_', '-')} must be a positive uint32")
    for name in ("null_ratio", "empty_ratio", "duplicate_ratio"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            parser.error(
                f"--{name.replace('_', '-')} must be finite and between 0 and 1"
            )
    # The numeric catalog validates ranges for the selected distribution only.
    if args.hot_doc_ratio is not None and not math.isfinite(args.hot_doc_ratio):
        parser.error("--hot-doc-ratio must be finite")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be a nonnegative signed 64-bit integer")
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("--warmup-seconds must be finite and nonnegative")
    if not math.isfinite(args.measure_seconds) or args.measure_seconds <= 0:
        parser.error("--measure-seconds must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output, metadata = None, None
    try:
        binaries = select_binaries(args)
        profiles = select_profiles(read_catalogs(binaries), args)
        if (
            args.num_docs is not None
            and any(profile["suite"] == "numeric" for profile in profiles)
            and args.num_docs not in (100000, 10000000)
        ):
            fail("Numeric --num-docs must be 100000 or 10000000")
        overrides = configuration(args, profiles)
        profiles = resolve_profiles(binaries, profiles, overrides)
        selected = select_cases(profiles, args.case)
        if args.list:
            for profile, cases in selected:
                suffix = " (supplemental)" if profile["supplemental"] else ""
                print(
                    f"{profile['suite']}/{profile['type']}/{profile['name']}{suffix}: "
                    f"{', '.join(cases)}"
                )
            return 0
        if any(profile["suite"] == "numeric" for profile, _ in selected):
            if args.type is None:
                fail("Numeric runs require an explicit --type, such as int32 or int64")
            if (
                args.seed != NUMERIC_SEED
                or args.warmup_seconds != NUMERIC_WARMUP_SECONDS
                or args.measure_seconds != NUMERIC_MEASURE_SECONDS
            ):
                fail(
                    "Numeric runs use fixed seed 20261115, warmup 0.2 s, and "
                    "measurement 1.0 s; timing and seed overrides require the array suite"
                )
        run_id = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        output = (
            (args.output or ROOT / "build/results/inverted" / run_id)
            .expanduser()
            .resolve()
        )
        output.mkdir(parents=True, exist_ok=False)
        selected_suites = {profile["suite"] for profile, _ in selected}
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "running",
            "started_at": timestamp(),
            "binaries": {
                suite: binary_record(binary)
                for suite, binary in binaries.items()
                if suite in selected_suites
            },
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected": [
                {
                    "suite": p["suite"],
                    "type": p["type"],
                    "profile": p["name"],
                    "cases": cases,
                }
                for p, cases in selected
            ],
            "configuration_overrides": overrides,
        }
        write_json(output / "run.json", metadata)
        summaries = [
            run_profile(
                binaries[profile["suite"]], output, profile, cases, args, overrides
            )
            for profile, cases in selected
        ]
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "repetitions": args.repetitions,
            "profiles": summaries,
        }
        write_json(output / "summary.json", summary)
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
