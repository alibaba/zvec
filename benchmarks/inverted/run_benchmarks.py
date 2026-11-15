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


def select_profiles(catalog, args):
    if catalog.get("schema_version") != 1:
        fail("Unsupported benchmark catalog version")
    if args.type != "all" and args.type not in {
        suite["name"] for suite in catalog["suites"]
    }:
        fail(f"Unknown type: {args.type}")
    requested_profiles, requested_cases = names(args.profile), names(args.case)
    candidates = []
    for suite in catalog["suites"]:
        suite_name = safe_name(suite["name"])
        if args.type not in ("all", suite_name):
            continue
        for profile in suite["profiles"]:
            safe_name(profile["name"])
            if profile["type"] != suite_name:
                fail("Catalog profile and suite types disagree")
            if requested_profiles and profile["name"] not in requested_profiles:
                continue
            if not requested_profiles and profile["supplemental"] and not args.list:
                continue
            for case in profile["cases"]:
                safe_name(case)
            candidates.append(profile)
    missing_profiles = requested_profiles - {p["name"] for p in candidates}
    if missing_profiles:
        fail(
            f"Unknown profiles for selected type: {', '.join(sorted(missing_profiles))}"
        )
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


def run_worker(binary, phase, report, log, **flags):
    command = [str(binary), f"--mode={phase}", f"--output={report}"]
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
    if result.get("phase") != phase:
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
        key: profile[key]
        for key in ("name", "type", "distribution", "cardinality", "array_length")
    }
    expected.update(num_docs=args.num_docs, seed=args.seed)
    for key, value in expected.items():
        if dataset.get(key) != value:
            fail(f"Build dataset mismatch: {key}")
    if (
        not isinstance(dataset.get("data_fingerprint"), str)
        or not dataset["data_fingerprint"]
    ):
        fail("Build is missing its data fingerprint")
    for key in (
        "insert_seconds",
        "seal_seconds",
        "build_seconds",
        "docs_per_second",
        "index_file_bytes",
        "bytes_per_doc",
    ):
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
        if not case["inputs"] or len(result["counts"]) != len(case["inputs"]):
            fail(f"Incomplete verification counts: {name}")
        if len(result["id_fingerprints"]) != len(case["inputs"]):
            fail(f"Incomplete verification fingerprints: {name}")
        for count in result["counts"]:
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
    if query["match_counts"] != verified["counts"]:
        fail(f"Query hit counts differ from verification: {name}")
    if type(query["completed_queries"]) is not int:
        fail(f"Invalid completed query count: {name}")
    for key in ("completed_queries", "elapsed_seconds", "mean_ns_per_query", "qps"):
        number(query[key], key, positive=True)


def summarize_metrics(samples, keys):
    result = {}
    for key in keys:
        values = []
        for sample in samples:
            value = sample
            for component in key.split("."):
                value = value.get(component) if isinstance(value, dict) else None
            if value is not None:
                values.append(number(value, key))
        result[key] = (
            {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            if len(values) == len(samples)
            else None
        )
    return result


def run_profile(binary, output, profile, selected, args):
    profile_dir = output / profile["type"] / profile["name"]
    profile_dir.mkdir(parents=True)
    builds, queries = [], {name: [] for name in selected}
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
        print(f"{prefix} build", flush=True)
        build = run_worker(
            binary,
            "build",
            build_path,
            logs / "build.log",
            type=profile["type"],
            profile=profile["name"],
            num_docs=args.num_docs,
            seed=args.seed,
            index=index,
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
        verification = run_worker(
            binary,
            "verify",
            verify_path,
            logs / "verify.log",
            index=index,
            manifest=build_path,
            case=",".join(selected),
        )
        verified = validate_verification(verification, build, cases, selected)
        builds.append(build)
        for name in selected:
            print(f"{prefix} {name}", flush=True)
            query = run_worker(
                binary,
                "query",
                directory / f"query-{name}.json",
                logs / f"query-{name}.log",
                index=index,
                manifest=build_path,
                verification=verify_path,
                case=name,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
            )
            validate_query(query, name, build, verified[name])
            queries[name].append(query)
    build_keys = (
        "insert_seconds",
        "seal_seconds",
        "build_seconds",
        "docs_per_second",
        "index_file_bytes",
        "index_allocated_bytes",
        "bytes_per_doc",
        "memory_before.rss_bytes",
        "memory_after.rss_bytes",
        "memory_after.peak_rss_bytes",
    )
    query_keys = (
        "mean_ns_per_query",
        "qps",
        "memory_before.rss_bytes",
        "memory_after.rss_bytes",
        "memory_after.peak_rss_bytes",
    )
    return {
        "type": profile["type"],
        "profile": profile["name"],
        "dataset": builds[0]["dataset"],
        "directory": str(profile_dir.relative_to(output)),
        "build": summarize_metrics(builds, build_keys),
        "cases": [
            {
                "name": name,
                "query_fingerprint": expected_queries[name],
                "match_counts": samples[0]["match_counts"],
                "metrics": summarize_metrics(samples, query_keys),
            }
            for name, samples in queries.items()
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=ROOT / "build/bin/inverted_bench")
    parser.add_argument(
        "--type", default="all", help="Type suite name from --list, or all (default)"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        choices=(100000, 10000000),
        help="Required when running benchmarks",
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
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    parser.add_argument("--measure-seconds", type=float, default=1.0)
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
        binary = args.bin.expanduser().resolve()
        completed = subprocess.run(
            [str(binary), "--mode=list"], capture_output=True, text=True, check=False
        )
        if completed.returncode:
            fail(f"Cannot read benchmark catalog: {completed.stderr.strip()}")
        selected = select_profiles(load_json(completed.stdout), args)
        if args.list:
            for profile, cases in selected:
                suffix = " (supplemental)" if profile["supplemental"] else ""
                print(
                    f"{profile['type']}/{profile['name']}{suffix}: {', '.join(cases)}"
                )
            return 0
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
        with binary.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        metadata = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at": timestamp(),
            "binary": str(binary),
            "binary_sha256": digest.hexdigest(),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "selected": [
                {"type": p["type"], "profile": p["name"], "cases": cases}
                for p, cases in selected
            ],
        }
        write_json(output / "run.json", metadata)
        summaries = [
            run_profile(binary, output, profile, cases, args)
            for profile, cases in selected
        ]
        summary = {
            "schema_version": 1,
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
