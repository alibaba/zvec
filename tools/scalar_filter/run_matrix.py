#!/usr/bin/env python3
"""Validate and compare JIT/Arrow plus optional upstream and old hardcoded libraries."""

# CLI progress and JSON summaries intentionally use stdout.
# ruff: noqa: T201
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from runner_common import (
    environment,
    library,
    new_output,
    records,
    require,
    run,
    sha256,
)


def main():  # noqa: PLR0915 - Keep the CLI validation/run/report orchestration together.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        required=True,
        help="Public-only scalar_filter_benchmark, compatible with old libraries",
    )
    parser.add_argument(
        "--jit-binary",
        type=Path,
        required=True,
        help="scalar_filter_jit_benchmark with query-level activation counters",
    )
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--integrated-lib-dir", type=Path, required=True)
    parser.add_argument("--baseline-lib-dir", type=Path)
    parser.add_argument("--hardcode-lib-dir", type=Path)
    parser.add_argument("--dependency-lib-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--pool", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", default="1,8")
    parser.add_argument("--policies", default="fixed,varying")
    parser.add_argument("--expected-bindings-per-query", type=int, default=1)
    parser.add_argument("--include-unfiltered", action="store_true")
    parser.add_argument("--quality-only", action="store_true")
    args = parser.parse_args()
    require(
        min(args.queries, args.pool, args.repeats, args.expected_bindings_per_query)
        > 0,
        "queries, pool, repeats and expected bindings must be positive",
    )
    require(args.queries % args.pool == 0, "queries must be a multiple of pool")
    threads = tuple(int(value) for value in args.threads.split(","))
    policies = tuple(args.policies.split(","))
    require(threads and min(threads) > 0, "threads must be positive")
    require(
        policies and all(p in ("fixed", "varying") for p in policies),
        "policies must contain fixed and/or varying",
    )
    binary = args.binary.resolve(strict=True)
    jit_binary = args.jit_binary.resolve(strict=True)
    dataset = args.path.resolve(strict=True)
    out = new_output(args.output_dir)
    libraries = {
        "arrow": library(args.integrated_lib_dir),
        "jit": library(args.integrated_lib_dir),
    }
    if args.baseline_lib_dir:
        libraries["upstream"] = library(args.baseline_lib_dir)
    if args.hardcode_lib_dir:
        libraries["hardcode"] = library(args.hardcode_lib_dir)
    filtered_variants = tuple(libraries)
    if args.include_unfiltered:
        libraries["unfiltered"] = library(args.integrated_lib_dir)
    hashes = {name: sha256(path) for name, path in libraries.items()}
    binary_hashes = {str(path): sha256(path) for path in (binary, jit_binary)}

    def execute(case, variant, policy, clients, repeat, dump=None):
        internal = variant in ("arrow", "jit", "unfiltered")
        executable = jit_binary if internal else binary
        mode = "unfiltered" if variant == "unfiltered" else "filtered"
        query_count = 1 if dump else args.queries
        command = [
            str(executable),
            "bench",
            "--path",
            str(dataset),
            "--ratio",
            "0.5",
            "--pool",
            str(args.pool),
            "--warmup",
            str(1 if dump else args.pool),
            "--queries",
            str(query_count),
            "--sequence",
            "sequential",
            "--output-fields",
            "all-scalars",
            "--candidate-topk",
            "21",
            "--ef-search",
            "300",
            "--thresholds",
            policy,
            "--threads",
            str(clients),
            "--mode",
            mode,
        ]
        if dump:
            command += ["--dump", str(dump), "--dump-count", str(args.pool)]
        env = environment(
            libraries[variant],
            variant in ("jit", "unfiltered"),
            args.dependency_lib_dir,
            legacy_direct=variant == "hardcode",
        )
        stdout = out / f"{case}.stdout.jsonl"
        stderr = out / f"{case}.stderr.log"
        print(case, flush=True)
        run(command, env, stdout, stderr)
        found = [r for r in records(stdout) if r.get("event") == "bench"]
        require(len(found) == 1, f"{case}: expected one benchmark record")
        bench = found[0]
        require(
            bench["queries"] == query_count
            and bench["returned"] == 21 * query_count
            and not any(
                bench[k] for k in ("errors", "short_results", "filter_violations")
            )
            and bench["mode"] == mode
            and bench["ratio"] == 0.5
            and bench["threshold_policy"] == policy,
            f"{case}: query/result validation failed",
        )
        if internal:
            require(
                "jit_successful_bindings" in bench,
                f"{case}: missing internal activation statistics",
            )
            if variant == "jit":
                require(
                    bench["jit_successful_bindings"]
                    == query_count * args.expected_bindings_per_query,
                    f"{case}: JIT silently fell back or segment count differs; inspect successful bindings",
                )
                require(
                    bench["jit_cache_total_compilations"] == 1
                    and bench["jit_cache_compilations"] == 0,
                    f"{case}: expected one cached compile during warmup and zero timed compiles",
                )
            else:
                require(
                    bench["jit_successful_bindings"] == 0
                    and bench["jit_cache_lookups"] == 0,
                    f"{case}: disabled/unfiltered run unexpectedly used scalar JIT",
                )
        record = {
            "case": case,
            "variant": variant,
            "policy": policy,
            "threads": clients,
            "repeat": repeat,
            "command": command,
            "library": str(libraries[variant]),
            "library_sha256": hashes[variant],
            "binary_sha256": binary_hashes[str(executable)],
            "env": {
                key: env[key]
                for key in (
                    "DYLD_LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                    "ZVEC_SCALAR_JIT",
                    "ZVEC_ISSUE868_FORWARD_QUERY_LOCK",
                    "ZVEC_ISSUE868_BOUND_INT32",
                )
            },
            "stdout": str(stdout),
            "stderr": str(stderr),
            "bench": bench,
        }
        rows = None
        if dump:
            rows = [json.loads(line) for line in dump.read_text().splitlines()]
            require(len(rows) == args.pool, f"{case}: missing quality queries")
            for query, row in enumerate(rows):
                require(
                    row["query_id"] == query
                    and row["raw_returned"] == 21
                    and row["violations"] == 0
                    and all(
                        len(row[key]) == 21
                        for key in ("ids", "scores", "times", "pids", "mails")
                    )
                    and all(value > row["threshold"] for value in row["times"]),
                    f"{case}: incomplete/incorrect quality result at query {query}",
                )
                require(
                    all(
                        pid == int(pk) and mail == f"user{pk}@example.test"
                        for pk, pid, mail in zip(
                            row["ids"], row["pids"], row["mails"], strict=True
                        )
                    ),
                    f"{case}: scalar output did not match source doc at query {query}",
                )
            record["dump"] = str(dump)
        print(
            f"  API={bench['query_api_mean_ms']:.4f} ms QPS={bench['qps']:.2f}",
            flush=True,
        )
        return record, rows

    quality = []
    for policy in policies:
        reference = None
        quality_records = []
        for variant in filtered_variants:
            case = f"quality_{policy}_{variant}"
            record, rows = execute(case, variant, policy, 1, 0, out / f"{case}.jsonl")
            if reference is None:
                reference = rows
            require(
                rows == reference,
                f"{policy}/{variant}: IDs/order/scores/scalar outputs differ from Arrow fallback",
            )
            quality_records.append(record)
        quality.append(
            {
                "policy": policy,
                "queries_per_variant": args.pool,
                "exact_ids_order_scores_all_scalars_equal": True,
                "records": quality_records,
            }
        )
    (out / "quality_summary.json").write_text(json.dumps(quality, indent=2) + "\n")
    measurements = []
    if not args.quality_only:
        schedule = []
        for policy in policies:
            variants = filtered_variants + (
                ("unfiltered",)
                if args.include_unfiltered and policy == policies[0]
                else ()
            )
            for clients in threads:
                for repeat in range(1, args.repeats + 1):
                    order = (
                        variants[repeat % len(variants) :]
                        + variants[: repeat % len(variants)]
                    )
                    if repeat % 2 == 0:
                        order = tuple(reversed(order))
                    schedule.extend(
                        (variant, policy, clients, repeat) for variant in order
                    )
        with (out / "measurements.jsonl").open("w") as cumulative:
            for index, (variant, policy, clients, repeat) in enumerate(schedule, 1):
                print(f"[{index}/{len(schedule)}]", end=" ", flush=True)
                case = f"{policy}_{variant}_t{clients}_r{repeat}"
                record, _ = execute(case, variant, policy, clients, repeat)
                measurements.append(record)
                cumulative.write(json.dumps(record) + "\n")
                cumulative.flush()
    performance = {}
    for policy in policies:
        performance[policy] = {}
        for clients in threads:
            group = performance[policy][str(clients)] = {}
            for variant in libraries:
                samples = [
                    row["bench"]
                    for row in measurements
                    if row["policy"] == policy
                    and row["threads"] == clients
                    and row["variant"] == variant
                ]
                if not samples:
                    continue
                group[variant] = {
                    key: median(sample[key] for sample in samples)
                    for key in (
                        "qps",
                        "query_api_mean_ms",
                        "query_api_p50_ms",
                        "query_api_p95_ms",
                        "query_api_p99_ms",
                        "average_cpu_cores",
                    )
                }
            if "jit" in group:
                group["jit_qps_ratios"] = {
                    variant: group["jit"]["qps"] / value["qps"]
                    for variant, value in group.items()
                    if variant != "jit"
                }
    summary = {
        "quality": quality,
        "performance_medians": performance,
        "measured_queries": sum(row["bench"]["queries"] for row in measurements),
        "dataset": str(dataset),
        "note": "Fresh processes, fixed query multiset, scalar output validation. "
        "QPS includes result validation/disposal; query_api_* times only Collection::query. "
        "JIT activation and cache compilation counters are checked outside timed calls. "
        "Unfiltered results have different acceptance semantics.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(performance, indent=2), flush=True)


if __name__ == "__main__":
    main()
