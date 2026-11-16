"""Plan, run, and summarize paired same-binary non-vector storage benchmarks.

Local automation can consume the JSON plans and launch native commands directly.
CI can use --run. Failed and partial runs are retained, never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
from pathlib import Path


def read_result(path):
    if not path.exists():
        return {"cases": [], "complete": False}
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        marker = '"cases":['
        pos = text.find(marker)
        if pos < 0:
            return {"cases": [], "complete": False}
        pos += len(marker)
        decoder, rows = json.JSONDecoder(), []
        while pos < len(text):
            while pos < len(text) and text[pos] in " \n\r\t,":
                pos += 1
            try:
                row, pos = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            rows.append(row)
        return {"cases": rows, "complete": False}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def profiles():
    base = dict(
        rows=16384,
        **{
            "payload-bytes": 1024,
            "array-elements": 2,
            "segment-mib": 2,
            "optimize": 0,
            "mutated": 0,
        },
    )
    out = {}

    def add(name, profile="rich", **changes):
        out[name] = {**base, "profile": profile, **changes}

    add("scalar", "scalar", **{"segment-mib": 128, "optimize": 1})
    add("single", **{"segment-mib": 128, "optimize": 1})
    add("multi")
    add("mutated", mutated=1)
    add("mutated-optimized", mutated=1, optimize=1)
    # The legacy --segment-mib argument controls the write buffer, hence the
    # number of blocks within a segment. Iterator snapshots explicitly seal
    # segments; never infer cross-segment coverage from buffer size alone.
    add("scalar-cross", "scalar", **{"seal-rows": 4096})
    add("cross", **{"seal-rows": 4096})
    add("cross-mutated", mutated=1, **{"seal-rows": 4096})
    for null in (0, 50, 100):
        add(
            "wide-null" + str(null),
            "wide",
            rows=8193,
            **{"payload-bytes": 128, "array-elements": 8, "null-percent": null},
        )
    add(
        "wide-single",
        "wide",
        rows=8193,
        **{
            "payload-bytes": 128,
            "array-elements": 8,
            "null-percent": 50,
            "segment-mib": 128,
            "optimize": 1,
        },
    )
    add(
        "wide-cross",
        "wide",
        rows=8193,
        **{
            "payload-bytes": 128,
            "array-elements": 8,
            "null-percent": 50,
            "seal-rows": 2048,
        },
    )
    add(
        "wide-mutated",
        "wide",
        rows=10003,
        mutated=1,
        **{"payload-bytes": 256, "array-elements": 3, "null-percent": 50},
    )
    add(
        "empty-utf8",
        "wide",
        rows=8193,
        **{
            "payload-bytes": 256,
            "array-elements": 0,
            "null-percent": 0,
            "empty-string-percent": 50,
            "empty-array-percent": 50,
            "utf8": 1,
        },
    )
    add(
        "large-array",
        rows=8193,
        **{
            "payload-bytes": 128,
            "array-elements": 1024,
            "null-percent": 25,
            "empty-array-percent": 25,
            "segment-mib": 4,
        },
    )
    add(
        "long-string",
        rows=32768,
        **{
            "payload-bytes": 8192,
            "array-elements": 4,
            "null-percent": 0,
            "segment-mib": 8,
        },
    )
    add(
        "pressure",
        rows=131072,
        **{
            "payload-bytes": 4096,
            "array-elements": 16,
            "null-percent": 0,
            "segment-mib": 8,
        },
    )
    add(
        "wide-pressure",
        "wide",
        rows=32768,
        **{
            "payload-bytes": 128,
            "array-elements": 128,
            "null-percent": 25,
            "segment-mib": 4,
        },
    )
    add(
        "cross-pressure",
        rows=131072,
        **{
            "payload-bytes": 4096,
            "array-elements": 16,
            "null-percent": 0,
            "segment-mib": 8,
            "seal-rows": 8192,
        },
    )
    add("lifecycle-rich", rows=8193, **{"payload-bytes": 256, "array-elements": 8})
    add(
        "lifecycle-wide",
        "wide",
        rows=8193,
        **{"payload-bytes": 128, "array-elements": 8, "null-percent": 50},
    )
    add(
        "lifecycle-cross-rich",
        rows=8193,
        **{"payload-bytes": 256, "array-elements": 8, "seal-rows": 2048},
    )
    add(
        "lifecycle-cross-wide",
        "wide",
        rows=8193,
        **{
            "payload-bytes": 128,
            "array-elements": 8,
            "null-percent": 50,
            "seal-rows": 2048,
        },
    )
    return out


def experiments():
    out = []

    def add(
        name, data, workers, cases, phase="query", cache="warm", budget=4096, **options
    ):
        out.append(
            {
                "group": name,
                "data": data,
                "workers": workers,
                "cases": cases,
                "phase": phase,
                "cache": cache,
                "budget": budget,
                "options": options,
            }
        )

    scalar = [
        "eq_ids",
        "eq_key",
        "eq_payload",
        "bucket_ids",
        "bucket_all",
        "range128_all",
        "in32_ids",
        "or_ids",
        "not_in_ids",
        "scan_payload_like",
        "like_suffix",
        "like_contains",
        "fetch1_payload",
        "fetch8_key",
        "fetch128_payload",
        "fetch32_contiguous",
        "fetch32_hot",
        "fetch_mixed",
        "fetch_edges",
    ]
    rich = [
        "eq_key",
        "fetch1_all",
        "fetch1_nullable",
        "fetch32_null_array",
        "nullable_filter",
        "array_filter",
        "array_all_filter",
        "like_suffix",
        "like_contains",
        "scan_payload_like",
        "fts_term",
        "fts_filter",
        "fts_and",
        "fts_or",
        "fts_phrase",
        "fts_or_filter",
        "scan_weight",
    ]
    for data in (
        "scalar",
        "single",
        "multi",
        "mutated",
        "mutated-optimized",
        "scalar-cross",
        "cross",
        "cross-mutated",
    ):
        for workers in (1, 4, 8, 16):
            add(
                f"core-{data}-t{workers}",
                data,
                workers,
                scalar if data in ("scalar", "scalar-cross") else rich,
            )
    wide = [
        "fetch1_all",
        "fetch_all",
        "fetch_projected",
        "fetch_edges",
        "nullable_filter",
        "array_filter",
        "not_null",
        "payload_eq_empty",
    ]
    for data in (
        "wide-null0",
        "wide-null50",
        "wide-null100",
        "wide-single",
        "wide-mutated",
        "empty-utf8",
        "large-array",
        "long-string",
        "wide-cross",
    ):
        for workers in (
            (1, 4, 8, 16)
            if data in ("wide-null50", "wide-single", "wide-cross")
            else (1, 4)
        ):
            add(f"shape-{data}-t{workers}", data, workers, wide)
    typed = [
        "i32_filter",
        "i64_filter",
        "u32_filter",
        "u64_filter",
        "float_filter",
        "double_filter",
        "bool_filter",
        "string_filter",
        "array_i64_filter",
        "array_u32_filter",
        "array_u64_filter",
        "array_float_filter",
        "array_double_filter",
        "array_bool_filter",
        "array_string_filter",
    ]
    for data in ("wide-null0", "wide-null50", "wide-null100", "wide-cross"):
        for workers in (1, 4):
            add(f"typed-{data}-t{workers}", data, workers, typed)
    for fraction in (0, 1, 10, 50, 100):
        for data, prefix in (
            ("wide-null50", "selectivity"),
            ("wide-cross", "selectivity-cross"),
        ):
            for workers in (1, 4):
                add(
                    f"{prefix}-{fraction}-t{workers}",
                    data,
                    workers,
                    ["range_select_ids", "range_select_all", "scan_select"],
                    **{"selectivity-percent": fraction},
                )
    for topk in (1, 10, 100):
        for data in ("single", "multi", "cross"):
            for workers in (1, 4):
                add(
                    f"topk-{data}-{topk}-t{workers}",
                    data,
                    workers,
                    ["fts_topk", "fts_topk_filter"],
                    topk=topk,
                )
    for size in (1, 8, 32, 128, 512):
        for data, prefix in (
            ("wide-null50", "batch-wide"),
            ("wide-cross", "batch-wide-cross"),
        ):
            for workers in (1, 4):
                add(
                    f"{prefix}-{size}-t{workers}",
                    data,
                    workers,
                    ["fetch_all"],
                    **{"fetch-size": size},
                )
    for i, projection in enumerate(
        (
            "none",
            "key",
            "payload",
            "opt,tags,payload",
            "u32,u64,f32,bin,ai64,astr,abool,abin",
            "all",
        )
    ):
        for data, prefix in (
            ("wide-null50", "projection"),
            ("wide-cross", "projection-cross"),
        ):
            for workers in (1, 4):
                add(
                    f"{prefix}-{i}-t{workers}",
                    data,
                    workers,
                    ["fetch_projected"],
                    projection=projection,
                )
    pressure = [
        "fetch1_all",
        "fetch32_null_array",
        "fetch128_payload",
        "range128_all",
        "scan_payload_like",
        "fts_filter",
    ]
    for data in ("pressure", "wide-pressure", "cross-pressure"):
        for budget in (128, 256, 512, 4096):
            for workers in (1, 4):
                add(
                    f"pressure-{data}-{budget}-t{workers}",
                    data,
                    workers,
                    pressure,
                    budget=budget,
                )
        for budget in (256, 4096):
            for workers in (8, 16):
                add(
                    f"pressure-{data}-{budget}-t{workers}",
                    data,
                    workers,
                    ["fetch32_null_array", "scan_payload_like", "fts_filter"],
                    budget=budget,
                )
    for data in (
        "single",
        "multi",
        "wide-null50",
        "large-array",
        "long-string",
        "cross",
        "wide-cross",
    ):
        for workers in (1, 4):
            for case in (
                "fetch1_all",
                "fetch32_null_array",
                "scan_payload_like",
                "fts_filter",
            ):
                add(
                    f"cold-{data}-{case}-t{workers}",
                    data,
                    workers,
                    [case],
                    cache="cold",
                )
    for data in ("pressure", "cross-pressure"):
        for budget in (128, 512):
            for workers in (1, 4):
                for case in ("fetch32_null_array", "scan_payload_like"):
                    add(
                        f"cold-{data}-{budget}-{case}-t{workers}",
                        data,
                        workers,
                        [case],
                        cache="cold",
                        budget=budget,
                    )
    for data in (
        "lifecycle-rich",
        "lifecycle-wide",
        "lifecycle-cross-rich",
        "lifecycle-cross-wide",
    ):
        for maintenance in ("none", "flush", "optimize"):
            for workers in (1, 4):
                for case in ("fetch32_null_array", "fts_filter", "array_filter"):
                    add(
                        f"mixed-{data}-{maintenance}-{case}-t{workers}",
                        data,
                        workers,
                        [case],
                        phase="mixed",
                        maintenance=maintenance,
                    )
        for workers in (1, 4):
            add(
                f"lifecycle-{data}-t{workers}",
                data,
                workers,
                ["fetch32_null_array"],
                phase="lifecycle",
            )
    for workers in (8, 16):
        for maintenance in ("none", "optimize"):
            for data, prefix in (
                ("lifecycle-rich", "mixed-rich"),
                ("lifecycle-cross-rich", "mixed-cross-rich"),
            ):
                add(
                    f"{prefix}-{maintenance}-t{workers}",
                    data,
                    workers,
                    ["fetch32_null_array"],
                    phase="mixed",
                    maintenance=maintenance,
                )
    assert len({x["group"] for x in out}) == len(out)
    return out


def representative_experiments():
    """Cover scenario and parameter boundaries without a Cartesian product."""
    full = {x["group"]: x for x in experiments()}
    names = [
        "core-scalar-t1",
        "core-scalar-t4",
        "core-cross-t1",
        "core-cross-t4",
        "core-mutated-t4",
        "core-mutated-optimized-t4",
        "shape-wide-null50-t1",
        "shape-wide-single-t4",
        "shape-wide-mutated-t4",
        "shape-wide-cross-t4",
        "shape-empty-utf8-t1",
        "shape-large-array-t4",
        "shape-long-string-t1",
        "typed-wide-null0-t1",
        "typed-wide-null50-t4",
        "typed-wide-null100-t1",
        "typed-wide-cross-t4",
        "selectivity-0-t1",
        "selectivity-1-t4",
        "selectivity-10-t1",
        "selectivity-50-t4",
        "selectivity-100-t1",
        "selectivity-cross-50-t4",
        "topk-cross-1-t1",
        "topk-cross-10-t4",
        "topk-cross-100-t4",
        "batch-wide-cross-1-t1",
        "batch-wide-cross-8-t4",
        "batch-wide-cross-32-t4",
        "batch-wide-cross-128-t4",
        "batch-wide-cross-512-t1",
        "projection-cross-0-t1",
        "projection-cross-1-t4",
        "projection-cross-2-t1",
        "projection-cross-3-t4",
        "projection-cross-4-t4",
        "projection-cross-5-t1",
        "pressure-pressure-128-t1",
        "pressure-pressure-256-t1",
        "pressure-pressure-512-t1",
        "pressure-pressure-4096-t1",
        "pressure-cross-pressure-128-t4",
        "pressure-cross-pressure-4096-t4",
        "pressure-wide-pressure-128-t4",
        "pressure-wide-pressure-4096-t4",
        "cold-cross-fetch32_null_array-t4",
        "cold-wide-cross-fts_filter-t1",
        "cold-long-string-scan_payload_like-t1",
        "core-scalar-t8",
        "core-scalar-t16",
        "core-cross-t16",
        "pressure-cross-pressure-4096-t16",
    ]
    for data, cases in (
        ("lifecycle-rich", ["fetch32_null_array", "fts_filter", "array_filter"]),
        ("lifecycle-wide", ["array_filter", "fetch32_null_array", "fts_filter"]),
        ("lifecycle-cross-rich", ["fts_filter", "array_filter", "fetch32_null_array"]),
        ("lifecycle-cross-wide", ["fetch32_null_array", "fts_filter", "array_filter"]),
    ):
        for index, case in enumerate(cases):
            maintenance = ("none", "flush", "optimize")[index]
            names.append(f"mixed-{data}-{maintenance}-{case}-t4")
        names.append(f"lifecycle-{data}-t1")
    names.append("mixed-lifecycle-cross-rich-none-fts_filter-t1")
    out = []
    for name in names:
        spec = dict(full[name])
        if name.startswith("pressure-"):
            spec["cases"] = [
                "fetch1_all",
                "fetch32_null_array",
                "scan_payload_like",
                "fts_filter",
            ]
            if spec["workers"] == 16:
                spec["cases"] = ["scan_payload_like"]
            elif spec["budget"] in (256, 512):
                spec["cases"] = ["fetch32_null_array"]
        elif name in ("core-scalar-t8", "core-scalar-t16"):
            spec["cases"] = ["fetch1_payload", "fetch32_contiguous"]
        elif name == "core-cross-t16":
            spec["cases"] = ["fetch32_null_array", "fts_filter"]
        out.append(spec)
    assert {case for x in out for case in x["cases"]} == {
        case for x in full.values() for case in x["cases"]
    }, "Representative tier must retain every query scenario"
    for option in ("topk", "fetch-size", "projection", "selectivity-percent"):
        assert {
            x["options"][option] for x in full.values() if option in x["options"]
        } == {x["options"][option] for x in out if option in x["options"]}, (
            f"Representative tier must retain every {option} value"
        )
    for dimension in ("workers", "budget", "phase", "cache"):
        assert {x[dimension] for x in full.values()} == {x[dimension] for x in out}
    return out


def reuse_completed(plan, args):
    reused, reused_status, reused_hashes = [], [], {}
    if args.reuse_status:
        assert args.reuse_plan, "--reuse-status requires --reuse-plan"
        previous_policy = json.loads(
            args.reuse_plan.with_name(
                args.reuse_plan.stem.removesuffix("-plan") + "-policy.json"
            ).read_text()
        )
        binary = args.binary.resolve()
        expected_hash = previous_policy.get("binary_sha256") or previous_policy.get(
            "files", {}
        ).get(str(binary))
        assert expected_hash == hashlib.sha256(binary.read_bytes()).hexdigest(), (
            "Reusing results requires the exact same binary"
        )
        old_plan = json.loads(args.reuse_plan.read_text())
        completed = {
            x["name"]: x
            for x in json.loads(args.reuse_status.read_text())
            if x["exit_code"] == 0
        }

        def comparable(item):
            cmd = item["command"]
            start = cmd.index("--rows")
            opts = {cmd[i]: cmd[i + 1] for i in range(start, len(cmd), 2)}
            opts.pop("--output")
            opts.pop("--cases")
            return cmd[:start], opts

        pending = []
        for item in plan:
            previous = next(
                (
                    old
                    for old in old_plan
                    if old["name"] in completed
                    and old["phase"] == item["phase"] == "query"
                    and old["group"] == item["group"]
                    and old["arm"] == item["arm"]
                    and old["repeat"] == item["repeat"]
                    and set(item["cases"]) <= set(old["cases"])
                    and comparable(old) == comparable(item)
                ),
                None,
            )
            if previous:
                cmd = previous["command"]
                raw = Path(cmd[cmd.index("--output") + 1])
                result = read_result(raw)
                assert result["complete"], "Only complete results may be reused"
                assert all(c["ids_validated"] for c in result["cases"])
                reused.append(previous)
                reused_status.append(completed[previous["name"]])
                reused_hashes[str(raw)] = hashlib.sha256(raw.read_bytes()).hexdigest()
            else:
                pending.append(item)
        plan = reused + pending
    return plan, reused_status, reused_hashes


def make_plans(args):
    root, binary = args.root.resolve(), args.binary.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    destination = root / tag
    destination.mkdir()
    data = root / "data"
    data.mkdir(exist_ok=True)
    modes = [("B", 0, "bp"), ("M", 1, "mmap")]

    def command(opts, mode, phase, path, output):
        cmd = [str(binary)]
        values = {
            **opts,
            "mmap": mode,
            "phase": phase,
            "path": str(path),
            "output": str(output),
        }
        for key, value in values.items():
            cmd += ["--" + key, str(value)]
        if os.name != "nt":
            cmd = ["/usr/bin/env", "-i", "LC_ALL=C.UTF-8", "MallocNanoZone=0", *cmd]
        return cmd

    preparation = []
    for name, opts in profiles().items():
        (data / name).mkdir(parents=True, exist_ok=True)
        for arm, mode, dirname in modes:
            path = data / name / dirname
            for phase in ("build", "validate"):
                output = root / f"prepare-{name}-{arm}-{phase}.json"
                preparation.append(
                    {
                        "name": f"prepare-{name}-{arm}-{phase}",
                        "phase": phase,
                        "cases": [],
                        "group": name,
                        "arm": arm,
                        "repeat": 1,
                        "log": str(output.with_suffix(".log")),
                        "command": command(
                            {**opts, "workers": 1, "budget-mib": 4096},
                            mode,
                            phase,
                            path,
                            output,
                        ),
                    }
                )
    candidates = (
        experiments()
        if args.tier == "full" or args.groups
        else representative_experiments()
    )
    selected = [x for x in candidates if not args.groups or x["group"] in args.groups]
    plan = []
    for repeat in range(1, args.rounds + 1):
        ordered = selected[repeat - 1 :] + selected[: repeat - 1]
        for position, spec in enumerate(ordered):
            sequence = modes if (repeat + position) % 2 else list(reversed(modes))
            for arm, mode, dirname in sequence:
                name = f"{tag}-{spec['group']}-r{repeat}-{arm}"
                output = destination / (name + ".json")
                cases = spec["cases"] if repeat % 2 else list(reversed(spec["cases"]))
                opts = {
                    **profiles()[spec["data"]],
                    **spec["options"],
                    "workers": spec["workers"],
                    "budget-mib": spec["budget"],
                    "cases": ",".join(cases),
                    "warmups": 0 if spec["cache"] == "cold" else 64,
                    "seconds": args.mixed_seconds
                    if spec["phase"] == "mixed"
                    else args.seconds,
                }
                if spec["cache"] == "cold":
                    opts["first-only"] = 1
                path = data / spec["data"] / dirname
                if spec["phase"] in ("mixed", "lifecycle"):
                    path = destination / (name + "-data")
                plan.append(
                    {
                        "name": name,
                        "group": spec["group"],
                        "arm": arm,
                        "mode": dirname,
                        "repeat": repeat,
                        "cases": cases,
                        "phase": spec["phase"],
                        "cache": spec["cache"],
                        "configuration": spec,
                        "command": command(opts, mode, spec["phase"], path, output),
                    }
                )
    plan, reused_status, reused_hashes = reuse_completed(plan, args)
    sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    save(root / (tag + "-plan.json"), plan)
    status_path = root / (tag + "-status.json")
    assert not status_path.exists()
    save(status_path, reused_status)
    save(root / (tag + "-preparation-plan.json"), preparation)
    if not (root / "preparation-plan.json").exists():
        save(root / "preparation-plan.json", preparation)
    save(
        root / (tag + "-policy.json"),
        {
            "binary": str(binary),
            "binary_sha256": sha,
            "rounds": args.rounds,
            "seconds": args.seconds,
            "mixed_seconds": args.mixed_seconds,
            "scenario_profiles": sum(len(x["cases"]) for x in selected),
            "experiments": len(selected),
            "cache_cold": "Fresh process and zero warmups; OS page-cache state not claimed cold.",
            "no_retries": True,
            "same_binary_both_modes": True,
            "tier": args.tier,
            "reused_completed_processes": len(reused_status),
            "reused_raw_hashes": reused_hashes,
            "topology": "segment-mib is the legacy write-buffer argument; seal-rows creates true segments; numeric segment directories with nonempty scalar blocks are audited separately.",
            "pressure": "Warmups do not guarantee residency under pressure; cache budgets are supported public configuration values, not an RSS bound.",
            "request_scheduler": "global-prefix64-v1: minimum64 timed requests globally; every worker participates; histogram proves a deterministic complete prefix. Cold first-only requests retain worker-specific offsets.",
            "matrix": selected,
            "data": profiles(),
        },
    )
    print(  # noqa: T201 -- command-line result
        json.dumps(
            {
                "plan": str(root / (tag + "-plan.json")),
                "status": str(status_path),
                "processes": len(plan),
                "scenario_profiles": sum(len(x["cases"]) for x in selected),
            }
        )
    )


def run_plan(args):
    plan = json.loads(args.plan.read_text())
    status = args.status or args.plan.with_name(
        args.plan.stem.removesuffix("-plan") + "-status.json"
    )
    records = json.loads(status.read_text()) if status.exists() else []
    assert [x["name"] for x in records] == [x["name"] for x in plan[: len(records)]]
    for item in plan[len(records) :]:
        cmd = item["command"]
        output = Path(cmd[cmd.index("--output") + 1])
        log = Path(item.get("log", str(output.with_suffix(".log"))))
        assert not log.exists(), "Never overwrite an attempted log"
        assert not output.exists(), "Never overwrite an attempted result"
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP")
        }
        with log.open("x") as stream:
            result = subprocess.run(
                cmd, check=False, stdout=stream, stderr=subprocess.STDOUT, env=env
            )
        records.append(
            {"name": item["name"], "exit_code": result.returncode, "log": str(log)}
        )
        save(status, records)
        print(len(records), "/", len(plan), item["name"], result.returncode, flush=True)  # noqa: T201 -- progress


def checked_metrics(row, result):
    assert row["queries"] > 0
    assert row["elapsed_s"] > 0
    scheduler = result.get("request_scheduler", "per-worker64-legacy")
    if scheduler == "global-prefix64-v1":
        histogram = row["request_counts"]
        n = row["queries"]
        assert len(histogram) == 64
        assert sum(histogram) == n
        if not result["first_only"]:
            assert n >= 64
            assert histogram == [n // 64 + (q < n % 64) for q in range(64)]
    return {
        **row,
        "request_scheduler": scheduler,
        "writer_cycles_s": row["writer_cycles"]
        / row.get("writer_elapsed_s", row["elapsed_s"]),
    }


def summarize(args):
    plan = json.loads(args.plan.read_text())
    status = json.loads(args.status.read_text())
    lookup = {x["name"]: x for x in plan}
    values, failures, raw, lifecycle = {}, [], {}, {}
    assert len({x["name"] for x in status}) == len(status), "Duplicate attempts"
    for rec in status:
        item = lookup[rec["name"]]
        cmd = item["command"]
        path = Path(cmd[cmd.index("--output") + 1])
        result = read_result(path)
        valid_parent = not rec["exit_code"] and result.get("complete", False)
        if item["phase"] in ("mixed", "lifecycle"):
            valid_parent = valid_parent and result.get("lifecycle", {}).get(
                "validated", False
            )
        if (
            item["phase"] == "mixed"
            and item["configuration"]["options"].get("maintenance", "none") != "none"
        ):
            # The serial writer may finish an operation after readers stop.
            # Subtract its entire drain interval to get a conservative lower
            # bound on maintenance time overlapping the readers.
            overlaps = [
                x["maintenance_us"]
                - max(0, x["writer_elapsed_s"] - x["elapsed_s"]) * 1e6
                for x in result.get("cases", [])
            ]
            valid_parent = valid_parent and bool(overlaps) and min(overlaps) > 1000
        if not valid_parent:
            failures.append(
                {**rec, "completed_cases": [x["case"] for x in result.get("cases", [])]}
            )
        if path.exists():
            raw[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if valid_parent and "lifecycle" in result:
            for metric, value in result["lifecycle"].items():
                if metric != "validated":
                    lifecycle.setdefault((item["group"], metric), {}).setdefault(
                        item["arm"], {}
                    )[item["repeat"]] = value
        for measured in result.get("cases", []):
            assert measured["ids_validated"]
            # Preserve failed processes as evidence, but exclude their timings.
            if not valid_parent:
                continue
            row = checked_metrics(measured, result)
            key = item["group"], row["case"]
            values.setdefault(key, {}).setdefault(item["arm"], {})[item["repeat"]] = {
                **row,
                "process_exit_code": rec["exit_code"],
            }
    summary = []
    for (group, case), arms in sorted(values.items()):
        item = next(i for i in plan if i["group"] == group)
        cold = item.get("cache") == "cold"
        metric = "p50_us" if cold else "qps"
        row = {
            "group": group,
            "case": case,
            "metric": metric,
            "values": {
                k: statistics.median(v[metric] for v in r.values())
                for k, r in arms.items()
            },
            "observations": {k: len(r) for k, r in arms.items()},
            "diagnostics": {
                k: {
                    m: statistics.median(v[m] for v in r.values())
                    for m in (
                        "writer_cycles_s",
                        "rss_bytes",
                        "evictions",
                        "capacity_waits",
                        "capacity_timeouts",
                        "disk_read_bytes",
                        "major_faults",
                    )
                }
                for k, r in arms.items()
            },
        }
        if "B" in arms and "M" in arms:
            pairs = sorted(arms["B"].keys() & arms["M"].keys())
            assert all(
                (
                    arms["B"][r]["request_fingerprint"],
                    arms["B"][r]["expected_docs_per_cycle"],
                )
                == (
                    arms["M"][r]["request_fingerprint"],
                    arms["M"][r]["expected_docs_per_cycle"],
                )
                for r in pairs
            )
            assert all(
                arms["B"][r]["request_scheduler"] == arms["M"][r]["request_scheduler"]
                for r in pairs
            )
            logs = [
                math.log(arms["B"][r][metric] / arms["M"][r][metric])
                * (-1 if cold else 1)
                for r in pairs
            ]
            row.update(
                pairs=len(pairs),
                bp_mmap_speed=math.exp(statistics.mean(logs)),
                parent_process_failure_pairs=sum(
                    arms["B"][r]["process_exit_code"] != 0
                    or arms["M"][r]["process_exit_code"] != 0
                    for r in pairs
                ),
            )
            if len(pairs) > 1:
                rng = random.Random(20261115)
                boot = sorted(
                    math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
                    for _ in range(20000)
                )
                row["bootstrap95"] = [boot[499], boot[19499]]
        summary.append(row)
    save(args.plan.with_name(args.plan.stem + "-summary.json"), summary)
    save(
        args.plan.with_name(args.plan.stem + "-lifecycle.json"),
        [
            {
                "group": group,
                "metric": metric,
                "values": {
                    arm: statistics.median(repeats.values())
                    for arm, repeats in arms.items()
                },
                "observations": {arm: len(repeats) for arm, repeats in arms.items()},
            }
            for (group, metric), arms in sorted(lifecycle.items())
        ],
    )
    save(
        args.plan.with_name(args.plan.stem + "-validation.json"),
        {
            "attempted": len(status),
            "planned": len(plan),
            "failures": failures,
            "raw_hashes": raw,
            "partial_timings_excluded": True,
        },
    )
    print(  # noqa: T201 -- command-line result
        json.dumps(
            {
                "attempted": len(status),
                "planned": len(plan),
                "failures": len(failures),
                "scenarios": len(summary),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["plan", "run", "summarize"])
    p.add_argument("--root", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--tag", default="baseline")
    p.add_argument(
        "--tier", choices=["representative", "full"], default="representative"
    )
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--mixed-seconds", type=float, default=2)
    p.add_argument("--groups", nargs="*")
    p.add_argument("--reuse-plan", type=Path)
    p.add_argument("--reuse-status", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--status", type=Path)
    args = p.parse_args()
    if args.action == "plan":
        make_plans(args)
    elif args.action == "run":
        run_plan(args)
    else:
        summarize(args)
