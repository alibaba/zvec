#!/usr/bin/env python3
"""Interleaved, fresh-process runs and exact-result checks for PR #889."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--binary", required=True, type=Path)
p.add_argument("--collection", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--hardcode-binary", type=Path)
p.add_argument("--baseline-binary", type=Path)
p.add_argument("--runs", type=int, default=3)
p.add_argument("--count", type=int, default=4096)
p.add_argument("--seconds", type=float, default=0)
p.add_argument(
    "--sweep",
    action="store_true",
    help="Time historical strategies only: none, inverted, upstream, post.",
)
p.add_argument(
    "--skip-quality",
    action="store_true",
    help="Use only after the same binaries/fixture passed archived quality checks.",
)
p.add_argument("--warmup", type=int, default=1024)
p.add_argument("--quality-count", type=int, default=1024)
p.add_argument("--clients", nargs="+", type=int, default=[1, 8])
p.add_argument("--post", action="store_true")
p.add_argument("--quality-only", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
binary = a.binary.resolve()
variants = [
    ("none", "none", "1", 21, 300, binary),
    ("arrow", "forward", "0", 21, 300, binary),
    ("jit", "forward", "1", 21, 300, binary),
    ("inverted", "inverted", "1", 21, 300, binary),
]
if a.hardcode_binary:
    variants.append(("hardcode", "forward", "1", 21, 300, a.hardcode_binary.resolve()))
if a.baseline_binary:
    variants.append(("upstream", "forward", "0", 21, 300, a.baseline_binary.resolve()))
if a.post:
    variants += [
        (f"post-{k}-{ef}", "post", "1", k, ef, binary)
        for k, ef in [
            (64, 300),
            (64, 400),
            (64, 600),
            (128, 1200),
            (256, 2048),
            (300, 300),
            (400, 400),
            (600, 600),
        ]
    ]


if a.sweep:
    if not a.baseline_binary:
        raise ValueError(
            "--sweep requires --baseline-binary for the upstream legacy path"
        )
    variants = [
        (name, strategy, "0", topk, ef, a.baseline_binary.resolve())
        for name, strategy, jit, topk, ef, exe in variants
        if name in ["none", "inverted", "upstream"] or strategy == "post"
    ]
quality_paths = [
    ("arrow", "forward", "0", 21, 300, binary),
    ("jit", "forward", "1", 21, 300, binary),
]
quality_paths += [v for v in variants if v[0] in ["hardcode", "upstream", "inverted"]]


def execute(name, variant, mode, extra):
    _, strategy, jit, k, ef, exe = variant
    output = a.output / (name + ".json")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing run {output}")
    cmd = [
        str(exe),
        "--mode",
        mode,
        "--path",
        str(a.collection.resolve()),
        "--strategy",
        strategy,
        "--candidate-topk",
        str(k),
        "--ef",
        str(ef),
        "--workers",
        "8",
        "--output",
        str(output.resolve()),
        *extra,
    ]
    env = os.environ.copy()
    env["ZVEC_SCALAR_JIT"] = jit
    start = time.time()
    event = {"name": name, "argv": cmd, "jit_env": jit, "started_unix": start}
    with (a.output / (name + ".log")).open("w") as log:
        result = subprocess.run(
            cmd, check=False, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    event.update(exit_code=result.returncode, wall_seconds=time.time() - start)
    with (a.output / "commands.jsonl").open("a") as f:
        f.write(json.dumps(event) + "\n")
    if result.returncode:
        raise RuntimeError(f"Failed {name}; see {a.output / (name + '.log')}")
    obj = json.loads(output.read_text())
    print(name, "QPS=" + str(obj.get("qps", "quality")), flush=True)  # noqa: T201 - CLI progress
    return obj


source_dir = Path(__file__).resolve().parent
source_hashes = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in source_dir.iterdir()
    if f.is_file()
}
manifest = {
    "binary_sha256": {
        str(v[5]): hashlib.sha256(v[5].read_bytes()).hexdigest()
        for v in variants + quality_paths
    },
    "harness_source_sha256": source_hashes,
    "collection": str(a.collection.resolve()),
    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
}
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

if not a.skip_quality:
    quality = {}
    for vary in [False, True]:
        selected = quality_paths
        results = {
            v[0]: execute(
                "quality-" + ("vary-" if vary else "fixed-") + v[0],
                v,
                "dump",
                ["--count", str(a.quality_count), "--vary", str(int(vary))],
            )
            for v in selected
        }
        ref = results["arrow"]["queries"]
        summary = {}
        for name, result in results.items():
            assert len(result["queries"]) == len(ref)
            same = sum(x == y for x, y in zip(ref, result["queries"], strict=True))
            summary[name] = {
                "exact_queries": same,
                "total_queries": len(ref),
                "cache": result["cache"],
            }
            if name == "jit" and result["cache"] is not None:
                assert result["cache"]["compilations"] > 0, result["cache"]
                assert result["cache"]["failed_compilations"] == 0, result["cache"]
                assert result["cache"]["hits"] >= len(ref) - 1, result["cache"]
            if same != len(ref):
                raise AssertionError(
                    f"{name} differs from Arrow on {len(ref) - same} queries (vary={vary})"
                )
        quality["vary" if vary else "fixed"] = summary

    ref = json.loads((a.output / "quality-fixed-arrow.json").read_text())["queries"]
    post_results = {}
    quality_variants = [v for v in variants if v[1] == "post"]
    if a.post:
        # The PR has quality-only topk=ef=1200/2048, never substitute timings
        # from 128/1200 or 256/2048 for these two configurations.
        quality_variants += [
            (
                f"post-{ef}-{ef}",
                "post",
                "0" if a.sweep else "1",
                ef,
                ef,
                a.baseline_binary.resolve() if a.sweep else binary,
            )
            for ef in (1200, 2048)
        ]
    for variant in quality_variants:
        result = execute(
            "quality-" + variant[0], variant, "dump", ["--count", str(a.quality_count)]
        )
        post_results[variant[0]] = result["queries"]
        overlap = [
            len({d["id"] for d in q["results"]} & {d["id"] for d in r["results"]}) / 21
            for q, r in zip(result["queries"], ref, strict=True)
        ]
        comparisons = [
            [
                x["score"] - y["score"]
                for x, y in zip(q["results"], r["results"], strict=True)
            ]
            for q, r in zip(result["queries"], ref, strict=True)
        ]
        quality[variant[0]] = {
            "top21_overlap": statistics.mean(overlap),
            "queries": len(overlap),
            "every_rank_no_worse_queries": sum(
                all(d <= 0 for d in diff) for diff in comparisons
            ),
            "at_least_one_better_rank_queries": sum(
                any(d < 0 for d in diff) for diff in comparisons
            ),
            "score_direction": "COSINE distance; lower is better",
            "note": "Agreement with filtered ANN reference; not brute-force recall.",
        }
    quality["same_ef"] = {}
    for low, high in [
        ("post-64-300", "post-300-300"),
        ("post-64-400", "post-400-400"),
        ("post-64-600", "post-600-600"),
        ("post-128-1200", "post-1200-1200"),
        ("post-256-2048", "post-2048-2048"),
    ]:
        if low in post_results and high in post_results:
            left, right = post_results[low], post_results[high]
            quality["same_ef"][low + " vs " + high] = {
                "exact_queries": sum(x == y for x, y in zip(left, right, strict=True)),
                "total_queries": len(left),
            }
    (a.output / "quality-summary.json").write_text(json.dumps(quality, indent=2) + "\n")

    if a.quality_only:
        raise SystemExit(0)

records = []
# Rotate the order each repetition to limit fixed-order drift. Fresh processes
# ensure independent cold compilation and prevent changes to the JIT env cache.
for repetition in range(a.runs):
    order = (
        variants[repetition % len(variants) :] + variants[: repetition % len(variants)]
    )
    for clients in a.clients:
        for variant in order:
            name = f"run-{repetition + 1}-c{clients}-{variant[0]}"
            result = execute(
                name,
                variant,
                "run",
                [
                    "--clients",
                    str(clients),
                    "--count",
                    str(a.count),
                    "--seconds",
                    str(a.seconds),
                    "--warmup",
                    str(a.warmup),
                ],
            )
            before, after = result["cache_before_timing"], result["cache_after_timing"]
            if before and after:
                delta = {k: after[k] - before[k] for k in after}
                result["cache_timed_delta"] = delta
                if variant[0] == "jit":
                    assert delta["compilations"] == 0, delta
                    assert delta["hits"] >= result["count"], delta
                    assert after["failed_compilations"] == 0, after
                if variant[0] in ["arrow", "none", "hardcode"]:
                    assert after["compilations"] == 0, after
            result.update(variant=variant[0], repetition=repetition + 1)
            records.append(result)
            (a.output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
summary = {}
for variant in variants:
    for clients in a.clients:
        rows = [
            r for r in records if r["variant"] == variant[0] and r["clients"] == clients
        ]
        summary[f"{variant[0]}-c{clients}"] = {
            "qps": statistics.median(r["qps"] for r in rows),
            "mean_api_ms": statistics.median(r["mean_api_ms"] for r in rows),
            "mean_request_ms": statistics.median(r["mean_request_ms"] for r in rows),
            "p95_api_ms": statistics.median(r["p95_api_ms"] for r in rows),
            "qps_runs": [r["qps"] for r in rows],
            "mean_api_ms_runs": [r["mean_api_ms"] for r in rows],
            "mean_request_ms_runs": [r["mean_request_ms"] for r in rows],
            "requested_seconds": a.seconds,
            "count": sum(r["count"] for r in rows),
            "cache_timed_deltas": [r.get("cache_timed_delta") for r in rows],
        }
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (a.output / "table.md").open("w") as f:
    f.write(
        "| Path | Clients | Median QPS | Median mean API ms | Median mean request ms |\n|---|---:|---:|---:|---:|\n"
    )
    for key, value in summary.items():
        label, clients = key.rsplit("-c", 1)
        f.write(
            f"| {label} | {clients} | {value['qps']:,.2f} | {value['mean_api_ms']:.4f} | {value['mean_request_ms']:.4f} |\n"
        )
