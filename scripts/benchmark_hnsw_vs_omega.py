#!/usr/bin/env python3
"""Benchmark Zvec HNSW vs OMEGA without VectorDBBench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_lib import (
    BenchmarkResult,
    build_hnsw_profile,
    build_index,
    build_omega_profile,
    compute_recall_with_zvec,
    discover_index_files,
    get_offline_load_duration,
    load_dataset_config,
    merge_omega_detailed_profile,
    must_get,
    prepare_dataset_artifacts,
    print_header,
    resolve_core_tools,
    resolve_dataset_spec,
    resolve_index_path,
    resolve_paths,
    run_concurrency_benchmark,
    run_profile_benchmark,
    validate_profile_output,
    write_grouped_profiling_summaries,
    write_offline_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Zvec HNSW vs OMEGA")
    parser.add_argument("--config", required=True, help="Path to benchmark JSON config")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset key to run from the top-level JSON config map",
    )
    parser.add_argument(
        "--target-recalls",
        type=str,
        default=None,
        help="Optional comma-separated override for omega.target_recalls in the JSON config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--skip-hnsw", action="store_true", help="Skip HNSW benchmark")
    parser.add_argument("--skip-omega", action="store_true", help="Skip OMEGA benchmark")
    parser.add_argument("--build-only", action="store_true", help="Only build index, skip search")
    parser.add_argument("--search-only", action="store_true", help="Only run search on existing index")
    parser.add_argument(
        "--retrain-only",
        action="store_true",
        help="Reuse existing OMEGA index and only retrain the model during the build phase",
    )
    parser.add_argument(
        "--zvec-root",
        type=str,
        default=None,
        help="Path to zvec repo root (default: auto-detect from this script)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Directory used to store benchmark artifacts",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Root directory containing the raw dataset files",
    )
    return parser.parse_args()


def run_hnsw(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    dataset_spec: dict[str, object],
    dataset_artifacts: dict[str, object],
    bench_bin: Path,
    hnsw_path: Path,
    hnsw_db_label: str,
    common: dict[str, object],
    hnsw_config: dict[str, object],
    profiling_config: dict[str, object],
) -> BenchmarkResult:
    print_header("HNSW Benchmark")

    hnsw_specific_args = hnsw_config.get("args", {})
    if not args.search_only:
        print("\n[Phase 1] Building HNSW index...")
        offline_metrics = build_index(
            index_kind="HNSW",
            index_path=hnsw_path,
            dataset_spec=dataset_spec,
            dataset_artifacts=dataset_artifacts,
            common_args=common,
            specific_args=hnsw_specific_args,
            retrain_only=False,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            write_offline_summary(hnsw_path, hnsw_db_label, offline_metrics)

    if args.build_only:
        return BenchmarkResult(
            type="HNSW",
            path=str(hnsw_path),
            success=True,
            target_recall=None,
            load_duration=get_offline_load_duration(hnsw_path),
        )

    index_files = discover_index_files(hnsw_path)
    recall = compute_recall_with_zvec(
        index_kind="HNSW",
        index_path=hnsw_path,
        dataset_artifacts=dataset_artifacts,
        common_args=common,
        target_recall=None,
        dry_run=args.dry_run,
    )

    benchmark = run_concurrency_benchmark(
        bench_bin=bench_bin,
        index_files=index_files,
        dataset_artifacts=dataset_artifacts,
        dataset_spec=dataset_spec,
        common_args=common,
        target_recall=None,
        dry_run=args.dry_run,
    )
    online = benchmark["summary"]
    success = online.get("retcode", 0) == 0

    hnsw_profile = None
    if success and not args.dry_run:
        print("\n[Profiling] Running HNSW single-thread profiling pass...")
        profile_ret, profile_output, profile_bench = run_profile_benchmark(
            bench_bin=bench_bin,
            index_files=index_files,
            dataset_artifacts=dataset_artifacts,
            dataset_spec=dataset_spec,
            common_args=common,
            target_recall=None,
            dry_run=False,
            extra_env={
                "ZVEC_LOG_LEVEL": "INFO",
                "ZVEC_HNSW_LOG_QUERY_STATS": "1",
                "ZVEC_HNSW_LOG_QUERY_LIMIT": str(profiling_config.get("hnsw_query_limit", 2000)),
            },
        )
        validate_profile_output("HNSW", profile_ret, profile_output, "HNSW query stats:")
        hnsw_profile = build_hnsw_profile(
            {"qps": online.get("qps"), "recall": recall},
            profile_output,
            profile_bench,
        )

    return BenchmarkResult(
        type="HNSW",
        path=str(hnsw_path),
        success=success,
        target_recall=None,
        load_duration=get_offline_load_duration(hnsw_path),
        qps=online.get("qps"),
        avg_latency_ms=online.get("avg_latency_ms"),
        p50_latency_ms=online.get("p50_latency_ms"),
        p90_latency_ms=online.get("p90_latency_ms"),
        p95_latency_ms=online.get("p95_latency_ms"),
        p99_latency_ms=online.get("p99_latency_ms"),
        recall=recall,
        profiling=hnsw_profile,
    )


def run_omega(
    *,
    args: argparse.Namespace,
    dataset_spec: dict[str, object],
    dataset_artifacts: dict[str, object],
    bench_bin: Path,
    omega_path: Path,
    omega_db_label: str,
    common: dict[str, object],
    omega_config: dict[str, object],
    profiling_config: dict[str, object],
    hnsw_profile: dict[str, object] | None,
    target_recalls: list[float],
) -> list[BenchmarkResult]:
    print_header("OMEGA Benchmark")

    omega_specific_args = omega_config.get("args", {})
    if not args.search_only:
        if args.retrain_only:
            print("\n[Phase 1] Retraining OMEGA model only (reusing existing index)...")
        else:
            print("\n[Phase 1] Building OMEGA index + training model...")
        offline_metrics = build_index(
            index_kind="OMEGA",
            index_path=omega_path,
            dataset_spec=dataset_spec,
            dataset_artifacts=dataset_artifacts,
            common_args=common,
            specific_args=omega_specific_args,
            retrain_only=args.retrain_only,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            write_offline_summary(
                omega_path,
                omega_db_label,
                offline_metrics,
                retrain_only=args.retrain_only,
            )

    if args.build_only:
        return [
            BenchmarkResult(
                type="OMEGA",
                path=str(omega_path),
                success=True,
                target_recall=target_recall,
                load_duration=get_offline_load_duration(omega_path),
            )
            for target_recall in target_recalls
        ]

    results: list[BenchmarkResult] = []
    index_files = discover_index_files(omega_path)
    omega_common = dict(common)
    omega_common["k"] = int(common["k"])

    for target_recall in target_recalls:
        print_header(f"OMEGA Search Benchmark (target_recall={target_recall})")
        recall = compute_recall_with_zvec(
            index_kind="OMEGA",
            index_path=omega_path,
            dataset_artifacts=dataset_artifacts,
            common_args=common,
            target_recall=target_recall,
            dry_run=args.dry_run,
        )
        benchmark = run_concurrency_benchmark(
            bench_bin=bench_bin,
            index_files=index_files,
            dataset_artifacts=dataset_artifacts,
            dataset_spec=dataset_spec,
            common_args=omega_common,
            target_recall=target_recall,
            dry_run=args.dry_run,
        )
        online = benchmark["summary"]
        success = online.get("retcode", 0) == 0

        omega_profile = None
        if success and not args.dry_run:
            print("\n[Profiling] Running OMEGA single-thread profiling pass...")
            profile_env = {
                "ZVEC_LOG_LEVEL": "INFO",
                "ZVEC_OMEGA_LOG_QUERY_STATS": "1",
                "ZVEC_OMEGA_LOG_QUERY_LIMIT": str(profiling_config.get("omega_query_limit", 2000)),
            }
            profile_ret, profile_output, profile_bench = run_profile_benchmark(
                bench_bin=bench_bin,
                index_files=index_files,
                dataset_artifacts=dataset_artifacts,
                dataset_spec=dataset_spec,
                common_args=omega_common,
                target_recall=target_recall,
                dry_run=False,
                extra_env=profile_env,
            )
            validate_profile_output("OMEGA", profile_ret, profile_output, "OMEGA query stats:")
            omega_profile = build_omega_profile(
                {"qps": online.get("qps"), "recall": recall},
                profile_output,
                profile_bench,
                hnsw_profile,
            )
            if profiling_config.get("omega_profile_control_timing", True):
                print("\n[Profiling] Running OMEGA control-timing pass...")
                detailed_ret, detailed_output, detailed_bench = run_profile_benchmark(
                    bench_bin=bench_bin,
                    index_files=index_files,
                    dataset_artifacts=dataset_artifacts,
                    dataset_spec=dataset_spec,
                    common_args=omega_common,
                    target_recall=target_recall,
                    dry_run=False,
                    extra_env={**profile_env, "ZVEC_OMEGA_PROFILE_CONTROL_TIMING": "1"},
                )
                validate_profile_output(
                    "OMEGA", detailed_ret, detailed_output, "OMEGA query stats:"
                )
                detailed_profile = build_omega_profile(
                    {"qps": online.get("qps"), "recall": recall},
                    detailed_output,
                    detailed_bench,
                    hnsw_profile,
                )
                omega_profile = merge_omega_detailed_profile(omega_profile, detailed_profile)

        results.append(
            BenchmarkResult(
                type="OMEGA",
                path=str(omega_path),
                success=success,
                target_recall=target_recall,
                load_duration=get_offline_load_duration(omega_path),
                qps=online.get("qps"),
                avg_latency_ms=online.get("avg_latency_ms"),
                p50_latency_ms=online.get("p50_latency_ms"),
                p90_latency_ms=online.get("p90_latency_ms"),
                p95_latency_ms=online.get("p95_latency_ms"),
                p99_latency_ms=online.get("p99_latency_ms"),
                recall=recall,
                profiling=omega_profile,
            )
        )

    return results


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_dataset_config(config_path, args.dataset)
    zvec_root, benchmark_dir = resolve_paths(
        Path(__file__).resolve(),
        config,
        args.zvec_root,
        args.benchmark_dir,
    )
    dataset_spec = resolve_dataset_spec(args.dataset, config, args.dataset_root)
    dataset_artifacts = prepare_dataset_artifacts(
        args.dataset, dataset_spec, benchmark_dir, dry_run=args.dry_run
    )
    bench_bin, recall_bin = resolve_core_tools(zvec_root)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    common = must_get(config, "common")
    hnsw_config = must_get(config, "hnsw")
    omega_config = must_get(config, "omega")
    profiling_config = config.get("profiling", {})

    hnsw_path = resolve_index_path(benchmark_dir, must_get(hnsw_config, "path"))
    omega_path = resolve_index_path(benchmark_dir, must_get(omega_config, "path"))
    hnsw_db_label = must_get(hnsw_config, "db_label")
    omega_db_label = must_get(omega_config, "db_label")
    target_recalls = omega_config.get("target_recalls", [])
    if args.target_recalls:
        target_recalls = [float(value) for value in args.target_recalls.split(",") if value]
    if not target_recalls:
        raise ValueError("omega.target_recalls must be a non-empty list")

    print("=" * 70)
    print(f"Zvec HNSW vs OMEGA ({dataset_name})")
    print(f"Config: {config_path}")
    print("=" * 70)
    print(f"zvec_root: {zvec_root}")
    print(f"benchmark_dir: {benchmark_dir}")
    print(f"dataset_dir: {dataset_spec['dataset_dir']}")
    print(f"bench_bin: {bench_bin}")
    print(f"recall_bin: {recall_bin}")
    print(f"hnsw_path: {hnsw_path}")
    print(f"omega_path: {omega_path}")
    print(f"target_recalls: {target_recalls}")
    print("=" * 70)

    results: list[BenchmarkResult] = []
    hnsw_profile = None

    if not args.skip_hnsw:
        hnsw_result = run_hnsw(
            args=args,
            dataset_name=dataset_name,
            dataset_spec=dataset_spec,
            dataset_artifacts=dataset_artifacts,
            bench_bin=bench_bin,
            hnsw_path=hnsw_path,
            hnsw_db_label=hnsw_db_label,
            common=common,
            hnsw_config=hnsw_config,
            profiling_config=profiling_config,
        )
        results.append(hnsw_result)
        hnsw_profile = hnsw_result.profiling

    if not args.skip_omega:
        results.extend(
            run_omega(
                args=args,
                dataset_spec=dataset_spec,
                dataset_artifacts=dataset_artifacts,
                bench_bin=bench_bin,
                omega_path=omega_path,
                omega_db_label=omega_db_label,
                common=common,
                omega_config=omega_config,
                profiling_config=profiling_config,
                hnsw_profile=hnsw_profile,
                target_recalls=target_recalls,
            )
        )

    if results:
        written_summary_paths = (
            write_grouped_profiling_summaries(dataset_name, results)
            if not args.dry_run
            else []
        )
        print("\n\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(
            f"{'Type':<10} {'target_recall':<15} {'load_dur(s)':<12} "
            f"{'qps':<8} {'avg_latency(ms)':<16} {'p95_latency(ms)':<16} "
            f"{'recall':<10} {'Status':<10}"
        )
        print("-" * 100)
        for result in results:
            tr = f"{result.target_recall:.2f}" if result.target_recall is not None else "N/A"
            status = "OK" if result.success else "FAILED"
            ld = f"{result.load_duration:.1f}" if result.load_duration is not None else "N/A"
            qps = f"{result.qps:.1f}" if result.qps is not None else "N/A"
            avg_latency = (
                f"{result.avg_latency_ms:.3f}" if result.avg_latency_ms is not None else "N/A"
            )
            p95_latency = (
                f"{result.p95_latency_ms:.3f}" if result.p95_latency_ms is not None else "N/A"
            )
            recall = f"{result.recall:.4f}" if result.recall is not None else "N/A"
            print(
                f"{result.type:<10} {tr:<15} {ld:<12} {qps:<8} "
                f"{avg_latency:<16} {p95_latency:<16} {recall:<10} {status:<10}"
            )

        print("\nProfiling Summary")
        print("-" * 75)
        print(
            f"{'Type':<10} {'target_recall':<15} {'avg_lat(ms)':<12} "
            f"{'avg_cmps':<12} {'avg_pred_calls':<16} {'avg_model_ms':<14} "
            f"{'saved_cmps':<12}"
        )
        for result in results:
            profile = result.profiling or {}
            tr = f"{result.target_recall:.2f}" if result.target_recall is not None else "N/A"
            avg_lat = profile.get("profile_avg_end2end_latency_ms")
            avg_cmps = profile.get("profile_avg_cmps")
            avg_pred_calls = profile.get("profile_avg_prediction_calls")
            avg_model_ms = profile.get("profile_avg_model_overhead_ms")
            saved_cmps = profile.get("profile_avg_early_stop_saved_cmps")
            print(
                f"{result.type:<10} "
                f"{tr:<15} "
                f"{(f'{avg_lat:.3f}' if avg_lat is not None else 'N/A'):<12} "
                f"{(f'{avg_cmps:.1f}' if avg_cmps is not None else 'N/A'):<12} "
                f"{(f'{avg_pred_calls:.2f}' if avg_pred_calls is not None else 'N/A'):<16} "
                f"{(f'{avg_model_ms:.3f}' if avg_model_ms is not None else 'N/A'):<14} "
                f"{(f'{saved_cmps:.1f}' if saved_cmps is not None else 'N/A'):<12}"
            )

        print()
        for path in written_summary_paths:
            print(f"Summary JSON: {path}")

    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
