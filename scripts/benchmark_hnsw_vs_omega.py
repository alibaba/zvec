from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from benchmark_lib import (
    BenchmarkResult,
    build_index,
    compute_recall_with_zvec,
    discover_index_files,
    emit,
    get_offline_load_duration,
    load_dataset_config,
    must_get,
    prepare_dataset_artifacts,
    print_header,
    resolve_core_tools,
    resolve_dataset_spec,
    resolve_index_path,
    resolve_paths,
    run_concurrency_benchmark,
    summarize_omega_prediction_profile,
    write_grouped_online_summaries,
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
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without executing"
    )
    parser.add_argument("--skip-hnsw", action="store_true", help="Skip HNSW benchmark")
    parser.add_argument(
        "--skip-omega", action="store_true", help="Skip OMEGA benchmark"
    )
    parser.add_argument(
        "--build-only", action="store_true", help="Only build index, skip search"
    )
    parser.add_argument(
        "--search-only", action="store_true", help="Only run search on existing index"
    )
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
    parser.add_argument(
        "--omega-profile-duration",
        type=int,
        default=5,
        help=(
            "Short diagnostic pass duration, in seconds, used to collect OMEGA "
            "prediction profile metrics for the summary JSON"
        ),
    )
    return parser.parse_args()


def run_hnsw(
    *,
    args: argparse.Namespace,
    dataset_spec: dict[str, object],
    dataset_artifacts: dict[str, object],
    bench_bin: Path,
    hnsw_path: Path,
    hnsw_db_label: str,
    common: dict[str, object],
    hnsw_config: dict[str, object],
) -> BenchmarkResult:
    print_header("HNSW Benchmark")

    hnsw_specific_args = hnsw_config.get("args", {})
    if not args.search_only:
        emit("\n[Phase 1] Building HNSW index...")
        offline_metrics, _ = build_index(
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
    target_recalls: list[float],
) -> list[BenchmarkResult]:
    print_header("OMEGA Benchmark")

    omega_specific_args = omega_config.get("args", {})
    omega_collection = None
    if not args.search_only:
        phase_message = (
            "\n[Phase 1] Retraining OMEGA model only (reusing existing index)..."
            if args.retrain_only
            else "\n[Phase 1] Building OMEGA index + training model..."
        )
        emit(phase_message)
        offline_metrics, omega_collection = build_index(
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

    return _run_omega_searches(
        args=args,
        dataset_spec=dataset_spec,
        dataset_artifacts=dataset_artifacts,
        bench_bin=bench_bin,
        omega_path=omega_path,
        common=common,
        target_recalls=target_recalls,
        collection=omega_collection,
    )


def _run_omega_searches(
    *,
    args: argparse.Namespace,
    dataset_spec: dict[str, object],
    dataset_artifacts: dict[str, object],
    bench_bin: Path,
    omega_path: Path,
    common: dict[str, object],
    target_recalls: list[float],
    collection: Any = None,
) -> list[BenchmarkResult]:
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
            collection=collection,
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
        omega_prediction_profile = None
        if online.get("retcode", 0) == 0 and not args.dry_run:
            profile_thread_count = online.get("thread_count")
            if profile_thread_count is not None:
                profile_path = (
                    omega_path
                    / f"omega_prediction_profile_tr{int(target_recall * 10000)}.jsonl"
                )
                profile_path.unlink(missing_ok=True)
                profile_common = dict(omega_common)
                profile_common["num_concurrency"] = str(profile_thread_count)
                profile_common["concurrency_duration"] = max(
                    1, int(args.omega_profile_duration)
                )
                emit(
                    "\n[Diagnostics] Collecting OMEGA prediction profile "
                    f"({profile_common['concurrency_duration']}s, "
                    f"threads={profile_thread_count})..."
                )
                profile_benchmark = run_concurrency_benchmark(
                    bench_bin=bench_bin,
                    index_files=index_files,
                    dataset_artifacts=dataset_artifacts,
                    dataset_spec=dataset_spec,
                    common_args=profile_common,
                    target_recall=target_recall,
                    dry_run=args.dry_run,
                    extra_env={
                        "ZVEC_OMEGA_PROFILE_PREDICTION": "1",
                        "ZVEC_OMEGA_PROFILE_OUTPUT": str(profile_path),
                    },
                )
                profile_summary = profile_benchmark["summary"]
                if profile_summary.get("retcode", 0) == 0:
                    omega_prediction_profile = summarize_omega_prediction_profile(
                        profile_path, profile_summary
                    )
                    if omega_prediction_profile is None:
                        omega_prediction_profile = {
                            "profile_path": str(profile_path),
                            "error": "OMEGA prediction profile file was empty or missing",
                        }
                else:
                    omega_prediction_profile = {
                        "profile_path": str(profile_path),
                        "error": "OMEGA prediction profile pass failed",
                        "retcode": profile_summary.get("retcode"),
                    }
        results.append(
            BenchmarkResult(
                type="OMEGA",
                path=str(omega_path),
                success=online.get("retcode", 0) == 0,
                target_recall=target_recall,
                load_duration=get_offline_load_duration(omega_path),
                qps=online.get("qps"),
                avg_latency_ms=online.get("avg_latency_ms"),
                p50_latency_ms=online.get("p50_latency_ms"),
                p90_latency_ms=online.get("p90_latency_ms"),
                p95_latency_ms=online.get("p95_latency_ms"),
                p99_latency_ms=online.get("p99_latency_ms"),
                recall=recall,
                omega_prediction_profile=omega_prediction_profile,
            )
        )

    return results


def _parse_target_recalls(
    args: argparse.Namespace, omega_config: dict[str, object]
) -> list[float]:
    target_recalls = omega_config.get("target_recalls", [])
    if args.target_recalls:
        target_recalls = [
            float(value) for value in args.target_recalls.split(",") if value
        ]
    if not target_recalls:
        raise ValueError("omega.target_recalls must be a non-empty list")
    return list(target_recalls)


def _emit_run_header(
    *,
    dataset_name: str,
    config_path: Path,
    zvec_root: Path,
    benchmark_dir: Path,
    dataset_spec: dict[str, object],
    bench_bin: Path,
    recall_bin: Path,
    hnsw_path: Path,
    omega_path: Path,
    target_recalls: list[float],
) -> None:
    emit("=" * 70)
    emit(f"Zvec HNSW vs OMEGA ({dataset_name})")
    emit(f"Config: {config_path}")
    emit("=" * 70)
    emit(f"zvec_root: {zvec_root}")
    emit(f"benchmark_dir: {benchmark_dir}")
    emit(f"dataset_dir: {dataset_spec['dataset_dir']}")
    emit(f"bench_bin: {bench_bin}")
    emit(f"recall_bin: {recall_bin}")
    emit(f"hnsw_path: {hnsw_path}")
    emit(f"omega_path: {omega_path}")
    emit(f"target_recalls: {target_recalls}")
    emit("=" * 70)


def _format_summary_row(result: BenchmarkResult) -> str:
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
    return (
        f"{result.type:<10} {tr:<15} {ld:<12} {qps:<8} "
        f"{avg_latency:<16} {p95_latency:<16} {recall:<10} {status:<10}"
    )


def _emit_result_summary(
    results: list[BenchmarkResult], summary_paths: list[Path]
) -> None:
    emit(f"\n\n{'=' * 70}")
    emit("Benchmark Summary")
    emit("=" * 70)
    emit(
        f"{'Type':<10} {'target_recall':<15} {'load_dur(s)':<12} "
        f"{'qps':<8} {'avg_latency(ms)':<16} {'p95_latency(ms)':<16} "
        f"{'recall':<10} {'Status':<10}"
    )
    emit("-" * 100)
    for result in results:
        emit(_format_summary_row(result))

    emit()
    for path in summary_paths:
        emit(f"Summary JSON: {path}")


def _resolve_run_context(args: argparse.Namespace) -> dict[str, Any]:
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

    hnsw_config = must_get(config, "hnsw")
    omega_config = must_get(config, "omega")

    return {
        "config_path": config_path,
        "dataset_name": args.dataset,
        "dataset_spec": dataset_spec,
        "dataset_artifacts": dataset_artifacts,
        "zvec_root": zvec_root,
        "benchmark_dir": benchmark_dir,
        "bench_bin": bench_bin,
        "recall_bin": recall_bin,
        "common": must_get(config, "common"),
        "hnsw_config": hnsw_config,
        "omega_config": omega_config,
        "hnsw_path": resolve_index_path(benchmark_dir, must_get(hnsw_config, "path")),
        "omega_path": resolve_index_path(benchmark_dir, must_get(omega_config, "path")),
        "hnsw_db_label": must_get(hnsw_config, "db_label"),
        "omega_db_label": must_get(omega_config, "db_label"),
        "target_recalls": _parse_target_recalls(args, omega_config),
    }


def main() -> int:
    args = parse_args()
    context = _resolve_run_context(args)
    _emit_run_header(
        dataset_name=context["dataset_name"],
        config_path=context["config_path"],
        zvec_root=context["zvec_root"],
        benchmark_dir=context["benchmark_dir"],
        dataset_spec=context["dataset_spec"],
        bench_bin=context["bench_bin"],
        recall_bin=context["recall_bin"],
        hnsw_path=context["hnsw_path"],
        omega_path=context["omega_path"],
        target_recalls=context["target_recalls"],
    )

    results: list[BenchmarkResult] = []
    if not args.skip_hnsw:
        results.append(
            run_hnsw(
                args=args,
                dataset_spec=context["dataset_spec"],
                dataset_artifacts=context["dataset_artifacts"],
                bench_bin=context["bench_bin"],
                hnsw_path=context["hnsw_path"],
                hnsw_db_label=context["hnsw_db_label"],
                common=context["common"],
                hnsw_config=context["hnsw_config"],
            )
        )

    if not args.skip_omega:
        results.extend(
            run_omega(
                args=args,
                dataset_spec=context["dataset_spec"],
                dataset_artifacts=context["dataset_artifacts"],
                bench_bin=context["bench_bin"],
                omega_path=context["omega_path"],
                omega_db_label=context["omega_db_label"],
                common=context["common"],
                omega_config=context["omega_config"],
                target_recalls=context["target_recalls"],
            )
        )

    if results:
        summary_paths = (
            write_grouped_online_summaries(context["dataset_name"], results)
            if not args.dry_run
            else []
        )
        _emit_result_summary(results, summary_paths)

    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
