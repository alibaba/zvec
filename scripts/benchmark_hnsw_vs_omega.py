#!/usr/bin/env python3
"""Generic VectorDBBench runner for Zvec HNSW vs Zvec+OMEGA."""

import argparse
import sys
from pathlib import Path
from benchmark_lib import (
    BenchmarkResult,
    build_base_command,
    build_hnsw_profile,
    build_omega_profile,
    get_offline_load_duration,
    get_run_result,
    latency_summary_from_profile,
    load_dataset_config,
    merge_omega_detailed_profile,
    must_get,
    print_header,
    resolve_index_path,
    resolve_paths,
    resolve_vectordbbench_command,
    run_command,
    run_command_capture,
    snapshot_result_files,
    validate_profile_output,
    write_grouped_profiling_summaries,
    write_offline_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic VectorDBBench runner for Zvec HNSW vs OMEGA"
    )
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
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
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
        "--vectordbbench-root",
        type=str,
        default=None,
        help="Path to VectorDBBench repo root (default: $VECTORDBBENCH_ROOT or sibling repo)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Directory used to store benchmark artifacts "
        "(default: config benchmark_dir, $ZVEC_BENCHMARK_DIR, or <zvec_root>/benchmark_results)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing VectorDBBench JSON result files",
    )
    parser.add_argument(
        "--concurrent-warmup",
        action="store_true",
        help="Run a concurrent-only warmup pass before the measured search benchmark",
    )
    parser.add_argument(
        "--warmup-duration",
        type=int,
        default=None,
        help="Warmup concurrency duration in seconds "
        "(default: config warmup.duration or 15)",
    )
    parser.add_argument(
        "--warmup-num-concurrency",
        type=str,
        default=None,
        help="Warmup concurrency list, e.g. '4' or '4,8' "
        "(default: config warmup.num_concurrency or the first configured concurrency)",
    )
    return parser.parse_args()


def resolve_warmup_settings(
    args: argparse.Namespace, common: dict[str, object], config: dict[str, object]
) -> tuple[bool, int, str]:
    warmup_config = config.get("warmup", {})
    enabled = args.concurrent_warmup or bool(warmup_config.get("enabled", False))
    configured_concurrency = str(common.get("num_concurrency", "1"))
    default_num_concurrency = str(
        warmup_config.get("num_concurrency", configured_concurrency.split(",")[0])
    )
    num_concurrency = args.warmup_num_concurrency or default_num_concurrency
    duration = args.warmup_duration or int(warmup_config.get("duration", 15))
    return enabled, duration, num_concurrency


def run_concurrent_warmup(
    *,
    label: str,
    vectordbbench_cmd: list[str],
    client_name: str,
    path: Path,
    db_label: str,
    case_type: str,
    common_args: dict[str, object],
    specific_args: dict[str, object],
    vectordbbench_root: Path,
    dry_run: bool,
    extra_flags: list[str] | None = None,
) -> int:
    print(f"\n[Warmup] Running concurrent-only warmup for {label}...")
    warmup_flags = ["skip-drop-old", "skip-load", "skip-search-serial"]
    if extra_flags:
        warmup_flags.extend(extra_flags)
    warmup_cmd = build_base_command(
        vectordbbench_cmd,
        client_name,
        path,
        db_label,
        case_type,
        common_args,
        specific_args,
        warmup_flags,
    )
    return run_command(warmup_cmd, vectordbbench_root, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_dataset_config(config_path, args.dataset)
    zvec_root, vectordbbench_root, benchmark_dir, results_dir = resolve_paths(
        Path(__file__).resolve(),
        config,
        args.zvec_root,
        args.vectordbbench_root,
        args.benchmark_dir,
        args.results_dir,
    )
    vectordbbench_cmd = resolve_vectordbbench_command()
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    common = must_get(config, "common")
    hnsw_config = must_get(config, "hnsw")
    omega_config = must_get(config, "omega")
    profiling_config = config.get("profiling", {})
    warmup_enabled, warmup_duration, warmup_num_concurrency = resolve_warmup_settings(
        args, common, config
    )

    case_type = must_get(common, "case_type")
    hnsw_path = resolve_index_path(benchmark_dir, must_get(hnsw_config, "path"))
    omega_path = resolve_index_path(benchmark_dir, must_get(omega_config, "path"))
    hnsw_db_label = must_get(hnsw_config, "db_label")
    omega_db_label = must_get(omega_config, "db_label")
    target_recalls = omega_config.get("target_recalls", [])
    if args.target_recalls:
        target_recalls = [float(value) for value in args.target_recalls.split(",") if value]
    if not target_recalls:
        raise ValueError("omega.target_recalls must be a non-empty list")

    hnsw_common_args = {k: v for k, v in common.items() if k != "case_type"}
    hnsw_specific_args = hnsw_config.get("args", {})
    omega_specific_args = omega_config.get("args", {})
    warmup_common_args = dict(hnsw_common_args)
    warmup_common_args["num_concurrency"] = warmup_num_concurrency
    warmup_common_args["concurrency_duration"] = warmup_duration

    print("=" * 70)
    print(f"VectorDBBench: Zvec HNSW vs OMEGA ({dataset_name})")
    print(f"Config: {config_path}")
    print("=" * 70)
    print(f"zvec_root: {zvec_root}")
    print(f"vectordbbench_root: {vectordbbench_root}")
    print(f"vectordbbench_cmd: {' '.join(vectordbbench_cmd)}")
    print(f"benchmark_dir: {benchmark_dir}")
    print(f"results_dir: {results_dir}")
    print(f"hnsw_path: {hnsw_path}")
    print(f"omega_path: {omega_path}")
    print(f"target_recalls: {target_recalls}")
    print(
        "concurrent_warmup: "
        + (
            f"enabled (num_concurrency={warmup_num_concurrency}, duration={warmup_duration}s)"
            if warmup_enabled
            else "disabled"
        )
    )
    print(
        "build_mode: "
        + ("retrain model only (reuse existing index)" if args.retrain_only else "build index + train model")
    )
    print("=" * 70)

    results: list[BenchmarkResult] = []

    if not args.skip_hnsw:
        print_header("HNSW Benchmark")

        if not args.search_only:
            print("\n[Phase 1] Building HNSW index...")
            before_files = snapshot_result_files(results_dir)
            cmd = build_base_command(
                vectordbbench_cmd,
                "zvec",
                hnsw_path,
                hnsw_db_label,
                case_type,
                hnsw_common_args,
                hnsw_specific_args,
                ["skip-search-serial", "skip-search-concurrent"],
            )
            ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
            if ret != 0 and not args.dry_run:
                print("ERROR: HNSW build failed!")
                return 1
            if not args.dry_run:
                write_offline_summary(
                    hnsw_path,
                    hnsw_db_label,
                    get_run_result(hnsw_db_label, before_files, results_dir),
                )

        if not args.build_only:
            if warmup_enabled:
                warmup_ret = run_concurrent_warmup(
                    label="HNSW",
                    vectordbbench_cmd=vectordbbench_cmd,
                    client_name="zvec",
                    path=hnsw_path,
                    db_label=hnsw_db_label,
                    case_type=case_type,
                    common_args=warmup_common_args,
                    specific_args=hnsw_specific_args,
                    vectordbbench_root=vectordbbench_root,
                    dry_run=args.dry_run,
                )
                if warmup_ret != 0 and not args.dry_run:
                    print("ERROR: HNSW concurrent warmup failed!")
                    return 1
            print("\n[Phase 2] Running HNSW search benchmark...")
            before_files = snapshot_result_files(results_dir)
            cmd = build_base_command(
                vectordbbench_cmd,
                "zvec",
                hnsw_path,
                hnsw_db_label,
                case_type,
                hnsw_common_args,
                hnsw_specific_args,
                ["skip-drop-old", "skip-load"],
            )
            ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
            metrics = get_run_result(hnsw_db_label, before_files, results_dir) if not args.dry_run else {}
            load_duration = get_offline_load_duration(hnsw_path)
            hnsw_profile = None
            if ret == 0 and not args.dry_run:
                print("\n[Profiling] Running HNSW serial-only profiling pass...")
                profile_cmd = build_base_command(
                    vectordbbench_cmd,
                    "zvec",
                    hnsw_path,
                    hnsw_db_label,
                    case_type,
                    hnsw_common_args,
                    hnsw_specific_args,
                    ["skip-drop-old", "skip-load", "skip-search-concurrent"],
                )
                profile_ret, profile_output = run_command_capture(
                    profile_cmd,
                    vectordbbench_root,
                    dry_run=False,
                    extra_env={
                        "ZVEC_LOG_LEVEL": "INFO",
                        "ZVEC_HNSW_LOG_QUERY_STATS": "1",
                        "ZVEC_HNSW_LOG_QUERY_LIMIT": str(profiling_config.get("hnsw_query_limit", 2000)),
                    },
                )
                validate_profile_output("HNSW", profile_ret, profile_output, "HNSW query stats:")
                hnsw_profile = build_hnsw_profile(metrics, profile_output)
            latency_summary = latency_summary_from_profile(hnsw_profile)
            results.append(
                BenchmarkResult(
                    type="HNSW",
                    path=str(hnsw_path),
                    success=ret == 0,
                    target_recall=None,
                    load_duration=load_duration if load_duration is not None else metrics.get("load_duration"),
                    qps=metrics.get("qps"),
                    avg_latency_ms=latency_summary["avg_latency_ms"],
                    p50_latency_ms=latency_summary["p50_latency_ms"],
                    p90_latency_ms=latency_summary["p90_latency_ms"],
                    p95_latency_ms=latency_summary["p95_latency_ms"],
                    p99_latency_ms=latency_summary["p99_latency_ms"],
                    recall=metrics.get("recall"),
                    profiling=hnsw_profile,
                )
            )

    if not args.skip_omega:
        build_target_recall = target_recalls[0]
        print_header("OMEGA Benchmark")

        if not args.search_only:
            if args.retrain_only:
                print("\n[Phase 1] Retraining OMEGA model only (reusing existing index)...")
            else:
                print("\n[Phase 1] Building OMEGA index + training model...")
            print(
                f"Build-time target_recall is ignored by training; using first requested value "
                f"for CLI compatibility: {build_target_recall}"
            )
            before_files = snapshot_result_files(results_dir)
            build_flags = ["skip-search-serial", "skip-search-concurrent"]
            if args.retrain_only:
                build_flags.extend(["skip-drop-old", "skip-load", "retrain-only"])
            cmd = build_base_command(
                vectordbbench_cmd,
                "zvecomega",
                omega_path,
                omega_db_label,
                case_type,
                hnsw_common_args,
                {**omega_specific_args, "target_recall": build_target_recall},
                build_flags,
            )
            ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
            if ret != 0 and not args.dry_run:
                print("ERROR: OMEGA build failed!")
                return 1
            if not args.dry_run:
                write_offline_summary(
                    omega_path,
                    omega_db_label,
                    get_run_result(omega_db_label, before_files, results_dir),
                    retrain_only=args.retrain_only,
                )

        if not args.build_only:
            for target_recall in target_recalls:
                print_header(f"OMEGA Search Benchmark (target_recall={target_recall})")
                if warmup_enabled:
                    warmup_extra_flags = ["retrain-only"] if args.retrain_only else None
                    warmup_ret = run_concurrent_warmup(
                        label=f"OMEGA target_recall={target_recall}",
                        vectordbbench_cmd=vectordbbench_cmd,
                        client_name="zvecomega",
                        path=omega_path,
                        db_label=omega_db_label,
                        case_type=case_type,
                        common_args=warmup_common_args,
                        specific_args={**omega_specific_args, "target_recall": target_recall},
                        vectordbbench_root=vectordbbench_root,
                        dry_run=args.dry_run,
                        extra_flags=warmup_extra_flags,
                    )
                    if warmup_ret != 0 and not args.dry_run:
                        print("ERROR: OMEGA concurrent warmup failed!")
                        return 1
                before_files = snapshot_result_files(results_dir)
                search_flags = ["skip-drop-old", "skip-load"]
                if args.retrain_only:
                    search_flags.append("retrain-only")
                cmd = build_base_command(
                    vectordbbench_cmd,
                    "zvecomega",
                    omega_path,
                    omega_db_label,
                    case_type,
                    hnsw_common_args,
                    {**omega_specific_args, "target_recall": target_recall},
                    search_flags,
                )
                ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
                metrics = get_run_result(omega_db_label, before_files, results_dir) if not args.dry_run else {}
                load_duration = get_offline_load_duration(omega_path)
                omega_profile = None
                if ret == 0 and not args.dry_run:
                    print("\n[Profiling] Running OMEGA serial-only latency pass...")
                    profile_flags = ["skip-drop-old", "skip-load", "skip-search-concurrent"]
                    if args.retrain_only:
                        profile_flags.append("retrain-only")
                    profile_cmd = build_base_command(
                        vectordbbench_cmd,
                        "zvecomega",
                        omega_path,
                        omega_db_label,
                        case_type,
                        hnsw_common_args,
                        {**omega_specific_args, "target_recall": target_recall},
                        profile_flags,
                    )
                    profile_env = {
                        "ZVEC_LOG_LEVEL": "INFO",
                        "ZVEC_OMEGA_LOG_QUERY_STATS": "1",
                        "ZVEC_OMEGA_LOG_QUERY_LIMIT": str(profiling_config.get("omega_query_limit", 2000)),
                    }
                    profile_ret, profile_output = run_command_capture(
                        profile_cmd,
                        vectordbbench_root,
                        dry_run=False,
                        extra_env=profile_env,
                    )
                    validate_profile_output("OMEGA", profile_ret, profile_output, "OMEGA query stats:")
                    baseline_profile = next(
                        (result.profiling for result in results if result.type == "HNSW" and result.profiling),
                        None,
                    )
                    omega_profile = build_omega_profile(metrics, profile_output, baseline_profile)
                    if profiling_config.get("omega_profile_control_timing", True):
                        print("\n[Profiling] Running OMEGA detailed control-timing pass...")
                        detailed_env = dict(profile_env)
                        detailed_env["ZVEC_OMEGA_PROFILE_CONTROL_TIMING"] = "1"
                        detailed_ret, detailed_output = run_command_capture(
                            profile_cmd,
                            vectordbbench_root,
                            dry_run=False,
                            extra_env=detailed_env,
                        )
                        validate_profile_output(
                            "OMEGA", detailed_ret, detailed_output, "OMEGA query stats:"
                        )
                        detailed_profile = build_omega_profile(
                            metrics, detailed_output, baseline_profile
                        )
                        omega_profile = merge_omega_detailed_profile(
                            omega_profile, detailed_profile
                        )
                latency_summary = latency_summary_from_profile(omega_profile)
                results.append(
                    BenchmarkResult(
                        type="OMEGA",
                        path=str(omega_path),
                        success=ret == 0,
                        target_recall=target_recall,
                        load_duration=load_duration if load_duration is not None else metrics.get("load_duration"),
                        qps=metrics.get("qps"),
                        avg_latency_ms=latency_summary["avg_latency_ms"],
                        p50_latency_ms=latency_summary["p50_latency_ms"],
                        p90_latency_ms=latency_summary["p90_latency_ms"],
                        p95_latency_ms=latency_summary["p95_latency_ms"],
                        p99_latency_ms=latency_summary["p99_latency_ms"],
                        recall=metrics.get("recall"),
                        profiling=omega_profile,
                    )
                )

    if results:
        written_summary_paths = write_grouped_profiling_summaries(dataset_name, results)
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
            ld = f"{result.load_duration:.1f}" if result.load_duration else "N/A"
            qps = f"{result.qps:.1f}" if result.qps else "N/A"
            avg_latency = f"{result.avg_latency_ms:.3f}" if result.avg_latency_ms is not None else "N/A"
            p95_latency = f"{result.p95_latency_ms:.3f}" if result.p95_latency_ms is not None else "N/A"
            recall = f"{result.recall:.4f}" if result.recall else "N/A"
            print(
                f"{result.type:<10} {tr:<15} {ld:<12} {qps:<8} "
                f"{avg_latency:<16} {p95_latency:<16} {recall:<10} {status:<10}"
            )

        print("\nProfiling Summary")
        print("-" * 75)
        print(f"{'Type':<10} {'target_recall':<15} {'avg_lat(ms)':<12} {'avg_cmps':<12} {'avg_pred_calls':<16} {'avg_model_ms':<14} {'saved_cmps':<12}")
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
            print(f"Profiling JSON: {path}")

    print("\nTo view results:")
    print("  vectordbbench results")
    print("\nOr start the web UI:")
    print("  vectordbbench start")
    print()

    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
