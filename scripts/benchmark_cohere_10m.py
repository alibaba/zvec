#!/usr/bin/env python3
"""
VectorDBBench: Zvec vs Zvec+OMEGA Comparison on Cohere-10M

Based on official zvec.org Cohere-10M benchmark parameters.

Usage:
    python benchmark_cohere_10m.py [--dry-run] [--target-recalls 0.90,0.95]
"""

import argparse
import json
import subprocess
import sys
import os
import importlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkResult:
    type: str
    ef_search: int
    target_recall: float | None
    path: str
    success: bool
    load_duration: float | None = None
    qps: float | None = None
    recall: float | None = None
    profiling: dict | None = None


def resolve_paths(
    zvec_root_arg: str | None,
    vectordbbench_root_arg: str | None,
    benchmark_dir_arg: str | None,
    results_dir_arg: str | None,
) -> tuple[Path, Path, Path, Path]:
    script_path = Path(__file__).resolve()
    zvec_root = Path(zvec_root_arg).resolve() if zvec_root_arg else script_path.parent.parent
    vectordbbench_root = (
        Path(vectordbbench_root_arg).resolve()
        if vectordbbench_root_arg
        else Path(os.environ.get("VECTORDBBENCH_ROOT", zvec_root.parent / "VectorDBBench")).resolve()
    )
    benchmark_dir = (
        Path(benchmark_dir_arg).resolve()
        if benchmark_dir_arg
        else Path(os.environ.get("ZVEC_BENCHMARK_DIR", zvec_root / "benchmark_results")).resolve()
    )
    if results_dir_arg:
        results_dir = Path(results_dir_arg).resolve()
    else:
        results_dir = None
        try:
            config = importlib.import_module("vectordb_bench").config
            results_dir = Path(config.RESULTS_LOCAL_DIR).resolve() / "Zvec"
        except Exception:
            results_dir = vectordbbench_root / "vectordb_bench" / "results" / "Zvec"
    return zvec_root, vectordbbench_root, benchmark_dir, results_dir


def resolve_vectordbbench_command() -> list[str]:
    return [sys.executable, "-m", "vectordb_bench"]


KV_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s,]+)")


def parse_scalar(value: str):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict:
    return {key: parse_scalar(value) for key, value in KV_PATTERN.findall(line)}


def avg_metric(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return None
    return sum(values) / len(values)


def parse_serial_runner_summary(output: str) -> dict:
    summary = {}
    for line in output.splitlines():
        if "search entire test_data:" not in line:
            continue
        summary = parse_key_values(line)
    return summary


def parse_query_records(output: str, prefix: str) -> list[dict]:
    records = []
    for line in output.splitlines():
        if prefix not in line:
            continue
        records.append(parse_key_values(line))
    return records


def build_hnsw_profile(metrics: dict, output: str) -> dict:
    query_records = parse_query_records(output, "HNSW query stats:")
    serial_summary = parse_serial_runner_summary(output)
    return {
        "query_count": len(query_records),
        "recall": metrics.get("recall"),
        "qps": metrics.get("qps"),
        "avg_end2end_latency_ms": avg_metric(query_records, "latency_ms"),
        "avg_cmps": avg_metric(query_records, "pairwise_dist_cnt"),
        "avg_scan_cmps": avg_metric(query_records, "cmps"),
        "serial_avg_latency_s": serial_summary.get("avg_latency"),
        "serial_p99_s": serial_summary.get("p99"),
        "serial_p95_s": serial_summary.get("p95"),
        "serial_avg_recall": serial_summary.get("avg_recall"),
    }


def build_omega_profile(metrics: dict, output: str, hnsw_profile: dict | None) -> dict:
    query_records = parse_query_records(output, "OMEGA query stats:")
    serial_summary = parse_serial_runner_summary(output)

    avg_pairwise_dist_cnt = avg_metric(query_records, "pairwise_dist_cnt")
    avg_pure_search_ms = avg_metric(query_records, "pure_search_ms")
    avg_omega_control_ms = avg_metric(query_records, "omega_control_ms")

    cmp_time_ms = None
    if avg_pairwise_dist_cnt and avg_pairwise_dist_cnt > 0 and avg_pure_search_ms is not None:
        cmp_time_ms = avg_pure_search_ms / avg_pairwise_dist_cnt

    model_overhead_cmp_equiv = None
    if cmp_time_ms and cmp_time_ms > 0 and avg_omega_control_ms is not None:
        model_overhead_cmp_equiv = avg_omega_control_ms / cmp_time_ms

    avg_saved_cmps = None
    if hnsw_profile and hnsw_profile.get("avg_cmps") is not None and avg_pairwise_dist_cnt is not None:
        avg_saved_cmps = hnsw_profile["avg_cmps"] - avg_pairwise_dist_cnt

    return {
        "query_count": len(query_records),
        "recall": metrics.get("recall"),
        "qps": metrics.get("qps"),
        "avg_end2end_latency_ms": avg_metric(query_records, "total_ms"),
        "avg_cmps": avg_pairwise_dist_cnt,
        "avg_scan_cmps": avg_metric(query_records, "scan_cmps"),
        "avg_omega_cmps": avg_metric(query_records, "omega_cmps"),
        "avg_prediction_calls": avg_metric(query_records, "prediction_calls"),
        "avg_should_stop_calls": avg_metric(query_records, "should_stop_calls"),
        "avg_advance_calls": avg_metric(query_records, "advance_calls"),
        "avg_model_overhead_ms": avg_omega_control_ms,
        "avg_should_stop_ms": avg_metric(query_records, "should_stop_ms"),
        "avg_prediction_eval_ms": avg_metric(query_records, "prediction_eval_ms"),
        "avg_feature_prep_ms": avg_metric(query_records, "feature_prep_ms"),
        "avg_pure_search_ms": avg_pure_search_ms,
        "avg_model_overhead_cmp_equiv": model_overhead_cmp_equiv,
        "avg_early_stop_saved_cmps": avg_saved_cmps,
        "avg_early_stop_hit_rate": avg_metric(query_records, "early_stop_hit"),
        "serial_avg_latency_s": serial_summary.get("avg_latency"),
        "serial_p99_s": serial_summary.get("p99"),
        "serial_p95_s": serial_summary.get("p95"),
        "serial_avg_recall": serial_summary.get("avg_recall"),
    }


def profiling_output_path(benchmark_dir: Path) -> Path:
    return benchmark_dir / "cohere_10m_profiling_summary.json"


def write_profiling_summary(benchmark_dir: Path, payload: dict) -> None:
    with open(profiling_output_path(benchmark_dir), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def get_latest_result(db_label: str, results_dir: Path) -> dict:
    if not results_dir.exists():
        return {}

    result_files = sorted(
        results_dir.glob("result_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
            for result in data.get("results", []):
                task_config = result.get("task_config", {})
                db_config = task_config.get("db_config", {})
                if db_config.get("db_label") == db_label:
                    metrics = result.get("metrics", {})
                    return {
                        "insert_duration": metrics.get("insert_duration"),
                        "optimize_duration": metrics.get("optimize_duration"),
                        "load_duration": metrics.get("load_duration"),
                        "qps": metrics.get("qps"),
                        "recall": metrics.get("recall"),
                    }
        except Exception:
            continue

    return {}


def snapshot_result_files(results_dir: Path) -> set[str]:
    if not results_dir.exists():
        return set()
    return {str(p) for p in results_dir.glob("result_*.json")}


def extract_result_from_file(result_file: Path, db_label: str) -> dict:
    try:
        with open(result_file) as f:
            data = json.load(f)
        for result in data.get("results", []):
            task_config = result.get("task_config", {})
            db_config = task_config.get("db_config", {})
            if db_config.get("db_label") == db_label:
                metrics = result.get("metrics", {})
                return {
                    "insert_duration": metrics.get("insert_duration"),
                    "optimize_duration": metrics.get("optimize_duration"),
                    "load_duration": metrics.get("load_duration"),
                    "qps": metrics.get("qps"),
                    "recall": metrics.get("recall"),
                }
    except Exception:
        return {}
    return {}


def get_run_result(db_label: str, before_files: set[str], results_dir: Path) -> dict:
    if not results_dir.exists():
        return {}

    current_files = {str(p) for p in results_dir.glob("result_*.json")}
    new_files = sorted(
        [Path(p) for p in current_files - before_files],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for result_file in new_files:
        metrics = extract_result_from_file(result_file, db_label)
        if metrics:
            return metrics

    return get_latest_result(db_label, results_dir)


def offline_summary_path(index_path: Path) -> Path:
    return index_path / "offline_benchmark_summary.json"


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def find_omega_model_dir(index_path: Path) -> Path | None:
    candidates = sorted(index_path.glob("*/omega_model"))
    return candidates[0] if candidates else None


def sum_timing_ms(data: dict) -> int:
    return sum(v for v in data.values() if isinstance(v, (int, float)))


def build_offline_summary(
    index_path: Path,
    db_label: str,
    metrics: dict,
    retrain_only: bool = False,
) -> dict:
    previous_summary = read_json_if_exists(offline_summary_path(index_path)) if retrain_only else {}
    previous_offline = previous_summary.get("offline", {})
    previous_omega_training = previous_summary.get("omega_training", {})

    insert_duration = metrics.get("insert_duration")
    optimize_duration = metrics.get("optimize_duration")
    load_duration = metrics.get("load_duration")

    omega_model_dir = find_omega_model_dir(index_path)
    omega_training = {}
    if omega_model_dir is not None:
        omega_training = {
            "collection_timing_ms": read_json_if_exists(
                omega_model_dir / "training_collection_timing.json"
            ),
            "lightgbm_timing_ms": read_json_if_exists(
                omega_model_dir / "lightgbm_training_timing.json"
            ),
        }

    if retrain_only:
        insert_duration = previous_offline.get("insert_duration_s")
        old_optimize_duration = previous_offline.get("optimize_duration_s")
        old_training_s = (
            sum_timing_ms(previous_omega_training.get("collection_timing_ms", {}))
            + sum_timing_ms(previous_omega_training.get("lightgbm_timing_ms", {}))
        ) / 1000.0
        new_training_s = (
            sum_timing_ms(omega_training.get("collection_timing_ms", {}))
            + sum_timing_ms(omega_training.get("lightgbm_timing_ms", {}))
        ) / 1000.0
        if old_optimize_duration is not None:
            optimize_duration = round(old_optimize_duration - old_training_s + new_training_s, 4)
        else:
            optimize_duration = metrics.get("optimize_duration")
        load_duration = (
            round(insert_duration + optimize_duration, 4)
            if insert_duration is not None and optimize_duration is not None
            else metrics.get("load_duration")
        )

    summary = {
        "db_label": db_label,
        "index_path": str(index_path),
        "generated_at": datetime.now().isoformat(),
        "offline": {
            "insert_duration_s": insert_duration,
            "optimize_duration_s": optimize_duration,
            "load_duration_s": load_duration,
        },
    }

    if omega_training:
        summary["omega_training"] = omega_training

    return summary


def write_offline_summary(
    index_path: Path,
    db_label: str,
    metrics: dict,
    retrain_only: bool = False,
) -> None:
    summary = build_offline_summary(index_path, db_label, metrics, retrain_only=retrain_only)
    with open(offline_summary_path(index_path), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def get_offline_load_duration(index_path: Path) -> float | None:
    summary = read_json_if_exists(offline_summary_path(index_path))
    return summary.get("offline", {}).get("load_duration_s")


def run_command(
    cmd: list[str],
    vectordbbench_root: Path,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'=' * 60}")
    print(f"Command:\n{cmd_str}")
    print(f"{'=' * 60}\n")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return 0, ""

    cwd = vectordbbench_root if vectordbbench_root.exists() else None
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    return process.wait(), "".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zvec HNSW vs OMEGA on Cohere-10M dataset"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--target-recalls",
        type=str,
        default="0.95",
        help="Comma-separated target recalls for OMEGA (default: 0.95)",
    )
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
        help="Path to the zvec repository root (default: auto-detect from this script)",
    )
    parser.add_argument(
        "--vectordbbench-root",
        type=str,
        default=None,
        help="Path to the VectorDBBench repository root "
             "(default: $VECTORDBBENCH_ROOT or sibling repo next to zvec)",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Directory used to store built benchmark artifacts "
             "(default: $ZVEC_BENCHMARK_DIR or <zvec_root>/benchmark_results)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing VectorDBBench JSON result files "
             "(default: runtime vectordb_bench.config.RESULTS_LOCAL_DIR/Zvec)",
    )
    args = parser.parse_args()

    zvec_root, vectordbbench_root, benchmark_dir, results_dir = resolve_paths(
        args.zvec_root, args.vectordbbench_root, args.benchmark_dir, args.results_dir
    )
    vectordbbench_cmd = resolve_vectordbbench_command()
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    CASE_TYPE = "Performance768D10M"
    M = 50
    EF_SEARCH = 118
    QUANTIZE_TYPE = "int8"
    USE_REFINER = True
    NUM_CONCURRENCY = "12,14,16,18,20"
    CONCURRENCY_DURATION = 30
    K = 100

    MIN_VECTOR_THRESHOLD = 100000
    NUM_TRAINING_QUERIES = 4000
    EF_TRAINING = 300
    WINDOW_SIZE = 100
    EF_GROUNDTRUTH = 500

    target_recalls = [float(x) for x in args.target_recalls.split(",")]

    hnsw_path = benchmark_dir / "cohere_10m_hnsw"
    omega_path = benchmark_dir / "cohere_10m_omega"

    print("=" * 70)
    print("VectorDBBench: Zvec HNSW vs OMEGA (Cohere-10M)")
    print("Based on official zvec.org benchmark parameters")
    print("=" * 70)
    print()
    print("Official HNSW Parameters:")
    print(f"  M: {M}")
    print(f"  ef_search: {EF_SEARCH}")
    print(f"  quantize_type: {QUANTIZE_TYPE}")
    print(f"  is_using_refiner: {USE_REFINER}")
    print(f"  num_concurrency: {NUM_CONCURRENCY}")
    print()
    print("OMEGA Parameters:")
    print(f"  min_vector_threshold: {MIN_VECTOR_THRESHOLD}")
    print(f"  num_training_queries: {NUM_TRAINING_QUERIES}")
    print(f"  ef_training: {EF_TRAINING}")
    print(f"  window_size: {WINDOW_SIZE}")
    print(f"  ef_groundtruth: {EF_GROUNDTRUTH}")
    print(f"  target_recalls: {target_recalls}")
    print(f"  build_mode: {'retrain model only (reuse existing index)' if args.retrain_only else 'build index + train model'}")
    print(f"zvec_root: {zvec_root}")
    print(f"vectordbbench_root: {vectordbbench_root}")
    print(f"vectordbbench_cmd: {' '.join(vectordbbench_cmd)}")
    print(f"benchmark_dir: {benchmark_dir}")
    print(f"results_dir: {results_dir}")
    print("=" * 70)

    results: list[BenchmarkResult] = []

    if not args.skip_hnsw:
        print(f"\n\n{'#' * 70}")
        print("# HNSW Benchmark")
        print(f"{'#' * 70}")

        hnsw_db_label = "16c64g-v0.1"

        common_hnsw_args = [
            *vectordbbench_cmd,
            "zvec",
            "--path",
            str(hnsw_path),
            "--db-label",
            hnsw_db_label,
            "--case-type",
            CASE_TYPE,
            "--num-concurrency",
            NUM_CONCURRENCY,
            "--quantize-type",
            QUANTIZE_TYPE,
            "--m",
            str(M),
            "--ef-search",
            str(EF_SEARCH),
            "--k",
            str(K),
            "--concurrency-duration",
            str(CONCURRENCY_DURATION),
        ]
        if USE_REFINER:
            common_hnsw_args.append("--is-using-refiner")

        if not args.search_only:
            print("\n[Phase 1] Building HNSW index...")
            before_files = snapshot_result_files(results_dir)
            cmd = common_hnsw_args + [
                "--skip-search-serial",
                "--skip-search-concurrent",
            ]
            ret, _ = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
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
            print("\n[Phase 2] Running HNSW search benchmark...")
            before_files = snapshot_result_files(results_dir)
            cmd = common_hnsw_args + [
                "--skip-drop-old",
                "--skip-load",
            ]
            ret, _ = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
            metrics = get_run_result(hnsw_db_label, before_files, results_dir) if not args.dry_run else {}
            load_duration = get_offline_load_duration(hnsw_path)
            hnsw_profile = None
            if ret == 0 and not args.dry_run:
                print("\n[Profiling] Running HNSW serial-only profiling pass...")
                profile_cmd = common_hnsw_args + [
                    "--skip-drop-old",
                    "--skip-load",
                    "--skip-search-concurrent",
                ]
                _, profile_output = run_command(
                    profile_cmd,
                    vectordbbench_root,
                    dry_run=False,
                    extra_env={
                        "ZVEC_HNSW_LOG_QUERY_STATS": "1",
                        "ZVEC_HNSW_LOG_QUERY_LIMIT": "2000",
                    },
                )
                hnsw_profile = build_hnsw_profile(metrics, profile_output)
            results.append(
                BenchmarkResult(
                    type="HNSW",
                    ef_search=EF_SEARCH,
                    target_recall=None,
                    path=str(hnsw_path),
                    success=ret == 0,
                    load_duration=load_duration if load_duration is not None else metrics.get("load_duration"),
                    qps=metrics.get("qps"),
                    recall=metrics.get("recall"),
                    profiling=hnsw_profile,
                )
            )

    if not args.skip_omega:
        omega_db_label = f"omega-m{M}-ef{EF_SEARCH}-refiner-int8"
        build_target_recall = target_recalls[0]

        common_omega_args = [
            *vectordbbench_cmd,
            "zvecomega",
            "--path",
            str(omega_path),
            "--db-label",
            omega_db_label,
            "--case-type",
            CASE_TYPE,
            "--num-concurrency",
            NUM_CONCURRENCY,
            "--quantize-type",
            QUANTIZE_TYPE,
            "--m",
            str(M),
            "--ef-search",
            str(EF_SEARCH),
            "--k",
            str(K),
            "--concurrency-duration",
            str(CONCURRENCY_DURATION),
            "--min-vector-threshold",
            str(MIN_VECTOR_THRESHOLD),
            "--num-training-queries",
            str(NUM_TRAINING_QUERIES),
            "--ef-training",
            str(EF_TRAINING),
            "--window-size",
            str(WINDOW_SIZE),
            "--ef-groundtruth",
            str(EF_GROUNDTRUTH),
        ]
        if USE_REFINER:
            common_omega_args.append("--is-using-refiner")

        if not args.search_only:
            print(f"\n\n{'#' * 70}")
            print("# OMEGA Offline Phase")
            print(f"{'#' * 70}")
            if args.retrain_only:
                print("\n[Phase 1] Retraining OMEGA model only (reusing existing index)...")
                print(
                    f"Reusing existing OMEGA path/db_label: "
                    f"path={omega_path}, db_label={omega_db_label}"
                )
            else:
                print("\n[Phase 1] Building OMEGA index + training model...")
                print(
                    f"Using shared OMEGA path/db_label for all target recalls: "
                    f"path={omega_path}, db_label={omega_db_label}"
                )
            print(
                "Build-time target_recall is ignored by training; "
                f"using first requested value for CLI compatibility: {build_target_recall}"
            )
            before_files = snapshot_result_files(results_dir)
            cmd = common_omega_args + [
                "--target-recall",
                str(build_target_recall),
                "--skip-search-serial",
                "--skip-search-concurrent",
            ]
            if args.retrain_only:
                cmd += [
                    "--skip-drop-old",
                    "--skip-load",
                    "--retrain-only",
                ]
            ret, _ = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
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
                print(f"\n\n{'#' * 70}")
                print(f"# OMEGA Benchmark (target_recall={target_recall})")
                print(f"{'#' * 70}")
                print("\n[Phase 2] Running OMEGA search benchmark...")
                if args.retrain_only:
                    print("Search is using the newly retrained model on the existing index.")
                before_files = snapshot_result_files(results_dir)
                cmd = common_omega_args + [
                    "--target-recall",
                    str(target_recall),
                    "--skip-drop-old",
                    "--skip-load",
                ]
                if args.retrain_only:
                    cmd.append("--retrain-only")
                ret, _ = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)
                metrics = get_run_result(omega_db_label, before_files, results_dir) if not args.dry_run else {}
                load_duration = get_offline_load_duration(omega_path)
                omega_profile = None
                if ret == 0 and not args.dry_run:
                    print("\n[Profiling] Running OMEGA serial-only profiling pass...")
                    profile_cmd = common_omega_args + [
                        "--target-recall",
                        str(target_recall),
                        "--skip-drop-old",
                        "--skip-load",
                        "--skip-search-concurrent",
                    ]
                    if args.retrain_only:
                        profile_cmd.append("--retrain-only")
                    _, profile_output = run_command(
                        profile_cmd,
                        vectordbbench_root,
                        dry_run=False,
                        extra_env={
                            "ZVEC_OMEGA_LOG_QUERY_STATS": "1",
                            "ZVEC_OMEGA_LOG_QUERY_LIMIT": "2000",
                        },
                    )
                    baseline_profile = next(
                        (result.profiling for result in results if result.type == "HNSW" and result.profiling),
                        None,
                    )
                    omega_profile = build_omega_profile(metrics, profile_output, baseline_profile)
                results.append(
                    BenchmarkResult(
                        type="OMEGA",
                        ef_search=EF_SEARCH,
                        target_recall=target_recall,
                        path=str(omega_path),
                        success=ret == 0,
                        load_duration=load_duration if load_duration is not None else metrics.get("load_duration"),
                        qps=metrics.get("qps"),
                        recall=metrics.get("recall"),
                        profiling=omega_profile,
                    )
                )

    if results:
        write_profiling_summary(
            benchmark_dir,
            {
                "generated_at": datetime.now().isoformat(),
                "dataset": "cohere_10m",
                "results": [
                    {
                        "type": result.type,
                        "target_recall": result.target_recall,
                        "path": result.path,
                        "load_duration_s": result.load_duration,
                        "qps": result.qps,
                        "recall": result.recall,
                        "profiling": result.profiling,
                    }
                    for result in results
                ],
            },
        )
        print("\n\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print()
        print(f"{'Type':<10} {'target_recall':<15} {'load_dur(s)':<12} {'qps':<12} {'recall':<10} {'Status':<10}")
        print("-" * 75)
        for r in results:
            tr = f"{r.target_recall:.2f}" if r.target_recall else "N/A"
            status = "OK" if r.success else "FAILED"
            ld = f"{r.load_duration:.1f}" if r.load_duration else "N/A"
            qps = f"{r.qps:.1f}" if r.qps else "N/A"
            recall = f"{r.recall:.4f}" if r.recall else "N/A"
            print(f"{r.type:<10} {tr:<15} {ld:<12} {qps:<12} {recall:<10} {status:<10}")

        print()
        print("Profiling Summary")
        print("-" * 75)
        print(f"{'Type':<10} {'target_recall':<15} {'avg_lat(ms)':<12} {'avg_cmps':<12} {'avg_pred_calls':<16} {'avg_model_ms':<14} {'saved_cmps':<12}")
        for r in results:
            profile = r.profiling or {}
            tr = f"{r.target_recall:.2f}" if r.target_recall else "N/A"
            avg_lat = profile.get("avg_end2end_latency_ms")
            avg_cmps = profile.get("avg_cmps")
            avg_pred_calls = profile.get("avg_prediction_calls")
            avg_model_ms = profile.get("avg_model_overhead_ms")
            saved_cmps = profile.get("avg_early_stop_saved_cmps")
            print(
                f"{r.type:<10} "
                f"{tr:<15} "
                f"{(f'{avg_lat:.3f}' if avg_lat is not None else 'N/A'):<12} "
                f"{(f'{avg_cmps:.1f}' if avg_cmps is not None else 'N/A'):<12} "
                f"{(f'{avg_pred_calls:.2f}' if avg_pred_calls is not None else 'N/A'):<16} "
                f"{(f'{avg_model_ms:.3f}' if avg_model_ms is not None else 'N/A'):<14} "
                f"{(f'{saved_cmps:.1f}' if saved_cmps is not None else 'N/A'):<12}"
            )
        print()
        print(f"Profiling JSON: {profiling_output_path(benchmark_dir)}")

    print()
    print("To view results:")
    print("  vectordbbench results")
    print()
    print("Or start the web UI:")
    print("  vectordbbench start")
    print()

    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
