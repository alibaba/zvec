#!/usr/bin/env python3
"""
VectorDBBench: Zvec vs Zvec+OMEGA Comparison on Cohere-1M

Based on official zvec.org benchmark parameters.

Usage:
    python benchmark_cohere_1m.py [--dry-run] [--target-recalls 0.90,0.95,0.98]
"""

import argparse
import json
import subprocess
import sys
import os
import importlib
import re
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Parsed benchmark result from VectorDBBench output."""
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
    source_results_dir = vectordbbench_root / "vectordb_bench" / "results" / "Zvec"

    if results_dir_arg:
        results_dir = Path(results_dir_arg).resolve()
    elif source_results_dir.exists():
        results_dir = source_results_dir
    else:
        results_dir = None
        try:
            config = importlib.import_module("vectordb_bench").config
            results_dir = Path(config.RESULTS_LOCAL_DIR).resolve() / "Zvec"
        except Exception:
            results_dir = vectordbbench_root / "vectordb_bench" / "results" / "Zvec"
    return zvec_root, vectordbbench_root, benchmark_dir, results_dir


def resolve_vectordbbench_command() -> list[str]:
    return [sys.executable, "-m", "vectordb_bench.cli.vectordbbench"]


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
        "benchmark_recall": metrics.get("recall"),
        "benchmark_qps": metrics.get("qps"),
        "profile_query_count": len(query_records),
        "profile_avg_end2end_latency_ms": avg_metric(query_records, "latency_ms"),
        "profile_avg_cmps": avg_metric(query_records, "pairwise_dist_cnt"),
        "profile_avg_scan_cmps": avg_metric(query_records, "cmps"),
        "profile_avg_pure_search_ms": avg_metric(query_records, "pure_search_ms"),
        "profile_serial_avg_latency_s": serial_summary.get("avg_latency"),
        "profile_serial_p99_s": serial_summary.get("p99"),
        "profile_serial_p95_s": serial_summary.get("p95"),
        "profile_serial_avg_recall": serial_summary.get("avg_recall"),
    }


def build_omega_profile(metrics: dict, output: str, hnsw_profile: dict | None) -> dict:
    query_records = parse_query_records(output, "OMEGA query stats:")
    serial_summary = parse_serial_runner_summary(output)

    avg_pairwise_dist_cnt = avg_metric(query_records, "pairwise_dist_cnt")
    avg_core_search_ms = avg_metric(query_records, "core_search_ms")
    avg_pure_search_ms = avg_metric(query_records, "pure_search_ms")
    avg_omega_control_ms = avg_metric(query_records, "omega_control_ms")
    avg_search_only_ms = (
        avg_pure_search_ms if avg_pure_search_ms is not None else avg_core_search_ms
    )

    cmp_time_ms = None
    if avg_pairwise_dist_cnt and avg_pairwise_dist_cnt > 0 and avg_search_only_ms is not None:
        cmp_time_ms = avg_search_only_ms / avg_pairwise_dist_cnt

    model_overhead_cmp_equiv = None
    if cmp_time_ms and cmp_time_ms > 0 and avg_omega_control_ms is not None:
        model_overhead_cmp_equiv = avg_omega_control_ms / cmp_time_ms

    avg_saved_cmps = None
    if hnsw_profile and hnsw_profile.get("profile_avg_cmps") is not None and avg_pairwise_dist_cnt is not None:
        avg_saved_cmps = hnsw_profile["profile_avg_cmps"] - avg_pairwise_dist_cnt

    return {
        "benchmark_recall": metrics.get("recall"),
        "benchmark_qps": metrics.get("qps"),
        "profile_query_count": len(query_records),
        "profile_avg_end2end_latency_ms": avg_metric(query_records, "total_ms"),
        "profile_avg_cmps": avg_pairwise_dist_cnt,
        "profile_avg_scan_cmps": avg_metric(query_records, "scan_cmps"),
        "profile_avg_omega_cmps": avg_metric(query_records, "omega_cmps"),
        "profile_avg_prediction_calls": avg_metric(query_records, "prediction_calls"),
        "profile_avg_should_stop_calls": avg_metric(query_records, "should_stop_calls"),
        "profile_avg_advance_calls": avg_metric(query_records, "advance_calls"),
        "profile_avg_model_overhead_ms": avg_omega_control_ms,
        "profile_avg_setup_ms": avg_metric(query_records, "setup_ms"),
        "profile_avg_should_stop_ms": avg_metric(query_records, "should_stop_ms"),
        "profile_avg_prediction_eval_ms": avg_metric(query_records, "prediction_eval_ms"),
        "profile_avg_core_search_ms": avg_core_search_ms,
        "profile_avg_pure_search_ms": avg_pure_search_ms,
        "profile_avg_model_overhead_cmp_equiv": model_overhead_cmp_equiv,
        "profile_avg_early_stop_saved_cmps": avg_saved_cmps,
        "profile_avg_early_stop_hit_rate": avg_metric(query_records, "early_stop_hit"),
        "profile_serial_avg_latency_s": serial_summary.get("avg_latency"),
        "profile_serial_p99_s": serial_summary.get("p99"),
        "profile_serial_p95_s": serial_summary.get("p95"),
        "profile_serial_avg_recall": serial_summary.get("avg_recall"),
    }


def profiling_output_path(index_path: Path) -> Path:
    return index_path / "online_benchmark_summary.json"


def write_profiling_summary(index_path: Path, payload: dict) -> None:
    with open(profiling_output_path(index_path), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_grouped_profiling_summaries(dataset: str, results: list[BenchmarkResult]) -> list[Path]:
    written_paths: list[Path] = []
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.path, []).append(result)

    for path_str, grouped_results in grouped.items():
        index_path = Path(path_str)
        write_profiling_summary(
            index_path,
            {
                "generated_at": datetime.now().isoformat(),
                "dataset": dataset,
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
                    for result in grouped_results
                ],
            },
        )
        written_paths.append(profiling_output_path(index_path))

    return written_paths


def get_latest_result(db_label: str, results_dir: Path) -> dict:
    """Get the latest benchmark result for a given db_label from VectorDBBench."""
    if not results_dir.exists():
        return {}

    # Find all result files, sorted by modification time (newest first)
    result_files = sorted(
        results_dir.glob("result_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )

    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Check each result in this file
            for result in data.get("results", []):
                task_config = result.get("task_config", {})
                db_config = task_config.get("db_config", {})
                if db_config.get("db_label") == db_label:
                    metrics = result.get("metrics", {})
                    return {
                        'insert_duration': metrics.get('insert_duration'),
                        'optimize_duration': metrics.get('optimize_duration'),
                        'load_duration': metrics.get('load_duration'),
                        'qps': metrics.get('qps'),
                        'recall': metrics.get('recall'),
                    }
        except Exception:
            # Skip files that can't be parsed
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
) -> int:
    """Run a command and return the exit code."""
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'='*60}")
    print(f"Command:\n{cmd_str}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return 0

    cwd = vectordbbench_root if vectordbbench_root.exists() else None
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def run_command_capture(
    cmd: list[str],
    vectordbbench_root: Path,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'='*60}")
    print(f"Command:\n{cmd_str}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY RUN] Command not executed")
        return 0, ""

    cwd = vectordbbench_root if vectordbbench_root.exists() else None
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log") as tmp:
        tmp_path = Path(tmp.name)

    try:
        with tmp_path.open("w+") as tmp:
            result = subprocess.run(cmd, cwd=cwd, env=env, stdout=tmp, stderr=subprocess.STDOUT, text=True)
            tmp.flush()
            tmp.seek(0)
            output = tmp.read()
        print(output, end="" if output.endswith("\n") or not output else "\n")
        return result.returncode, output
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Zvec HNSW vs OMEGA on Cohere-1M dataset"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--target-recalls", type=str, default="0.95",
                        help="Comma-separated target recalls for OMEGA (default: 0.95)")
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

    # Configuration - based on official zvec.org parameters
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Official parameters from zvec.org for Cohere-1M
    CASE_TYPE = "Performance768D1M"
    M = 15
    EF_SEARCH = 180
    QUANTIZE_TYPE = "int8"
    NUM_CONCURRENCY = "12,14,16,18,20"
    CONCURRENCY_DURATION = 30
    K = 100

    # OMEGA parameters
    MIN_VECTOR_THRESHOLD = 100000
    NUM_TRAINING_QUERIES = 4000
    EF_TRAINING = 500
    WINDOW_SIZE = 100
    EF_GROUNDTRUTH = 1000

    # Parse target recalls
    target_recalls = [float(x) for x in args.target_recalls.split(",")]

    # Paths
    hnsw_path = benchmark_dir / "cohere_1m_hnsw"
    omega_path = benchmark_dir / "cohere_1m_omega"

    print("=" * 70)
    print("VectorDBBench: Zvec HNSW vs OMEGA (Cohere-1M)")
    print("Based on official zvec.org benchmark parameters")
    print("=" * 70)
    print()
    print("Official HNSW Parameters:")
    print(f"  M: {M}")
    print(f"  ef_search: {EF_SEARCH}")
    print(f"  quantize_type: {QUANTIZE_TYPE}")
    print()
    print("OMEGA Parameters:")
    print(f"  min_vector_threshold: {MIN_VECTOR_THRESHOLD}")
    print(f"  num_training_queries: {NUM_TRAINING_QUERIES}")
    print(f"  ef_training: {EF_TRAINING}")
    print(f"  window_size: {WINDOW_SIZE}")
    print(f"  ef_groundtruth: {EF_GROUNDTRUTH} (HNSW-based ground truth)")
    print(f"  target_recalls: {target_recalls}")
    print(f"  build_mode: {'retrain model only (reuse existing index)' if args.retrain_only else 'build index + train model'}")
    print()
    print(f"Concurrency: {NUM_CONCURRENCY}")
    print(f"zvec_root: {zvec_root}")
    print(f"vectordbbench_root: {vectordbbench_root}")
    print(f"vectordbbench_cmd: {' '.join(vectordbbench_cmd)}")
    print(f"benchmark_dir: {benchmark_dir}")
    print(f"results_dir: {results_dir}")
    print("=" * 70)

    results: list[BenchmarkResult] = []

    # ============ HNSW Benchmark ============
    if not args.skip_hnsw:
        print(f"\n\n{'#'*70}")
        print(f"# HNSW Benchmark")
        print(f"{'#'*70}")

        hnsw_db_label = "16c64g-v0.1"

        if not args.search_only:
            # Phase 1: Build Index
            print("\n[Phase 1] Building HNSW index...")
            before_files = snapshot_result_files(results_dir)
            cmd = [
                *vectordbbench_cmd, "zvec",
                "--path", str(hnsw_path),
                "--db-label", hnsw_db_label,
                "--case-type", CASE_TYPE,
                "--m", str(M),
                "--ef-search", str(EF_SEARCH),
                "--quantize-type", QUANTIZE_TYPE,
                "--num-concurrency", NUM_CONCURRENCY,
                "--concurrency-duration", str(CONCURRENCY_DURATION),
                "--k", str(K),
                "--skip-search-serial",
                "--skip-search-concurrent",
            ]
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
            # Phase 2: Run Search Benchmark
            print("\n[Phase 2] Running HNSW search benchmark...")
            before_files = snapshot_result_files(results_dir)
            cmd = [
                *vectordbbench_cmd, "zvec",
                "--path", str(hnsw_path),
                "--db-label", hnsw_db_label,
                "--case-type", CASE_TYPE,
                "--m", str(M),
                "--ef-search", str(EF_SEARCH),
                "--quantize-type", QUANTIZE_TYPE,
                "--num-concurrency", NUM_CONCURRENCY,
                "--concurrency-duration", str(CONCURRENCY_DURATION),
                "--k", str(K),
                "--skip-drop-old",
                "--skip-load",
            ]
            ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)

            # Get results from VectorDBBench
            metrics = get_run_result(hnsw_db_label, before_files, results_dir) if not args.dry_run else {}
            load_duration = get_offline_load_duration(hnsw_path)
            hnsw_profile = None
            if ret == 0 and not args.dry_run:
                print("\n[Profiling] Running HNSW serial-only profiling pass...")
                profile_cmd = [
                    *vectordbbench_cmd, "zvec",
                    "--path", str(hnsw_path),
                    "--db-label", hnsw_db_label,
                    "--case-type", CASE_TYPE,
                    "--m", str(M),
                    "--ef-search", str(EF_SEARCH),
                    "--quantize-type", QUANTIZE_TYPE,
                    "--num-concurrency", NUM_CONCURRENCY,
                    "--concurrency-duration", str(CONCURRENCY_DURATION),
                    "--k", str(K),
                    "--skip-drop-old",
                    "--skip-load",
                    "--skip-search-concurrent",
                ]
                _, profile_output = run_command_capture(
                    profile_cmd,
                    vectordbbench_root,
                    dry_run=False,
                    extra_env={
                        "ZVEC_LOG_LEVEL": "INFO",
                        "ZVEC_HNSW_LOG_QUERY_STATS": "1",
                        "ZVEC_HNSW_LOG_QUERY_LIMIT": "2000",
                    },
                )
                hnsw_profile = build_hnsw_profile(metrics, profile_output)
            results.append(BenchmarkResult(
                type="HNSW",
                ef_search=EF_SEARCH,
                target_recall=None,
                path=str(hnsw_path),
                success=ret == 0,
                load_duration=load_duration if load_duration is not None else metrics.get('load_duration'),
                qps=metrics.get('qps'),
                recall=metrics.get('recall'),
                profiling=hnsw_profile,
            ))

    # ============ OMEGA Benchmarks ============
    if not args.skip_omega:
        omega_db_label = f"omega-m{M}-ef{EF_SEARCH}-int8"
        build_target_recall = target_recalls[0]

        if not args.search_only:
            print(f"\n\n{'#'*70}")
            print("# OMEGA Offline Phase")
            print(f"{'#'*70}")
            if args.retrain_only:
                print("\n[Phase 1] Retraining OMEGA model only (reusing existing index)...")
                print(
                    f"Reusing existing OMEGA path/db_label: path={omega_path}, db_label={omega_db_label}"
                )
            else:
                print("\n[Phase 1] Building OMEGA index + training model...")
                print(
                    f"Using shared OMEGA path/db_label for all target recalls: path={omega_path}, db_label={omega_db_label}"
                )
            print(
                f"Build-time target_recall is ignored by training; using first requested value for CLI compatibility: {build_target_recall}"
            )
            before_files = snapshot_result_files(results_dir)
            cmd = [
                *vectordbbench_cmd, "zvecomega",
                "--path", str(omega_path),
                "--db-label", omega_db_label,
                "--case-type", CASE_TYPE,
                "--m", str(M),
                "--ef-search", str(EF_SEARCH),
                "--quantize-type", QUANTIZE_TYPE,
                "--min-vector-threshold", str(MIN_VECTOR_THRESHOLD),
                "--num-training-queries", str(NUM_TRAINING_QUERIES),
                "--ef-training", str(EF_TRAINING),
                "--window-size", str(WINDOW_SIZE),
                "--ef-groundtruth", str(EF_GROUNDTRUTH),
                "--target-recall", str(build_target_recall),
                "--num-concurrency", NUM_CONCURRENCY,
                "--concurrency-duration", str(CONCURRENCY_DURATION),
                "--k", str(K),
                "--skip-search-serial",
                "--skip-search-concurrent",
            ]
            if args.retrain_only:
                cmd.extend([
                    "--skip-drop-old",
                    "--skip-load",
                    "--retrain-only",
                ])
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
                print(f"\n\n{'#'*70}")
                print(f"# OMEGA Benchmark (target_recall={target_recall})")
                print(f"{'#'*70}")

                # Phase 2: Run Search Benchmark
                print("\n[Phase 2] Running OMEGA search benchmark...")
                if args.retrain_only:
                    print("Search is using the newly retrained model on the existing index.")
                before_files = snapshot_result_files(results_dir)
                cmd = [
                    *vectordbbench_cmd, "zvecomega",
                    "--path", str(omega_path),
                    "--db-label", omega_db_label,
                    "--case-type", CASE_TYPE,
                    "--m", str(M),
                    "--ef-search", str(EF_SEARCH),
                    "--quantize-type", QUANTIZE_TYPE,
                    "--min-vector-threshold", str(MIN_VECTOR_THRESHOLD),
                    "--num-training-queries", str(NUM_TRAINING_QUERIES),
                    "--ef-training", str(EF_TRAINING),
                    "--window-size", str(WINDOW_SIZE),
                    "--ef-groundtruth", str(EF_GROUNDTRUTH),
                    "--target-recall", str(target_recall),
                    "--num-concurrency", NUM_CONCURRENCY,
                    "--concurrency-duration", str(CONCURRENCY_DURATION),
                    "--k", str(K),
                    "--skip-drop-old",
                    "--skip-load",
                ]
                if args.retrain_only:
                    cmd.append("--retrain-only")
                ret = run_command(cmd, vectordbbench_root, dry_run=args.dry_run)

                metrics = get_run_result(omega_db_label, before_files, results_dir) if not args.dry_run else {}
                load_duration = get_offline_load_duration(omega_path)
                omega_profile = None
                if ret == 0 and not args.dry_run:
                    print("\n[Profiling] Running OMEGA serial-only profiling pass...")
                    profile_cmd = [
                        *vectordbbench_cmd, "zvecomega",
                        "--path", str(omega_path),
                        "--db-label", omega_db_label,
                        "--case-type", CASE_TYPE,
                        "--m", str(M),
                        "--ef-search", str(EF_SEARCH),
                        "--quantize-type", QUANTIZE_TYPE,
                        "--min-vector-threshold", str(MIN_VECTOR_THRESHOLD),
                        "--num-training-queries", str(NUM_TRAINING_QUERIES),
                        "--ef-training", str(EF_TRAINING),
                        "--window-size", str(WINDOW_SIZE),
                        "--ef-groundtruth", str(EF_GROUNDTRUTH),
                        "--target-recall", str(target_recall),
                        "--num-concurrency", NUM_CONCURRENCY,
                        "--concurrency-duration", str(CONCURRENCY_DURATION),
                        "--k", str(K),
                        "--skip-drop-old",
                        "--skip-load",
                        "--skip-search-concurrent",
                    ]
                    if args.retrain_only:
                        profile_cmd.append("--retrain-only")
                    _, profile_output = run_command_capture(
                        profile_cmd,
                        vectordbbench_root,
                        dry_run=False,
                        extra_env={
                            "ZVEC_LOG_LEVEL": "INFO",
                            "ZVEC_OMEGA_PROFILE_CONTROL_TIMING": "1",
                            "ZVEC_OMEGA_LOG_QUERY_STATS": "1",
                            "ZVEC_OMEGA_LOG_QUERY_LIMIT": "2000",
                        },
                    )
                    baseline_profile = next(
                        (result.profiling for result in results if result.type == "HNSW" and result.profiling),
                        None,
                    )
                    omega_profile = build_omega_profile(metrics, profile_output, baseline_profile)
                results.append(BenchmarkResult(
                    type="OMEGA",
                    ef_search=EF_SEARCH,
                    target_recall=target_recall,
                    path=str(omega_path),
                    success=ret == 0,
                    load_duration=load_duration if load_duration is not None else metrics.get('load_duration'),
                    qps=metrics.get('qps'),
                    recall=metrics.get('recall'),
                    profiling=omega_profile,
                ))

    # ============ Summary ============
    if results:
        written_summary_paths = write_grouped_profiling_summaries("cohere_1m", results)
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
            avg_lat = profile.get("profile_avg_end2end_latency_ms")
            avg_cmps = profile.get("profile_avg_cmps")
            avg_pred_calls = profile.get("profile_avg_prediction_calls")
            avg_model_ms = profile.get("profile_avg_model_overhead_ms")
            saved_cmps = profile.get("profile_avg_early_stop_saved_cmps")
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
        for path in written_summary_paths:
            print(f"Profiling JSON: {path}")

    print()
    print("To view results:")
    print("  vectordbbench results")
    print()
    print("Or start the web UI:")
    print("  vectordbbench start")
    print()

    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
