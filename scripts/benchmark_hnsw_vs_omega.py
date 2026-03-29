#!/usr/bin/env python3
"""
Generic VectorDBBench runner for Zvec HNSW vs Zvec+OMEGA.

Configuration is loaded from a JSON file so datasets, paths, and all
benchmark/index parameters can be changed without editing the script.
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    type: str
    path: str
    success: bool
    target_recall: float | None
    load_duration: float | None = None
    qps: float | None = None
    recall: float | None = None
    avg_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p90_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    profiling: dict | None = None


KV_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s,]+)")


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
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_dataset_config(path: Path, dataset_name: str) -> dict[str, Any]:
    root = load_json(path)
    if dataset_name not in root:
        available = ", ".join(sorted(root.keys()))
        raise ValueError(
            f"Dataset '{dataset_name}' not found in {path}. Available datasets: {available}"
        )
    dataset_config = root[dataset_name]
    if not isinstance(dataset_config, dict):
        raise ValueError(f"Dataset config for '{dataset_name}' must be a JSON object")
    return dataset_config


def resolve_paths(
    config: dict[str, Any],
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

    config_benchmark_dir = config.get("benchmark_dir")
    if benchmark_dir_arg:
        benchmark_dir = Path(benchmark_dir_arg).resolve()
    elif config_benchmark_dir:
        benchmark_dir = Path(config_benchmark_dir).expanduser().resolve()
    else:
        benchmark_dir = Path(os.environ.get("ZVEC_BENCHMARK_DIR", zvec_root / "benchmark_results")).resolve()

    source_results_dir = vectordbbench_root / "vectordb_bench" / "results" / "Zvec"
    if results_dir_arg:
        results_dir = Path(results_dir_arg).resolve()
    elif config.get("results_dir"):
        results_dir = Path(config["results_dir"]).expanduser().resolve()
    elif source_results_dir.exists():
        results_dir = source_results_dir
    else:
        try:
            bench_config = importlib.import_module("vectordb_bench").config
            results_dir = Path(bench_config.RESULTS_LOCAL_DIR).resolve() / "Zvec"
        except Exception:
            results_dir = source_results_dir

    return zvec_root, vectordbbench_root, benchmark_dir, results_dir


def resolve_vectordbbench_command() -> list[str]:
    return [sys.executable, "-m", "vectordb_bench.cli.vectordbbench"]


def parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_key_values(line: str) -> dict[str, Any]:
    return {key: parse_scalar(value) for key, value in KV_PATTERN.findall(line)}


def avg_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    if not values:
        return None
    return sum(values) / len(values)


def percentile_metric(records: list[dict[str, Any]], key: str, percentile: float) -> float | None:
    values = sorted(float(record[key]) for record in records if key in record)
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def parse_serial_runner_summary(output: str) -> dict[str, Any]:
    summary = {}
    for line in output.splitlines():
        if "search entire test_data:" not in line:
            continue
        summary = parse_key_values(line)
    return summary


def parse_query_records(output: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    for line in output.splitlines():
        if prefix not in line:
            continue
        records.append(parse_key_values(line))
    return records


def build_hnsw_profile(metrics: dict[str, Any], output: str) -> dict[str, Any]:
    query_records = parse_query_records(output, "HNSW query stats:")
    serial_summary = parse_serial_runner_summary(output)
    avg_latency_ms = avg_metric(query_records, "latency_ms")
    p50_latency_ms = percentile_metric(query_records, "latency_ms", 50)
    p90_latency_ms = percentile_metric(query_records, "latency_ms", 90)
    p95_latency_ms = percentile_metric(query_records, "latency_ms", 95)
    p99_latency_ms = percentile_metric(query_records, "latency_ms", 99)
    return {
        "benchmark_recall": metrics.get("recall"),
        "benchmark_qps": metrics.get("qps"),
        "profile_query_count": len(query_records),
        "profile_avg_end2end_latency_ms": avg_latency_ms,
        "profile_p50_end2end_latency_ms": p50_latency_ms,
        "profile_p90_end2end_latency_ms": p90_latency_ms,
        "profile_p95_end2end_latency_ms": p95_latency_ms,
        "profile_p99_end2end_latency_ms": p99_latency_ms,
        "profile_avg_cmps": avg_metric(query_records, "pairwise_dist_cnt"),
        "profile_avg_scan_cmps": avg_metric(query_records, "cmps"),
        "profile_avg_pure_search_ms": avg_metric(query_records, "pure_search_ms"),
        "profile_serial_avg_latency_s": serial_summary.get("avg_latency"),
        "profile_serial_p99_s": serial_summary.get("p99"),
        "profile_serial_p95_s": serial_summary.get("p95"),
        "profile_serial_avg_recall": serial_summary.get("avg_recall"),
    }


def build_omega_profile(
    metrics: dict[str, Any], output: str, hnsw_profile: dict[str, Any] | None
) -> dict[str, Any]:
    query_records = parse_query_records(output, "OMEGA query stats:")
    serial_summary = parse_serial_runner_summary(output)
    avg_latency_ms = avg_metric(query_records, "total_ms")
    p50_latency_ms = percentile_metric(query_records, "total_ms", 50)
    p90_latency_ms = percentile_metric(query_records, "total_ms", 90)
    p95_latency_ms = percentile_metric(query_records, "total_ms", 95)
    p99_latency_ms = percentile_metric(query_records, "total_ms", 99)

    avg_pairwise_dist_cnt = avg_metric(query_records, "pairwise_dist_cnt")
    avg_core_search_ms = avg_metric(query_records, "core_search_ms")
    avg_pure_search_ms = avg_metric(query_records, "pure_search_ms")
    avg_hook_total_ms = avg_metric(query_records, "hook_total_ms")
    avg_search_only_ms = (
        avg_pure_search_ms if avg_pure_search_ms is not None else avg_core_search_ms
    )

    cmp_time_ms = None
    if avg_pairwise_dist_cnt and avg_pairwise_dist_cnt > 0 and avg_search_only_ms is not None:
        cmp_time_ms = avg_search_only_ms / avg_pairwise_dist_cnt

    model_overhead_cmp_equiv = None
    if cmp_time_ms and cmp_time_ms > 0 and avg_hook_total_ms is not None:
        model_overhead_cmp_equiv = avg_hook_total_ms / cmp_time_ms

    avg_saved_cmps = None
    if (
        hnsw_profile
        and hnsw_profile.get("profile_avg_cmps") is not None
        and avg_pairwise_dist_cnt is not None
    ):
        avg_saved_cmps = hnsw_profile["profile_avg_cmps"] - avg_pairwise_dist_cnt

    return {
        "benchmark_recall": metrics.get("recall"),
        "benchmark_qps": metrics.get("qps"),
        "profile_query_count": len(query_records),
        "profile_avg_end2end_latency_ms": avg_latency_ms,
        "profile_p50_end2end_latency_ms": p50_latency_ms,
        "profile_p90_end2end_latency_ms": p90_latency_ms,
        "profile_p95_end2end_latency_ms": p95_latency_ms,
        "profile_p99_end2end_latency_ms": p99_latency_ms,
        "profile_avg_cmps": avg_pairwise_dist_cnt,
        "profile_avg_scan_cmps": avg_metric(query_records, "scan_cmps"),
        "profile_avg_omega_cmps": avg_metric(query_records, "omega_cmps"),
        "profile_avg_prediction_calls": avg_metric(query_records, "prediction_calls"),
        "profile_avg_should_stop_calls": avg_metric(query_records, "should_stop_calls"),
        "profile_avg_advance_calls": avg_metric(query_records, "advance_calls"),
        "profile_avg_model_overhead_ms": avg_hook_total_ms,
        "profile_avg_setup_ms": avg_metric(query_records, "setup_ms"),
        "profile_avg_should_stop_ms": avg_metric(query_records, "should_stop_ms"),
        "profile_avg_prediction_eval_ms": avg_metric(query_records, "prediction_eval_ms"),
        "profile_avg_core_search_ms": avg_core_search_ms,
        "profile_avg_pure_search_ms": avg_pure_search_ms,
        "profile_avg_hook_total_ms": avg_hook_total_ms,
        "profile_avg_hook_body_ms": avg_metric(query_records, "hook_body_ms"),
        "profile_avg_hook_dispatch_ms": avg_metric(query_records, "hook_dispatch_ms"),
        "profile_avg_report_visit_candidate_ms": avg_metric(query_records, "report_visit_candidate_ms"),
        "profile_avg_should_predict_ms": avg_metric(query_records, "should_predict_ms"),
        "profile_avg_report_hop_ms": avg_metric(query_records, "report_hop_ms"),
        "profile_avg_update_top_candidates_ms": avg_metric(query_records, "update_top_candidates_ms"),
        "profile_avg_push_traversal_window_ms": avg_metric(query_records, "push_traversal_window_ms"),
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


def write_profiling_summary(index_path: Path, payload: dict[str, Any]) -> None:
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
                        "avg_latency_ms": result.avg_latency_ms,
                        "p50_latency_ms": result.p50_latency_ms,
                        "p90_latency_ms": result.p90_latency_ms,
                        "p95_latency_ms": result.p95_latency_ms,
                        "p99_latency_ms": result.p99_latency_ms,
                        "recall": result.recall,
                        "profiling": result.profiling,
                    }
                    for result in grouped_results
                ],
            },
        )
        written_paths.append(profiling_output_path(index_path))

    return written_paths


def get_latest_result(db_label: str, results_dir: Path) -> dict[str, Any]:
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
                        "avg_latency_ms": metrics.get("serial_latency_avg"),
                        "p95_latency_ms": metrics.get("serial_latency_p95"),
                        "p99_latency_ms": metrics.get("serial_latency_p99"),
                        "recall": metrics.get("recall"),
                    }
        except Exception:
            continue
    return {}


def latency_summary_from_profile(profile: dict[str, Any] | None) -> dict[str, float | None]:
    profile = profile or {}
    return {
        "avg_latency_ms": profile.get("profile_avg_end2end_latency_ms"),
        "p50_latency_ms": profile.get("profile_p50_end2end_latency_ms"),
        "p90_latency_ms": profile.get("profile_p90_end2end_latency_ms"),
        "p95_latency_ms": profile.get("profile_p95_end2end_latency_ms"),
        "p99_latency_ms": profile.get("profile_p99_end2end_latency_ms"),
    }


def merge_omega_detailed_profile(
    summary_profile: dict[str, Any], detailed_profile: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(summary_profile)
    detailed_keys = [
        "profile_avg_model_overhead_ms",
        "profile_avg_should_stop_ms",
        "profile_avg_prediction_eval_ms",
        "profile_avg_core_search_ms",
        "profile_avg_pure_search_ms",
        "profile_avg_hook_total_ms",
        "profile_avg_hook_body_ms",
        "profile_avg_hook_dispatch_ms",
        "profile_avg_report_visit_candidate_ms",
        "profile_avg_should_predict_ms",
        "profile_avg_report_hop_ms",
        "profile_avg_update_top_candidates_ms",
        "profile_avg_push_traversal_window_ms",
        "profile_avg_model_overhead_cmp_equiv",
    ]
    for key in detailed_keys:
        merged[key] = detailed_profile.get(key)
    return merged


def snapshot_result_files(results_dir: Path) -> set[str]:
    if not results_dir.exists():
        return set()
    return {str(p) for p in results_dir.glob("result_*.json")}


def extract_result_from_file(result_file: Path, db_label: str) -> dict[str, Any]:
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


def get_run_result(db_label: str, before_files: set[str], results_dir: Path) -> dict[str, Any]:
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


def read_json_if_exists(path: Path) -> dict[str, Any]:
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


def sum_timing_ms(data: dict[str, Any]) -> int:
    return sum(v for v in data.values() if isinstance(v, (int, float)))


def build_offline_summary(
    index_path: Path,
    db_label: str,
    metrics: dict[str, Any],
    retrain_only: bool = False,
) -> dict[str, Any]:
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
    metrics: dict[str, Any],
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
    cmd_str = " \\\n    ".join(cmd)
    print(f"\n{'=' * 60}")
    print(f"Command:\n{cmd_str}")
    print(f"{'=' * 60}\n")
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


def must_get(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    return config[key]


def resolve_index_path(benchmark_dir: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (benchmark_dir / path).resolve()


def append_option(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if isinstance(value, list):
        cmd.extend([flag, ",".join(str(v) for v in value)])
    else:
        cmd.extend([flag, str(value)])


def extend_with_args(cmd: list[str], args_map: dict[str, Any] | None) -> None:
    if not args_map:
        return
    for key, value in args_map.items():
        append_option(cmd, key, value)


def extend_with_flags(cmd: list[str], flags: list[str] | None) -> None:
    if not flags:
        return
    for flag in flags:
        cmd.append(f"--{flag}")


def build_base_command(
    vectordbbench_cmd: list[str],
    client_name: str,
    path: Path,
    db_label: str,
    case_type: str,
    common_args: dict[str, Any],
    specific_args: dict[str, Any] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    cmd = [
        *vectordbbench_cmd,
        client_name,
        "--path",
        str(path),
        "--db-label",
        db_label,
        "--case-type",
        case_type,
    ]
    extend_with_args(cmd, common_args)
    extend_with_args(cmd, specific_args)
    extend_with_flags(cmd, extra_flags)
    return cmd


def validate_profile_output(profile_name: str, ret: int, output: str, prefix: str) -> None:
    if ret != 0:
        raise RuntimeError(f"{profile_name} profiling pass failed with exit code {ret}")
    if not parse_query_records(output, prefix):
        raise RuntimeError(
            f"{profile_name} profiling pass completed without any '{prefix}' records in stdout"
        )


def print_header(title: str) -> None:
    print("\n\n" + "#" * 70)
    print(f"# {title}")
    print("#" * 70)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_dataset_config(config_path, args.dataset)
    zvec_root, vectordbbench_root, benchmark_dir, results_dir = resolve_paths(
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

    case_type = must_get(common, "case_type")
    hnsw_path = resolve_index_path(benchmark_dir, must_get(hnsw_config, "path"))
    omega_path = resolve_index_path(benchmark_dir, must_get(omega_config, "path"))
    hnsw_db_label = must_get(hnsw_config, "db_label")
    omega_db_label = must_get(omega_config, "db_label")
    target_recalls = omega_config.get("target_recalls", [])
    if not target_recalls:
        raise ValueError("omega.target_recalls must be a non-empty list")

    hnsw_common_args = {k: v for k, v in common.items() if k != "case_type"}
    hnsw_specific_args = hnsw_config.get("args", {})
    omega_specific_args = omega_config.get("args", {})

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
