#!/usr/bin/env python3

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
    profiling: dict[str, Any] | None = None


KV_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s,]+)")


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
    script_path: Path,
    config: dict[str, Any],
    zvec_root_arg: str | None,
    vectordbbench_root_arg: str | None,
    benchmark_dir_arg: str | None,
    results_dir_arg: str | None,
) -> tuple[Path, Path, Path, Path]:
    zvec_root = Path(zvec_root_arg).resolve() if zvec_root_arg else script_path.parent.parent
    vectordbbench_root = (
        Path(vectordbbench_root_arg).resolve()
        if vectordbbench_root_arg
        else Path(
            os.environ.get("VECTORDBBENCH_ROOT", zvec_root.parent / "VectorDBBench")
        ).resolve()
    )

    config_benchmark_dir = config.get("benchmark_dir")
    if benchmark_dir_arg:
        benchmark_dir = Path(benchmark_dir_arg).resolve()
    elif config_benchmark_dir:
        benchmark_dir = Path(config_benchmark_dir).expanduser().resolve()
    else:
        benchmark_dir = Path(
            os.environ.get("ZVEC_BENCHMARK_DIR", zvec_root / "benchmark_results")
        ).resolve()

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


def percentile_metric(
    records: list[dict[str, Any]], key: str, percentile: float
) -> float | None:
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
        "profile_avg_report_visit_candidate_ms": avg_metric(
            query_records, "report_visit_candidate_ms"
        ),
        "profile_avg_should_predict_ms": avg_metric(query_records, "should_predict_ms"),
        "profile_avg_report_hop_ms": avg_metric(query_records, "report_hop_ms"),
        "profile_avg_update_top_candidates_ms": avg_metric(
            query_records, "update_top_candidates_ms"
        ),
        "profile_avg_push_traversal_window_ms": avg_metric(
            query_records, "push_traversal_window_ms"
        ),
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


def write_grouped_profiling_summaries(
    dataset: str, results: list[BenchmarkResult]
) -> list[Path]:
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


def get_run_result(
    db_label: str, before_files: set[str], results_dir: Path
) -> dict[str, Any]:
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
    previous_summary = (
        read_json_if_exists(offline_summary_path(index_path)) if retrain_only else {}
    )
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
            optimize_duration = round(
                old_optimize_duration - old_training_s + new_training_s, 4
            )
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
            result = subprocess.run(
                cmd, cwd=cwd, env=env, stdout=tmp, stderr=subprocess.STDOUT, text=True
            )
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
