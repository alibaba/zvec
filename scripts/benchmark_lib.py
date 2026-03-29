#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
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
AVG_LINE_PATTERN = re.compile(r"Avg latency:\s*([0-9.]+)ms qps:\s*([0-9.]+)")
PERCENTILE_PATTERN = re.compile(r"(\d+)\s+Percentile:\s*([0-9.]+)\s+ms")
PROCESS_LINE_PATTERN = re.compile(
    r"Process query:\s*(\d+), total process time:\s*(\d+)ms, duration:\s*(\d+)ms"
)
RECALL_PATTERN = re.compile(r"Recall@(\d+):\s*([0-9.]+)")

_ZVEC_INITIALIZED = False

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "cohere_1m": {
        "dataset_dirname": "cohere/cohere_medium_1m",
        "remote_dirname": "cohere_medium_1m",
        "dimension": 768,
        "metric_type": "COSINE",
        "train_files": ["shuffle_train.parquet"],
    },
    "cohere_10m": {
        "dataset_dirname": "cohere/cohere_large_10m",
        "remote_dirname": "cohere_large_10m",
        "dimension": 768,
        "metric_type": "COSINE",
        "train_files": [f"shuffle_train-{idx:02d}-of-10.parquet" for idx in range(10)],
    },
}

DATASET_DOWNLOAD_BASE_URLS = {
    "S3": "https://assets.zilliz.com/benchmark",
    "ALIYUNOSS": "https://assets.zilliz.com.cn/benchmark",
}


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


def must_get(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def resolve_paths(
    script_path: Path,
    config: dict[str, Any],
    zvec_root_arg: str | None,
    benchmark_dir_arg: str | None,
) -> tuple[Path, Path]:
    zvec_root = Path(zvec_root_arg).resolve() if zvec_root_arg else script_path.parent.parent

    config_benchmark_dir = config.get("benchmark_dir")
    if benchmark_dir_arg:
        benchmark_dir = Path(benchmark_dir_arg).resolve()
    elif config_benchmark_dir:
        benchmark_dir = Path(config_benchmark_dir).expanduser().resolve()
    else:
        benchmark_dir = (zvec_root / "benchmark_results").resolve()

    return zvec_root, benchmark_dir


def resolve_index_path(benchmark_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (benchmark_dir / path).resolve()


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


def parse_query_records(output: str, prefix: str) -> list[dict[str, Any]]:
    records = []
    for line in output.splitlines():
        if prefix in line:
            records.append(parse_key_values(line))
    return records


def parse_bench_output(output: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in output.splitlines():
        if (match := PROCESS_LINE_PATTERN.search(line)) is not None:
            metrics["process_query_count"] = int(match.group(1))
            metrics["total_process_time_ms"] = int(match.group(2))
            metrics["duration_ms"] = int(match.group(3))
        elif (match := AVG_LINE_PATTERN.search(line)) is not None:
            metrics["avg_latency_ms"] = float(match.group(1))
            metrics["qps"] = float(match.group(2))
        elif (match := PERCENTILE_PATTERN.search(line)) is not None:
            metrics[f"p{match.group(1)}_latency_ms"] = float(match.group(2))
    return metrics


def parse_recall_output(output: str, topk: int) -> float | None:
    recall_by_k: dict[int, float] = {}
    for line in output.splitlines():
        match = RECALL_PATTERN.search(line)
        if match is not None:
            recall_by_k[int(match.group(1))] = float(match.group(2))
    if topk in recall_by_k:
        return recall_by_k[topk]
    if recall_by_k:
        return recall_by_k[max(recall_by_k)]
    return None


def build_hnsw_profile(
    metrics: dict[str, Any], output: str, bench_summary: dict[str, Any]
) -> dict[str, Any]:
    query_records = parse_query_records(output, "HNSW query stats:")
    return {
        "benchmark_recall": metrics.get("recall"),
        "benchmark_qps": metrics.get("qps"),
        "profile_query_count": len(query_records),
        "profile_avg_end2end_latency_ms": bench_summary.get("avg_latency_ms"),
        "profile_p50_end2end_latency_ms": bench_summary.get("p50_latency_ms"),
        "profile_p90_end2end_latency_ms": bench_summary.get("p90_latency_ms"),
        "profile_p95_end2end_latency_ms": bench_summary.get("p95_latency_ms"),
        "profile_p99_end2end_latency_ms": bench_summary.get("p99_latency_ms"),
        "profile_avg_cmps": avg_metric(query_records, "pairwise_dist_cnt"),
        "profile_avg_scan_cmps": avg_metric(query_records, "cmps"),
        "profile_avg_pure_search_ms": avg_metric(query_records, "pure_search_ms"),
        "profile_serial_avg_latency_s": (
            bench_summary["avg_latency_ms"] / 1000.0
            if bench_summary.get("avg_latency_ms") is not None
            else None
        ),
        "profile_serial_p99_s": (
            bench_summary["p99_latency_ms"] / 1000.0
            if bench_summary.get("p99_latency_ms") is not None
            else None
        ),
        "profile_serial_p95_s": (
            bench_summary["p95_latency_ms"] / 1000.0
            if bench_summary.get("p95_latency_ms") is not None
            else None
        ),
        "profile_serial_avg_recall": metrics.get("recall"),
    }


def build_omega_profile(
    metrics: dict[str, Any],
    output: str,
    bench_summary: dict[str, Any],
    hnsw_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    query_records = parse_query_records(output, "OMEGA query stats:")

    avg_pairwise_dist_cnt = avg_metric(query_records, "pairwise_dist_cnt")
    avg_core_search_ms = avg_metric(query_records, "core_search_ms")
    avg_pure_search_ms = avg_metric(query_records, "pure_search_ms")
    avg_hook_total_ms = avg_metric(query_records, "hook_total_ms")
    avg_search_only_ms = avg_pure_search_ms if avg_pure_search_ms is not None else avg_core_search_ms

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
        "profile_avg_end2end_latency_ms": bench_summary.get("avg_latency_ms"),
        "profile_p50_end2end_latency_ms": bench_summary.get("p50_latency_ms"),
        "profile_p90_end2end_latency_ms": bench_summary.get("p90_latency_ms"),
        "profile_p95_end2end_latency_ms": bench_summary.get("p95_latency_ms"),
        "profile_p99_end2end_latency_ms": bench_summary.get("p99_latency_ms"),
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
        "profile_serial_avg_latency_s": (
            bench_summary["avg_latency_ms"] / 1000.0
            if bench_summary.get("avg_latency_ms") is not None
            else None
        ),
        "profile_serial_p99_s": (
            bench_summary["p99_latency_ms"] / 1000.0
            if bench_summary.get("p99_latency_ms") is not None
            else None
        ),
        "profile_serial_p95_s": (
            bench_summary["p95_latency_ms"] / 1000.0
            if bench_summary.get("p95_latency_ms") is not None
            else None
        ),
        "profile_serial_avg_recall": metrics.get("recall"),
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
            "lightgbm_training_metrics": read_json_if_exists(
                omega_model_dir / "lightgbm_training_metrics.json"
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
        load_duration = (
            round(insert_duration + optimize_duration, 4)
            if insert_duration is not None and optimize_duration is not None
            else None
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
    index_path: Path, db_label: str, metrics: dict[str, Any], retrain_only: bool = False
) -> Path:
    summary = build_offline_summary(index_path, db_label, metrics, retrain_only=retrain_only)
    path = offline_summary_path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return path


def get_offline_load_duration(index_path: Path) -> float | None:
    return read_json_if_exists(offline_summary_path(index_path)).get("offline", {}).get(
        "load_duration_s"
    )


def latency_summary_from_profile(profile: dict[str, Any] | None) -> dict[str, float | None]:
    profile = profile or {}
    return {
        "avg_latency_ms": profile.get("profile_avg_end2end_latency_ms"),
        "p50_latency_ms": profile.get("profile_p50_end2end_latency_ms"),
        "p90_latency_ms": profile.get("profile_p90_end2end_latency_ms"),
        "p95_latency_ms": profile.get("profile_p95_end2end_latency_ms"),
        "p99_latency_ms": profile.get("profile_p99_end2end_latency_ms"),
    }


def resolve_dataset_spec(
    dataset_name: str, config: dict[str, Any], dataset_root_arg: str | None
) -> dict[str, Any]:
    default = DATASET_SPECS.get(dataset_name, {})
    dataset_root = None
    if dataset_root_arg:
        dataset_root = Path(dataset_root_arg).expanduser().resolve()
    elif config.get("dataset_root"):
        dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    elif os.environ.get("DATASET_LOCAL_DIR"):
        dataset_root = Path(os.environ["DATASET_LOCAL_DIR"]).expanduser().resolve()
    else:
        dataset_root = Path("/tmp/zvec/dataset").resolve()

    dataset_dirname = config.get("dataset_dirname", default.get("dataset_dirname"))
    if not dataset_dirname:
        raise ValueError(
            f"Dataset directory name is not configured for {dataset_name}."
        )

    dimension = int(config.get("dimension", default.get("dimension", 0)))
    metric_type = str(config.get("metric_type", default.get("metric_type", "COSINE"))).upper()
    remote_dirname = str(config.get("remote_dirname", default.get("remote_dirname", "")))
    train_files = list(config.get("train_files", default.get("train_files", [])))
    dataset_source = str(config.get("dataset_source", os.environ.get("ZVEC_DATASET_SOURCE", "S3")))
    download_base_url = str(
        config.get(
            "dataset_base_url",
            os.environ.get(
                "ZVEC_DATASET_BASE_URL",
                DATASET_DOWNLOAD_BASE_URLS.get(dataset_source.upper(), DATASET_DOWNLOAD_BASE_URLS["S3"]),
            ),
        )
    )
    if dimension <= 0:
        raise ValueError(f"Missing dataset dimension for {dataset_name}")

    dataset_dir = (dataset_root / dataset_dirname).resolve()
    return {
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dimension": dimension,
        "metric_type": metric_type,
        "remote_dirname": remote_dirname,
        "train_files": train_files,
        "download_base_url": download_base_url.rstrip("/"),
    }


def _require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "This script requires polars in the active Python environment."
        ) from exc
    return pl


def _sorted_train_files(dataset_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in [
        "shuffle_train-*.parquet",
        "train-*.parquet",
        "shuffle_train.parquet",
        "train.parquet",
    ]:
        candidates.extend(sorted(dataset_dir.glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _dataset_required_files(dataset_name: str, dataset_spec: dict[str, Any]) -> list[str]:
    required = list(dataset_spec.get("train_files", []))
    if not required:
        raise ValueError(
            f"Dataset {dataset_name} does not define train_files for auto-download"
        )
    required.extend(["test.parquet", "neighbors.parquet"])
    return required


def _download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_dataset_available(dataset_name: str, dataset_spec: dict[str, Any], dry_run: bool) -> None:
    dataset_dir = dataset_spec["dataset_dir"]
    required_files = _dataset_required_files(dataset_name, dataset_spec)
    missing_files = [name for name in required_files if not (dataset_dir / name).exists()]
    if not missing_files:
        return

    remote_dirname = dataset_spec.get("remote_dirname")
    if not remote_dirname:
        raise FileNotFoundError(
            f"Dataset directory is incomplete and auto-download is not configured: {dataset_dir}"
        )

    base_url = dataset_spec["download_base_url"]
    print(f"Dataset files missing under {dataset_dir}, downloading from {base_url}/{remote_dirname} ...")
    if dry_run:
        for name in missing_files:
            print(f"[Dry-run] download {base_url}/{remote_dirname}/{name} -> {dataset_dir / name}")
        return

    for name in missing_files:
        url = f"{base_url}/{remote_dirname}/{name}"
        output_path = dataset_dir / name
        print(f"Downloading {url}")
        _download_file(url, output_path)


def prepare_dataset_artifacts(
    dataset_name: str,
    dataset_spec: dict[str, Any],
    benchmark_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    dataset_dir = dataset_spec["dataset_dir"]
    query_parquet = dataset_dir / "test.parquet"
    gt_parquet = dataset_dir / "neighbors.parquet"
    ensure_dataset_available(dataset_name, dataset_spec, dry_run)
    train_files = _sorted_train_files(dataset_dir)
    if not dry_run:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not query_parquet.exists():
            raise FileNotFoundError(f"Missing query parquet: {query_parquet}")
        if not gt_parquet.exists():
            raise FileNotFoundError(f"Missing ground-truth parquet: {gt_parquet}")
        if not train_files:
            raise FileNotFoundError(f"No train parquet files found under: {dataset_dir}")

    cache_dir = (benchmark_dir / "_dataset_cache" / dataset_name).resolve()
    query_txt = cache_dir / "query.txt"
    gt_txt = cache_dir / "groundtruth.txt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        refresh_query = (not query_txt.exists()) or query_txt.stat().st_mtime < query_parquet.stat().st_mtime
        refresh_gt = (not gt_txt.exists()) or gt_txt.stat().st_mtime < gt_parquet.stat().st_mtime
        if refresh_query:
            _write_query_text(query_parquet, query_txt)
        if refresh_gt:
            _write_groundtruth_text(gt_parquet, gt_txt)

    return {
        "dataset_dir": dataset_dir,
        "query_parquet": query_parquet,
        "gt_parquet": gt_parquet,
        "query_txt": query_txt,
        "groundtruth_txt": gt_txt,
        "train_files": train_files,
    }


def _write_query_text(query_parquet: Path, output_path: Path) -> None:
    pl = _require_polars()
    frame = pl.read_parquet(query_parquet).sort("id")
    with open(output_path, "w") as f:
        for row in frame.iter_rows(named=True):
            vector = row["emb"]
            vector_text = " ".join(str(round(float(v), 16)) for v in vector)
            f.write(f"{int(row['id'])};{vector_text};\n")


def _write_groundtruth_text(gt_parquet: Path, output_path: Path) -> None:
    pl = _require_polars()
    frame = pl.read_parquet(gt_parquet).sort("id")
    with open(output_path, "w") as f:
        for row in frame.iter_rows(named=True):
            neighbors = " ".join(str(int(v)) for v in row["neighbors_id"])
            f.write(f"{int(row['id'])};{neighbors}\n")


def _ensure_zvec_initialized() -> None:
    global _ZVEC_INITIALIZED
    if _ZVEC_INITIALIZED:
        return
    import zvec

    zvec.init(log_level=zvec.LogLevel.WARN)
    _ZVEC_INITIALIZED = True


def _quantize_type_from_name(name: str):
    import zvec

    normalized = str(name).upper()
    mapping = {
        "": zvec.QuantizeType.UNDEFINED,
        "UNDEFINED": zvec.QuantizeType.UNDEFINED,
        "FP16": zvec.QuantizeType.FP16,
        "INT8": zvec.QuantizeType.INT8,
        "INT4": zvec.QuantizeType.INT4,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported quantize type: {name}")
    return mapping[normalized]


def _metric_type_from_name(name: str):
    import zvec

    normalized = str(name).upper()
    mapping = {
        "COSINE": zvec.MetricType.COSINE,
        "IP": zvec.MetricType.IP,
        "L2": zvec.MetricType.L2,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported metric type: {name}")
    return mapping[normalized]


def _maybe_destroy_collection(path: Path) -> None:
    import zvec

    if not path.exists():
        return
    try:
        zvec.open(str(path)).destroy()
        return
    except Exception:
        pass
    shutil.rmtree(path, ignore_errors=True)


def _build_schema(
    index_kind: str,
    dimension: int,
    metric_type: str,
    common_args: dict[str, Any],
    specific_args: dict[str, Any],
):
    import zvec

    quantize_type = _quantize_type_from_name(common_args.get("quantize_type", ""))
    metric = _metric_type_from_name(metric_type)
    if index_kind == "OMEGA":
        index_param = zvec.OmegaIndexParam(
            metric_type=metric,
            m=int(common_args["m"]),
            ef_construction=int(specific_args.get("ef_construction", 500)),
            quantize_type=quantize_type,
            min_vector_threshold=int(specific_args["min_vector_threshold"]),
            num_training_queries=int(specific_args["num_training_queries"]),
            ef_training=int(specific_args["ef_training"]),
            window_size=int(specific_args["window_size"]),
            ef_groundtruth=int(specific_args["ef_groundtruth"]),
            k_train=int(specific_args.get("k_train", 1)),
        )
    else:
        index_param = zvec.HnswIndexParam(
            metric_type=metric,
            m=int(common_args["m"]),
            ef_construction=int(specific_args.get("ef_construction", 500)),
            quantize_type=quantize_type,
        )

    return zvec.CollectionSchema(
        name=f"{index_kind.lower()}_benchmark",
        fields=[
            zvec.FieldSchema(
                "id",
                zvec.DataType.INT64,
                nullable=False,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            zvec.VectorSchema(
                "dense",
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=index_param,
            )
        ],
    )


def build_index(
    *,
    index_kind: str,
    index_path: Path,
    dataset_spec: dict[str, Any],
    dataset_artifacts: dict[str, Any],
    common_args: dict[str, Any],
    specific_args: dict[str, Any],
    retrain_only: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        print(f"[Dry-run] Build {index_kind} at {index_path}")
        return {"insert_duration": None, "optimize_duration": None, "load_duration": None}

    _ensure_zvec_initialized()
    import zvec

    if retrain_only:
        collection = zvec.open(
            str(index_path), zvec.CollectionOption(read_only=False, enable_mmap=True)
        )
        insert_duration = None
    else:
        _maybe_destroy_collection(index_path)
        schema = _build_schema(
            index_kind,
            dataset_spec["dimension"],
            dataset_spec["metric_type"],
            common_args,
            specific_args,
        )
        collection = zvec.create_and_open(
            str(index_path),
            schema,
            zvec.CollectionOption(read_only=False, enable_mmap=True),
        )
        insert_duration = _insert_training_data(collection, dataset_artifacts["train_files"])

    optimize_start = time.perf_counter()
    collection.optimize(option=zvec.OptimizeOption(retrain_only=retrain_only))
    optimize_duration = time.perf_counter() - optimize_start
    try:
        collection.flush()
    except Exception:
        pass
    del collection

    load_duration = None
    if insert_duration is not None:
        load_duration = insert_duration + optimize_duration
    elif optimize_duration is not None:
        load_duration = optimize_duration

    return {
        "insert_duration": round(insert_duration, 4) if insert_duration is not None else None,
        "optimize_duration": round(optimize_duration, 4) if optimize_duration is not None else None,
        "load_duration": round(load_duration, 4) if load_duration is not None else None,
    }


def _insert_training_data(collection, train_files: list[Path], batch_size: int = 1000) -> float:
    import zvec

    pl = _require_polars()
    start = time.perf_counter()
    for train_file in train_files:
        frame = pl.read_parquet(train_file)
        for offset in range(0, frame.height, batch_size):
            batch = frame.slice(offset, batch_size)
            ids = batch["id"].to_list()
            vectors = batch["emb"].to_list()
            docs = [
                zvec.Doc(
                    id=str(int(doc_id)),
                    fields={"id": int(doc_id)},
                    vectors={"dense": vector},
                )
                for doc_id, vector in zip(ids, vectors, strict=True)
            ]
            collection.insert(docs)
    return time.perf_counter() - start


def compute_recall_with_zvec(
    *,
    index_kind: str,
    index_path: Path,
    dataset_artifacts: dict[str, Any],
    common_args: dict[str, Any],
    target_recall: float | None,
    dry_run: bool,
) -> float | None:
    if dry_run:
        return None

    _ensure_zvec_initialized()
    import zvec

    pl = _require_polars()
    query_frame = pl.read_parquet(dataset_artifacts["query_parquet"]).sort("id")
    gt_frame = pl.read_parquet(dataset_artifacts["gt_parquet"]).sort("id")
    gt_map = {
        int(row["id"]): [int(value) for value in row["neighbors_id"][: int(common_args["k"])]]
        for row in gt_frame.iter_rows(named=True)
    }

    option = zvec.CollectionOption(read_only=True, enable_mmap=True)
    collection = zvec.open(str(index_path), option)
    use_refiner = bool(common_args.get("is_using_refiner", False))
    if index_kind == "OMEGA":
        query_param = zvec.OmegaQueryParam(
            ef=int(common_args["ef_search"]),
            target_recall=float(target_recall),
            is_using_refiner=use_refiner,
        )
    else:
        query_param = zvec.HnswQueryParam(
            ef=int(common_args["ef_search"]),
            is_using_refiner=use_refiner,
        )

    recall_sum = 0.0
    query_count = 0
    topk = int(common_args["k"])
    for row in query_frame.iter_rows(named=True):
        query_id = int(row["id"])
        gt = gt_map.get(query_id)
        if not gt:
            continue
        results = collection.query(
            vectors=zvec.VectorQuery(field_name="dense", vector=row["emb"], param=query_param),
            topk=topk,
            output_fields=[],
        )
        pred = [int(doc.id) for doc in results[:topk]]
        recall_sum += len(set(pred) & set(gt)) / float(topk)
        query_count += 1

    del collection
    if query_count == 0:
        return None
    return recall_sum / query_count


def resolve_core_tools(zvec_root: Path) -> tuple[Path, Path]:
    bench_bin = (zvec_root / "build/bin/bench").resolve()
    recall_bin = (zvec_root / "build/bin/recall").resolve()
    if not bench_bin.exists():
        raise FileNotFoundError(f"bench binary not found: {bench_bin}")
    if not recall_bin.exists():
        raise FileNotFoundError(f"recall binary not found: {recall_bin}")
    return bench_bin, recall_bin


def _metric_type_name_for_core(metric_type: str) -> str:
    mapping = {
        "COSINE": "kCosine",
        "IP": "kInnerProduct",
        "L2": "kL2sq",
    }
    normalized = str(metric_type).upper()
    if normalized not in mapping:
        raise ValueError(f"Unsupported metric type: {metric_type}")
    return mapping[normalized]


def _quantizer_json(quantize_type: str) -> dict[str, Any] | None:
    normalized = str(quantize_type).upper()
    if normalized in {"", "UNDEFINED"}:
        return None
    mapping = {
        "FP16": "kFP16",
        "INT8": "kInt8",
        "INT4": "kInt4",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported quantize type: {quantize_type}")
    return {"type": mapping[normalized]}


def build_core_index_config_json(
    *,
    index_type: str,
    metric_type: str,
    dimension: int,
    m: int,
    ef_construction: int,
    quantize_type: str,
) -> str:
    payload: dict[str, Any] = {
        "index_type": index_type,
        "metric_type": _metric_type_name_for_core(metric_type),
        "dimension": int(dimension),
        "version": 0,
        "is_sparse": False,
        "data_type": "DT_FP32",
        "use_id_map": False,
        "is_huge_page": False,
        "m": int(m),
        "ef_construction": int(ef_construction),
    }
    quantizer = _quantizer_json(quantize_type)
    if quantizer is not None:
        payload["quantizer_param"] = quantizer
    return json.dumps(payload, separators=(",", ":"))


def build_core_query_param_json(
    *,
    index_type: str,
    ef_search: int,
    topk: int,
    target_recall: float | None = None,
) -> str:
    payload: dict[str, Any] = {
        "index_type": index_type,
        "topk": int(topk),
        "fetch_vector": False,
        "radius": 0.0,
        "is_linear": False,
        "ef_search": int(ef_search),
    }
    if target_recall is not None:
        payload["target_recall"] = float(target_recall)
    return json.dumps(payload, separators=(",", ":"))


def discover_index_files(index_path: Path) -> dict[str, Path | None]:
    coarse_candidates = sorted(index_path.glob("*/dense.qindex.*.proxima"))
    full_candidates = sorted(index_path.glob("*/dense.index.*.proxima"))
    primary = coarse_candidates[0] if coarse_candidates else (full_candidates[0] if full_candidates else None)
    reference = full_candidates[0] if full_candidates else None
    if primary is None:
        raise FileNotFoundError(f"No core index file found under {index_path}")
    return {"primary": primary, "reference": reference}


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write_core_config(
    *,
    path: Path,
    index_path: Path,
    index_config_json: str,
    query_param_json: str,
    query_file: Path,
    topk: int,
    use_refiner: bool,
    reference_index_path: Path | None,
    metric_type: str,
    dimension: int,
    m: int,
    ef_construction: int,
    bench_thread_count: int | None = None,
    bench_secs: int | None = None,
    recall_thread_count: int | None = None,
    groundtruth_file: Path | None = None,
) -> None:
    lines = [
        "IndexCommon:",
        f"  IndexPath: {_yaml_quote(str(index_path))}",
        f"  IndexConfig: {_yaml_quote(index_config_json)}",
        f"  TopK: {_yaml_quote(str(topk))}",
        f"  QueryFile: {_yaml_quote(str(query_file))}",
        "  QueryType: 'float'",
        "  QueryFirstSep: ';'",
        "  QuerySecondSep: ' '",
    ]
    if bench_thread_count is not None:
        lines.append(f"  BenchThreadCount: {bench_thread_count}")
    if bench_secs is not None:
        lines.append(f"  BenchSecs: {bench_secs}")
        lines.append("  BenchIterCount: 1000000000")
    if recall_thread_count is not None:
        lines.append(f"  RecallThreadCount: {recall_thread_count}")
        lines.append(f"  RecallGTCount: {topk}")
        lines.append("  CompareById: true")
    if groundtruth_file is not None:
        lines.append(f"  GroundTruthFile: {_yaml_quote(str(groundtruth_file))}")
        lines.append("  GroundTruthFirstSep: ';'")
        lines.append("  GroundTruthSecondSep: ' '")

    lines.extend(
        [
            "QueryConfig:",
            f"  QueryParam: {_yaml_quote(query_param_json)}",
        ]
    )

    if use_refiner:
        if reference_index_path is None:
            raise ValueError("Refiner requested but reference index is missing")
        reference_config = build_core_index_config_json(
            index_type="kHNSW",
            metric_type=metric_type,
            dimension=dimension,
            m=m,
            ef_construction=ef_construction,
            quantize_type="UNDEFINED",
        )
        lines.extend(
            [
                "  RefinerConfig:",
                "    ScaleFactor: 2",
                "    ReferenceIndex:",
                f"      Config: {_yaml_quote(reference_config)}",
                f"      Path: {_yaml_quote(str(reference_index_path))}",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def run_command_capture(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    printable = " ".join(str(token) for token in cmd)
    print(printable)
    if dry_run:
        return 0, ""

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0 and completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return completed.returncode, completed.stdout


def _temporary_config_path(prefix: str, parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return parent / f"{prefix}_{int(time.time() * 1000)}.yaml"


def run_bench(
    *,
    bench_bin: Path,
    index_file: Path,
    query_file: Path,
    metric_type: str,
    dimension: int,
    m: int,
    ef_construction: int,
    quantize_type: str,
    ef_search: int,
    topk: int,
    bench_thread_count: int,
    bench_secs: int,
    use_refiner: bool,
    reference_index_path: Path | None,
    target_recall: float | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    config_path = _temporary_config_path("bench", index_file.parent)
    try:
        _write_core_config(
            path=config_path,
            index_path=index_file,
            index_config_json=build_core_index_config_json(
                index_type="kOMEGA" if target_recall is not None else "kHNSW",
                metric_type=metric_type,
                dimension=dimension,
                m=m,
                ef_construction=ef_construction,
                quantize_type=quantize_type,
            ),
            query_param_json=build_core_query_param_json(
                index_type="kOMEGA" if target_recall is not None else "kHNSW",
                ef_search=ef_search,
                topk=topk,
                target_recall=target_recall,
            ),
            query_file=query_file,
            topk=topk,
            use_refiner=use_refiner,
            reference_index_path=reference_index_path,
            metric_type=metric_type,
            dimension=dimension,
            m=m,
            ef_construction=ef_construction,
            bench_thread_count=bench_thread_count,
            bench_secs=bench_secs,
        )
        ret, output = run_command_capture(
            [str(bench_bin), str(config_path)],
            dry_run=dry_run,
            extra_env=extra_env,
        )
        return ret, output, parse_bench_output(output)
    finally:
        if config_path.exists():
            config_path.unlink()


def run_recall(
    *,
    recall_bin: Path,
    index_file: Path,
    query_file: Path,
    groundtruth_file: Path,
    metric_type: str,
    dimension: int,
    m: int,
    ef_construction: int,
    quantize_type: str,
    ef_search: int,
    topk: int,
    use_refiner: bool,
    reference_index_path: Path | None,
    target_recall: float | None = None,
    dry_run: bool = False,
) -> tuple[int, str, float | None]:
    config_path = _temporary_config_path("recall", index_file.parent)
    try:
        _write_core_config(
            path=config_path,
            index_path=index_file,
            index_config_json=build_core_index_config_json(
                index_type="kOMEGA" if target_recall is not None else "kHNSW",
                metric_type=metric_type,
                dimension=dimension,
                m=m,
                ef_construction=ef_construction,
                quantize_type=quantize_type,
            ),
            query_param_json=build_core_query_param_json(
                index_type="kOMEGA" if target_recall is not None else "kHNSW",
                ef_search=ef_search,
                topk=topk,
                target_recall=target_recall,
            ),
            query_file=query_file,
            topk=topk,
            use_refiner=use_refiner,
            reference_index_path=reference_index_path,
            metric_type=metric_type,
            dimension=dimension,
            m=m,
            ef_construction=ef_construction,
            recall_thread_count=1,
            groundtruth_file=groundtruth_file,
        )
        ret, output = run_command_capture([str(recall_bin), str(config_path)], dry_run=dry_run)
        return ret, output, parse_recall_output(output, topk)
    finally:
        if config_path.exists():
            config_path.unlink()


def run_concurrency_benchmark(
    *,
    bench_bin: Path,
    index_files: dict[str, Path | None],
    dataset_artifacts: dict[str, Any],
    dataset_spec: dict[str, Any],
    common_args: dict[str, Any],
    target_recall: float | None,
    dry_run: bool,
) -> dict[str, Any]:
    ef_search = int(common_args["ef_search"])
    topk = int(common_args["k"])
    m = int(common_args["m"])
    ef_construction = int(common_args.get("ef_construction", 500))
    quantize_type = str(common_args.get("quantize_type", "UNDEFINED"))
    use_refiner = bool(common_args.get("is_using_refiner", False))
    duration = int(common_args["concurrency_duration"])
    thread_counts = [int(value) for value in str(common_args["num_concurrency"]).split(",") if value]

    best_summary: dict[str, Any] | None = None
    best_output = ""
    for thread_count in thread_counts:
        ret, output, summary = run_bench(
            bench_bin=bench_bin,
            index_file=index_files["primary"],
            query_file=dataset_artifacts["query_txt"],
            metric_type=dataset_spec["metric_type"],
            dimension=dataset_spec["dimension"],
            m=m,
            ef_construction=ef_construction,
            quantize_type=quantize_type,
            ef_search=ef_search,
            topk=topk,
            bench_thread_count=thread_count,
            bench_secs=duration,
            use_refiner=use_refiner,
            reference_index_path=index_files["reference"],
            target_recall=target_recall,
            dry_run=dry_run,
        )
        summary["thread_count"] = thread_count
        summary["retcode"] = ret
        if best_summary is None or (summary.get("qps") or 0.0) > (best_summary.get("qps") or 0.0):
            best_summary = summary
            best_output = output

    return {"summary": best_summary or {}, "output": best_output}


def run_profile_benchmark(
    *,
    bench_bin: Path,
    index_files: dict[str, Path | None],
    dataset_artifacts: dict[str, Any],
    dataset_spec: dict[str, Any],
    common_args: dict[str, Any],
    target_recall: float | None,
    dry_run: bool,
    extra_env: dict[str, str] | None,
) -> tuple[int, str, dict[str, Any]]:
    return run_bench(
        bench_bin=bench_bin,
        index_file=index_files["primary"],
        query_file=dataset_artifacts["query_txt"],
        metric_type=dataset_spec["metric_type"],
        dimension=dataset_spec["dimension"],
        m=int(common_args["m"]),
        ef_construction=int(common_args.get("ef_construction", 500)),
        quantize_type=str(common_args.get("quantize_type", "UNDEFINED")),
        ef_search=int(common_args["ef_search"]),
        topk=int(common_args["k"]),
        bench_thread_count=1,
        bench_secs=max(1, int(common_args.get("profiling_duration", 1))),
        use_refiner=bool(common_args.get("is_using_refiner", False)),
        reference_index_path=index_files["reference"],
        target_recall=target_recall,
        dry_run=dry_run,
        extra_env=extra_env,
    )


def validate_profile_output(label: str, retcode: int, output: str, expected_prefix: str) -> None:
    if retcode != 0:
        raise RuntimeError(f"{label} profiling command failed with exit code {retcode}")
    if expected_prefix not in output:
        tail = "\n".join(output.splitlines()[-40:]) if output else "<empty output>"
        raise RuntimeError(
            f"{label} profiling output does not contain '{expected_prefix}'. "
            f"Last output lines:\n{tail}"
        )
