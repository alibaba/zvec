from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import zvec
except ImportError:
    zvec = None


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
    omega_prediction_profile: dict[str, Any] | None = None


KV_PATTERN = re.compile(r"([A-Za-z_]+)=([^\s,]+)")
AVG_LINE_PATTERN = re.compile(r"Avg latency:\s*([0-9.]+)ms qps:\s*([0-9.]+)")
PERCENTILE_PATTERN = re.compile(r"(\d+)\s+Percentile:\s*([0-9.]+)\s+ms")
PROCESS_LINE_PATTERN = re.compile(
    r"Process query:\s*(\d+), total process time:\s*(\d+)ms, duration:\s*(\d+)ms"
)
RECALL_PATTERN = re.compile(r"Recall@(\d+):\s*([0-9.]+)")

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
    "test_small": {
        "dataset_dirname": "test_small",
        "dimension": 128,
        "metric_type": "COSINE",
        "train_files": ["train.parquet"],
    },
}

DATASET_DOWNLOAD_BASE_URLS = {
    "S3": "https://assets.zilliz.com/benchmark",
    "ALIYUNOSS": "https://assets.zilliz.com.cn/benchmark",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
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
    emit(f"\n{'=' * 70}")
    emit(title)
    emit("=" * 70)


def emit(message: str = "") -> None:
    sys.stdout.write(f"{message}\n")


def resolve_paths(
    script_path: Path,
    config: dict[str, Any],
    zvec_root_arg: str | None,
    benchmark_dir_arg: str | None,
) -> tuple[Path, Path]:
    zvec_root = (
        Path(zvec_root_arg).resolve() if zvec_root_arg else script_path.parent.parent
    )

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


def _numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _sum_numeric(records: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(records, key)
    if not values:
        return None
    return sum(values)


def _avg_numeric(records: list[dict[str, Any]], key: str) -> float | None:
    values = _numeric_values(records, key)
    if not values:
        return None
    return sum(values) / len(values)


def _percentile_numeric(
    records: list[dict[str, Any]], key: str, percentile: float
) -> float | None:
    values = sorted(_numeric_values(records, key))
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


def read_omega_prediction_profile_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid OMEGA profile JSONL at {path}:{lineno}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def summarize_omega_prediction_profile(
    profile_path: Path, benchmark: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    records = read_omega_prediction_profile_records(profile_path)
    if not records:
        return None

    query_count = len(records)
    saved_values = _numeric_values(records, "saved_comparisons")
    baseline_values = _numeric_values(records, "baseline_comparisons")
    omega_values = _numeric_values(records, "omega_comparisons")

    total_model_time_ns = _sum_numeric(records, "model_time_ns") or 0.0
    total_decision_time_ns = _sum_numeric(records, "decision_time_ns") or 0.0
    total_model_calls = _sum_numeric(records, "model_calls") or 0.0
    total_prediction_checks = _sum_numeric(records, "prediction_checks") or 0.0

    summary: dict[str, Any] = {
        "profile_path": str(profile_path),
        "profile_query_count": query_count,
        "profile_thread_count": (benchmark or {}).get("thread_count"),
        "profile_duration_s": (benchmark or {}).get("duration_s"),
        "metric_scope": "per_query",
        "saved_comparisons_definition": (
            "baseline_comparisons - omega_comparisons, where baseline_comparisons "
            "is collected by a profiling-only shadow HNSW run on the same query "
            "with OMEGA early stop disabled"
        ),
        "model_calls_total": int(total_model_calls),
        "prediction_checks_total": int(total_prediction_checks),
        "model_time_ns_total": int(total_model_time_ns),
        "decision_time_ns_total": int(total_decision_time_ns),
        "model_calls_per_query": total_model_calls / query_count,
        "prediction_checks_per_query": total_prediction_checks / query_count,
        "model_time_us_per_query": total_model_time_ns / query_count / 1000.0,
        "decision_time_us_per_query": total_decision_time_ns / query_count / 1000.0,
        "model_time_us_per_call": (
            total_model_time_ns / total_model_calls / 1000.0
            if total_model_calls > 0
            else None
        ),
        "decision_time_us_per_check": (
            total_decision_time_ns / total_prediction_checks / 1000.0
            if total_prediction_checks > 0
            else None
        ),
        "omega_comparisons_per_query": _avg_numeric(records, "omega_comparisons"),
        "omega_comparisons_p50": _percentile_numeric(records, "omega_comparisons", 50),
        "omega_comparisons_p95": _percentile_numeric(records, "omega_comparisons", 95),
        "early_stop_hit_count": sum(
            1 for record in records if bool(record.get("early_stop_hit"))
        ),
        "early_stop_hit_ratio": sum(
            1 for record in records if bool(record.get("early_stop_hit"))
        )
        / query_count,
    }

    if baseline_values:
        summary.update(
            {
                "baseline_profile_query_count": len(baseline_values),
                "baseline_comparisons_per_query": sum(baseline_values)
                / len(baseline_values),
                "saved_comparisons_per_query": (
                    sum(saved_values) / len(saved_values) if saved_values else None
                ),
                "saved_comparisons_p50": _percentile_numeric(
                    records, "saved_comparisons", 50
                ),
                "saved_comparisons_p95": _percentile_numeric(
                    records, "saved_comparisons", 95
                ),
                "saved_comparisons_total": (
                    int(sum(saved_values)) if saved_values else None
                ),
                "saved_comparisons_ratio": (
                    sum(saved_values) / sum(baseline_values)
                    if saved_values and sum(baseline_values) > 0
                    else None
                ),
            }
        )
    else:
        summary.update(
            {
                "baseline_profile_query_count": 0,
                "baseline_comparisons_per_query": None,
                "saved_comparisons_per_query": None,
                "saved_comparisons_p50": None,
                "saved_comparisons_p95": None,
                "saved_comparisons_total": None,
                "saved_comparisons_ratio": None,
            }
        )

    if omega_values and baseline_values:
        summary["comparison_count_delta_per_query"] = (
            sum(omega_values) / len(omega_values)
            - sum(baseline_values) / len(baseline_values)
        )

    return summary


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


def online_summary_path(index_path: Path) -> Path:
    return index_path / "online_benchmark_summary.json"


def write_online_summary(index_path: Path, payload: dict[str, Any]) -> None:
    with online_summary_path(index_path).open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def online_result_payload(result: BenchmarkResult) -> dict[str, Any]:
    payload = {
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
    }
    if result.omega_prediction_profile is not None:
        payload["omega_prediction_profile"] = result.omega_prediction_profile
    return payload


def write_grouped_online_summaries(
    dataset: str, results: list[BenchmarkResult]
) -> list[Path]:
    written_paths: list[Path] = []
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.path, []).append(result)

    for path_str, grouped_results in grouped.items():
        index_path = Path(path_str)
        write_online_summary(
            index_path,
            {
                "generated_at": datetime.now().isoformat(),
                "dataset": dataset,
                "results": [online_result_payload(result) for result in grouped_results],
            },
        )
        written_paths.append(online_summary_path(index_path))

    return written_paths


def offline_summary_path(index_path: Path) -> Path:
    return index_path / "offline_benchmark_summary.json"


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
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
            optimize_duration = round(
                old_optimize_duration - old_training_s + new_training_s, 4
            )
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
    summary = build_offline_summary(
        index_path, db_label, metrics, retrain_only=retrain_only
    )
    path = offline_summary_path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return path


def get_offline_load_duration(index_path: Path) -> float | None:
    return (
        read_json_if_exists(offline_summary_path(index_path))
        .get("offline", {})
        .get("load_duration_s")
    )


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
    metric_type = str(
        config.get("metric_type", default.get("metric_type", "COSINE"))
    ).upper()
    dataset_format = str(
        config.get("format", config.get("dataset_format", default.get("format", "parquet")))
    ).lower()
    remote_dirname = str(
        config.get("remote_dirname", default.get("remote_dirname", ""))
    )
    train_files = list(config.get("train_files", default.get("train_files", [])))
    base_files = list(config.get("base_files", default.get("base_files", [])))
    query_file = str(config.get("query_file", default.get("query_file", "")))
    groundtruth_file = str(
        config.get("groundtruth_file", default.get("groundtruth_file", ""))
    )
    query_count = config.get("query_count", default.get("query_count"))
    recall_query_count = config.get(
        "recall_query_count", default.get("recall_query_count")
    )
    dataset_source = str(
        config.get("dataset_source", os.environ.get("ZVEC_DATASET_SOURCE", "S3"))
    )
    download_base_url = str(
        config.get(
            "dataset_base_url",
            os.environ.get(
                "ZVEC_DATASET_BASE_URL",
                DATASET_DOWNLOAD_BASE_URLS.get(
                    dataset_source.upper(), DATASET_DOWNLOAD_BASE_URLS["S3"]
                ),
            ),
        )
    )
    if dimension <= 0:
        raise ValueError(f"Missing dataset dimension for {dataset_name}")

    dataset_dir = (dataset_root / dataset_dirname).resolve()
    return {
        "format": dataset_format,
        "dataset_root": dataset_root,
        "dataset_dir": dataset_dir,
        "dimension": dimension,
        "metric_type": metric_type,
        "remote_dirname": remote_dirname,
        "train_files": train_files,
        "base_files": base_files,
        "query_file": query_file,
        "groundtruth_file": groundtruth_file,
        "query_count": query_count,
        "recall_query_count": recall_query_count,
        "vector_dtype": str(
            config.get("vector_dtype", default.get("vector_dtype", "float32"))
        ),
        "insert_batch_size": int(
            config.get("insert_batch_size", default.get("insert_batch_size", 1000))
        ),
        "download_base_url": download_base_url.rstrip("/"),
    }


def _require_polars():
    if pl is None:
        raise RuntimeError(
            "This script requires polars in the active Python environment."
        )
    return pl


def _require_numpy():
    if np is None:
        raise RuntimeError(
            "Big-ann binary datasets require numpy in the active Python environment."
        )
    return np


def _require_zvec():
    if zvec is None:
        raise RuntimeError(
            "This script requires zvec in the active Python environment."
        )
    return zvec


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


def _dataset_required_files(
    dataset_name: str, dataset_spec: dict[str, Any]
) -> list[str]:
    if dataset_spec.get("format") == "bigann":
        required = list(dataset_spec.get("base_files", []))
        query_file = dataset_spec.get("query_file")
        groundtruth_file = dataset_spec.get("groundtruth_file")
        if query_file:
            required.append(str(query_file))
        if groundtruth_file:
            required.append(str(groundtruth_file))
        if not required or not dataset_spec.get("base_files"):
            raise ValueError(
                f"Big-ann dataset {dataset_name} must define base_files, "
                "query_file, and groundtruth_file"
            )
        return required

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
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_dataset_available(
    dataset_name: str, dataset_spec: dict[str, Any], dry_run: bool
) -> None:
    dataset_dir = dataset_spec["dataset_dir"]
    required_files = _dataset_required_files(dataset_name, dataset_spec)
    missing_files = [
        name for name in required_files if not (dataset_dir / name).exists()
    ]
    if not missing_files:
        return

    remote_dirname = dataset_spec.get("remote_dirname")
    if not remote_dirname:
        raise FileNotFoundError(
            f"Dataset directory is incomplete and auto-download is not configured: {dataset_dir}"
        )

    base_url = dataset_spec["download_base_url"]
    emit(
        f"Dataset files missing under {dataset_dir}, downloading from {base_url}/{remote_dirname} ..."
    )
    if dry_run:
        for name in missing_files:
            emit(
                f"[Dry-run] download {base_url}/{remote_dirname}/{name} -> {dataset_dir / name}"
            )
        return

    for name in missing_files:
        url = f"{base_url}/{remote_dirname}/{name}"
        output_path = dataset_dir / name
        emit(f"Downloading {url}")
        _download_file(url, output_path)


def prepare_dataset_artifacts(
    dataset_name: str,
    dataset_spec: dict[str, Any],
    benchmark_dir: Path,
    dry_run: bool = False,
) -> dict[str, Path]:
    if dataset_spec.get("format") == "bigann":
        return _prepare_bigann_dataset_artifacts(
            dataset_name, dataset_spec, benchmark_dir, dry_run=dry_run
        )
    if dataset_spec.get("format", "parquet") != "parquet":
        raise ValueError(f"Unsupported dataset format: {dataset_spec.get('format')}")

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
            raise FileNotFoundError(
                f"No train parquet files found under: {dataset_dir}"
            )

    cache_dir = (benchmark_dir / "_dataset_cache" / dataset_name).resolve()
    query_txt = cache_dir / "query.txt"
    gt_txt = cache_dir / "groundtruth.txt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        refresh_query = (
            not query_txt.exists()
            or query_txt.stat().st_mtime < query_parquet.stat().st_mtime
        )
        refresh_gt = (
            not gt_txt.exists()
        ) or gt_txt.stat().st_mtime < gt_parquet.stat().st_mtime
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


def _as_optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set")
    return parsed


def _numpy_dtype(dtype_name: str):
    module = _require_numpy()
    normalized = str(dtype_name).lower()
    mapping = {
        "float32": module.float32,
        "f32": module.float32,
        "uint8": module.uint8,
        "u8": module.uint8,
        "int8": module.int8,
        "i8": module.int8,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported big-ann vector dtype: {dtype_name}")
    return mapping[normalized]


def _read_xbin_header(path: Path) -> tuple[int, int]:
    module = _require_numpy()
    header = module.fromfile(path, dtype=module.uint32, count=2)
    if header.size != 2:
        raise ValueError(f"Invalid xbin header: {path}")
    return int(header[0]), int(header[1])


def _xbin_mmap(path: Path, dtype_name: str, maxn: int | None = None):
    module = _require_numpy()
    dtype = _numpy_dtype(dtype_name)
    n, d = _read_xbin_header(path)
    if maxn is not None:
        n = min(n, int(maxn))
    return module.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def _knn_ids_mmap(path: Path, maxn: int | None = None):
    module = _require_numpy()
    n, k = _read_xbin_header(path)
    if maxn is not None:
        n = min(n, int(maxn))
    return module.memmap(path, dtype=module.int32, mode="r", offset=8, shape=(n, k))


def _needs_refresh(output_path: Path, source_paths: list[Path]) -> bool:
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime
    return any(output_mtime < source_path.stat().st_mtime for source_path in source_paths)


def _prepare_bigann_dataset_artifacts(
    dataset_name: str,
    dataset_spec: dict[str, Any],
    benchmark_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    ensure_dataset_available(dataset_name, dataset_spec, dry_run)

    dataset_dir = dataset_spec["dataset_dir"]
    base_paths = [dataset_dir / name for name in dataset_spec["base_files"]]
    query_path = dataset_dir / str(dataset_spec["query_file"])
    gt_path = dataset_dir / str(dataset_spec["groundtruth_file"])
    vector_dtype = str(dataset_spec.get("vector_dtype", "float32"))
    dimension = int(dataset_spec["dimension"])

    query_count = _as_optional_positive_int(dataset_spec.get("query_count"), "query_count")
    recall_query_count = _as_optional_positive_int(
        dataset_spec.get("recall_query_count"), "recall_query_count"
    )

    if dry_run:
        resolved_query_count = query_count or 0
        resolved_recall_query_count = recall_query_count or resolved_query_count
    else:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        for base_path in base_paths:
            if not base_path.exists():
                raise FileNotFoundError(f"Missing big-ann base file: {base_path}")
            _, base_dimension = _read_xbin_header(base_path)
            if base_dimension != dimension:
                raise ValueError(
                    f"Big-ann base dimension mismatch for {base_path}: "
                    f"expected {dimension}, got {base_dimension}"
                )
        if not query_path.exists():
            raise FileNotFoundError(f"Missing big-ann query file: {query_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing big-ann ground-truth file: {gt_path}")

        query_n, query_dimension = _read_xbin_header(query_path)
        gt_n, _gt_k = _read_xbin_header(gt_path)
        if query_dimension != dimension:
            raise ValueError(
                f"Big-ann query dimension mismatch for {query_path}: "
                f"expected {dimension}, got {query_dimension}"
            )
        resolved_query_count = min(query_n, gt_n)
        if query_count is not None:
            resolved_query_count = min(resolved_query_count, query_count)
        resolved_recall_query_count = min(
            resolved_query_count, recall_query_count or resolved_query_count
        )

    cache_dir = (benchmark_dir / "_dataset_cache" / dataset_name).resolve()
    query_txt = cache_dir / f"query_{resolved_query_count}.txt"
    gt_txt = cache_dir / f"groundtruth_{resolved_query_count}.txt"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        if _needs_refresh(query_txt, [query_path]):
            _write_bigann_query_text(
                query_path,
                query_txt,
                vector_dtype=vector_dtype,
                query_count=resolved_query_count,
            )
        if _needs_refresh(gt_txt, [gt_path]):
            _write_bigann_groundtruth_text(
                gt_path, gt_txt, query_count=resolved_query_count
            )

    return {
        "format": "bigann",
        "dataset_dir": dataset_dir,
        "query_file": query_path,
        "gt_file": gt_path,
        "query_txt": query_txt,
        "groundtruth_txt": gt_txt,
        "train_files": base_paths,
        "query_count": resolved_query_count,
        "recall_query_count": resolved_recall_query_count,
        "vector_dtype": vector_dtype,
        "dimension": dimension,
    }


def _write_bigann_query_text(
    query_path: Path,
    output_path: Path,
    *,
    vector_dtype: str,
    query_count: int,
) -> None:
    vectors = _xbin_mmap(query_path, vector_dtype, maxn=query_count)
    with output_path.open("w") as f:
        for query_id in range(query_count):
            vector_text = " ".join(
                str(round(float(value), 16)) for value in vectors[query_id]
            )
            f.write(f"{query_id};{vector_text};\n")


def _write_bigann_groundtruth_text(
    gt_path: Path, output_path: Path, *, query_count: int
) -> None:
    ids = _knn_ids_mmap(gt_path, maxn=query_count)
    with output_path.open("w") as f:
        for query_id in range(query_count):
            neighbors = " ".join(str(int(value)) for value in ids[query_id])
            f.write(f"{query_id};{neighbors}\n")


def _write_query_text(query_parquet: Path, output_path: Path) -> None:
    polars = _require_polars()
    frame = polars.read_parquet(query_parquet).sort("id")
    with output_path.open("w") as f:
        for row in frame.iter_rows(named=True):
            vector = row["emb"]
            vector_text = " ".join(str(round(float(v), 16)) for v in vector)
            f.write(f"{int(row['id'])};{vector_text};\n")


def _write_groundtruth_text(gt_parquet: Path, output_path: Path) -> None:
    polars = _require_polars()
    frame = polars.read_parquet(gt_parquet).sort("id")
    with output_path.open("w") as f:
        for row in frame.iter_rows(named=True):
            neighbors = " ".join(str(int(v)) for v in row["neighbors_id"])
            f.write(f"{int(row['id'])};{neighbors}\n")


@lru_cache(maxsize=1)
def _initialized_zvec():
    module = _require_zvec()
    module.init(log_level=module.LogLevel.WARN)
    return module


def _quantize_type_from_name(name: str):
    module = _initialized_zvec()
    normalized = str(name).upper()
    mapping = {
        "": module.QuantizeType.UNDEFINED,
        "UNDEFINED": module.QuantizeType.UNDEFINED,
        "FP16": module.QuantizeType.FP16,
        "INT8": module.QuantizeType.INT8,
        "INT4": module.QuantizeType.INT4,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported quantize type: {name}")
    return mapping[normalized]


def _metric_type_from_name(name: str):
    module = _initialized_zvec()
    normalized = str(name).upper()
    mapping = {
        "COSINE": module.MetricType.COSINE,
        "IP": module.MetricType.IP,
        "L2": module.MetricType.L2,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported metric type: {name}")
    return mapping[normalized]


def _maybe_destroy_collection(path: Path) -> None:
    module = _initialized_zvec()
    if not path.exists():
        return
    try:
        module.open(str(path)).destroy()
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
    module = _initialized_zvec()
    quantize_type = _quantize_type_from_name(common_args.get("quantize_type", ""))
    metric = _metric_type_from_name(metric_type)
    if index_kind == "OMEGA":
        index_param = module.OmegaIndexParam(
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
        index_param = module.HnswIndexParam(
            metric_type=metric,
            m=int(common_args["m"]),
            ef_construction=int(specific_args.get("ef_construction", 500)),
            quantize_type=quantize_type,
        )

    return module.CollectionSchema(
        name=f"{index_kind.lower()}_benchmark",
        fields=[
            module.FieldSchema(
                "id",
                module.DataType.INT64,
                nullable=False,
                index_param=module.InvertIndexParam(enable_range_optimization=True),
            )
        ],
        vectors=[
            module.VectorSchema(
                "dense",
                module.DataType.VECTOR_FP32,
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
) -> tuple[dict[str, Any], Any]:
    if dry_run:
        emit(f"[Dry-run] Build {index_kind} at {index_path}")
        return (
            {
                "insert_duration": None,
                "optimize_duration": None,
                "load_duration": None,
            },
            None,
        )

    module = _initialized_zvec()

    if retrain_only:
        collection = module.open(
            str(index_path), module.CollectionOption(read_only=False, enable_mmap=True)
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
        collection = module.create_and_open(
            str(index_path),
            schema,
            module.CollectionOption(read_only=False, enable_mmap=True),
        )
        insert_duration = _insert_training_data(
            collection,
            dataset_artifacts,
            batch_size=int(dataset_spec.get("insert_batch_size", 1000)),
        )

    optimize_start = time.perf_counter()
    if retrain_only:
        collection.retrain_omega()
    else:
        collection.optimize(option=module.OptimizeOption())
    optimize_duration = time.perf_counter() - optimize_start
    with contextlib.suppress(Exception):
        collection.flush()

    # For OMEGA indexes, keep collection open to work around reopening bug
    # For other indexes, close as before
    if index_kind == "OMEGA":
        omega_collection = collection
    else:
        del collection
        omega_collection = None

    load_duration = None
    if insert_duration is not None:
        load_duration = insert_duration + optimize_duration
    elif optimize_duration is not None:
        load_duration = optimize_duration

    return (
        {
            "insert_duration": round(insert_duration, 4)
            if insert_duration is not None
            else None,
            "optimize_duration": round(optimize_duration, 4)
            if optimize_duration is not None
            else None,
            "load_duration": round(load_duration, 4)
            if load_duration is not None
            else None,
        },
        omega_collection,
    )


def _insert_training_data(
    collection, dataset_artifacts: dict[str, Any], batch_size: int = 1000
) -> float:
    if dataset_artifacts.get("format") == "bigann":
        return _insert_bigann_training_data(
            collection,
            dataset_artifacts["train_files"],
            vector_dtype=str(dataset_artifacts.get("vector_dtype", "float32")),
            dimension=int(dataset_artifacts["dimension"]),
            batch_size=batch_size,
        )

    module = _initialized_zvec()
    polars = _require_polars()
    start = time.perf_counter()
    train_files = dataset_artifacts["train_files"]
    for train_file in train_files:
        frame = polars.read_parquet(train_file)
        for offset in range(0, frame.height, batch_size):
            batch = frame.slice(offset, batch_size)
            ids = batch["id"].to_list()
            vectors = batch["emb"].to_list()
            docs = [
                module.Doc(
                    id=str(int(doc_id)),
                    fields={"id": int(doc_id)},
                    vectors={"dense": vector},
                )
                for doc_id, vector in zip(ids, vectors, strict=True)
            ]
            collection.insert(docs)
    return time.perf_counter() - start


def _insert_bigann_training_data(
    collection,
    train_files: list[Path],
    *,
    vector_dtype: str,
    dimension: int,
    batch_size: int,
) -> float:
    module = _initialized_zvec()
    start = time.perf_counter()
    next_id = 0
    for train_file in train_files:
        vectors = _xbin_mmap(train_file, vector_dtype)
        if vectors.shape[1] != dimension:
            raise ValueError(
                f"Big-ann base dimension mismatch for {train_file}: "
                f"expected {dimension}, got {vectors.shape[1]}"
            )
        for offset in range(0, vectors.shape[0], batch_size):
            end = min(offset + batch_size, vectors.shape[0])
            batch_vectors = vectors[offset:end].tolist()
            docs = [
                module.Doc(
                    id=str(next_id + offset + local_idx),
                    fields={"id": next_id + offset + local_idx},
                    vectors={"dense": vector},
                )
                for local_idx, vector in enumerate(batch_vectors)
            ]
            collection.insert(docs)
        next_id += vectors.shape[0]
    return time.perf_counter() - start


def compute_recall_with_zvec(
    *,
    index_kind: str,
    index_path: Path,
    dataset_artifacts: dict[str, Any],
    common_args: dict[str, Any],
    target_recall: float | None,
    dry_run: bool,
    collection: Any = None,
) -> float | None:
    if dry_run:
        return None
    if dataset_artifacts.get("format") == "bigann":
        return _compute_bigann_recall_with_zvec(
            index_kind=index_kind,
            index_path=index_path,
            dataset_artifacts=dataset_artifacts,
            common_args=common_args,
            target_recall=target_recall,
            collection=collection,
        )

    module = _initialized_zvec()
    polars = _require_polars()
    query_frame = polars.read_parquet(dataset_artifacts["query_parquet"]).sort("id")
    gt_frame = polars.read_parquet(dataset_artifacts["gt_parquet"]).sort("id")
    gt_map = {
        int(row["id"]): [
            int(value) for value in row["neighbors_id"][: int(common_args["k"])]
        ]
        for row in gt_frame.iter_rows(named=True)
    }

    # Use pre-opened collection if provided (workaround for OMEGA reopening bug)
    # Otherwise open the collection normally
    if collection is None:
        option = module.CollectionOption(
            read_only=(index_kind != "OMEGA"), enable_mmap=True
        )
        collection = module.open(str(index_path), option)
        should_close = True
    else:
        should_close = False
    use_refiner = bool(common_args.get("is_using_refiner", False))
    if index_kind == "OMEGA":
        query_param = module.OmegaQueryParam(
            ef=int(common_args["ef_search"]),
            target_recall=float(target_recall),
            is_using_refiner=use_refiner,
        )
    else:
        query_param = module.HnswQueryParam(
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
            vectors=module.VectorQuery(
                field_name="dense", vector=row["emb"], param=query_param
            ),
            topk=topk,
            output_fields=[],
        )
        pred = [int(doc.id) for doc in results[:topk]]
        recall_sum += len(set(pred) & set(gt)) / float(topk)
        query_count += 1

    if should_close:
        del collection
    if query_count == 0:
        return None
    return recall_sum / query_count


def _compute_bigann_recall_with_zvec(
    *,
    index_kind: str,
    index_path: Path,
    dataset_artifacts: dict[str, Any],
    common_args: dict[str, Any],
    target_recall: float | None,
    collection: Any = None,
) -> float | None:
    module = _initialized_zvec()
    query_count = min(
        int(dataset_artifacts["query_count"]),
        int(dataset_artifacts.get("recall_query_count") or dataset_artifacts["query_count"]),
    )
    query_vectors = _xbin_mmap(
        dataset_artifacts["query_file"],
        str(dataset_artifacts.get("vector_dtype", "float32")),
        maxn=query_count,
    )
    gt_ids = _knn_ids_mmap(dataset_artifacts["gt_file"], maxn=query_count)

    if collection is None:
        option = module.CollectionOption(
            read_only=(index_kind != "OMEGA"), enable_mmap=True
        )
        collection = module.open(str(index_path), option)
        should_close = True
    else:
        should_close = False

    use_refiner = bool(common_args.get("is_using_refiner", False))
    if index_kind == "OMEGA":
        query_param = module.OmegaQueryParam(
            ef=int(common_args["ef_search"]),
            target_recall=float(target_recall),
            is_using_refiner=use_refiner,
        )
    else:
        query_param = module.HnswQueryParam(
            ef=int(common_args["ef_search"]),
            is_using_refiner=use_refiner,
        )

    recall_sum = 0.0
    topk = int(common_args["k"])
    for query_id in range(query_count):
        gt = [int(value) for value in gt_ids[query_id, :topk]]
        results = collection.query(
            vectors=module.VectorQuery(
                field_name="dense",
                vector=query_vectors[query_id].tolist(),
                param=query_param,
            ),
            topk=topk,
            output_fields=[],
        )
        pred = [int(doc.id) for doc in results[:topk]]
        recall_sum += len(set(pred) & set(gt)) / float(topk)

    if should_close:
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
    primary = (
        coarse_candidates[0]
        if coarse_candidates
        else (full_candidates[0] if full_candidates else None)
    )
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
    emit(printable)
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
        sys.stdout.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            sys.stdout.write("\n")
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
        ret, output = run_command_capture(
            [str(recall_bin), str(config_path)], dry_run=dry_run
        )
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
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    ef_search = int(common_args["ef_search"])
    topk = int(common_args["k"])
    m = int(common_args["m"])
    ef_construction = int(common_args.get("ef_construction", 500))
    quantize_type = str(common_args.get("quantize_type", "UNDEFINED"))
    use_refiner = bool(common_args.get("is_using_refiner", False))
    duration = int(common_args["concurrency_duration"])
    thread_counts = [
        int(value) for value in str(common_args["num_concurrency"]).split(",") if value
    ]

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
            extra_env=extra_env,
        )
        summary["thread_count"] = thread_count
        summary["duration_s"] = duration
        summary["retcode"] = ret
        if best_summary is None or (summary.get("qps") or 0.0) > (
            best_summary.get("qps") or 0.0
        ):
            best_summary = summary
            best_output = output

    return {"summary": best_summary or {}, "output": best_output}
