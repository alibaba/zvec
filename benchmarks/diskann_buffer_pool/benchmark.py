#!/usr/bin/env python3
"""DiskANN mmap versus Buffer Pool benchmark through the public Python API."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def proc_snapshot() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSize"}:
                values[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass

    try:
        fields = Path("/proc/self/stat").read_text().split()
        values["minor_faults"] = int(fields[9])
        values["major_faults"] = int(fields[11])
    except (OSError, ValueError, IndexError):
        pass

    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"read_bytes", "write_bytes", "syscr", "syscw"}:
                values[key] = int(value.strip())
    except (OSError, ValueError):
        pass
    return values


class RssSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "RssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def _sample(self) -> None:
        rss = proc_snapshot().get("VmRSS", 0)
        self.peak_rss = max(self.peak_rss, rss)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(latencies_ns: list[int], wall_seconds: float) -> dict[str, Any]:
    latencies_ms = [value / 1_000_000.0 for value in latencies_ns]
    return {
        "queries": len(latencies_ns),
        "wall_seconds": wall_seconds,
        "qps": len(latencies_ns) / wall_seconds if wall_seconds else 0.0,
        "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "max_ms": max(latencies_ms, default=0.0),
    }


def build_workload_plan(
    vectors: np.ndarray,
    repetitions: int,
    workload: str,
    hot_queries: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a deterministic request sequence shared by both storage modes."""
    query_count = len(vectors)
    if query_count == 0 or repetitions <= 0:
        raise ValueError("workload requires queries and repetitions > 0")

    request_count = query_count * repetitions
    rng = np.random.default_rng(seed)
    query_ids = np.arange(query_count, dtype=np.int64)
    hot_ids = np.empty(0, dtype=np.int64)

    if workload == "uniform_unique":
        # Every query appears exactly once per cycle; a fixed permutation
        # avoids giving any storage mode a favorable source-file order.
        cycle = rng.permutation(query_ids)
        requests = np.tile(cycle, repetitions)
    elif workload == "semantic80":
        hot_count = hot_queries or max(1, int(np.ceil(query_count * 0.20)))
        if hot_count >= query_count:
            raise ValueError("semantic80 hot query count must be < query count")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.maximum(norms, np.finfo(np.float32).tiny)
        seed_ids = np.linspace(
            0, query_count - 1, num=min(5, query_count), dtype=np.int64
        )
        similarities = normalized @ normalized[seed_ids].T
        semantic_scores = similarities.max(axis=1)
        semantic_order = np.argsort(-semantic_scores, kind="stable")
        hot_ids = semantic_order[:hot_count]
        tail_ids = semantic_order[hot_count:]
        choose_hot = rng.random(request_count) < 0.80
        requests = np.empty(request_count, dtype=np.int64)
        requests[choose_hot] = rng.choice(hot_ids, choose_hot.sum())
        requests[~choose_hot] = rng.choice(tail_ids, (~choose_hot).sum())
    elif workload == "zipf1":
        ranked_ids = rng.permutation(query_ids)
        weights = 1.0 / np.arange(1, query_count + 1, dtype=np.float64)
        weights /= weights.sum()
        ranks = rng.choice(query_count, size=request_count, p=weights)
        requests = ranked_ids[ranks]
    elif workload == "exact90":
        hot_count = hot_queries or min(5, query_count)
        if hot_count >= query_count:
            raise ValueError("exact90 hot query count must be < query count")
        shuffled = rng.permutation(query_ids)
        hot_ids = shuffled[:hot_count]
        tail_ids = shuffled[hot_count:]
        choose_hot = rng.random(request_count) < 0.90
        requests = np.empty(request_count, dtype=np.int64)
        requests[choose_hot] = rng.choice(hot_ids, choose_hot.sum())
        requests[~choose_hot] = rng.choice(tail_ids, (~choose_hot).sum())
    else:
        raise ValueError(f"unknown workload: {workload}")

    counts = np.bincount(requests, minlength=query_count)
    top_count = max(1, int(np.ceil(query_count * 0.20)))
    top_share = float(np.sort(counts)[-top_count:].sum() / request_count)
    hot_share = 0.0
    if len(hot_ids):
        hot_share = float(counts[hot_ids].sum() / request_count)
    return requests, {
        "workload": workload,
        "request_count": int(request_count),
        "unique_queries": int(np.count_nonzero(counts)),
        "query_top20_share": top_share,
        "configured_hot_share": hot_share,
        "hot_query_count": int(len(hot_ids)),
        "workload_seed": seed,
    }


def result_fingerprint(results: list[list[str]]) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update("\0".join(result).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def emit(event: str, **values: Any) -> None:
    payload = {"event": event, "monotonic_ns": time.monotonic_ns(), **values}
    print(json.dumps(payload, sort_keys=True), flush=True)


def import_zvec() -> Any:
    import zvec

    return zvec


def make_schema(zvec: Any, dimension: int) -> Any:
    from zvec.typing import QuantizeType

    return zvec.CollectionSchema(
        name="diskann_buffer_pool_benchmark",
        fields=[zvec.FieldSchema("ordinal", zvec.DataType.INT64, nullable=False)],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=zvec.DiskAnnIndexParam(
                    metric_type=zvec.MetricType.L2,
                    max_degree=64,
                    list_size=100,
                    pq_chunk_num=0,
                    quantize_type=QuantizeType.UNDEFINED,
                ),
            )
        ],
    )


def build_collection(args: argparse.Namespace) -> None:
    zvec = import_zvec()
    zvec.init(
        memory_limit_mb=args.build_memory_mb,
        query_threads=args.threads,
        optimize_threads=args.threads,
        log_type=zvec.LogType.CONSOLE,
        log_level=zvec.LogLevel.INFO,
    )

    path = Path(args.collection)
    if path.exists():
        raise RuntimeError(f"collection already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    query_rng = np.random.default_rng(args.seed + 1)
    query_indices = sorted(
        int(value)
        for value in query_rng.choice(
            args.count, size=min(args.query_vectors, args.count), replace=False
        )
    )
    query_index_set = set(query_indices)
    query_vectors: dict[int, np.ndarray] = {}

    collection = zvec.create_and_open(
        path=str(path),
        schema=make_schema(zvec, args.dimension),
        option=zvec.CollectionOption(
            read_only=False, enable_mmap=args.mode == "mmap"
        ),
    )
    emit("build_open", mode=args.mode, memory=proc_snapshot())

    insert_start = time.monotonic()
    for first in range(0, args.count, args.insert_batch):
        size = min(args.insert_batch, args.count - first)
        vectors = rng.standard_normal((size, args.dimension), dtype=np.float32)
        docs = []
        for local_index, vector in enumerate(vectors):
            ordinal = first + local_index
            if ordinal in query_index_set:
                query_vectors[ordinal] = vector.copy()
            docs.append(
                zvec.Doc(
                    id=str(ordinal),
                    fields={"ordinal": ordinal},
                    vectors={"embedding": vector.tolist()},
                )
            )
        results = collection.insert(docs)
        if len(results) != size or not all(result.ok() for result in results):
            raise RuntimeError(f"insert failed at document {first}")
        if first == 0 or first + size == args.count or (first + size) % 10_000 == 0:
            emit(
                "build_insert_progress",
                inserted=first + size,
                elapsed_seconds=time.monotonic() - insert_start,
                memory=proc_snapshot(),
            )

    collection.flush()
    emit(
        "build_insert_done",
        elapsed_seconds=time.monotonic() - insert_start,
        memory=proc_snapshot(),
    )

    optimize_start = time.monotonic()
    collection.optimize(option=zvec.OptimizeOption(concurrency=args.threads))
    emit(
        "build_optimize_done",
        elapsed_seconds=time.monotonic() - optimize_start,
        memory=proc_snapshot(),
        stats={
            "doc_count": collection.stats.doc_count,
            "index_completeness": dict(collection.stats.index_completeness),
        },
    )

    ordered_vectors = np.stack([query_vectors[index] for index in query_indices])
    query_file = Path(args.query_file)
    query_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        query_file,
        ids=np.asarray(query_indices, dtype=np.int64),
        vectors=ordered_vectors,
    )
    del collection
    gc.collect()
    emit("build_done", mode=args.mode, memory=proc_snapshot())


def run_query_pass(
    collection: Any,
    zvec: Any,
    ids: np.ndarray,
    vectors: np.ndarray,
    request_indices: np.ndarray,
    list_size: int,
    topk: int,
    client_threads: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    if client_threads <= 0:
        raise ValueError("client_threads must be > 0")
    result_ids: list[list[str] | None] = [None] * len(request_indices)

    def worker(thread_id: int) -> tuple[list[int], int]:
        latencies_ns: list[int] = []
        self_hits = 0
        for position in range(thread_id, len(request_indices), client_threads):
            query_index = int(request_indices[position])
            expected_id = ids[query_index]
            vector = vectors[query_index]
            query = zvec.Query(
                field_name="embedding",
                vector=vector.tolist(),
                param=zvec.DiskAnnQueryParam(list_size=list_size),
            )
            query_start = time.perf_counter_ns()
            result = collection.query(
                queries=query,
                topk=topk,
                output_fields=[],
                include_vector=False,
            )
            latencies_ns.append(time.perf_counter_ns() - query_start)
            current_ids = [doc.id for doc in result]
            result_ids[position] = current_ids
            if str(int(expected_id)) in current_ids:
                self_hits += 1
        return latencies_ns, self_hits

    start = time.monotonic()
    worker_results: list[tuple[list[int], int]] = []
    if client_threads == 1:
        worker_results.append(worker(0))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=client_threads
        ) as executor:
            futures = [executor.submit(worker, tid) for tid in range(client_threads)]
            worker_results = [future.result() for future in futures]
    wall_seconds = time.monotonic() - start

    latencies_ns = [
        latency
        for thread_latencies, _ in worker_results
        for latency in thread_latencies
    ]
    self_hits = sum(hits for _, hits in worker_results)
    if any(value is None for value in result_ids):
        raise RuntimeError("workload execution left incomplete query results")
    summary = latency_summary(latencies_ns, wall_seconds)
    summary["self_recall_at_k"] = self_hits / len(latencies_ns)
    return summary, [value for value in result_ids if value is not None]


def query_collection(args: argparse.Namespace) -> None:
    zvec = import_zvec()
    zvec.init(
        memory_limit_mb=args.memory_mb,
        query_threads=args.threads,
        optimize_threads=1,
        log_type=zvec.LogType.CONSOLE,
        log_level=zvec.LogLevel.INFO,
    )
    query_data = np.load(args.query_file)
    ids = query_data["ids"]
    vectors = query_data["vectors"]
    first_requests, first_workload = build_workload_plan(
        vectors,
        repetitions=1,
        workload=args.workload,
        hot_queries=args.hot_queries,
        seed=args.workload_seed,
    )
    steady_requests, steady_workload = build_workload_plan(
        vectors,
        repetitions=args.repetitions,
        workload=args.workload,
        hot_queries=args.hot_queries,
        seed=args.workload_seed,
    )
    emit(
        "process_ready",
        mode=args.mode,
        platform=platform.platform(),
        machine=platform.machine(),
        memory_budget_mb=args.memory_mb,
        engine_threads=args.threads,
        client_threads=args.client_threads,
        workload=steady_workload,
        memory=proc_snapshot(),
    )

    open_start = time.monotonic()
    with RssSampler() as open_sampler:
        collection = zvec.open(
            path=args.collection,
            option=zvec.CollectionOption(
                read_only=True, enable_mmap=args.mode == "mmap"
            ),
        )
    emit(
        "open_done",
        mode=args.mode,
        elapsed_seconds=time.monotonic() - open_start,
        peak_rss=open_sampler.peak_rss,
        memory=proc_snapshot(),
        persisted_enable_mmap=collection.option.enable_mmap,
    )

    with RssSampler() as first_sampler:
        first_summary, first_results = run_query_pass(
            collection,
            zvec,
            ids,
            vectors,
            request_indices=first_requests,
            list_size=args.list_size,
            topk=args.topk,
            client_threads=args.client_threads,
        )
    first_summary.update(first_workload)
    emit(
        "first_pass_done",
        mode=args.mode,
        metrics=first_summary,
        peak_rss=first_sampler.peak_rss,
        memory=proc_snapshot(),
    )

    with RssSampler() as steady_sampler:
        steady_summary, steady_results = run_query_pass(
            collection,
            zvec,
            ids,
            vectors,
            request_indices=steady_requests,
            list_size=args.list_size,
            topk=args.topk,
            client_threads=args.client_threads,
        )
    steady_summary.update(steady_workload)
    emit(
        "steady_pass_done",
        mode=args.mode,
        metrics=steady_summary,
        peak_rss=steady_sampler.peak_rss,
        memory=proc_snapshot(),
        result_fingerprint=result_fingerprint(first_results + steady_results),
    )

    del collection
    gc.collect()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    emit(
        "query_done",
        mode=args.mode,
        memory=proc_snapshot(),
        rusage={
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
            "max_rss_kib": usage.ru_maxrss,
            "minor_faults": usage.ru_minflt,
            "major_faults": usage.ru_majflt,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--mode", choices=("mmap", "buffer_pool"), required=True)
    build.add_argument("--collection", required=True)
    build.add_argument("--query-file", required=True)
    build.add_argument("--count", type=int, default=100_000)
    build.add_argument("--dimension", type=int, default=128)
    build.add_argument("--insert-batch", type=int, default=1_000)
    build.add_argument("--query-vectors", type=int, default=200)
    build.add_argument("--seed", type=int, default=20260807)
    build.add_argument("--threads", type=int, default=4)
    build.add_argument("--build-memory-mb", type=int, default=4_096)
    build.set_defaults(func=build_collection)

    query = subparsers.add_parser("query")
    query.add_argument("--mode", choices=("mmap", "buffer_pool"), required=True)
    query.add_argument("--collection", required=True)
    query.add_argument("--query-file", required=True)
    query.add_argument("--memory-mb", type=int, default=128)
    query.add_argument("--threads", type=int, default=1)
    query.add_argument("--client-threads", type=int, default=1)
    query.add_argument("--list-size", type=int, default=100)
    query.add_argument("--topk", type=int, default=10)
    query.add_argument("--repetitions", type=int, default=5)
    query.add_argument(
        "--workload",
        choices=("uniform_unique", "semantic80", "zipf1", "exact90"),
        default="uniform_unique",
    )
    query.add_argument("--hot-queries", type=int, default=0)
    query.add_argument("--workload-seed", type=int, default=20260825)
    query.set_defaults(func=query_collection)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
