#!/usr/bin/env python3
"""Diagnose per-query RSS growth without retaining query results."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import sys
import time

import numpy as np
from benchmark import proc_snapshot


def emit(event: str, **values: object) -> None:
    sys.stdout.write(json.dumps({"event": event, **values}, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mmap", "buffer_pool"), required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--memory-mb", type=int, default=128)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--list-size", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--malloc-trim", action="store_true")
    args = parser.parse_args()

    zvec = importlib.import_module("zvec")

    zvec.init(
        memory_limit_mb=args.memory_mb,
        query_threads=1,
        optimize_threads=1,
        log_type=zvec.LogType.CONSOLE,
        log_level=zvec.LogLevel.INFO,
    )
    data = np.load(args.query_file)
    ids = data["ids"]
    vectors = data["vectors"]
    emit("soak_ready", mode=args.mode, memory=proc_snapshot())
    collection = zvec.open(
        path=args.collection,
        option=zvec.CollectionOption(read_only=True, enable_mmap=args.mode == "mmap"),
    )
    emit("soak_open", mode=args.mode, memory=proc_snapshot())

    libc = ctypes.CDLL(None)
    malloc_trim = getattr(libc, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int

    digest = hashlib.sha256()
    total_queries = 0
    total_hits = 0
    for batch in range(1, args.batches + 1):
        started = time.monotonic()
        for expected_id, vector in zip(ids, vectors, strict=True):
            result = collection.query(
                queries=zvec.Query(
                    field_name="embedding",
                    vector=vector.tolist(),
                    param=zvec.DiskAnnQueryParam(list_size=args.list_size),
                ),
                topk=args.topk,
                output_fields=[],
                include_vector=False,
            )
            current_ids = [doc.id for doc in result]
            digest.update("\0".join(current_ids).encode())
            digest.update(b"\n")
            total_hits += str(int(expected_id)) in current_ids
            total_queries += 1
            del result, current_ids

        elapsed = time.monotonic() - started
        gc.collect()
        before_trim = proc_snapshot()
        trim_result = None
        if args.malloc_trim and malloc_trim is not None:
            trim_result = malloc_trim(0)
        emit(
            "soak_batch",
            mode=args.mode,
            batch=batch,
            batch_queries=len(ids),
            qps=len(ids) / elapsed,
            total_queries=total_queries,
            self_recall_at_k=total_hits / total_queries,
            before_trim=before_trim,
            trim_result=trim_result,
            after_trim=proc_snapshot(),
        )

    emit(
        "soak_done",
        mode=args.mode,
        total_queries=total_queries,
        result_fingerprint=digest.hexdigest(),
        memory=proc_snapshot(),
    )


if __name__ == "__main__":
    main()
