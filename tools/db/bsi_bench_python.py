# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public Python API companion to the native issue-868 benchmark.

Scalar timings include Collection.query and result materialization. They must
not be confused with the native tool's isolated InvertedColumnIndexer timings.
"""

from __future__ import annotations

# This CLI emits structured benchmark records and progress to stdout.
# ruff: noqa: T201
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import zvec


def mix(values):
    with np.errstate(over="ignore"):
        x = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> 31)


def timestamps(ids):
    return (mix(np.asarray(ids, dtype=np.uint64) + 868) & 0x7FFFFFFF).astype(np.int32)


def vectors(ids):
    ids = np.asarray(ids, dtype=np.uint64).reshape(-1, 1)
    bits = mix(ids * 64 + np.arange(62, dtype=np.uint64) + (868 ^ 0x98C81))
    return (bits >> 40).astype(np.float32) * np.float32(1.0 / 8388608.0) - np.float32(1)


def emit(args, event, **values):
    print(
        json.dumps(
            {
                "event": event,
                "mode": args.mode,
                "layout": args.layout,
                "rows": args.rows,
                "api": "python_collection",
                **values,
            }
        ),
        flush=True,
    )


def build(args, params):
    if args.path.exists():
        raise RuntimeError("Build directory already exists")
    hybrid = args.mode == "hybrid_build"
    field = zvec.FieldSchema(
        "dataTime", zvec.DataType.INT32, nullable=True, index_param=params
    )
    schema = zvec.CollectionSchema("issue868", fields=[] if hybrid else [field])
    if hybrid:
        schema = zvec.CollectionSchema(
            "issue868",
            fields=[
                zvec.FieldSchema(
                    "pid",
                    zvec.DataType.INT32,
                    nullable=True,
                    index_param=zvec.InvertIndexParam(True),
                )
            ],
            vectors=[
                zvec.VectorSchema(
                    "vec",
                    zvec.DataType.VECTOR_FP32,
                    62,
                    index_param=zvec.HnswIndexParam(
                        metric_type=zvec.MetricType.COSINE,
                        m=16,
                        ef_construction=200,
                        quantize_type=zvec.QuantizeType.INT8,
                    ),
                )
            ],
        )
    collection = zvec.create_and_open(str(args.path), schema)
    if hybrid:
        collection.add_column(field)
    start = time.perf_counter()
    for begin in range(0, args.rows, 512):
        ids = np.arange(begin, min(begin + 512, args.rows), dtype=np.uint64)
        times = timestamps(ids).tolist()
        vector_values = vectors(ids).tolist() if hybrid else None
        docs = []
        for offset, value in enumerate(times):
            fields = {"dataTime": value}
            if hybrid:
                fields["pid"] = (begin + offset) % 1024
            docs.append(
                zvec.Doc(
                    id=str(begin + offset),
                    fields=fields,
                    vectors={"vec": vector_values[offset]} if hybrid else {},
                )
            )
        statuses = collection.insert(docs)
        if not all(status.ok() for status in statuses):
            raise RuntimeError(f"Insertion failed: {statuses}")
        if begin // 100000 != (begin + len(ids)) // 100000:
            print(
                f"progress {begin + len(ids)} {time.perf_counter() - start:.3f}s",
                flush=True,
            )
    emit(args, "insert", seconds=time.perf_counter() - start)
    start = time.perf_counter()
    collection.optimize()
    collection.close()
    size = sum(path.stat().st_size for path in args.path.rglob("*") if path.is_file())
    emit(args, "optimize", seconds=time.perf_counter() - start, bytes=size)


def query(args):
    collection = zvec.open(str(args.path), option=zvec.CollectionOption(read_only=True))
    assert collection.schema.field("dataTime").index_param.layout.name == args.layout
    hybrid = args.mode == "hybrid_query"
    vector_queries = [
        zvec.Query("vec", vector=v, param=zvec.HnswQueryParam(ef=300))
        for v in vectors(np.arange(args.rows + 10000, args.rows + 10256)).tolist()
    ]

    def execute(i):
        return collection.query(
            queries=vector_queries[i % 256] if hybrid else None,
            topk=21,
            filter=None if args.unfiltered else f"dataTime > {args.bound}",
            output_fields=["pid", "dataTime"] if hybrid else [],
        )

    signatures = []
    for i in range(256 if hybrid else 1):
        found = execute(i)
        assert len(found) == 21
        ids = [int(doc.id) for doc in found]
        assert all(0 <= doc_id < args.rows for doc_id in ids)
        assert args.unfiltered or np.all(timestamps(ids) > args.bound)
        signatures.append("".join(f"{doc.id}," for doc in found))
    if args.reference:
        path = Path(args.reference)
        if path.exists():
            assert path.read_text().splitlines() == signatures, "ANN layouts disagree"
        else:
            path.write_text("\n".join(signatures) + "\n")
    emit(
        args,
        "ann_verified_256_queries" if hybrid else "scalar_result_verified",
        seconds=0,
    )
    for i in range(64):
        execute(i)
    for round_id in range(args.rounds):
        barrier = threading.Barrier(args.threads + 1)
        start = [0.0]

        def worker(offset, barrier=barrier, start=start):
            samples = []
            barrier.wait()
            while time.perf_counter() - start[0] < args.seconds:
                before = time.perf_counter()
                found = execute(offset)
                assert len(found) == 21
                del found
                samples.append((time.perf_counter() - before) * 1000)
                offset += args.threads
            return samples

        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            tasks = [pool.submit(worker, i) for i in range(args.threads)]
            start[0] = time.perf_counter()
            barrier.wait()
            samples = [sample for task in tasks for sample in task.result()]
            wall = time.perf_counter() - start[0]
        emit(
            args,
            "query",
            threads=args.threads,
            round=round_id,
            bound=args.bound,
            unfiltered=args.unfiltered,
            queries=len(samples),
            wall_s=wall,
            qps=len(samples) / wall,
            mean_ms=float(np.mean(samples)),
            p50_ms=float(np.percentile(samples, 50)),
            p95_ms=float(np.percentile(samples, 95)),
            p99_ms=float(np.percentile(samples, 99)),
        )
    collection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--bound", type=int, default=536870911)
    parser.add_argument("--reference", default="")
    parser.add_argument("--unfiltered", action="store_true")
    args = parser.parse_args()
    zvec.init(
        log_level=zvec.LogLevel.ERROR,
        memory_limit_mb=8192,
        query_threads=8,
        optimize_threads=8,
        invert_to_forward_scan_ratio=0.9,
    )
    params = zvec.InvertIndexParam(layout=getattr(zvec.InvertIndexLayout, args.layout))
    if args.mode.endswith("_build"):
        build(args, params)
    elif args.mode == "switch":
        collection = zvec.open(str(args.path))
        start = time.perf_counter()
        collection.create_index("dataTime", params)
        collection.close()
        size = sum(
            path.stat().st_size for path in args.path.rglob("*") if path.is_file()
        )
        emit(
            args,
            "rebuild_scalar_index",
            seconds=time.perf_counter() - start,
            bytes=size,
        )
    else:
        query(args)


if __name__ == "__main__":
    main()
