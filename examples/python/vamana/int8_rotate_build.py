"""
Zvec Python API — Vamana INT8 + Random Rotation Build Example
==============================================================

Builds a zvec Collection with Vamana graph index, INT8 quantization and
random rotation (CosineInt8Converter + FhtKacRotator) enabled.

The key configuration is:
    quantize_type=QuantizeType.INT8, enable_rotate=True

This triggers the C++ CosineInt8Converter to:
  1. Create a FhtKacRotator (random orthogonal rotation matrix)
  2. Rotate all data vectors before INT8 quantization
  3. Store the rotator state in the index storage for search-side query rotation

The reformer (CosineInt8Reformer) is automatically loaded from storage during
collection.open(), ensuring query vectors are rotated with the same matrix.

Input : /root/data/cohere/1m/cohere_train_vector_1m.norm.zvec.vecs
Output: /root/data/cohere/1m/db/cohere_vamana_cosine_int8_rotate

Usage::

    conda activate baseline
    python int8_rotate_build.py
"""

from __future__ import annotations

import mmap
import os
import shutil
import struct
import time

import numpy as np

import zvec
from zvec import (
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    InvertIndexParam,
    LogLevel,
    LogType,
    MetricType,
    OptimizeOption,
    QuantizeType,
    QuantizerParam,
    VamanaIndexParam,
    VectorSchema,
)

# ==================== Configuration ====================

VECS_FILE = "/root/data/cohere/1m/cohere_train_vector_1m.norm.zvec.vecs"
COLLECTION_PATH = "/root/data/cohere/1m/db/cohere_vamana_cosine_int8_rotate"

DIMENSION = 768
METRIC_TYPE = MetricType.COSINE
MAX_DEGREE = 64
SEARCH_LIST_SIZE = 100

INSERT_BATCH_SIZE = 1000

# ==================== .zvec.vecs Parser ====================

VECS_HEADER_FMT = "<QHHI" + "Q" * 11
VECS_HEADER_SIZE = struct.calcsize(VECS_HEADER_FMT)


def parse_vecs_file(path: str):
    """Parse a .zvec.vecs file and return (num_vecs, meta_size, data_start, offsets)."""
    with open(path, "rb") as f:
        header_bytes = f.read(VECS_HEADER_SIZE)

    vals = struct.unpack(VECS_HEADER_FMT, header_bytes)
    num_vecs = vals[0]
    meta_size = vals[3]

    offsets = {
        "key":       (vals[5],  vals[6]),
        "dense":     (vals[7],  vals[8]),
        "sparse":    (vals[9],  vals[10]),
        "partition": (vals[11], vals[12]),
        "taglist":   (vals[13], vals[14]),
    }

    data_start = VECS_HEADER_SIZE + meta_size
    return num_vecs, meta_size, data_start, offsets


# ==================== Main ====================

def main() -> None:
    print("=" * 60)
    print("  Zvec Python API — Vamana INT8 + Rotate Build Example")
    print("=" * 60)

    # ---- Step 1: Init zvec ----
    print("\n[Step 1] Initializing zvec ...")
    zvec.init(log_type=LogType.CONSOLE, log_level=LogLevel.INFO)
    print("  Done.")

    # ---- Step 2: Parse .zvec.vecs header ----
    print(f"\n[Step 2] Parsing vecs file: {VECS_FILE}")
    num_vecs, meta_size, data_start, offsets = parse_vecs_file(VECS_FILE)
    dense_offset, dense_size = offsets["dense"]
    key_offset, key_size = offsets["key"]

    elem_size = dense_size // num_vecs
    vec_dim_floats = elem_size // 4
    print(f"  num_vecs: {num_vecs:,}, dim: {vec_dim_floats}")
    assert vec_dim_floats == DIMENSION

    # ---- Step 3: Create collection with INT8 + enable_rotate ----
    print(f"\n[Step 3] Creating collection at {COLLECTION_PATH} ...")
    print(f"  index_type    = Vamana (graph-based ANN)")
    print(f"  quantize_type = QuantizeType.INT8 + enable_rotate=True")
    print(f"  metric_type   = MetricType.COSINE")
    print(f"  → CosineInt8Converter + FhtKacRotator")

    index_param = VamanaIndexParam(
        metric_type=METRIC_TYPE,
        max_degree=MAX_DEGREE,
        search_list_size=SEARCH_LIST_SIZE,
        quantize_type=QuantizeType.INT8,
        quantizer_param=QuantizerParam(enable_rotate=True),
    )
    print(f"  index_param   = {index_param}")

    schema = zvec.CollectionSchema(
        name="cohere_vamana_cosine_int8_rotate",
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=index_param,
            ),
        ],
    )

    os.makedirs(os.path.dirname(COLLECTION_PATH), exist_ok=True)

    if os.path.exists(COLLECTION_PATH):
        print(f"  Removing existing collection ...")
        shutil.rmtree(COLLECTION_PATH)

    collection = zvec.create_and_open(
        path=COLLECTION_PATH,
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    print(f"  Collection created: {collection.schema.name}")

    # ---- Step 4: Read vectors via mmap and insert ----
    print(f"\n[Step 4] Inserting {num_vecs:,} vectors "
          f"(batch_size={INSERT_BATCH_SIZE}) ...")

    insert_start = time.perf_counter()
    total_inserted = 0

    with open(VECS_FILE, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            dense_abs = data_start + dense_offset
            key_abs = data_start + key_offset

            batch_docs: list[Doc] = []

            for i in range(num_vecs):
                key_val = struct.unpack_from("<Q", mm, key_abs + i * 8)[0]
                vec_offset = dense_abs + i * elem_size
                vec = np.frombuffer(
                    mm, dtype=np.float32, count=DIMENSION, offset=vec_offset
                ).tolist()

                doc = Doc(
                    id=str(key_val),
                    fields={"id": int(key_val)},
                    vectors={"embedding": vec},
                )
                batch_docs.append(doc)

                if len(batch_docs) >= INSERT_BATCH_SIZE:
                    results = collection.insert(batch_docs)
                    ok = sum(1 for r in results if r.ok())
                    total_inserted += ok
                    if (total_inserted % 50_000) == 0 or total_inserted == num_vecs:
                        elapsed = time.perf_counter() - insert_start
                        speed = total_inserted / elapsed if elapsed > 0 else 0
                        print(f"  [{total_inserted:>8,} / {num_vecs:,}] "
                              f"{speed:.0f} docs/s")
                    batch_docs.clear()

            if batch_docs:
                results = collection.insert(batch_docs)
                ok = sum(1 for r in results if r.ok())
                total_inserted += ok

    insert_elapsed = time.perf_counter() - insert_start
    print(f"\n  Insert complete: {total_inserted:,} docs "
          f"in {insert_elapsed:.1f}s "
          f"({total_inserted / insert_elapsed:.0f} docs/s)")

    # ---- Step 5: Optimize (build Vamana graph with INT8 + rotation) ----
    print(f"\n[Step 5] Optimizing collection (building Vamana graph with "
          f"INT8 + rotation) ...")
    opt_start = time.perf_counter()
    collection.optimize(option=OptimizeOption())
    opt_elapsed = time.perf_counter() - opt_start
    print(f"  Optimize done in {opt_elapsed:.1f}s")

    # ---- Step 6: Flush ----
    print(f"\n[Step 6] Flushing collection ...")
    collection.flush()
    print(f"  Doc count: {collection.stats.doc_count:,}")
    print("  Done.")

    print(f"\n{'=' * 60}")
    print(f"  Build complete!")
    print(f"  Collection saved to: {COLLECTION_PATH}")
    print(f"  Run int8_rotate_query.py to search and evaluate.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
