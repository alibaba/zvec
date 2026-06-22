"""
Zvec Python API — INT8 + Random Rotation Query Example
=======================================================

Opens an INT8 + rotation enabled collection (built by int8_rotate_build.py),
runs vector searches, and evaluates recall against ground truth.

The reformer (CosineInt8Reformer) is automatically loaded from the stored
index meta during collection.open(), which rotates query vectors using the
same FhtKacRotator that was used during build.

Equivalent C++ config (search_current2.yaml):
    IndexConfig: '{"quantizer_param":{"type":"kInt8"},"metric_type":"kCosine","m":15,...}'
    QueryConfig: '{"index_type":"kHNSW","ef_search":180}'

Configuration:
    Collection : /root/data/cohere/1m/db/cohere_cosine_int8_rotate
    TopK       : 100
    QueryFile  : /root/data/cohere/1m/cohere_test_vector_1m.1000.norm.txt
    GroundTruth: /root/data/cohere/1m/neighbors.txt
    ef_search  : 180

Usage::

    conda activate baseline
    python int8_rotate_query.py
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

import zvec
from zvec import (
    CollectionOption,
    HnswQueryParam,
    LogLevel,
    LogType,
    Query,
)

# ==================== Configuration ====================

COLLECTION_PATH = "/root/data/cohere/1m/db/cohere_cosine_int8_rotate"
QUERY_FILE = "/root/data/cohere/1m/cohere_test_vector_1m.1000.norm.txt"
GROUNDTRUTH_FILE = "/root/data/cohere/1m/neighbors.txt"

DIMENSION = 768
TOPK = 100
EF_SEARCH = 180
MAX_QUERIES = 1000
WARMUP_ROUNDS = 1
MEASURE_ROUNDS = 3


# ==================== File Parsers ====================

def parse_query_file(
    path: str,
    dimension: int,
    first_sep: str = ";",
    second_sep: str = " ",
    max_queries: int = 0,
) -> list[tuple[Optional[str], np.ndarray]]:
    """Parse query file in ``key;v1 v2 v3 ...`` format."""
    queries: list[tuple[Optional[str], np.ndarray]] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(first_sep, 1)
            key = parts[0].strip() if parts else None

            vec_str = parts[1].strip().rstrip(first_sep).strip() if len(parts) > 1 else ""
            vec_strs = vec_str.split(second_sep) if vec_str else []
            vector = np.array([float(v) for v in vec_strs], dtype=np.float32)

            if len(vector) != dimension:
                print(f"  Warning: query {key} has dim={len(vector)}, "
                      f"expected {dimension}, skipping")
                continue

            queries.append((key, vector))
            if max_queries and len(queries) >= max_queries:
                break

    return queries


def parse_groundtruth_file(
    path: str,
    first_sep: str = ";",
    second_sep: str = " ",
) -> dict[str, list[str]]:
    """Parse ground truth file in ``key;id1 id2 id3 ...`` format."""
    gt: dict[str, list[str]] = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(first_sep, 1)
            key = parts[0].strip()

            ids_str = parts[1].strip().rstrip(first_sep).strip() if len(parts) > 1 else ""
            ids = ids_str.split(second_sep) if ids_str else []
            gt[key] = ids

    return gt


# ==================== Main ====================

def main() -> None:
    print("=" * 60)
    print("  Zvec Python API — INT8 + Rotate Query Example")
    print("=" * 60)

    # ---- Step 1: Init zvec ----
    print("\n[Step 1] Initializing zvec ...")
    zvec.init(log_type=LogType.CONSOLE, log_level=LogLevel.INFO)
    print("  Done.")

    # ---- Step 2: Open collection ----
    print(f"\n[Step 2] Opening collection: {COLLECTION_PATH}")
    collection = zvec.open(
        path=COLLECTION_PATH,
        option=CollectionOption(read_only=True, enable_mmap=True),
    )
    print(f"  Collection : {collection.schema.name}")
    print(f"  Doc count  : {collection.stats.doc_count:,}")
    print(f"  Dimension  : {DIMENSION}")
    print(f"  TopK       : {TOPK}")
    print(f"  ef_search  : {EF_SEARCH}")

    # ---- Step 3: Load queries ----
    print(f"\n[Step 3] Loading queries from: {QUERY_FILE}")
    queries = parse_query_file(QUERY_FILE, DIMENSION,
                                max_queries=MAX_QUERIES)
    num_queries = len(queries)
    print(f"  Loaded {num_queries} queries.")

    # ---- Step 4: Load ground truth ----
    gt: dict[str, list[str]] = {}
    if os.path.exists(GROUNDTRUTH_FILE):
        print(f"\n[Step 4] Loading ground truth from: {GROUNDTRUTH_FILE}")
        gt = parse_groundtruth_file(GROUNDTRUTH_FILE)
        print(f"  Loaded ground truth for {len(gt)} queries.")
    else:
        print(f"\n[Step 4] Ground truth not found, skipping recall eval.")

    # ---- Step 5: Search and collect first results ----
    total_rounds = WARMUP_ROUNDS + MEASURE_ROUNDS
    print(f"\n[Step 5] Running {total_rounds} rounds "
          f"({WARMUP_ROUNDS} warmup + {MEASURE_ROUNDS} measured), "
          f"{num_queries} queries/round ...")

    round_qps_list: list[float] = []
    round_recall_list: list[float] = []
    first_results: list[Optional[str]] = []  # first result id per query (last measured round)

    for rnd in range(total_rounds):
        is_warmup = rnd < WARMUP_ROUNDS
        label = "warmup" if is_warmup else f"measured-{rnd - WARMUP_ROUNDS + 1}"

        search_start = time.perf_counter()
        total_recall = 0.0
        matched = 0
        if not is_warmup:
            first_results.clear()

        for idx, (key, vec) in enumerate(queries):
            vq = Query(
                field_name="embedding",
                vector=vec.tolist(),
                param=HnswQueryParam(ef=EF_SEARCH),
            )
            results = collection.query(queries=vq, topk=TOPK)
            qid = key if key is not None else str(idx)

            # Record first result id
            if not is_warmup and results:
                first_results.append(results[0].id)
            elif not is_warmup:
                first_results.append(None)

            if qid in gt:
                gt_ids = set(gt[qid][:TOPK])
                if gt_ids:
                    hit = sum(1 for d in results if d.id in gt_ids)
                    recall = hit / len(gt_ids)
                    total_recall += recall
                    matched += 1

        search_elapsed = time.perf_counter() - search_start
        rnd_qps = num_queries / search_elapsed if search_elapsed > 0 else 0
        rnd_recall = (total_recall / matched * 100) if matched > 0 else 0.0

        if is_warmup:
            print(f"  [Round {rnd + 1}/{total_rounds}] {label}: "
                  f"QPS={rnd_qps:.1f}  recall@{TOPK}={rnd_recall:.2f}%  (discarded)")
        else:
            round_qps_list.append(rnd_qps)
            round_recall_list.append(rnd_recall)
            print(f"  [Round {rnd + 1}/{total_rounds}] {label}: "
                  f"QPS={rnd_qps:.1f}  recall@{TOPK}={rnd_recall:.2f}%")

    # Print first 10 first-results for quick inspection
    print(f"\n  First results (showing up to 10):")
    for i, rid in enumerate(first_results[:10]):
        qid = queries[i][0] if queries[i][0] is not None else str(i)
        print(f"    query={qid}  ->  first_result_id={rid}")

    # Save all first results to file for cross-comparison
    result_file = "/tmp/int8_rotate_first_results.txt"
    with open(result_file, "w") as f:
        for i, rid in enumerate(first_results):
            qid = queries[i][0] if queries[i][0] is not None else str(i)
            f.write(f"{qid}\t{rid}\n")
    print(f"  Saved first results to {result_file}")

    # ---- Step 6: Compare with INT8 (no-rotate) first results ----
    int8_result_file = "/tmp/int8_first_results.txt"
    if os.path.exists(int8_result_file):
        print(f"\n[Step 6] Comparing first results with INT8 (no-rotate) ...")
        int8_results: dict[str, Optional[str]] = {}
        with open(int8_result_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    int8_results[parts[0]] = parts[1]

        same_count = 0
        diff_count = 0
        diffs: list[tuple[str, Optional[str], Optional[str]]] = []
        for i, rid in enumerate(first_results):
            qid = queries[i][0] if queries[i][0] is not None else str(i)
            int8_rid = int8_results.get(qid)
            if rid == int8_rid:
                same_count += 1
            else:
                diff_count += 1
                if len(diffs) < 10:
                    diffs.append((qid, int8_rid, rid))

        total_compared = same_count + diff_count
        print(f"  Compared {total_compared} queries:")
        print(f"    Same first result : {same_count} ({same_count / total_compared * 100:.1f}%)")
        print(f"    Different         : {diff_count} ({diff_count / total_compared * 100:.1f}%)")
        if diffs:
            print(f"  First few differences:")
            for qid, int8_rid, rot_rid in diffs:
                print(f"    query={qid}  int8={int8_rid}  int8_rotate={rot_rid}")
    else:
        print(f"\n[Step 6] INT8 result file not found ({int8_result_file}),")
        print(f"          Run int8_query.py first to enable comparison.")

    # ---- Step 7: Summary ----
    avg_qps = sum(round_qps_list) / len(round_qps_list) if round_qps_list else 0
    avg_recall = sum(round_recall_list) / len(round_recall_list) if round_recall_list else 0
    min_qps = min(round_qps_list) if round_qps_list else 0
    max_qps = max(round_qps_list) if round_qps_list else 0

    print(f"\n[Step 7] Summary")
    print(f"  Warmup rounds   : {WARMUP_ROUNDS}")
    print(f"  Measured rounds : {MEASURE_ROUNDS}")
    print(f"  Queries/round   : {num_queries}")
    print(f"  Avg QPS         : {avg_qps:.1f}  (min={min_qps:.1f}, max={max_qps:.1f})")
    if round_recall_list:
        print(f"  Avg recall@{TOPK}  : {avg_recall:.2f}%")
    else:
        print(f"  Avg recall@{TOPK}  : N/A (no ground truth)")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
