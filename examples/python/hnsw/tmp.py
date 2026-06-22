"""
Zvec Python API — PCA Index Search + Refiner
============================================

Primary search uses a PCA-reduced (320d) HNSW index.  One zvec query per
vector (topk=EF_SEARCH) feeds two comparison modes:

  Normal  : Take the first TOPK results from the PCA search directly.
  Refiner : Fetch original (768d) vectors for all EF_SEARCH candidates via
            collection.fetch() from the refiner collection, compute exact
            cosine similarity against the original query vector, re-rank to
            TOPK.

Data flow
---------
  PCA query  (320d)  -->  cohere-1m-pca-exp  -->  EF_SEARCH candidate IDs
                                                         |
  Original query (768d)  <--  test.parquet              |
                                    |                    v
                            refiner_rerank()  <--  cohere-10m-exp.fetch(ids)
                                    |
                                    v
                              TOPK final IDs  -->  recall@TOPK

Configuration:
    COLLECTION_PATH      : /root/code/VectorDBBench/db/cohere-1m-pca-exp
    REFINER_COLL_PATH    : /root/code/VectorDBBench/db/cohere-10m-exp
    QUERY_FILE           : .../cohere_pca_1m/test.parquet       (320d PCA queries)
    ORIGINAL_QUERY_FILE  : .../cohere_medium_1m/test.parquet    (768d original queries)
    GROUNDTRUTH_FILE     : .../cohere_pca_1m/neighbors.parquet
    TopK                 : 100
    EF_SEARCH            : 180  (also used as refiner candidate count)

Usage::

    conda activate baseline
    python tmp.py
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

import zvec
from zvec import (
    CollectionOption,
    HnswQueryParam,
    LogLevel,
    LogType,
    Query,
)

# ==================== Configuration ====================

# Primary HNSW index — PCA-reduced vectors (320d)
COLLECTION_PATH = "/root/code/VectorDBBench/db/openai-500k-pca-exp"
# Refiner index — original float32 vectors (768d), used for exact re-ranking
REFINER_COLL_PATH = "/root/code/VectorDBBench/db/openai-500k-exp"

# PCA query vectors (320d) for main HNSW search
QUERY_FILE = "/tmp/vectordb_bench/dataset/openai/openai_pca_500k/test.parquet"
# Original query vectors (768d) for refiner cosine re-ranking
ORIGINAL_QUERY_FILE = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k/test.parquet"

GROUNDTRUTH_FILE = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k/neighbors.parquet"

EMBEDDING_FIELD = "dense"   # VectorDBBench uses "dense" as the vector field name
TOPK = 100
EF_SEARCH = 180
MAX_QUERIES = 1000
WARMUP_ROUNDS = 0
MEASURE_ROUNDS = 1


# ==================== Parquet Loaders ====================

def load_queries_from_parquet(
    path: str,
    max_rows: Optional[int] = None,
) -> list[tuple[str, np.ndarray]]:
    """
    Load query vectors from a parquet file (columns: id, emb).

    Uses the *row index* as the query key (str) so that it matches the
    row-indexed ground truth returned by load_groundtruth_from_parquet().

    Returns:
        list of (row_index_str, float32_vector) pairs.
    """
    tbl = pq.read_table(path, columns=["id", "emb"], use_threads=True)
    emb_col = tbl.column("emb").combine_chunks()
    dim = len(emb_col[0].as_py())
    vecs = np.asarray(emb_col.values, dtype=np.float32).reshape(-1, dim)
    if max_rows is not None:
        vecs = vecs[:max_rows]
    return [(str(i), vecs[i]) for i in range(len(vecs))]


def load_orig_queries_from_parquet(
    path: str,
    max_rows: Optional[int] = None,
) -> Optional[list[np.ndarray]]:
    """
    Load original (non-PCA) query vectors from parquet, in row order.
    Returns None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    tbl = pq.read_table(path, columns=["id", "emb"], use_threads=True)
    emb_col = tbl.column("emb").combine_chunks()
    dim = len(emb_col[0].as_py())
    vecs = np.asarray(emb_col.values, dtype=np.float32).reshape(-1, dim)
    if max_rows is not None:
        vecs = vecs[:max_rows]
    return list(vecs)


def load_groundtruth_from_parquet(
    path: str,
    max_rows: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Load ground truth neighbors from parquet.

    Scans for the first list-typed column and treats each row as the neighbor
    ID list for the corresponding query (row-aligned with QUERY_FILE).

    Returns:
        dict mapping str(row_index) -> [str(neighbor_id), ...].
        Neighbor IDs are converted to str so they can be compared directly
        with the string IDs returned by collection.query().
    """
    tbl = pq.read_table(path, use_threads=True)
    print(f"  neighbors schema: {[f.name for f in tbl.schema]}")

    nb_col = None
    for col_name in tbl.schema.names:
        col = tbl.column(col_name).combine_chunks()
        if len(col) > 0:
            sample = col[0]
            if hasattr(sample, "as_py") and isinstance(sample.as_py(), list):
                nb_col = col
                print(
                    f"  Using column '{col_name}' for neighbors, "
                    f"{len(col[0].as_py())} per query"
                )
                break

    if nb_col is None:
        raise ValueError(f"No list column found in {path}")

    limit = min(max_rows, len(nb_col)) if max_rows else len(nb_col)
    return {
        str(i): [str(n) for n in nb_col[i].as_py()]
        for i in range(limit)
    }


# ==================== Refiner ====================

def refiner_rerank(
    query_orig: np.ndarray,
    candidate_ids: list[str],
    refiner_collection,
    topk: int,
) -> list[str]:
    """
    Re-rank *candidate_ids* using exact cosine similarity from the refiner
    collection (original, non-PCA vectors).

    Steps:
      1. collection.fetch(candidate_ids) — retrieve original vectors by ID.
      2. L2-normalise both query and fetched vectors, then compute dot product
         (= cosine similarity). Normalisation is mandatory because the raw
         Cohere embeddings are NOT unit-norm.
      3. Return top-*topk* IDs in descending similarity order.

    Args:
        query_orig       : Original query vector (may be un-normalised).
        candidate_ids    : Result IDs from the initial PCA HNSW search.
        refiner_collection: zvec Collection opened on original-vector index.
        topk             : Number of final results to return.

    Returns:
        List of top-*topk* document IDs after exact re-ranking.
    """
    docs = refiner_collection.fetch(candidate_ids, include_vector=True)

    # L2-normalise query once (outside the inner loop)
    q_norm_val = np.linalg.norm(query_orig)
    q = query_orig / q_norm_val if q_norm_val > 1e-10 else query_orig

    valid_ids: list[str] = []
    scores: list[float] = []

    for cid in candidate_ids:
        if cid in docs:
            vec_list = docs[cid].vector(EMBEDDING_FIELD)
            if vec_list is not None:
                v = np.asarray(vec_list, dtype=np.float32)
                # L2-normalise fetched vector (stored vectors may be un-normalised)
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    v = v / v_norm
                scores.append(float(np.dot(v, q)))
                valid_ids.append(cid)

    if not valid_ids:
        # Fallback: no vectors fetched, keep original order
        return candidate_ids[:topk]

    scores_arr = np.array(scores, dtype=np.float32)
    k = min(topk, len(scores_arr))
    top_local = np.argpartition(scores_arr, -k)[-k:]
    top_local = top_local[np.argsort(scores_arr[top_local])[::-1]]
    return [valid_ids[i] for i in top_local]


# ==================== Main ====================

def main() -> None:
    print("=" * 65)
    print("  Zvec — PCA Index Search + Exact Refiner")
    print("=" * 65)

    # ---- Step 1: Init zvec ----
    print("\n[Step 1] Initializing zvec ...")
    zvec.init(log_type=LogType.CONSOLE, log_level=LogLevel.INFO)
    print("  Done.")

    # ---- Step 2: Open primary (PCA) collection ----
    print(f"\n[Step 2] Opening primary collection: {COLLECTION_PATH}")
    collection = zvec.open(
        path=COLLECTION_PATH,
        option=CollectionOption(read_only=True, enable_mmap=True),
    )
    # Auto-detect actual vector field name (overrides EMBEDDING_FIELD if needed)
    global EMBEDDING_FIELD
    vec_fields = collection.schema.vectors
    if vec_fields:
        detected = vec_fields[0].name
        if detected != EMBEDDING_FIELD:
            print(f"  [Info] Auto-detected vector field '{detected}' "
                  f"(config was '{EMBEDDING_FIELD}'), using '{detected}'.")
        EMBEDDING_FIELD = detected
    print(f"  Collection : {collection.schema.name}")
    print(f"  VectorField: {EMBEDDING_FIELD}  dim={vec_fields[0].dimension if vec_fields else '?'}")
    print(f"  Doc count  : {collection.stats.doc_count:,}")
    print(f"  TopK       : {TOPK}")
    print(f"  ef_search  : {EF_SEARCH}")

    # ---- Step 2b: Open refiner collection ----
    refiner_collection = None
    if os.path.exists(REFINER_COLL_PATH):
        print(f"\n[Step 2b] Opening refiner collection: {REFINER_COLL_PATH}")
        refiner_collection = zvec.open(
            path=REFINER_COLL_PATH,
            option=CollectionOption(read_only=True, enable_mmap=True),
        )
        print(f"  Refiner collection : {refiner_collection.schema.name}")
        print(f"  Refiner doc count  : {refiner_collection.stats.doc_count:,}")
    else:
        print(f"\n[Step 2b] Refiner collection not found ({REFINER_COLL_PATH}), "
              f"refiner disabled.")

    # ---- Step 3: Load PCA query vectors ----
    print(f"\n[Step 3] Loading PCA queries from: {QUERY_FILE}")
    t0 = time.perf_counter()
    queries = load_queries_from_parquet(QUERY_FILE, max_rows=MAX_QUERIES)
    num_queries = len(queries)
    pca_dim = queries[0][1].shape[0] if queries else 0
    print(f"  Loaded {num_queries} queries, dim={pca_dim}, "
          f"耗时 {time.perf_counter()-t0:.2f}s")

    # ---- Step 3b: Load original query vectors (for refiner) ----
    orig_queries: Optional[list[np.ndarray]] = None
    if refiner_collection is not None:
        print(f"\n[Step 3b] Loading original queries from: {ORIGINAL_QUERY_FILE}")
        t0 = time.perf_counter()
        orig_queries = load_orig_queries_from_parquet(
            ORIGINAL_QUERY_FILE, max_rows=MAX_QUERIES
        )
        if orig_queries is not None:
            orig_dim = orig_queries[0].shape[0]
            print(f"  Loaded {len(orig_queries)} original queries, dim={orig_dim}, "
                  f"耗时 {time.perf_counter()-t0:.2f}s")
        else:
            print(f"  Original query file not found — refiner disabled.")
            refiner_collection = None

    # ---- Step 4: Load ground truth ----
    gt: dict[str, list[str]] = {}
    if os.path.exists(GROUNDTRUTH_FILE):
        print(f"\n[Step 4] Loading ground truth from: {GROUNDTRUTH_FILE}")
        t0 = time.perf_counter()
        gt = load_groundtruth_from_parquet(GROUNDTRUTH_FILE, max_rows=MAX_QUERIES)
        print(f"  Loaded GT for {len(gt)} queries, 耗时 {time.perf_counter()-t0:.2f}s")
    else:
        print(f"\n[Step 4] Ground truth not found, skipping recall eval.")

    # ---- Step 5: Search and collect results ----
    #
    # ONE zvec query per vector with topk=EF_SEARCH.
    # Normal mode  : results[:TOPK]             — recall from PCA index alone.
    # Refiner mode : fetch original vectors via refiner_collection.fetch(),
    #               re-rank with exact cosine, return TOPK.
    #
    total_rounds = WARMUP_ROUNDS + MEASURE_ROUNDS
    print(
        f"\n[Step 5] Running {total_rounds} rounds "
        f"({WARMUP_ROUNDS} warmup + {MEASURE_ROUNDS} measured), "
        f"{num_queries} queries/round ..."
    )
    refiner_enabled = refiner_collection is not None and orig_queries is not None

    round_qps_list: list[float] = []
    round_recall_list: list[float] = []
    round_refiner_recall_list: list[float] = []
    first_results: list[Optional[str]] = []

    for rnd in range(total_rounds):
        is_warmup = rnd < WARMUP_ROUNDS
        label = "warmup" if is_warmup else f"measured-{rnd - WARMUP_ROUNDS + 1}"

        search_start = time.perf_counter()
        total_recall = 0.0
        total_refiner_recall = 0.0
        matched = 0
        matched_refiner = 0
        if not is_warmup:
            first_results.clear()

        for idx, (key, vec) in enumerate(queries):
            # Single zvec search: topk=EF_SEARCH to serve both modes
            vq = Query(
                field_name=EMBEDDING_FIELD,
                vector=vec.tolist(),
                param=HnswQueryParam(ef=EF_SEARCH),
            )
            results = collection.query(queries=vq, topk=EF_SEARCH)

            # Normal mode: first TOPK results as-is
            results_normal = results[:TOPK]

            if not is_warmup:
                first_results.append(results_normal[0].id if results_normal else None)

            if key in gt:
                gt_ids = set(gt[key][:TOPK])
                if gt_ids:
                    # ── Normal recall ──
                    hit = sum(1 for d in results_normal if d.id in gt_ids)
                    total_recall += hit / len(gt_ids)
                    matched += 1

                    # ── Refiner recall ──
                    if refiner_enabled and results:
                        cand_ids = [d.id for d in results]
                        reranked = refiner_rerank(
                            orig_queries[idx], cand_ids, refiner_collection, TOPK
                        )
                        hit_ref = sum(1 for rid in reranked if rid in gt_ids)
                        total_refiner_recall += hit_ref / len(gt_ids)
                        matched_refiner += 1

        search_elapsed = time.perf_counter() - search_start
        rnd_qps = num_queries / search_elapsed if search_elapsed > 0 else 0
        rnd_recall = (total_recall / matched * 100) if matched > 0 else 0.0
        rnd_ref_recall = (
            (total_refiner_recall / matched_refiner * 100)
            if matched_refiner > 0
            else 0.0
        )

        refiner_str = (
            f"  refiner_recall@{TOPK}={rnd_ref_recall:.2f}%"
            if refiner_enabled
            else ""
        )

        if is_warmup:
            print(
                f"  [Round {rnd + 1}/{total_rounds}] {label}: "
                f"QPS={rnd_qps:.1f}  "
                f"normal_recall@{TOPK}={rnd_recall:.2f}%"
                f"{refiner_str}  (discarded)"
            )
        else:
            round_qps_list.append(rnd_qps)
            round_recall_list.append(rnd_recall)
            round_refiner_recall_list.append(rnd_ref_recall)
            print(
                f"  [Round {rnd + 1}/{total_rounds}] {label}: "
                f"QPS={rnd_qps:.1f}  "
                f"normal_recall@{TOPK}={rnd_recall:.2f}%"
                f"{refiner_str}"
            )

    # Print first 10 first-results for quick inspection
    print(f"\n  First results (showing up to 10):")
    for i, rid in enumerate(first_results[:10]):
        qid = queries[i][0]
        print(f"    query={qid}  ->  first_result_id={rid}")

    # Save first results for cross-comparison
    result_file = "/tmp/pca_exp_first_results.txt"
    with open(result_file, "w") as f:
        for i, rid in enumerate(first_results):
            f.write(f"{queries[i][0]}\t{rid}\n")
    print(f"  Saved first results to {result_file}")

    # ---- Step 6: Compare with baseline first results ----
    baseline_result_file = "/tmp/int8_first_results.txt"
    if os.path.exists(baseline_result_file):
        print(f"\n[Step 6] Comparing with baseline ({baseline_result_file}) ...")
        baseline: dict[str, Optional[str]] = {}
        with open(baseline_result_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    baseline[parts[0]] = parts[1]

        same_count, diff_count = 0, 0
        diffs: list[tuple[str, Optional[str], Optional[str]]] = []
        for i, rid in enumerate(first_results):
            qid = queries[i][0]
            base_rid = baseline.get(qid)
            if rid == base_rid:
                same_count += 1
            else:
                diff_count += 1
                if len(diffs) < 10:
                    diffs.append((qid, base_rid, rid))

        total_compared = same_count + diff_count
        print(f"  Compared {total_compared} queries:")
        print(f"    Same  : {same_count} ({same_count / total_compared * 100:.1f}%)")
        print(f"    Diff  : {diff_count} ({diff_count / total_compared * 100:.1f}%)")
        if diffs:
            print("  First few differences:")
            for qid, base_rid, cur_rid in diffs:
                print(f"    query={qid}  baseline={base_rid}  pca_exp={cur_rid}")
    else:
        print(f"\n[Step 6] Baseline result file not found ({baseline_result_file}), skipped.")

    # ---- Step 7: Summary ----
    avg_qps = sum(round_qps_list) / len(round_qps_list) if round_qps_list else 0
    avg_recall = sum(round_recall_list) / len(round_recall_list) if round_recall_list else 0
    avg_ref_recall = (
        sum(round_refiner_recall_list) / len(round_refiner_recall_list)
        if round_refiner_recall_list
        else 0
    )
    min_qps = min(round_qps_list) if round_qps_list else 0
    max_qps = max(round_qps_list) if round_qps_list else 0

    print(f"\n[Step 7] Summary")
    print(f"  Warmup rounds        : {WARMUP_ROUNDS}")
    print(f"  Measured rounds      : {MEASURE_ROUNDS}")
    print(f"  Queries/round        : {num_queries}")
    print(f"  Avg QPS              : {avg_qps:.1f}  (min={min_qps:.1f}, max={max_qps:.1f})")
    if round_recall_list:
        print(f"  Avg normal recall@{TOPK} : {avg_recall:.2f}%  (PCA index, no refiner)")
    else:
        print(f"  Avg normal recall@{TOPK} : N/A (no ground truth)")
    if refiner_enabled and round_refiner_recall_list:
        gain = avg_ref_recall - avg_recall
        print(
            f"  Avg refiner recall@{TOPK}: {avg_ref_recall:.2f}%  "
            f"(candidates={EF_SEARCH} -> topk={TOPK})"
        )
        print(f"  Refiner gain         : {gain:+.2f}%")
    else:
        print(f"  Avg refiner recall@{TOPK}: N/A (refiner disabled)")

    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    main()
