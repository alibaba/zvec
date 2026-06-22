"""
Recall 对比测试：原始数据 vs PCA vs PCA+Refiner（cosine 场景）

三种模式：
  原始   : 用 768 维原始向量做 brute-force 余弦检索
  PCA    : 用降维后 PCA 向量做 brute-force 余弦检索
  Refiner: PCA 粗检索 top-REFINER_K 候选，再用原始向量精排 top-K

注意事项：
  - 余弦相似度 = L2 归一化后的内积
  - PCA 旋转时对原始向量做矩阵乘法（不归一化），cosine 正确流程参见 pca_transform_cosine.py
  - Refiner 精排使用归一化后的原始向量（同检索阶段保持一致）
"""

import os
import time
import numpy as np
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────
#  路径配置
# ──────────────────────────────────────────────
BASE_DIR   = "/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m"
PCA_DIR    = "/tmp/vectordb_bench/dataset/cohere/cohere_pca_1m"

# 原始数据库文件（优先使用 shuffle_train，不存在则 fallback 到 train）
_TRAIN_CANDIDATES = ["shuffle_train.parquet", "train.parquet"]

# ──────────────────────────────────────────────
#  检索配置
# ──────────────────────────────────────────────
N_QUERY   = 100    # 测试 query 数量
K         = 100     # 最终返回近邻数
REFINER_K = 180    # refiner 粗检索候选数（> K）
N_THREADS = 8      # 并行线程数


# ══════════════════════════════════════════════
#  数据 I/O
# ══════════════════════════════════════════════

def _find_train_file(base_dir):
    for name in _TRAIN_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"在 {base_dir} 中未找到训练数据文件: {_TRAIN_CANDIDATES}")


def read_vectors(path, max_rows=None):
    """
    读取 parquet 向量数据。
    返回: ids (np.int64, shape=[N]), vecs (np.float32, shape=[N, D])
    """
    tbl     = pq.read_table(path, columns=["id", "emb"], use_threads=True)
    ids     = tbl.column("id").to_numpy()
    emb_col = tbl.column("emb").combine_chunks()
    dim     = len(emb_col[0].as_py())
    vecs    = np.asarray(emb_col.values, dtype=np.float32).reshape(-1, dim)
    if max_rows is not None:
        ids  = ids[:max_rows]
        vecs = vecs[:max_rows]
    return ids, vecs


def read_neighbors(path, max_rows=None):
    """
    读取 ground truth neighbors.parquet。
    返回: list[list[int]]，每条 query 对应的 neighbor ID 列表（按相似度降序）
    """
    tbl = pq.read_table(path, use_threads=True)
    print(f"      neighbors schema: {[f.name for f in tbl.schema]}")

    for name in tbl.schema.names:
        col    = tbl.column(name).combine_chunks()
        sample = col[0]
        if hasattr(sample, "as_py") and isinstance(sample.as_py(), list):
            limit  = min(max_rows, len(col)) if max_rows else len(col)
            result = [col[i].as_py() for i in range(limit)]
            print(f"      ground truth 列='{name}', 每条 {len(result[0])} 个邻居")
            return result

    raise ValueError(f"在 {path} 中未找到 list 类型的 neighbors 列")


# ══════════════════════════════════════════════
#  向量运算工具
# ══════════════════════════════════════════════

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """行 L2 归一化（返回新数组）"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """返回分数最高的 k 个下标（降序排列）"""
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def recall_at_k(result_db_indices: np.ndarray,
                db_ids: np.ndarray,
                gt_ids: list,
                k: int) -> float:
    """
    计算 recall@k（基于 ID 集合匹配）。
    result_db_indices: 检索返回的数据库行下标（已排序，取前 k 个）
    db_ids           : 数据库 ID 数组
    gt_ids           : ground truth neighbor ID 列表（取前 k 个）
    """
    found  = set(db_ids[result_db_indices[:k]].tolist())
    gt_set = set(gt_ids[:k])
    return len(found & gt_set) / k


# ══════════════════════════════════════════════
#  多线程 worker
# ══════════════════════════════════════════════

def process_queries(query_indices,
                    db_orig_norm, db_pca_norm, db_ids,
                    q_orig_norm, q_pca_norm,
                    gt_neighbors, k, refiner_k):
    """
    对给定 query 子集执行三种检索，返回各模式的 recall 列表。

    三种模式：
      1. 原始数据 brute-force
      2. PCA 数据 brute-force
      3. PCA 粗检索 refiner_k 个候选 → 原始数据精排 → top-k
    """
    r_orig, r_pca, r_ref = [], [], []

    for i in query_indices:
        gt  = gt_neighbors[i]
        q_o = q_orig_norm[i]    # (D_orig,)
        q_p = q_pca_norm[i]     # (D_pca,)

        # ── 模式 1：原始数据 ──
        scores_orig = db_orig_norm @ q_o        # (N,)
        idx_orig    = topk_indices(scores_orig, k)
        r_orig.append(recall_at_k(idx_orig, db_ids, gt, k))

        # ── 模式 2：PCA 数据 ──
        scores_pca = db_pca_norm @ q_p          # (N,)
        idx_pca    = topk_indices(scores_pca, k)
        r_pca.append(recall_at_k(idx_pca, db_ids, gt, k))

        # ── 模式 3：PCA 粗检索 + 原始精排 ──
        cand_idx      = topk_indices(scores_pca, refiner_k)          # (refiner_k,)
        rerank_scores = db_orig_norm[cand_idx] @ q_o                 # (refiner_k,)
        top_k_local   = topk_indices(rerank_scores, k)               # (k,)
        r_ref.append(recall_at_k(cand_idx[top_k_local], db_ids, gt, k))

    return r_orig, r_pca, r_ref


# ══════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════

def main():
    SEP = "=" * 65
    print(SEP)
    print("  Recall 对比：原始数据  |  PCA  |  PCA+Refiner  (cosine)")
    print(SEP)
    print(f"  N_QUERY={N_QUERY}, K={K}, REFINER_K={REFINER_K}, N_THREADS={N_THREADS}")
    print()

    # ── 1. 加载原始数据库向量 ──
    print("[1/5] 加载原始数据库向量...")
    t0 = time.time()
    train_path  = _find_train_file(BASE_DIR)
    db_ids, db_orig = read_vectors(train_path)
    print(f"      文件={os.path.basename(train_path)}, shape={db_orig.shape}, "
          f"耗时 {time.time()-t0:.1f}s")

    # ── 2. 加载 PCA 数据库向量 ──
    print("[2/5] 加载 PCA 数据库向量...")
    t0 = time.time()
    _, db_pca = read_vectors(os.path.join(PCA_DIR, "train.parquet"))
    print(f"      shape={db_pca.shape}, 耗时 {time.time()-t0:.1f}s")

    # ── 3. 加载测试 query（前 N_QUERY 条）──
    print(f"[3/5] 加载测试 query（前 {N_QUERY} 条）...")
    t0 = time.time()
    _, q_orig = read_vectors(os.path.join(BASE_DIR, "test.parquet"), max_rows=N_QUERY)
    _, q_pca  = read_vectors(os.path.join(PCA_DIR,  "test.parquet"), max_rows=N_QUERY)
    print(f"      原始={q_orig.shape}, PCA={q_pca.shape}, 耗时 {time.time()-t0:.1f}s")

    # ── 4. 加载 ground truth ──
    print("[4/5] 加载 ground truth...")
    t0 = time.time()
    gt = read_neighbors(
        os.path.join(BASE_DIR, "neighbors.parquet"), max_rows=N_QUERY
    )
    print(f"      共 {len(gt)} 条 query 的 ground truth, 耗时 {time.time()-t0:.1f}s")

    # ── 5. L2 归一化（余弦相似度 = 归一化内积）──
    print("[5/5] L2 归一化...")
    t0 = time.time()
    db_orig_norm = l2_normalize(db_orig)
    db_pca_norm  = l2_normalize(db_pca)
    q_orig_norm  = l2_normalize(q_orig)
    q_pca_norm   = l2_normalize(q_pca)
    del db_orig, db_pca, q_orig, q_pca
    print(f"      耗时 {time.time()-t0:.1f}s")
    print()

    # ── 6. 多线程检索 ──
    print(f"开始多线程检索 ({N_THREADS} 线程 × {N_QUERY} queries)...")
    t0 = time.time()

    # 轮询分配：query i 分给线程 i % N_THREADS
    chunks = [list(range(i, N_QUERY, N_THREADS)) for i in range(N_THREADS)]

    all_r_orig, all_r_pca, all_r_ref = [], [], []

    with ThreadPoolExecutor(max_workers=N_THREADS) as exe:
        futures = [
            exe.submit(
                process_queries,
                chunk,
                db_orig_norm, db_pca_norm, db_ids,
                q_orig_norm, q_pca_norm,
                gt, K, REFINER_K,
            )
            for chunk in chunks
            if chunk
        ]
        for fut in as_completed(futures):
            r_o, r_p, r_r = fut.result()
            all_r_orig.extend(r_o)
            all_r_pca.extend(r_p)
            all_r_ref.extend(r_r)

    elapsed = time.time() - t0

    # ── 7. 汇总输出 ──
    d_orig = db_orig_norm.shape[1]
    d_pca  = db_pca_norm.shape[1]

    header_fmt = f"  {{:<30}}  {{:>10}}  {{:>7}}  {{:>8}}  {{:>7}}"
    row_fmt    = f"  {{:<30}}  {{:>9.2f}}%  {{:>6.1f}}%  {{:>7.1f}}%  {{:>6.1f}}%"

    print()
    print(SEP)
    print(f"  检索完成  耗时={elapsed:.2f}s  ({N_QUERY} queries)")
    print()
    print(header_fmt.format("模式", f"Recall@{K}", "min", "median", "max"))
    print(f"  {'-'*30}  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*7}")

    for label, vals in [
        (f"原始数据 ({d_orig}d)",           all_r_orig),
        (f"PCA 数据 ({d_pca}d)",            all_r_pca),
        (f"PCA+Refiner ({REFINER_K}→{K})", all_r_ref),
    ]:
        a = np.array(vals)
        print(row_fmt.format(
            label,
            a.mean() * 100,
            a.min()  * 100,
            np.median(a) * 100,
            a.max()  * 100,
        ))

    print(SEP)


if __name__ == "__main__":
    main()
