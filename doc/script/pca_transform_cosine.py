"""
对 openai 向量数据进行 PCA 降维（cosine 正确版本）：
  1. L2 归一化训练数据
  2. 在归一化数据上做标准 PCA，获取旋转矩阵 V
  3. 对原始数据通过 V 旋转
  4. 截取前 HALF_DIM 维
"""
import os
import shutil
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import time

BASE_DIR = "/tmp/vectordb_bench/dataset/openai/openai_medium_500k"
OUT_DIR  = "/tmp/vectordb_bench/dataset/openai/openai_pca_500k"
HALF_DIM = 256  # 1536 / 2

os.makedirs(OUT_DIR, exist_ok=True)


def read_vectors_fast(path):
    """快速读取 parquet 中的向量数据"""
    t = pq.read_table(path, columns=["id", "emb"], use_threads=True)
    ids = t.column("id").to_numpy()
    emb_col = t.column("emb").combine_chunks()
    values = np.asarray(emb_col.values, dtype=np.float32).reshape(-1, 1536)
    return ids, values


def l2_normalize(vecs):
    """L2 归一化"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)


def write_parquet_fast(ids, vecs, path):
    """快速写入 id + emb 格式的 parquet 文件（避免 tolist）"""
    n, d = vecs.shape
    # 直接从 numpy buffer 构造 pyarrow array
    flat = pa.array(vecs.ravel(), type=pa.float32())
    offsets = np.arange(0, (n + 1) * d, d, dtype=np.int64)
    emb_col = pa.LargeListArray.from_arrays(offsets, flat)
    table = pa.table({
        "id": pa.array(ids, type=pa.int64()),
        "emb": emb_col
    })
    pq.write_table(table, path, use_dictionary=False)
    print(f"      写入 {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")


# ── 1. 读取训练数据 ──
print("[1/6] 读取训练数据...")
t0 = time.time()
train_ids, train_raw = read_vectors_fast(os.path.join(BASE_DIR, "shuffle_train.parquet"))
n_train, n_dims = train_raw.shape
print(f"      train shape: ({n_train}, {n_dims}), 耗时 {time.time()-t0:.1f}s")

# ── 2. L2 归一化 ──
print("[2/6] L2 归一化训练数据...")
t0 = time.time()
train_normed = l2_normalize(train_raw.copy())
print(f"      归一化完成, 耗时 {time.time()-t0:.1f}s")

# ── 3. 标准 PCA 训练，获取旋转矩阵 ──
print(f"[3/6] 标准 PCA 训练 (n_components={HALF_DIM})，获取旋转矩阵...")
t0 = time.time()
pca = PCA(n_components=HALF_DIM, svd_solver='randomized', random_state=42)
pca.fit(train_normed)
rotation = pca.components_.astype(np.float32)  # (HALF_DIM, 1536)
del train_normed; gc.collect()
print(f"      PCA fit 完成, 耗时 {time.time()-t0:.1f}s")
print(f"      解释方差比累计: {pca.explained_variance_ratio_.sum()*100:.4f}%")
print(f"      旋转矩阵 shape: {rotation.shape}")

# ── 4. 对原始训练数据旋转 + 截断 ──
print(f"[4/6] 对原始训练数据旋转并截取前 {HALF_DIM} 维...")
t0 = time.time()
train_pca = np.ascontiguousarray(train_raw @ rotation.T)  # (N, HALF_DIM)
del train_raw; gc.collect()
print(f"      train_pca shape: {train_pca.shape}, 耗时 {time.time()-t0:.1f}s")

# ── 5. 对原始测试数据旋转 + 截断 ──
print(f"[5/6] 对原始测试数据旋转并截取前 {HALF_DIM} 维...")
t0 = time.time()
test_ids, test_raw = read_vectors_fast(os.path.join(BASE_DIR, "test.parquet"))
test_pca = np.ascontiguousarray(test_raw @ rotation.T)  # (M, HALF_DIM)
del test_raw; gc.collect()
print(f"      test_pca shape: {test_pca.shape}, 耗时 {time.time()-t0:.1f}s")

# ── 6. 写入 parquet（train 和 test 并行写入）──
print(f"[6/6] 写入输出目录: {OUT_DIR}")
t0 = time.time()
with ThreadPoolExecutor(max_workers=2) as pool:
    pool.submit(write_parquet_fast, train_ids, train_pca, os.path.join(OUT_DIR, "train.parquet"))
    pool.submit(write_parquet_fast, test_ids, test_pca, os.path.join(OUT_DIR, "test.parquet"))

for fname in ["neighbors.parquet", "scalar_labels.parquet"]:
    src = os.path.join(BASE_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT_DIR, fname))
        print(f"      复制 {fname}")

print(f"\n全部完成! 写入耗时 {time.time()-t0:.1f}s")
print(f"输出目录内容:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    print(f"  {f}: {os.path.getsize(fpath)/1024/1024:.1f} MB")
