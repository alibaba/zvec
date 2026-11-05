"""HNSW first builds, deletion compaction, and subsequent graph reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import zvec


@pytest.mark.parametrize("enable_mmap", [True, False])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.COSINE, zvec.MetricType.IP]
)
def test_hnsw_optimize_rebuild_preserves_vectors_and_ids(tmp_path, enable_mmap, metric):
    zvec.init()
    rng = np.random.default_rng(731)
    data = rng.normal(size=(312, 37)).astype(np.float32)
    data *= (np.arange(len(data)) % 7 + 1)[:, None]
    expected = {f"doc-{i}": vector for i, vector in enumerate(data)}
    schema = zvec.CollectionSchema(
        name="hnsw_optimize",
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                37,
                index_param=zvec.HnswIndexParam(
                    metric_type=metric, m=16, ef_construction=64
                ),
            )
        ],
    )
    path = str(tmp_path / "collection")
    collection = zvec.create_and_open(
        path, schema, zvec.CollectionOption(enable_mmap=enable_mmap)
    )
    try:
        for start in range(0, len(data), 64):
            statuses = collection.insert(
                [
                    zvec.Doc(id=f"doc-{i}", vectors={"vector": data[i].tolist()})
                    for i in range(start, min(start + 64, len(data)))
                ]
            )
            assert all(status.ok() for status in statuses)
        collection.close()
        collection = zvec.open(path)

        def check():
            query = data[-2]
            hits = collection.query(
                zvec.Query(
                    "vector",
                    vector=query.tolist(),
                    param=zvec.HnswQueryParam(is_linear=True),
                ),
                topk=1000,
                include_vector=True,
            )
            assert collection.stats.doc_count == len(expected)
            assert {hit.id for hit in hits} == set(expected)
            for hit in hits:
                vector = expected[hit.id]
                np.testing.assert_allclose(
                    hit.vector("vector"), vector, rtol=2e-6, atol=2e-6
                )
                if metric == zvec.MetricType.L2:
                    score = np.sum((vector - query) ** 2)
                elif metric == zvec.MetricType.COSINE:
                    score = 1 - vector @ query / (
                        np.linalg.norm(vector) * np.linalg.norm(query)
                    )
                else:
                    score = vector @ query
                np.testing.assert_allclose(hit.score, score, rtol=2e-5, atol=2e-5)

        # Reopened Flat input exercises the first graph build.
        collection.optimize(zvec.OptimizeOption(concurrency=4))
        check()
        collection.close()
        collection = zvec.open(path)
        check()

        # More than 30% tombstones force physical deletion compaction. The
        # upsert supplies a Flat tail alongside the old graph during rebuild.
        removed = [f"doc-{i}" for i in range(128)]
        assert all(status.ok() for status in collection.delete(removed))
        for doc_id in removed:
            del expected[doc_id]
        replacement = rng.normal(size=37).astype(np.float32)
        assert collection.upsert(
            zvec.Doc(id="doc-311", vectors={"vector": replacement.tolist()})
        ).ok()
        expected["doc-311"] = replacement
        collection.optimize(zvec.OptimizeOption(concurrency=4))
        check()
        collection.close()
        collection = zvec.open(path)
        check()

        # A later optimize can reuse the packed graph as a mutable merge base.
        assert collection.insert(
            zvec.Doc(id="appended", vectors={"vector": data[0].tolist()})
        ).ok()
        expected["appended"] = data[0]
        collection.optimize(zvec.OptimizeOption(concurrency=2))
        check()
        fetched = collection.fetch(["doc-128", "doc-311", "appended", "doc-0"])
        assert set(fetched) == {"doc-128", "doc-311", "appended"}
        assert not list(Path(path).rglob("*.hnsw-*"))
    finally:
        collection.close()
