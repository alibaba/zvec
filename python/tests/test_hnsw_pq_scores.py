"""PQ inner-product scores and radii must use caller-facing dot products."""

import numpy as np
import pytest

import zvec


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("count", [300, 1100])
def test_hnsw_pq_inner_product_scores_and_radius(tmp_path, bits, rotate, count):
    rng = np.random.default_rng(903)
    vectors = rng.uniform(0.25, 1.5, size=(count, 16)).astype(np.float32)
    schema = zvec.CollectionSchema(
        "pq_ip_scores",
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                16,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.IP,
                    m=16,
                    ef_construction=100,
                    quantize_type=zvec.QuantizeType.PQ,
                    quantizer_param=zvec.QuantizerParam(
                        num_chunk=4,
                        num_bits=bits,
                        enable_rotate=rotate,
                        opq_iter=2,
                        opq_pq_iter=2,
                    ),
                ),
            )
        ],
    )
    path = str(tmp_path / "index")
    topk = 40
    writer = zvec.create_and_open(path, schema)
    query = zvec.Query("vector", vector=vectors[0], param=zvec.HnswQueryParam(ef=2000))
    try:
        for start in range(0, count, 200):
            assert all(
                status.ok()
                for status in writer.insert(
                    [
                        zvec.Doc(id=str(i), vectors={"vector": vectors[i]})
                        for i in range(start, min(start + 200, count))
                    ]
                )
            )
        writer.optimize()
        before = writer.query(query, topk=topk, output_fields=[])
        # Positive inputs have positive approximate inner products. Comparing
        # two query routes alone would miss a score-sign error shared by both.
        assert len(before) == topk
        assert all(doc.score > 0 for doc in before)
        assert [doc.score for doc in before] == sorted(
            (doc.score for doc in before), reverse=True
        )
    finally:
        writer.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True))
    try:
        after = reader.query(query, topk=topk, output_fields=[])
        assert [doc.id for doc in after] == [doc.id for doc in before]
        np.testing.assert_allclose(
            [doc.score for doc in after], [doc.score for doc in before]
        )
        for radius in (after[topk // 2].score, after[0].score + 1.0):
            limited = zvec.Query(
                "vector",
                vector=vectors[0],
                param=zvec.HnswQueryParam(ef=2000, radius=radius),
            )
            expected = {int(doc.id): doc.score for doc in after if doc.score >= radius}
            docs = reader.query(limited, topk=topk, output_fields=[])
            assert {int(doc.id): doc.score for doc in docs} == expected
            ids, scores = reader.query_internal_ids(
                limited, topk=topk, return_scores=True
            )
            assert dict(zip(ids[ids >= 0], scores[ids >= 0])) == expected
            assert np.all(ids[len(expected) :] == -1)
            assert np.all(np.isnan(scores[len(expected) :]))
    finally:
        reader.close()
