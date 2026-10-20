"""Public IVF PQ configuration, persistence and coarse/refine behavior."""

import pickle

import numpy as np
import pytest
import zvec


@pytest.mark.parametrize("bits,fast_scan", [(8, False), (4, False), (4, True)])
@pytest.mark.parametrize("rotate", [False, True])
def test_ivf_pq_parameter_round_trip(bits, fast_scan, rotate):
    qp = zvec.QuantizerParam(
        enable_rotate=rotate,
        num_chunk=4,
        num_bits=bits,
        fast_scan=fast_scan,
        opq_iter=3,
        opq_pq_iter=2,
    )
    param = zvec.IVFIndexParam(
        metric_type=zvec.MetricType.L2,
        n_list=4,
        n_iters=3,
        quantize_type=zvec.QuantizeType.PQ,
        quantizer_param=qp,
    )
    assert pickle.loads(pickle.dumps(qp)) == qp
    restored = pickle.loads(pickle.dumps(param))
    assert restored.quantizer_param == qp
    assert restored.to_dict() == param.to_dict()
    assert param.to_dict()["quantizer_param"] == qp.to_dict()
    assert '"fast_scan"' in repr(param)


def test_legacy_quantizer_and_ivf_pickle():
    qp = zvec.QuantizerParam.__new__(zvec.QuantizerParam)
    qp.__setstate__((True,))
    assert qp.enable_rotate
    assert qp.num_chunk == 8 and qp.num_bits == 8
    assert not qp.fast_scan
    for state in [
        (zvec.MetricType.L2, 4, 3, False, zvec.QuantizeType.INT8),
        (zvec.MetricType.L2, 4, 3, False, zvec.QuantizeType.INT8, True),
    ]:
        param = zvec.IVFIndexParam.__new__(zvec.IVFIndexParam)
        param.__setstate__(state)
        assert param.quantizer_param.enable_rotate == (len(state) == 6)
        assert param.quantizer_param.num_bits == 8


@pytest.mark.parametrize("bits,fast_scan", [(8, False), (4, False), (4, True)])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.IP, zvec.MetricType.COSINE]
)
def test_ivf_pq_collection_reopen_and_refine(tmp_path, bits, fast_scan, rotate, metric):
    rng = np.random.default_rng(903)
    vectors = rng.normal(size=(1100, 16)).astype(np.float32)
    query = rng.normal(size=16).astype(np.float32)
    qp = zvec.QuantizerParam(
        enable_rotate=rotate,
        num_chunk=4,
        num_bits=bits,
        fast_scan=fast_scan,
        opq_iter=2,
        opq_pq_iter=2,
    )
    schema = zvec.CollectionSchema(
        name="ivf_pq",
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                dimension=16,
                index_param=zvec.IVFIndexParam(
                    metric_type=metric,
                    n_list=4,
                    n_iters=3,
                    quantize_type=zvec.QuantizeType.PQ,
                    quantizer_param=qp,
                ),
            )
        ],
    )
    path = str(tmp_path / "index")
    coll = zvec.create_and_open(path, schema)
    try:
        for start in range(0, len(vectors), 200):
            statuses = coll.insert(
                [
                    zvec.Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                    for i in range(start, min(start + 200, len(vectors)))
                ]
            )
            assert all(status.ok() for status in statuses)
        coll.optimize()
        before = coll.query(
            zvec.Query(
                field_name="vector",
                vector=query.tolist(),
                param=zvec.IVFQueryParam(nprobe=4),
            ),
            topk=40,
            output_fields=[],
        )
    finally:
        coll.close()
    coll = zvec.open(path, zvec.CollectionOption(read_only=True, enable_mmap=True))
    try:
        restored = coll.schema.vector("vector").index_param
        assert restored.quantize_type == zvec.QuantizeType.PQ
        assert restored.quantizer_param == qp
        coarse = coll.query(
            zvec.Query(
                field_name="vector",
                vector=query.tolist(),
                param=zvec.IVFQueryParam(nprobe=4),
            ),
            topk=40,
            output_fields=[],
        )
        assert [doc.id for doc in coarse] == [doc.id for doc in before]
        np.testing.assert_allclose(
            [doc.score for doc in coarse], [doc.score for doc in before]
        )
        refined = coll.query(
            zvec.Query(
                field_name="vector",
                vector=query.tolist(),
                param=zvec.IVFQueryParam(
                    nprobe=4, is_using_refiner=True, scale_factor=4
                ),
            ),
            topk=10,
            output_fields=[],
        )
        ids = np.array([int(doc.id) for doc in coarse])
        candidates = vectors[ids]
        if metric == zvec.MetricType.L2:
            scores = np.sum((candidates - query) ** 2, axis=1)
            order = np.argsort(scores)[:10]
        else:
            scores = candidates @ query
            if metric == zvec.MetricType.COSINE:
                scores /= np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
            order = np.argsort(-scores)[:10]
            if metric == zvec.MetricType.COSINE:
                scores = 1.0 - scores
        # Verify optimize built compressed postings rather than leaving the raw Flat fallback.
        assert np.max(np.abs(np.array([doc.score for doc in coarse]) - scores)) > 1e-4
        assert [int(doc.id) for doc in refined] == ids[order].tolist()
        np.testing.assert_allclose(
            [doc.score for doc in refined], scores[order], rtol=1e-4, atol=1e-5
        )
    finally:
        coll.close()
