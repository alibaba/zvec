"""Public HNSW/IVF PQ configuration, persistence and coarse/refine behavior."""

import pickle

import numpy as np
import pytest
import zvec


def index_param(kind, **kwargs):
    if kind == "ivf":
        return zvec.IVFIndexParam(n_list=4, n_iters=3, **kwargs)
    return zvec.HnswIndexParam(m=16, ef_construction=100, **kwargs)


def query_param(kind, **kwargs):
    if kind == "ivf":
        return zvec.IVFQueryParam(nprobe=4, **kwargs)
    return zvec.HnswQueryParam(ef=2000, **kwargs)


@pytest.mark.parametrize(
    "kind,bits,fast_scan",
    [
        ("ivf", 8, False),
        ("ivf", 4, False),
        ("ivf", 4, True),
        ("hnsw", 8, False),
        ("hnsw", 4, False),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
def test_pq_parameter_round_trip(kind, bits, fast_scan, rotate):
    qp = zvec.QuantizerParam(
        enable_rotate=rotate,
        num_chunk=4,
        num_bits=bits,
        fast_scan=fast_scan,
        opq_iter=3,
        opq_pq_iter=2,
    )
    param = index_param(
        kind,
        metric_type=zvec.MetricType.L2,
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


@pytest.mark.parametrize(
    "kind,bits,fast_scan",
    [
        ("ivf", 8, False),
        ("ivf", 4, False),
        ("ivf", 4, True),
        ("hnsw", 8, False),
        ("hnsw", 4, False),
    ],
)
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.IP, zvec.MetricType.COSINE]
)
def test_pq_collection_reopen_and_refine(
    tmp_path, kind, bits, fast_scan, rotate, metric
):
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
                index_param=index_param(
                    kind,
                    metric_type=metric,
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
        # Untrained writer blocks remain searchable through raw Flat.
        assert coll.query(
            zvec.Query(field_name="vector", vector=query.tolist()), topk=5
        )
        coll.optimize()
        before = coll.query(
            zvec.Query(
                field_name="vector",
                vector=query.tolist(),
                param=query_param(kind),
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
                param=query_param(kind),
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
                param=query_param(kind, is_using_refiner=True, scale_factor=4),
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


def test_legacy_hnsw_pickle():
    for state in [
        (zvec.MetricType.L2, 16, 100, zvec.QuantizeType.INT8, False),
        (
            zvec.MetricType.L2,
            16,
            100,
            zvec.QuantizeType.INT8,
            False,
            True,
            True,
            zvec.DataType.VECTOR_FP16,
        ),
    ]:
        param = zvec.HnswIndexParam.__new__(zvec.HnswIndexParam)
        param.__setstate__(state)
        assert param.quantizer_param.enable_rotate == (len(state) == 8)
        assert param.quantizer_param.num_bits == 8


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("enable_mmap", [False, True])
def test_hnsw_pq_incremental_writes_and_retraining(tmp_path, bits, enable_mmap):
    zvec.init()
    rng = np.random.default_rng(137)
    vectors = rng.normal(size=(303, 16)).astype(np.float32)
    qp = zvec.QuantizerParam(num_chunk=4, num_bits=bits)
    param = zvec.HnswIndexParam(
        metric_type=zvec.MetricType.L2,
        m=16,
        ef_construction=100,
        quantize_type=zvec.QuantizeType.PQ,
        quantizer_param=qp,
        flat_data_type=zvec.DataType.VECTOR_FP16,
    )
    schema = zvec.CollectionSchema(
        name="pq_writes",
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, 16, index_param=param
            )
        ],
    )
    path = str(tmp_path / "index")
    option = zvec.CollectionOption(enable_mmap=enable_mmap)
    coll = zvec.create_and_open(path, schema, option=option)

    def check(ids):
        hits = coll.query(
            zvec.Query(
                "vector",
                vector=vectors[302].tolist(),
                param=zvec.HnswQueryParam(is_linear=True, is_using_refiner=True),
            ),
            topk=400,
        )
        assert {hit.id for hit in hits} == ids
        assert coll.schema.vector("vector").index_param.quantizer_param == qp

    try:
        for start in range(0, 300, 100):
            statuses = coll.insert(
                [
                    zvec.Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                    for i in range(start, start + 100)
                ]
            )
            assert all(status.ok() for status in statuses)
        coll.optimize()
        coll.close()
        coll = zvec.open(path, option)
        assert coll.insert(
            zvec.Doc(id="300", vectors={"vector": vectors[300].tolist()})
        ).ok()
        assert coll.update(
            zvec.Doc(id="1", vectors={"vector": vectors[301].tolist()})
        ).ok()
        assert coll.delete("0").ok()
        expected = {str(i) for i in range(1, 301)}
        check(expected)
        coll.optimize()
        check(expected)
        coll.close()
        coll = zvec.open(path, option)
        check(expected)
    finally:
        coll.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fast_scan": True, "num_bits": 4},
        {"num_chunk": 0},
        {"num_chunk": 17},
        {"num_bits": 6},
        {"num_chunk": 3, "num_bits": 4},
        {"enable_rotate": True, "opq_iter": 0},
    ],
)
def test_hnsw_pq_rejects_invalid_configuration(tmp_path, kwargs):
    schema = zvec.CollectionSchema(
        name="invalid_pq",
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                16,
                index_param=zvec.HnswIndexParam(
                    quantize_type=zvec.QuantizeType.PQ,
                    quantizer_param=zvec.QuantizerParam(**kwargs),
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="PQ"):
        coll = zvec.create_and_open(str(tmp_path / "index"), schema)
        coll.close()
