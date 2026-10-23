# Copyright 2025-present the zvec project
# Licensed under the Apache License, Version 2.0.
"""Public HNSW/IVF/DiskANN + Turbo RaBitQ: settings, training, mutations, and reopen."""

import math
import pickle
import platform
import random
import sys

import pytest
import zvec


def make_params(kind, metric, bits):
    q = zvec.QuantizerParam(
        total_bits=bits, num_clusters=4, sample_count=128, n_iters=6
    )
    if kind == "hnsw":
        return zvec.HnswIndexParam(
            metric_type=metric,
            m=16,
            ef_construction=100,
            quantize_type=zvec.QuantizeType.RABITQ,
            quantizer_param=q,
        )
    if kind == "diskann":
        return zvec.DiskAnnIndexParam(
            metric_type=metric,
            max_degree=16,
            list_size=100,
            quantize_type=zvec.QuantizeType.RABITQ,
            quantizer_param=q,
        )
    return zvec.IVFIndexParam(
        metric_type=metric,
        n_list=8,
        n_iters=10,
        quantize_type=zvec.QuantizeType.RABITQ,
        total_bits=bits,
        sample_count=128,
        quantizer_param=q,
    )


@pytest.mark.parametrize("kind", ["hnsw", "ivf", "diskann"])
def test_rabitq_parameter_pickle(kind):
    params = make_params(kind, zvec.MetricType.L2, 3)
    restored = pickle.loads(pickle.dumps(params))
    assert restored.to_dict() == params.to_dict()
    assert restored.quantizer_param == params.quantizer_param
    assert restored.quantizer_param.to_dict() == {
        "enable_rotate": False,
        "total_bits": 3,
        "num_clusters": 4,
        "sample_count": 128,
        "n_iters": 6,
    }


supported = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


@pytest.mark.skipif(not supported, reason="RaBitQ integration requires Linux x86_64")
@pytest.mark.parametrize("kind", ["hnsw", "ivf", "diskann"])
@pytest.mark.parametrize("bits", [3, 9])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.IP, zvec.MetricType.COSINE]
)
def test_turbo_rabitq_collection(tmp_path, kind, bits, metric):
    rng = random.Random(8121)
    vectors = [[rng.gauss(0, 1) for _ in range(128)] for _ in range(320)]
    for vector in vectors:
        norm = math.sqrt(sum(x * x for x in vector))
        vector[:] = [x / norm for x in vector]
    params = make_params(kind, metric, bits)
    schema = zvec.CollectionSchema(
        name="turbo_rabitq",
        fields=[zvec.FieldSchema("ordinal", zvec.DataType.INT64)],
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=params,
            )
        ],
    )
    path = str(tmp_path / "collection")
    coll = zvec.create_and_open(path=path, schema=schema)

    def docs(start, end):
        return [
            zvec.Doc(
                id=str(i), fields={"ordinal": i}, vectors={"embedding": vectors[i]}
            )
            for i in range(start, end)
        ]

    def query(i, **kwargs):
        qp = (
            zvec.HnswQueryParam(ef=320, is_using_refiner=True)
            if kind == "hnsw"
            else zvec.DiskAnnQueryParam(list_size=320)
            if kind == "diskann"
            else zvec.IVFQueryParam(nprobe=8, is_using_refiner=True)
        )
        return coll.query(
            queries=zvec.Query(field_name="embedding", vector=vectors[i], param=qp),
            topk=5,
            **kwargs,
        )

    def check_settings():
        actual = coll.schema.vector("embedding").index_param
        assert actual.type == params.type
        assert actual.quantize_type == zvec.QuantizeType.RABITQ
        assert actual.quantizer_param == params.quantizer_param
        if kind == "ivf":
            assert actual.total_bits == bits and actual.sample_count == 128

    try:
        assert all(s.ok() for s in coll.insert(docs(0, 256)))
        coll.optimize()
        assert coll.stats.index_completeness["embedding"] == 1
        check_settings()
        before = query(17, include_vector=True)
        assert before[0].id == "17"
        assert before[0].vector("embedding") == pytest.approx(vectors[17])
        filtered = query(17, filter="ordinal >= 128")
        assert filtered and all(int(d.id) >= 128 for d in filtered)
        coll.close()
        coll = zvec.open(path=path)
        check_settings()
        assert [d.id for d in query(17)] == [d.id for d in before]
        # Force another training/build pass with new vectors after reopen.
        assert all(s.ok() for s in coll.insert(docs(256, 320)))
        assert all(s.ok() for s in coll.delete(["17"]))
        vectors[18] = [-x for x in vectors[18]]
        assert all(s.ok() for s in coll.update(docs(18, 19)))
        coll.optimize()
        assert coll.stats.index_completeness["embedding"] == 1
        assert query(300)[0].id == "300"
        assert query(18)[0].id == "18"
        assert all(d.id != "17" for d in query(17))
        coll.close()
        coll = zvec.open(path=path)
        check_settings()
        assert query(300)[0].id == "300"
        assert query(18)[0].id == "18"
    finally:
        coll.close()


@pytest.mark.skipif(not supported, reason="RaBitQ integration requires Linux x86_64")
@pytest.mark.parametrize(
    "settings",
    [
        {"total_bits": 0},
        {"total_bits": 10},
        {"sample_count": -1},
        {"num_clusters": 0},
        {"n_iters": 0},
        {"enable_rotate": True},
    ],
)
@pytest.mark.parametrize("kind", ["hnsw", "diskann"])
def test_graph_rabitq_rejects_invalid_settings(tmp_path, settings, kind):
    cls = zvec.HnswIndexParam if kind == "hnsw" else zvec.DiskAnnIndexParam
    params = cls(
        quantize_type=zvec.QuantizeType.RABITQ,
        quantizer_param=zvec.QuantizerParam(**settings),
    )
    schema = zvec.CollectionSchema(
        name="invalid",
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=params,
            )
        ],
    )
    with pytest.raises(Exception, match="RABITQ|RabitQ"):
        zvec.create_and_open(path=str(tmp_path / "invalid"), schema=schema)


@pytest.mark.parametrize("kind", ["hnsw", "ivf", "diskann"])
def test_legacy_rabitq_parameter_state(kind):
    """Previously pickled indexes stored only enable_rotate at position 5."""
    if kind == "hnsw":
        cls = zvec.HnswIndexParam
        state = (
            zvec.MetricType.L2,
            16,
            100,
            zvec.QuantizeType.RABITQ,
            False,
            False,
            False,
            zvec.DataType.VECTOR_FP32,
        )
    elif kind == "diskann":
        cls = zvec.DiskAnnIndexParam
        state = (zvec.MetricType.L2, 16, 100, 0, zvec.QuantizeType.RABITQ, False)
    else:
        cls = zvec.IVFIndexParam
        state = (
            zvec.MetricType.L2,
            8,
            10,
            False,
            zvec.QuantizeType.RABITQ,
            False,
            3,
            128,
        )
    restored = cls.__new__(cls)
    restored.__setstate__(state)
    assert restored.quantizer_param.total_bits == 7
    assert restored.quantizer_param.num_clusters == 16
    assert restored.quantizer_param.n_iters == 20
    if kind == "ivf":
        assert restored.total_bits == 3 and restored.sample_count == 128
    if kind == "diskann":
        restored = cls.__new__(cls)
        restored.__setstate__(state[:5])
        assert restored.quantizer_param.total_bits == 7
        assert not restored.quantizer_param.enable_rotate


@pytest.mark.parametrize("cls", [zvec.FlatIndexParam, zvec.VamanaIndexParam])
def test_rabitq_rejects_unsupported_index(tmp_path, cls):
    params = cls(quantize_type=zvec.QuantizeType.RABITQ)
    schema = zvec.CollectionSchema(
        name="invalid",
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=params,
            )
        ],
    )
    with pytest.raises(Exception, match="RABITQ"):
        zvec.create_and_open(path=str(tmp_path / "invalid"), schema=schema)


@pytest.mark.skipif(not supported, reason="RaBitQ integration requires Linux x86_64")
def test_ivf_rabitq_rejects_additional_rotation(tmp_path):
    params = zvec.IVFIndexParam(
        quantize_type=zvec.QuantizeType.RABITQ,
        quantizer_param=zvec.QuantizerParam(enable_rotate=True),
    )
    schema = zvec.CollectionSchema(
        name="invalid",
        vectors=[
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=128,
                index_param=params,
            )
        ],
    )
    with pytest.raises(Exception, match="RABITQ"):
        zvec.create_and_open(path=str(tmp_path / "invalid"), schema=schema)
