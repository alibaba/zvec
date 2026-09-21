"""Validate every Python index family with a configurable Uniform quantizer.

HnswRabitqIndexParam and IvfRabitqIndexParam select RABITQ internally and do
not expose a quantize_type constructor argument.
"""

import numpy as np
import pytest

import zvec


@pytest.mark.parametrize(
    "param_type,data_type,metric",
    [
        (param_type, data_type, zvec.MetricType.L2)
        for param_type in (zvec.HnswIndexParam, zvec.VamanaIndexParam)
        for data_type in (zvec.DataType.VECTOR_FP16, zvec.DataType.VECTOR_INT8)
    ]
    + [
        (zvec.HnswIndexParam, data_type, zvec.MetricType.IP)
        for data_type in (
            zvec.DataType.SPARSE_VECTOR_FP32,
            zvec.DataType.SPARSE_VECTOR_FP16,
        )
    ],
)
@pytest.mark.parametrize(
    "quantizer",
    [
        zvec.QuantizeType.UNIFORM_UINT7,
        zvec.QuantizeType.UNIFORM_UINT8,
        zvec.QuantizeType.UNIFORM_UINT4,
    ],
)
def test_uniform_rejects_unsupported_field_types(
    tmp_path, param_type, data_type, metric, quantizer
):
    # Logical vector type is separate from flat_data_type. Dense FP32 fields
    # can use FP16/UINT8 Flat storage; Uniform does not accept other field types.
    schema = zvec.CollectionSchema(
        name="uniform_field_type_validation",
        vectors=[
            zvec.VectorSchema(
                "dense",
                data_type,
                32,
                index_param=param_type(metric_type=metric, quantize_type=quantizer),
            )
        ],
    )
    with pytest.raises(ValueError, match="Invalid schema:.*quantiz"):
        collection = zvec.create_and_open(str(tmp_path / "collection"), schema=schema)
        collection.close()


@pytest.mark.parametrize(
    "param_type,index_name,supported,query_type",
    [
        (zvec.FlatIndexParam, "FLAT", False, None),
        (zvec.IVFIndexParam, "IVF", False, None),
        (zvec.DiskAnnIndexParam, "DISKANN", False, None),
        (zvec.HnswIndexParam, "HNSW", True, zvec.HnswQueryParam),
        (zvec.VamanaIndexParam, "VAMANA", True, zvec.VamanaQueryParam),
    ],
)
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.IP, zvec.MetricType.COSINE]
)
@pytest.mark.parametrize(
    "quantizer,levels",
    [
        (zvec.QuantizeType.UNIFORM_UINT7, 127),
        (zvec.QuantizeType.UNIFORM_UINT8, 255),
        (zvec.QuantizeType.UNIFORM_UINT4, 15),
    ],
)
def test_uniform_index_support_matrix(
    tmp_path, param_type, index_name, supported, query_type, metric, quantizer, levels
):
    schema = zvec.CollectionSchema(
        name="uniform_support_matrix",
        vectors=[
            zvec.VectorSchema(
                "dense",
                zvec.DataType.VECTOR_FP32,
                32,
                index_param=param_type(metric_type=metric, quantize_type=quantizer),
            )
        ],
    )
    path = str(tmp_path / "collection")
    if not supported or metric != zvec.MetricType.L2:
        error = (
            f"quantization is not supported with {index_name}"
            if not supported
            else "only supports L2 metric"
        )
        error_type = ValueError
        if index_name == "DISKANN":
            error += "|DiskAnn is not supported on this platform"
            error_type = (ValueError, RuntimeError)
        with pytest.raises(error_type, match=error):
            collection = zvec.create_and_open(path=path, schema=schema)
            collection.close()
        return

    vectors = (
        np.random.default_rng(754).uniform(0.01, 0.99, (65, 32)).astype(np.float32)
    )
    # Endpoints each occupy >1% of the data, including after appending. Thus
    # both min/max training and UINT4's trimmed range produce exactly [0, 1].
    vectors[0] = 0
    vectors[1] = 1
    query_vector = np.full(32, 0.37, dtype=np.float32)
    raw_scores = np.sum((vectors - query_vector) ** 2, axis=1)
    quant_scores = np.sum(
        (np.rint(vectors * levels) - np.rint(query_vector * levels)) ** 2, axis=1
    ) / (levels * levels)
    assert np.max(np.abs(raw_scores - quant_scores)) > 1e-3
    collection = zvec.create_and_open(path=path, schema=schema)

    def check(count, trained):
        persisted_param = collection.schema.vectors[0].index_param
        assert persisted_param.quantize_type == quantizer
        assert persisted_param.to_dict()["quantize_type"] == quantizer.name
        hits = collection.query(
            zvec.Query(
                "dense", vector=query_vector.tolist(), param=query_type(is_linear=True)
            ),
            topk=count,
        )
        assert len(hits) == count
        assert {hit.id for hit in hits} == {str(i) for i in range(count)}
        for hit in hits:
            i = int(hit.id)
            expected = quant_scores[i] if i < trained else raw_scores[i]
            assert hit.score == pytest.approx(float(expected), rel=2e-5, abs=2e-5)
        # Also check normal graph search, independently of the exhaustive scan.
        for i in (0, count - 1):
            nearest = collection.query(
                zvec.Query("dense", vector=vectors[i].tolist(), param=query_type()),
                topk=1,
            )
            assert nearest[0].id == str(i)
            assert nearest[0].score == pytest.approx(0.0, abs=1e-5)

    try:
        assert all(
            status.ok()
            for status in collection.insert(
                [
                    zvec.Doc(id=str(i), vectors={"dense": v.tolist()})
                    for i, v in enumerate(vectors[:64])
                ]
            )
        )
        check(64, 0)
        assert collection.stats.index_completeness["dense"] == 0
        collection.optimize()
        check(64, 64)
        assert collection.stats.index_completeness["dense"] == 1
        assert collection.insert(
            zvec.Doc(id="64", vectors={"dense": vectors[64].tolist()})
        ).ok()
        check(65, 64)
        collection.flush()
        collection.close()
        collection = zvec.open(path=path)
        check(65, 64)
        collection.optimize()
        check(65, 65)
        collection.close()
        collection = zvec.open(path=path, option=zvec.CollectionOption(read_only=True))
        check(65, 65)
    finally:
        collection.close()
