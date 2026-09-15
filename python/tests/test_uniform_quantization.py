"""Exercise Uniform schema validation, training, writes and persisted indexes."""

import gc

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    Query,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType


@pytest.mark.parametrize("param_type", [HnswIndexParam, VamanaIndexParam])
@pytest.mark.parametrize("metric_type", [MetricType.IP, MetricType.COSINE])
@pytest.mark.parametrize(
    "quantize_type",
    [
        QuantizeType.UNIFORM_UINT7,
        QuantizeType.UNIFORM_UINT8,
        QuantizeType.UNIFORM_UINT4,
    ],
)
def test_uniform_quantization_rejects_non_l2(
    tmp_path, param_type, metric_type, quantize_type
):
    schema = CollectionSchema(
        name="uniform_invalid_metric",
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                32,
                index_param=param_type(
                    metric_type=metric_type, quantize_type=quantize_type
                ),
            )
        ],
    )
    with pytest.raises(ValueError, match="only supports L2 metric"):
        zvec.create_and_open(path=str(tmp_path / "invalid"), schema=schema)


@pytest.mark.parametrize(
    "quantize_type",
    [
        QuantizeType.UNIFORM_UINT7,
        QuantizeType.UNIFORM_UINT8,
        QuantizeType.UNIFORM_UINT4,
    ],
)
@pytest.mark.parametrize("index_type", ["vamana", "hnsw"])
@pytest.mark.parametrize(
    "flat_data_type",
    [DataType.VECTOR_FP32, DataType.VECTOR_FP16, DataType.VECTOR_UINT8],
    ids=["flat_fp32", "flat_fp16", "flat_uint8"],
)
@pytest.mark.parametrize("use_refiner", [False, True])
def test_uniform_quantization_queries_untrained_segments(
    tmp_path, quantize_type, index_type, flat_data_type, use_refiner
):
    """Raw-only writes stay searchable alongside trained graph segments."""
    dimension = 32
    initial_count = 128
    vectors = (
        np.random.default_rng(754)
        .integers(32, 224, size=(initial_count + 3, dimension))
        .astype(np.float32)
    )
    index_options = dict(
        metric_type=MetricType.L2,
        quantize_type=quantize_type,
        flat_data_type=flat_data_type,
        use_flat_contiguous_memory=True,
    )
    if index_type == "vamana":
        index_param = VamanaIndexParam(
            max_degree=16, search_list_size=64, **index_options
        )

        def query_param(is_linear=False):
            return VamanaQueryParam(
                ef_search=256, is_using_refiner=use_refiner, is_linear=is_linear
            )
    else:
        index_param = HnswIndexParam(m=16, ef_construction=64, **index_options)

        def query_param(is_linear=False):
            return HnswQueryParam(
                ef=256, is_using_refiner=use_refiner, is_linear=is_linear
            )

    schema = CollectionSchema(
        name="uniform_incremental",
        fields=[FieldSchema("ordinal", DataType.INT32)],
        vectors=[
            VectorSchema(
                "dense", DataType.VECTOR_FP32, dimension, index_param=index_param
            )
        ],
    )
    path = str(tmp_path / "collection")
    collection = zvec.create_and_open(path=path, schema=schema)

    def insert(start, end):
        results = collection.insert(
            [
                Doc(
                    id=str(i),
                    fields={"ordinal": i},
                    vectors={"dense": vectors[i].tolist()},
                )
                for i in range(start, end)
            ]
        )
        assert all(result.ok() for result in results)

    def assert_searchable(count):
        # Check both the trained segment and every subsequent raw-only block.
        for i in [0, *range(initial_count, count)]:
            hits = collection.query(
                Query("dense", vector=vectors[i].tolist(), param=query_param()),
                topk=1,
                include_vector=True,
            )
            assert [hit.id for hit in hits] == [str(i)]
            assert hits[0].vectors["dense"] == vectors[i].tolist()
            if use_refiner:
                assert hits[0].score == pytest.approx(0.0, abs=1e-5)

        # Exhaustive recall also verifies block offsets and result merging.
        query = Query(
            "dense", vector=vectors[0].tolist(), param=query_param(is_linear=True)
        )
        hits = collection.query(query, topk=count)
        assert len(hits) == count
        assert {hit.id for hit in hits} == {str(i) for i in range(count)}
        hits = collection.query(query, topk=count, filter="ordinal >= 128")
        assert len(hits) == count - initial_count
        assert {hit.id for hit in hits} == {str(i) for i in range(initial_count, count)}

    try:
        insert(0, initial_count)
        assert_searchable(initial_count)
        collection.optimize()
        assert_searchable(initial_count)

        for i in range(initial_count, len(vectors)):
            insert(i, i + 1)
            assert_searchable(i + 1)
            collection.flush()
            assert_searchable(i + 1)

        collection.close()
        collection = zvec.open(path=path, option=CollectionOption(read_only=True))
        assert_searchable(len(vectors))
        collection.close()
        collection = zvec.open(path=path)
        collection.optimize()
        assert_searchable(len(vectors))
    finally:
        collection.close()


@pytest.mark.parametrize(
    "quantize_type",
    [
        QuantizeType.UNIFORM_UINT7,
        QuantizeType.UNIFORM_UINT8,
        QuantizeType.UNIFORM_UINT4,
    ],
)
@pytest.mark.parametrize("index_type", ["vamana", "hnsw"])
def test_uniform_quantization_survives_reopen(tmp_path, quantize_type, index_type):
    vectors = (
        np.random.default_rng(20260909)
        .integers(0, 128, size=(1024, 32))
        .astype(np.float32)
    )
    if index_type == "vamana":
        index_param = VamanaIndexParam(
            metric_type=MetricType.L2,
            max_degree=16,
            search_list_size=64,
            quantize_type=quantize_type,
            use_contiguous_memory=True,
            two_pass_build=True,
        )
        query_param = VamanaQueryParam(ef_search=64, is_using_refiner=False)
    else:
        index_param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=16,
            ef_construction=64,
            quantize_type=quantize_type,
            use_contiguous_memory=True,
        )
        query_param = HnswQueryParam(ef=64, is_using_refiner=False)

    schema = CollectionSchema(
        name=f"uniform_{index_type}",
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=32,
                index_param=index_param,
            )
        ],
    )
    path = str(tmp_path / f"uniform_{index_type}")
    collection = zvec.create_and_open(path=path, schema=schema)
    for start in range(0, len(vectors), 256):
        results = collection.insert(
            [
                Doc(id=str(i), vectors={"dense": vectors[i].tolist()})
                for i in range(start, start + 256)
            ]
        )
        assert all(result.ok() for result in results)
    collection.optimize()

    query = Query(
        field_name="dense",
        vector=vectors[19].tolist(),
        param=query_param,
    )
    before = collection.query(query, topk=5)
    assert len(before) == 5
    assert len({doc.id for doc in before}) == 5
    before_ids = [doc.id for doc in before]
    before_scores = [doc.score for doc in before]
    del collection
    gc.collect()

    collection = zvec.open(
        path=path, option=CollectionOption(read_only=True, enable_mmap=True)
    )
    assert collection.schema.vectors[0].index_param.quantize_type == quantize_type
    after = collection.query(query, topk=5)
    assert [doc.id for doc in after] == before_ids
    assert [doc.score for doc in after] == pytest.approx(before_scores)
    assert all(np.isfinite(doc.score) for doc in after)


@pytest.mark.parametrize(
    "quantize_type",
    [
        QuantizeType.UNIFORM_UINT7,
        QuantizeType.UNIFORM_UINT8,
        QuantizeType.UNIFORM_UINT4,
    ],
)
@pytest.mark.parametrize("index_type", ["vamana", "hnsw"])
@pytest.mark.parametrize(
    "flat_data_type",
    [DataType.VECTOR_FP16, DataType.VECTOR_UINT8],
    ids=["flat_fp16", "flat_uint8"],
)
def test_uniform_quantizer_uses_flat_storage_vectors(
    tmp_path, quantize_type, index_type, flat_data_type
):
    """Uniform training and encoding must consume the configured Flat type."""
    dimension = 32
    doc_count = 128
    coordinates = np.arange(doc_count * dimension, dtype=np.float32).reshape(
        doc_count, dimension
    )
    input_vectors = (
        np.remainder(coordinates * 37.0, 241.0)
        + np.remainder(coordinates * 13.0, 11.0) * 0.071
    ).astype(np.float32)
    if flat_data_type == DataType.VECTOR_FP16:
        flat_vectors = input_vectors.astype(np.float16).astype(np.float32)
    else:
        flat_vectors = input_vectors.astype(np.uint8).astype(np.float32)
    assert np.any(input_vectors != flat_vectors)
    query_vector = flat_vectors[37]

    def build_and_search(label, vectors, flat_data_type):
        if index_type == "vamana":
            index_param = VamanaIndexParam(
                metric_type=MetricType.L2,
                max_degree=16,
                search_list_size=64,
                quantize_type=quantize_type,
                use_contiguous_memory=True,
                use_flat_contiguous_memory=True,
                flat_data_type=flat_data_type,
            )
            query_param = VamanaQueryParam(
                ef_search=doc_count,
                is_linear=True,
            )
        else:
            index_param = HnswIndexParam(
                metric_type=MetricType.L2,
                m=16,
                ef_construction=64,
                quantize_type=quantize_type,
                use_flat_contiguous_memory=True,
                flat_data_type=flat_data_type,
            )
            query_param = HnswQueryParam(ef=doc_count, is_linear=True)

        schema = CollectionSchema(
            name="uniform_flat_source",
            vectors=[
                VectorSchema(
                    "dense",
                    DataType.VECTOR_FP32,
                    dimension=dimension,
                    index_param=index_param,
                )
            ],
        )
        path = str(tmp_path / label)
        collection = zvec.create_and_open(path=path, schema=schema)
        try:
            results = collection.insert(
                [
                    Doc(id=str(i), vectors={"dense": vector.tolist()})
                    for i, vector in enumerate(vectors)
                ]
            )
            assert all(result.ok() for result in results)

            # Discard all writer state so optimize can only consume the
            # persisted Flat representation.
            collection = None
            gc.collect()
            collection = zvec.open(path=path)
            collection.optimize()

            hits = collection.query(
                Query(
                    field_name="dense",
                    vector=query_vector.tolist(),
                    param=query_param,
                ),
                topk=doc_count,
            )
            return [hit.id for hit in hits], np.asarray(
                [hit.score for hit in hits], dtype=np.float32
            )
        finally:
            if collection is not None:
                collection.destroy()

    name = f"{index_type}_{quantize_type.name.lower()}_{flat_data_type.name.lower()}"
    native_ids, native_scores = build_and_search(
        f"{name}_native", input_vectors, flat_data_type
    )
    reference_ids, reference_scores = build_and_search(
        f"{name}_rounded_fp32", flat_vectors, DataType.VECTOR_FP32
    )

    assert native_ids == reference_ids
    np.testing.assert_array_equal(native_scores, reference_scores)
