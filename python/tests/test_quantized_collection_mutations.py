"""Apply the same mutation/query contract to record and Uniform quantizers."""

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    MetricType,
    QuantizeType,
    Query,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
)


@pytest.mark.parametrize(
    "quantize_type",
    [
        QuantizeType.FP16,
        QuantizeType.INT8,
        QuantizeType.INT4,
        QuantizeType.UNIFORM_UINT7,
        QuantizeType.UNIFORM_UINT8,
        QuantizeType.UNIFORM_UINT4,
    ],
    ids=["fp16", "int8", "int4", "uniform_uint7", "uniform_uint8", "uniform_uint4"],
)
@pytest.mark.parametrize("index_kind", ["hnsw", "vamana"])
@pytest.mark.parametrize("use_refiner", [False, True])
def test_quantized_mutations_remain_searchable(
    tmp_path, quantize_type, index_kind, use_refiner
):
    # Non-mmap storage uses the buffer pool configured by library initialization.
    zvec.init()
    dimension = 32
    rng = np.random.default_rng(754)
    index_options = dict(
        metric_type=MetricType.L2,
        quantize_type=quantize_type,
        flat_data_type=DataType.VECTOR_FP16,
    )
    if index_kind == "hnsw":
        index_param = HnswIndexParam(m=16, ef_construction=64, **index_options)

        def query_param(is_linear=False):
            return HnswQueryParam(
                ef=128, is_using_refiner=use_refiner, is_linear=is_linear
            )
    else:
        index_param = VamanaIndexParam(
            max_degree=16, search_list_size=64, **index_options
        )

        def query_param(is_linear=False):
            return VamanaQueryParam(
                ef_search=128, is_using_refiner=use_refiner, is_linear=is_linear
            )

    schema = CollectionSchema(
        name="quantized_mutations",
        fields=[FieldSchema("ordinal", DataType.INT32, index_param=InvertIndexParam())],
        vectors=[
            VectorSchema(
                "dense", DataType.VECTOR_FP32, dimension, index_param=index_param
            )
        ],
    )
    path = str(tmp_path / "collection")
    option = CollectionOption(enable_mmap=False)
    collection = zvec.create_and_open(path=path, schema=schema, option=option)
    vectors = {}
    ordinals = {}

    def write(method, doc_id, ordinal):
        vector = rng.integers(16, 240, dimension).astype(np.float32) + 0.13
        doc = Doc(
            id=doc_id, fields={"ordinal": ordinal}, vectors={"dense": vector.tolist()}
        )
        assert method(doc).ok()
        vectors[doc_id] = vector.astype(np.float16).astype(np.float32)
        ordinals[doc_id] = ordinal

    def check(nearest_id="0"):
        query_vector = vectors[nearest_id]
        query = Query(
            "dense", vector=query_vector.tolist(), param=query_param(is_linear=True)
        )
        hits = collection.query(query, topk=128, include_vector=True)
        assert collection.stats.doc_count == len(vectors)
        assert len(hits) == len(vectors)
        assert {hit.id for hit in hits} == set(vectors)
        for hit in hits:
            np.testing.assert_array_equal(hit.vector("dense"), vectors[hit.id])
            if use_refiner:
                expected = float(np.sum((vectors[hit.id] - query_vector) ** 2))
                assert hit.score == pytest.approx(expected, rel=1e-5, abs=1e-5)

        # Check graph/Flat search without forcing a linear scan, and verify
        # inverted filtering against the current version of the document.
        query = Query("dense", vector=query_vector.tolist(), param=query_param())
        assert [hit.id for hit in collection.query(query, topk=1)] == [nearest_id]
        hits = collection.query(
            query, topk=10, filter=f"ordinal = {ordinals[nearest_id]}"
        )
        assert [hit.id for hit in hits] == [nearest_id]
        fetched = collection.fetch([nearest_id])[nearest_id]
        np.testing.assert_array_equal(fetched.vector("dense"), query_vector)

    try:
        for i in range(64):
            write(collection.insert, str(i), i)
        check()

        # The first iteration mutates untrained Flat blocks; the second also
        # invalidates documents in the previously optimized graph segment.
        for phase in range(2):
            write(collection.update, "0", 100 + phase)
            check()
            write(collection.upsert, "1", 200 + phase)
            check("1")
            new_id = f"new{phase}"
            write(collection.upsert, new_id, 300 + phase)
            check(new_id)

            assert collection.update(Doc(id="0", fields={"ordinal": 999})).ok()
            ordinals["0"] = 999
            check()

            deleted_id = str(2 + phase)
            assert collection.delete(deleted_id).ok()
            del vectors[deleted_id], ordinals[deleted_id]
            check()
            deleted_id = str(4 + phase)
            collection.delete_by_filter(f"ordinal = {ordinals[deleted_id]}")
            del vectors[deleted_id], ordinals[deleted_id]
            check()

            collection.flush()
            check(new_id)
            collection.close()
            collection = zvec.open(
                path=path, option=CollectionOption(read_only=True, enable_mmap=False)
            )
            check(new_id)
            collection.close()
            collection = zvec.open(path=path, option=option)
            collection.optimize()
            check(new_id)
    finally:
        collection.close()
