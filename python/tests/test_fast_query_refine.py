"""Fast search with native reference storage and optional refined scores."""

from functools import partial

import numpy as np
import pytest

import zvec
from zvec import CollectionOption, CollectionSchema, Doc, Query
from zvec import HnswIndexParam, HnswQueryParam
from zvec import VamanaIndexParam, VamanaQueryParam, VectorSchema
from zvec.typing import DataType, MetricType, QuantizeType


@pytest.mark.parametrize(
    "quantizer,flat_type",
    [
        (QuantizeType.INT8, DataType.VECTOR_FP16),
        (QuantizeType.FP16, DataType.VECTOR_FP16),
        (QuantizeType.INT4, DataType.VECTOR_FP16),
        (QuantizeType.UNIFORM_UINT4, DataType.VECTOR_UINT8),
        (QuantizeType.UNIFORM_UINT7, DataType.VECTOR_UINT8),
        (QuantizeType.UNIFORM_UINT8, DataType.VECTOR_UINT8),
    ],
    ids=[
        "record_int8",
        "record_fp16",
        "record_int4",
        "uniform4",
        "uniform7",
        "uniform8",
    ],
)
@pytest.mark.parametrize("deleted", [False, True], ids=["direct", "delete_fallback"])
@pytest.mark.parametrize("index_kind", ["vamana", "hnsw"])
def test_refine_candidate_counts_and_parameter_switches(
    tmp_path, quantizer, flat_type, deleted, index_kind
):
    rng = np.random.default_rng(20260911)
    vectors = rng.integers(0, 128, size=(1400, 128)).astype(np.float32)
    if flat_type == DataType.VECTOR_FP16:
        vectors = vectors / 129.3
    index_params = dict(
        metric_type=MetricType.L2,
        quantize_type=quantizer,
        use_contiguous_memory=True,
        use_flat_contiguous_memory=True,
        flat_data_type=flat_type,
    )
    if index_kind == "vamana":
        index = VamanaIndexParam(
            max_degree=32, search_list_size=100, two_pass_build=True, **index_params
        )
        query_param = partial(VamanaQueryParam, ef_search=64)
    else:
        index = HnswIndexParam(m=32, ef_construction=100, **index_params)
        query_param = partial(HnswQueryParam, ef=64)
    schema = CollectionSchema(
        name="fast_query_refine",
        vectors=[
            VectorSchema(
                "vector",
                DataType.VECTOR_FP32,
                dimension=128,
                index_param=index,
            )
        ],
    )
    path = str(tmp_path / "index")
    writer = zvec.create_and_open(path, schema)
    for start in range(0, len(vectors), 200):
        assert all(
            s.ok()
            for s in writer.insert(
                [
                    Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                    for i in range(start, min(start + 200, len(vectors)))
                ]
            )
        )
    writer.optimize()
    if deleted:
        assert all(s.ok() for s in writer.delete(["0"]))
    writer.close()
    coll = zvec.open(path, CollectionOption(read_only=True, enable_mmap=True))
    try:
        for refine, scale, candidates in [
            (True, None, 64),
            (True, 0.0, 64),
            (True, 0.5, 10),
            (True, 1.0, 10),
            (True, 1.1, 11),
            (True, 1.9, 19),
            (False, 0.0, 10),
            (True, 1.4, 14),
            (True, 2.6, 26),
            (False, 2.6, 10),
        ]:
            # Default/zero preserves main's max(topk, ef) candidate budget;
            # explicit positive factors use max(topk, floor(topk * factor)).
            options = {} if scale is None else {"scale_factor": scale}
            param = query_param(is_using_refiner=refine, **options)
            for row in [11, 89, 531]:
                query = np.ascontiguousarray(vectors[row] + np.float32(0.021))
                coarse_param = query_param()
                coarse_docs = coll.query(
                    Query(field_name="vector", vector=query, param=coarse_param),
                    topk=candidates,
                    output_fields=[],
                )
                coarse_ids = np.asarray(
                    [int(doc.id) for doc in coarse_docs], dtype=np.int64
                )
                output = coll.fast_query(
                    "vector",
                    query,
                    param,
                    topk=10,
                )
                scored_ids, scores = coll.fast_query(
                    "vector",
                    query,
                    param,
                    topk=10,
                    return_scores=True,
                )
                np.testing.assert_array_equal(scored_ids, output)
                # Ordinary query and fast_query consume exactly the same params.
                docs = coll.query(
                    Query(field_name="vector", vector=query, param=param), topk=10
                )
                np.testing.assert_array_equal(output, [int(doc.id) for doc in docs])
                np.testing.assert_allclose(
                    scores, [doc.score for doc in docs], rtol=2e-5, atol=2e-5
                )
                if refine:
                    native_dtype = (
                        np.float16 if flat_type == DataType.VECTOR_FP16 else np.uint8
                    )
                    native_query = query.astype(native_dtype).astype(np.float32)
                    native_rows = (
                        vectors[coarse_ids].astype(native_dtype).astype(np.float32)
                    )
                    distances = np.square(native_rows - native_query).sum(axis=1)
                    order = np.lexsort((coarse_ids, distances))[: len(output)]
                    np.testing.assert_array_equal(output, coarse_ids[order])
                    np.testing.assert_allclose(
                        scores, distances[order], rtol=2e-5, atol=2e-5
                    )
                else:
                    np.testing.assert_array_equal(output, coarse_ids)
                    np.testing.assert_allclose(
                        scores, [doc.score for doc in coarse_docs], rtol=2e-5, atol=2e-5
                    )
        for scale in (-1.0, float("nan"), float("inf")):
            param = query_param(is_using_refiner=True, scale_factor=scale)
            with pytest.raises((ValueError, RuntimeError)):
                coll.query(
                    Query(field_name="vector", vector=vectors[11], param=param), topk=10
                )
            with pytest.raises((ValueError, RuntimeError)):
                coll.fast_query("vector", vectors[11], param)

    finally:
        coll.close()
