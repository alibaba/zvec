"""Refine scale configuration and backward-compatible serialization."""

import pickle
import platform
import sys

import numpy as np
import pytest
import zvec

PARAMS = [
    (zvec.HnswQueryParam, 0.0),
    (zvec.VamanaQueryParam, 0.0),
    (zvec.HnswRabitqQueryParam, 0.0),
    (zvec.IVFQueryParam, 10.0),
    (zvec.IvfRabitqQueryParam, 10.0),
]


@pytest.mark.parametrize("param_type,default_scale", PARAMS)
def test_refine_scale_defaults_and_round_trip(param_type, default_scale):
    default = param_type()
    assert default.scale_factor == default_scale
    assert default.is_using_refiner is False
    param = param_type(is_using_refiner=True, scale_factor=2.8)
    assert param.scale_factor == pytest.approx(2.8)
    assert param.is_using_refiner is True
    assert "scale_factor" in repr(param)
    restored = pickle.loads(pickle.dumps(param))
    assert restored.__getstate__() == param.__getstate__()
    assert restored.scale_factor == param.scale_factor
    with pytest.raises(AttributeError):
        param.scale_factor = 4.0


@pytest.mark.parametrize("param_type", [zvec.HnswQueryParam, zvec.VamanaQueryParam])
@pytest.mark.parametrize("length", [4, 5, 6])
def test_old_graph_pickle_keeps_default_scale(param_type, length):
    state = (48, 0.0, False, True, 12, 1)[:length]
    restored = param_type.__new__(param_type)
    restored.__setstate__(state)
    assert restored.scale_factor == 0.0
    assert restored.is_using_refiner is True
    if length >= 5:
        assert restored.prefetch_offset == 12
    if length >= 6:
        assert restored.prefetch_lines == 1


@pytest.mark.parametrize(
    "param_type,state,scale",
    [
        (zvec.HnswRabitqQueryParam, (48, 0.0, False, True), 0.0),
        (zvec.IVFQueryParam, (12, 0.0, False), 10.0),
        (zvec.IvfRabitqQueryParam, (12, 0.0, False, True), 10.0),
        (zvec.IvfRabitqQueryParam, (12, 0.0, False, True, 3.5), 3.5),
    ],
)
def test_existing_pickle_formats(param_type, state, scale):
    restored = param_type.__new__(param_type)
    restored.__setstate__(state)
    assert restored.scale_factor == scale


@pytest.mark.parametrize("param_type", [zvec.HnswQueryParam, zvec.VamanaQueryParam])
def test_existing_positional_extra_params_still_work(param_type):
    param = param_type(64, 0.0, False, True, {"prefetch_offset": 12}, scale_factor=2.8)
    assert param.prefetch_offset == 12
    assert param.scale_factor == pytest.approx(2.8)


@pytest.mark.parametrize(
    "index_kind,quantizer,flat_type,native_dtype",
    [
        pytest.param(kind, quantizer, flat, dtype, id=f"{kind}-{label}")
        for kind in ("hnsw", "vamana")
        for quantizer, flat, dtype, label in (
            (
                zvec.QuantizeType.INT8,
                zvec.DataType.VECTOR_FP16,
                np.float16,
                "record_int8",
            ),
            (
                zvec.QuantizeType.UNIFORM_UINT4,
                zvec.DataType.VECTOR_UINT8,
                np.uint8,
                "uniform_uint4",
            ),
        )
    ]
    + [
        pytest.param(
            "hnsw_rabitq",
            None,
            zvec.DataType.VECTOR_FP32,
            np.float32,
            id="hnsw_rabitq",
            marks=pytest.mark.skipif(
                not (
                    sys.platform == "linux"
                    and platform.machine() in ("x86_64", "AMD64")
                ),
                reason="HNSW RaBitQ only supported on Linux x86_64",
            ),
        )
    ],
)
def test_query_refine_scale_controls_candidates(
    tmp_path, index_kind, quantizer, flat_type, native_dtype
):
    """Compare ordinary query refinement with exact distances on coarse hits."""
    rng = np.random.default_rng(20260915)
    vectors = rng.random((1400, 128), dtype=np.float32)
    queries = rng.random((16, 128), dtype=np.float32)
    if native_dtype == np.uint8:
        vectors = np.floor(vectors * 128)
        queries = np.floor(queries * 128)
    index_settings = dict(
        metric_type=zvec.MetricType.L2,
        quantize_type=quantizer,
        use_contiguous_memory=True,
        use_flat_contiguous_memory=True,
        flat_data_type=flat_type,
    )
    if index_kind == "hnsw":
        index = zvec.HnswIndexParam(m=32, ef_construction=100, **index_settings)
        param_type = zvec.HnswQueryParam
        query_settings = {"ef": 128}
    elif index_kind == "vamana":
        index = zvec.VamanaIndexParam(
            max_degree=32, search_list_size=100, two_pass_build=True, **index_settings
        )
        param_type = zvec.VamanaQueryParam
        query_settings = {"ef_search": 128}
    else:
        index = zvec.HnswRabitqIndexParam(
            metric_type=zvec.MetricType.L2,
            m=32,
            ef_construction=100,
            total_bits=4,
            num_clusters=4,
            sample_count=1000,
        )
        param_type = zvec.HnswRabitqQueryParam
        query_settings = {"ef": 128}
    schema = zvec.CollectionSchema(
        name="query_refine_scale",
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, dimension=128, index_param=index
            )
        ],
    )
    path = str(tmp_path / "index")
    writer = zvec.create_and_open(path, schema)
    try:
        for start in range(0, len(vectors), 200):
            statuses = writer.insert(
                [
                    zvec.Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                    for i in range(start, min(start + 200, len(vectors)))
                ]
            )
            assert all(status.ok() for status in statuses)
        writer.optimize()
    finally:
        writer.close()

    coll = zvec.open(path, zvec.CollectionOption(read_only=True, enable_mmap=True))
    try:

        def search(vector, param, topk=10):
            docs = coll.query(
                zvec.Query(field_name="vector", vector=vector, param=param),
                topk=topk,
                output_fields=[],
            )
            return np.array([int(doc.id) for doc in docs]), np.array(
                [doc.score for doc in docs]
            )

        coarse_param = param_type(**query_settings)
        saw_different_results = False
        for query in queries:
            minimum_ids = None
            for candidate_count in (10, 11, 19, 28, 55, 10):
                coarse_ids, _ = search(query, coarse_param, topk=candidate_count)
                raw_query = query.astype(native_dtype).astype(np.float32)
                raw_vectors = (
                    vectors[coarse_ids].astype(native_dtype).astype(np.float32)
                )
                distances = np.square(raw_vectors - raw_query).sum(axis=1)
                order = np.lexsort((coarse_ids, distances))[:10]
                param = param_type(
                    **query_settings,
                    is_using_refiner=True,
                    scale_factor=candidate_count / 10,
                )
                ids, scores = search(query, param)
                np.testing.assert_array_equal(ids, coarse_ids[order])
                np.testing.assert_allclose(
                    scores, distances[order], rtol=2e-5, atol=2e-5
                )
                if minimum_ids is None:
                    minimum_ids = ids
                elif candidate_count > 10:
                    saw_different_results |= not np.array_equal(ids, minimum_ids)
        # Ensure ignoring the multiplier cannot accidentally pass this dataset.
        assert saw_different_results

        query = queries[0]
        # Zero preserves the existing graph default: max(topk, ef) candidates.
        for ef in (6, 32, 128):
            settings = {next(iter(query_settings)): ef}
            coarse_ids, _ = search(query, param_type(**settings), topk=max(10, ef))
            raw_vectors = vectors[coarse_ids].astype(native_dtype).astype(np.float32)
            raw_query = query.astype(native_dtype).astype(np.float32)
            distances = np.square(raw_vectors - raw_query).sum(axis=1)
            order = np.lexsort((coarse_ids, distances))[:10]
            for overrides in ({}, {"scale_factor": 0.0}):
                param = param_type(**settings, is_using_refiner=True, **overrides)
                ids, scores = search(query, param)
                np.testing.assert_array_equal(ids, coarse_ids[order])
                np.testing.assert_allclose(
                    scores, distances[order], rtol=2e-5, atol=2e-5
                )

        minimum_param = param_type(
            **query_settings, is_using_refiner=True, scale_factor=1.0
        )
        minimum_ids, minimum_scores = search(query, minimum_param)
        for scale in (0.5, 1.0):
            param = param_type(
                **query_settings, is_using_refiner=True, scale_factor=scale
            )
            ids, scores = search(query, param)
            np.testing.assert_array_equal(ids, minimum_ids)
            np.testing.assert_allclose(scores, minimum_scores)

        coarse_ids, coarse_scores = search(query, coarse_param)
        for scale in (2.8, -1.0, float("nan"), float("inf"), 1e30):
            param = param_type(**query_settings, scale_factor=scale)
            ids, scores = search(query, param)
            np.testing.assert_array_equal(ids, coarse_ids)
            np.testing.assert_allclose(scores, coarse_scores)
            if scale != 2.8:
                refined_param = param_type(
                    **query_settings, is_using_refiner=True, scale_factor=scale
                )
                with pytest.raises((ValueError, RuntimeError)):
                    search(query, refined_param)
    finally:
        coll.close()
