"""Advanced dense search: collection queries, scores, fallback and lifetime."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    Query,
    HnswIndexParam,
    HnswQueryParam,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType


@pytest.mark.parametrize(
    "index_type", [zvec.FlatIndexParam, HnswIndexParam, VamanaIndexParam]
)
def test_empty_collection(tmp_path, index_type):
    schema = CollectionSchema(
        name="empty_fast_query",
        vectors=[
            VectorSchema("vector", DataType.VECTOR_FP32, 32, index_param=index_type())
        ],
    )
    path = str(tmp_path / "empty")
    writer = zvec.create_and_open(path, schema)
    writer.close()
    reader = zvec.open(path, CollectionOption(read_only=True))
    vector = np.zeros(32, dtype=np.float32)
    try:
        assert reader.query(Query("vector", vector=vector), topk=3) == []
        for topk in (3, 0, 1):
            ids, scores = reader.fast_query(
                "vector", vector, topk=topk, return_scores=True
            )
            np.testing.assert_array_equal(ids, np.full(topk, -1, dtype=np.int64))
            assert scores.shape == (topk,)
            assert np.all(np.isnan(scores))
            np.testing.assert_array_equal(
                ids, reader.fast_query("vector", vector, topk=topk)
            )
        # Empty state must still validate the field, vector and parameter type.
        with pytest.raises(ValueError, match="dense vector field"):
            reader.fast_query("missing", vector)
        with pytest.raises(ValueError, match="dtype|dimension"):
            reader.fast_query("vector", vector[:-1])
        wrong_param = (
            HnswQueryParam() if index_type is not HnswIndexParam else VamanaQueryParam()
        )
        with pytest.raises(ValueError, match="parameter type"):
            reader.fast_query("vector", vector, wrong_param)
    finally:
        reader.close()


@pytest.fixture(
    params=[(128, False), (1200, False), (1200, True)],
    ids=["brute_fallback", "vamana_graph", "hnsw_graph"],
)
def collection(tmp_path, request):
    """Cross the Vamana brute-force threshold to exercise both result paths."""
    rng = np.random.default_rng(721)
    vectors = rng.normal(size=(request.param[0], 32)).astype(np.float32)
    schema = CollectionSchema(
        name="fast_query",
        vectors=[
            VectorSchema(
                "vector",
                DataType.VECTOR_FP32,
                dimension=32,
                index_param=HnswIndexParam(
                    metric_type=MetricType.L2,
                    m=16,
                    ef_construction=64,
                    use_contiguous_memory=True,
                )
                if request.param[1]
                else VamanaIndexParam(
                    metric_type=MetricType.L2,
                    max_degree=16,
                    search_list_size=64,
                    use_contiguous_memory=True,
                    two_pass_build=True,
                    quantize_type=QuantizeType.UNDEFINED,
                ),
            )
        ],
    )
    path = str(tmp_path / "collection")
    writer = zvec.create_and_open(path, schema)
    for start in range(0, len(vectors), 512):
        statuses = writer.insert(
            [
                Doc(id=f"row-{i}", vectors={"vector": vectors[i].tolist()})
                for i in range(start, min(start + 512, len(vectors)))
            ]
        )
        assert all(status.ok() for status in statuses)
    with pytest.raises(ValueError, match="read-only"):
        writer.fast_query("vector", vectors[0])
    writer.optimize()
    writer.close()
    reader = zvec.open(path, CollectionOption(read_only=True, enable_mmap=True))
    try:
        yield reader, vectors, HnswQueryParam if request.param[1] else VamanaQueryParam
    finally:
        reader.close()


def test_fast_query_matches_query_and_owns_results(collection):
    coll, vectors, param_type = collection
    query = np.ascontiguousarray(vectors[17] + 0.013, dtype=np.float32)
    for ef in (24, 80):
        param = param_type(
            **({"ef": ef} if param_type is HnswQueryParam else {"ef_search": ef}),
            is_using_refiner=False,
        )
        expected_docs = coll.query(
            Query(field_name="vector", vector=query, param=param),
            topk=10,
            output_fields=[],
        )
        expected = np.array([int(doc.id[4:]) for doc in expected_docs])
        ids = coll.fast_query("vector", query, param, topk=10)
        np.testing.assert_array_equal(ids, expected)
        assert ids.dtype == np.int64
        scored_ids, scores = coll.fast_query(
            "vector", query, param, topk=10, return_scores=True
        )
        np.testing.assert_array_equal(scored_ids, expected)
        np.testing.assert_allclose(
            scores, [doc.score for doc in expected_docs], rtol=1e-5, atol=1e-5
        )
        # Another search must not overwrite a capsule-owned result array.
        coll.fast_query("vector", vectors[25], param, topk=10)
        np.testing.assert_array_equal(ids, expected)


def test_preconditions_and_invalid_vectors(collection):
    coll, vectors, param_type = collection
    with pytest.raises(ValueError, match="dense vector field"):
        coll.fast_query("missing", vectors[0])
    wrong_param = (
        VamanaQueryParam() if param_type is HnswQueryParam else HnswQueryParam()
    )
    with pytest.raises(ValueError, match="parameter type"):
        coll.fast_query("vector", vectors[0], wrong_param)
    param = param_type(is_using_refiner=False)
    for invalid_query in (
        vectors[0, :-1],
        vectors[0].astype(np.float64),
        vectors[0].astype(np.int32),
        vectors[0].astype(np.complex64),
        np.zeros(32, dtype=[("x", np.float32)]),
        vectors[0].astype(">f4"),
        vectors[0, ::2],
        vectors[0].reshape(1, -1),
    ):
        with pytest.raises(ValueError, match="dtype|dimension|1D"):
            coll.fast_query("vector", invalid_query, param)


def test_reused_inline_and_default_params_and_close(collection):
    coll, vectors, param_type = collection
    query = np.ascontiguousarray(vectors[17] + 0.013)
    for ef in (80, 16, 64):
        settings = {"ef": ef} if param_type is HnswQueryParam else {"ef_search": ef}
        param = param_type(**settings)
        for topk in (1, 21, 3, 10):
            ids, scores = coll.fast_query(
                "vector", query, param, topk=topk, return_scores=True
            )
            assert len(ids) == len(scores) == topk
            assert scores.dtype == np.float32
            np.testing.assert_array_equal(
                ids, coll.fast_query("vector", query, param_type(**settings), topk=topk)
            )
            docs = coll.query(
                Query(field_name="vector", vector=query, param=param), topk=topk
            )
            np.testing.assert_array_equal(ids, [int(doc.id[4:]) for doc in docs])
        # A call with None uses defaults regardless of the preceding query.
        docs = coll.query(Query(field_name="vector", vector=query), topk=10)
        np.testing.assert_array_equal(
            coll.fast_query("vector", query), [int(doc.id[4:]) for doc in docs]
        )
    raw = coll._obj
    saved_ids, saved_scores = ids.copy(), scores.copy()
    for topk in (0, -1):
        empty_ids, empty_scores = coll.fast_query(
            "vector", query, param, topk=topk, return_scores=True
        )
        assert empty_ids.shape == empty_scores.shape == (0,)
        assert empty_ids.dtype == np.int64
        assert empty_scores.dtype == np.float32
    coll.close()
    np.testing.assert_array_equal(ids, saved_ids)
    np.testing.assert_array_equal(scores, saved_scores)
    for obj in (coll, raw):
        with pytest.raises(ValueError, match="closed"):
            obj.fast_query("vector", query, param)


def test_concurrent_calls_keep_query_parameters_local(collection):
    coll, vectors, param_type = collection
    queries = np.ascontiguousarray(vectors[[3, 17, 29, 41]] + 0.013)
    params = [None]
    for ef in (16, 64, 80):
        settings = {"ef": ef} if param_type is HnswQueryParam else {"ef_search": ef}
        params.append(param_type(**settings))
    topks = [1, 21, 3, 10]
    expected = []
    for query, param, topk in zip(queries, params, topks):
        docs = coll.query(Query("vector", vector=query, param=param), topk=topk)
        expected.append(
            ([int(doc.id[4:]) for doc in docs], [doc.score for doc in docs])
        )

    ready = Barrier(4)

    def search(worker):
        # Exercise the first calls concurrently, as well as repeated queries
        # with different topk, defaults and graph search parameters.
        ready.wait()
        for repeat in range(40):
            i = (worker + repeat) % len(queries)
            ids, scores = coll.fast_query(
                "vector", queries[i], params[i], topk=topks[i], return_scores=True
            )
            np.testing.assert_array_equal(ids, expected[i][0])
            np.testing.assert_allclose(scores, expected[i][1], rtol=2e-5, atol=2e-5)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(search, range(4)))


@pytest.mark.parametrize("compact", [False, True], ids=["delete_filter", "compacted"])
def test_internal_ids_survive_deletion_and_compaction(tmp_path, compact):
    vectors = np.random.default_rng(823).normal(size=(64, 32)).astype(np.float32)
    schema = CollectionSchema(
        name="deleted_ordinals",
        vectors=[
            VectorSchema(
                "vector",
                DataType.VECTOR_FP32,
                dimension=32,
                index_param=VamanaIndexParam(
                    metric_type=MetricType.L2,
                    max_degree=16,
                    search_list_size=32,
                    use_contiguous_memory=True,
                ),
            )
        ],
    )
    path = str(tmp_path / "deleted")
    writer = zvec.create_and_open(path, schema)
    assert all(
        result.ok()
        for result in writer.insert(
            [
                Doc(id=f"row-{i}", vectors={"vector": vector.tolist()})
                for i, vector in enumerate(vectors)
            ]
        )
    )
    writer.optimize()
    assert all(result.ok() for result in writer.delete([f"row-{i}" for i in range(8)]))
    if compact:
        writer.optimize()
    writer.close()
    coll = zvec.open(path, CollectionOption(read_only=True, enable_mmap=True))
    try:
        param = VamanaQueryParam(ef_search=64, is_using_refiner=False)
        query = vectors[12]
        expected = [
            int(doc.id[4:])
            for doc in coll.query(
                Query(field_name="vector", vector=query, param=param), topk=10
            )
        ]
        assert all(row >= 8 for row in expected)
        np.testing.assert_array_equal(coll.fast_query("vector", query, param), expected)
        docs = coll.query(
            Query(field_name="vector", vector=query, param=param), topk=10
        )
        ids, scores = coll.fast_query("vector", query, param, return_scores=True)
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(scores, [doc.score for doc in docs], rtol=1e-5)
        # Padding and result ownership on the filtered / compacted routes.
        padded_ids, padded_scores = coll.fast_query(
            "vector", query, param, topk=70, return_scores=True
        )
        assert len(padded_ids) == len(padded_scores) == 70
        assert np.all(padded_ids[56:] == -1)
        assert np.all(np.isnan(padded_scores[56:]))
    finally:
        coll.close()


def test_ids_are_merged_across_segments(tmp_path):
    vectors = np.random.default_rng(984).normal(size=(1100, 32)).astype(np.float32)
    schema = CollectionSchema(
        name="multiple_segments",
        vectors=[
            VectorSchema(
                "vector",
                DataType.VECTOR_FP32,
                dimension=32,
                index_param=VamanaIndexParam(
                    metric_type=MetricType.L2,
                    max_degree=16,
                    search_list_size=32,
                    use_contiguous_memory=True,
                ),
            )
        ],
    )
    # The native schema persists this limit, but the Python constructor does
    # not expose it. Restore a schema with the smallest allowed segment size.
    native_schema = schema._get_object()
    name, fields, _ = native_schema.__getstate__()
    restored = type(native_schema).__new__(type(native_schema))
    restored.__setstate__((name, fields, 1000))
    schema = CollectionSchema._from_core(restored)
    path = str(tmp_path / "segments")
    writer = zvec.create_and_open(path, schema)
    for start in range(0, len(vectors), 500):
        assert all(
            result.ok()
            for result in writer.insert(
                [
                    Doc(id=f"row-{i}", vectors={"vector": vectors[i].tolist()})
                    for i in range(start, min(start + 500, len(vectors)))
                ]
            )
        )
    writer.close()
    coll = zvec.open(path, CollectionOption(read_only=True, enable_mmap=True))
    try:
        param = VamanaQueryParam(ef_search=64, is_using_refiner=False)
        # Both the persisted segment and the final writing segment must
        # participate, with the latter's segment-local IDs mapped globally.
        for query_id in (12, 1050):
            query = vectors[query_id]
            expected = [
                int(doc.id[4:])
                for doc in coll.query(
                    Query(field_name="vector", vector=query, param=param), topk=10
                )
            ]
            assert expected[0] == query_id
            np.testing.assert_array_equal(
                coll.fast_query("vector", query, param), expected
            )
        docs = coll.query(
            Query(field_name="vector", vector=query, param=param), topk=10
        )
        ids, scores = coll.fast_query("vector", query, param, return_scores=True)
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_allclose(scores, [doc.score for doc in docs], rtol=1e-5)
    finally:
        coll.close()


def test_field_and_collection_caches_are_independent(tmp_path):
    rng = np.random.default_rng(726)
    vectors = {
        "l2": rng.normal(size=(64, 32)).astype(np.float32),
        "ip": rng.normal(size=(64, 16)).astype(np.float32),
    }
    schema = CollectionSchema(
        name="two_fields",
        vectors=[
            VectorSchema(
                name,
                DataType.VECTOR_FP32,
                dimension=values.shape[1],
                index_param=zvec.FlatIndexParam(
                    metric_type=MetricType.L2 if name == "l2" else MetricType.IP
                ),
            )
            for name, values in vectors.items()
        ],
    )
    readers = []
    try:
        for number in range(2):
            path = str(tmp_path / f"collection-{number}")
            writer = zvec.create_and_open(path, schema)
            assert all(
                s.ok()
                for s in writer.insert(
                    [
                        Doc(
                            id=str(i),
                            vectors={
                                name: values[i if number == 0 else 63 - i].tolist()
                                for name, values in vectors.items()
                            },
                        )
                        for i in range(64)
                    ]
                )
            )
            writer.optimize()
            writer.close()
            readers.append(zvec.open(path, CollectionOption(read_only=True)))
        for name in ("l2", "ip", "l2", "ip"):
            query = vectors[name][17]
            for reader in readers:
                docs = reader.query(Query(field_name=name, vector=query), topk=10)
                ids, scores = reader.fast_query(
                    name, query, topk=10, return_scores=True
                )
                np.testing.assert_array_equal(ids, [int(doc.id) for doc in docs])
                np.testing.assert_allclose(
                    scores, [doc.score for doc in docs], rtol=1e-5
                )
    finally:
        for reader in readers:
            reader.close()


@pytest.mark.parametrize(
    "index_kind",
    ["flat", "hnsw", "vamana", "ivf", "hnsw_rabitq", "ivf_rabitq", "diskann"],
)
@pytest.mark.parametrize("metric", [MetricType.L2, MetricType.IP, MetricType.COSINE])
def test_fast_query_index_and_metric_dispatch(tmp_path, index_kind, metric):
    """Exercise every dense index dispatch, including score normalization."""
    record = dict(metric_type=metric, quantize_type=QuantizeType.INT8)
    factories = {
        "flat": lambda: (zvec.FlatIndexParam(**record), None),
        "hnsw": lambda: (
            HnswIndexParam(m=16, ef_construction=64, **record),
            HnswQueryParam(ef=64),
        ),
        "vamana": lambda: (
            VamanaIndexParam(max_degree=16, search_list_size=64, **record),
            VamanaQueryParam(ef_search=64),
        ),
        "ivf": lambda: (
            zvec.IVFIndexParam(n_list=4, n_iters=2, **record),
            zvec.IVFQueryParam(nprobe=4),
        ),
        "hnsw_rabitq": lambda: (
            zvec.HnswRabitqIndexParam(
                metric_type=metric,
                m=16,
                ef_construction=64,
                total_bits=4,
                num_clusters=4,
                sample_count=256,
            ),
            zvec.HnswRabitqQueryParam(ef=64),
        ),
        "ivf_rabitq": lambda: (
            zvec.IvfRabitqIndexParam(metric_type=metric, nlist=4, total_bits=4),
            zvec.IvfRabitqQueryParam(nprobe=4),
        ),
        "diskann": lambda: (
            zvec.DiskAnnIndexParam(
                metric_type=metric, max_degree=16, list_size=64, pq_chunk_num=4
            ),
            zvec.DiskAnnQueryParam(list_size=64),
        ),
    }
    index, query_param = factories[index_kind]()
    vectors = np.random.default_rng(754).normal(size=(512, 128)).astype(np.float32)
    schema = CollectionSchema(
        name="fast_index_dispatch",
        vectors=[VectorSchema("vector", DataType.VECTOR_FP32, 128, index_param=index)],
    )
    path = str(tmp_path / "collection")
    try:
        writer = zvec.create_and_open(path, schema)
    except RuntimeError as exc:
        if "not supported on this platform" in str(exc) or "RabitQ requires AVX" in str(
            exc
        ):
            pytest.skip(str(exc))
        raise
    try:
        assert all(
            s.ok()
            for s in writer.insert(
                [
                    Doc(id=str(i), vectors={"vector": v.tolist()})
                    for i, v in enumerate(vectors)
                ]
            )
        )
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(path, CollectionOption(read_only=True))
    try:
        params = [query_param, None, query_param]
        if index_kind in ("hnsw", "vamana"):
            # Exercise call-local refiner parameters through both direct L2/IP
            # search and the cosine fallback, interleaved with defaults.
            params.extend(
                [
                    type(query_param)(is_using_refiner=True, scale_factor=1.5),
                    None,
                    type(query_param)(is_using_refiner=True, scale_factor=4.0),
                    query_param,
                ]
            )
        for param in params:
            for row in (12, 41):
                query = np.ascontiguousarray(vectors[row] + np.float32(0.021))
                docs = reader.query(Query("vector", vector=query, param=param), topk=10)
                ids, scores = reader.fast_query(
                    "vector", query, param, return_scores=True
                )
                np.testing.assert_array_equal(ids, [int(d.id) for d in docs])
                np.testing.assert_allclose(
                    scores, [d.score for d in docs], rtol=2e-5, atol=2e-5
                )
                np.testing.assert_array_equal(
                    ids, reader.fast_query("vector", query, param)
                )
    finally:
        reader.close()


@pytest.mark.parametrize("index_kind", ["vamana", "hnsw"])
@pytest.mark.parametrize(
    "quantizer",
    [
        zvec.QuantizeType.UNIFORM_UINT4,
        zvec.QuantizeType.UNIFORM_UINT7,
        zvec.QuantizeType.UNIFORM_UINT8,
    ],
)
def test_uniform_raw_fallback(tmp_path, index_kind, quantizer):
    vectors = np.random.default_rng(945).normal(size=(130, 32)).astype(np.float32)
    options = dict(
        metric_type=zvec.MetricType.L2,
        quantize_type=quantizer,
        flat_data_type=zvec.DataType.VECTOR_FP16,
        use_flat_contiguous_memory=True,
    )
    if index_kind == "vamana":
        index = zvec.VamanaIndexParam(max_degree=16, search_list_size=64, **options)
        param_type = zvec.VamanaQueryParam
    else:
        index = zvec.HnswIndexParam(m=16, ef_construction=64, **options)
        param_type = zvec.HnswQueryParam
    schema = zvec.CollectionSchema(
        name="uniform_fallback",
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, 32, index_param=index
            )
        ],
    )
    path = str(tmp_path / "collection")
    writer = zvec.create_and_open(path, schema)
    assert all(
        s.ok()
        for s in writer.insert(
            [
                zvec.Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                for i in range(128)
            ]
        )
    )
    writer.close()
    for phase in ("untrained", "optimized", "new_writes"):
        if phase != "untrained":
            writer = zvec.open(path)
            if phase == "optimized":
                writer.optimize()
            else:
                assert all(
                    s.ok()
                    for s in writer.insert(
                        [
                            zvec.Doc(id=str(i), vectors={"vector": vectors[i].tolist()})
                            for i in (128, 129)
                        ]
                    )
                )
            writer.close()
        reader = zvec.open(path, zvec.CollectionOption(read_only=True))
        try:
            for refine in (False, True):
                param = param_type(is_using_refiner=refine)
                query = vectors[128] if phase == "new_writes" else vectors[17]
                docs = reader.query(
                    zvec.Query("vector", vector=query, param=param), topk=10
                )
                ids, scores = reader.fast_query(
                    "vector", query, param, return_scores=True
                )
                np.testing.assert_array_equal(ids, [int(d.id) for d in docs])
                np.testing.assert_allclose(
                    scores, [d.score for d in docs], rtol=2e-5, atol=2e-5
                )
                np.testing.assert_array_equal(
                    ids, reader.fast_query("vector", query, param)
                )
                if phase == "new_writes":
                    assert ids[0] == 128
        finally:
            reader.close()
