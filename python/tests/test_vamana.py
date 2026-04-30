# Copyright 2025-present the zvec project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the Python entry point of the Vamana (DiskANN) dense vector index.

Mirrors the structure of ``test_hnsw_contiguous_memory.py`` (the closest
hnsw dense reference), and is split into two parts:

1. **Surface tests** — verify that ``VamanaIndexParam`` / ``VamanaQueryParam``
   are correctly bound: construction defaults, readonly properties,
   ``to_dict``, ``__repr__``, pickle round-trip, and that they appear in the
   public ``zvec`` namespace with the expected ``IndexType.VAMANA`` value.

2. **End-to-end tests** — build a collection that uses Vamana on a dense
   FP32 column, insert deterministic documents, then run a top-k query
   through ``VamanaQueryParam`` on both the writer segment and the
   persisted (post-``optimize()``) segment.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest

import zvec
from zvec import (
    Collection,
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    InvertIndexParam,
    VamanaIndexParam,
    VamanaQueryParam,
    VectorQuery,
    VectorSchema,
)
from zvec.typing import DataType, IndexType, MetricType, QuantizeType

DIMENSION = 32
NUM_DOCS = 128
TOPK = 5

# Defaults pulled from src/include/zvec/core/interface/constants.h. Keep
# in sync with kDefaultVamana* if the engine defaults ever change.
DEFAULT_MAX_DEGREE = 64
DEFAULT_SEARCH_LIST_SIZE = 100
DEFAULT_ALPHA = 1.2
DEFAULT_EF_SEARCH = 200
DEFAULT_SATURATE_GRAPH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_schema(
    name: str,
    *,
    metric_type: MetricType = MetricType.IP,
    max_degree: int = 32,
    search_list_size: int = 64,
    alpha: float = 1.2,
    use_contiguous_memory: bool = False,
) -> CollectionSchema:
    """Create a simple schema with a single FP32 Vamana vector column."""
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema(
                "id",
                DataType.INT64,
                nullable=False,
                index_param=InvertIndexParam(enable_range_optimization=True),
            ),
        ],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=VamanaIndexParam(
                    metric_type=metric_type,
                    max_degree=max_degree,
                    search_list_size=search_list_size,
                    alpha=alpha,
                    use_contiguous_memory=use_contiguous_memory,
                ),
            ),
        ],
    )


def _generate_docs(rng: np.random.Generator, num: int = NUM_DOCS) -> list[Doc]:
    """Produce deterministic documents for insertion."""
    docs: list[Doc] = []
    for i in range(num):
        vec = rng.standard_normal(DIMENSION).astype(np.float32)
        docs.append(
            Doc(
                id=str(i),
                fields={"id": i},
                vectors={"dense": vec.tolist()},
            )
        )
    return docs


def _query_topk(
    coll: Collection, query_vec: list[float], *, ef_search: int = 64
) -> list[str]:
    """Run a top-k vector query and return the returned ids in order."""
    vector_query = VectorQuery(
        field_name="dense",
        vector=query_vec,
        param=VamanaQueryParam(ef_search=ef_search),
    )
    hits = coll.query(vectors=vector_query, topk=TOPK)
    assert hits is not None, "query returned None"
    assert len(hits) >= 1, f"expected at least one hit, got {hits!r}"
    return [doc.id for doc in hits]


# ---------------------------------------------------------------------------
# 1) Surface: construction / property / to_dict / repr / pickle / namespace
# ---------------------------------------------------------------------------


class TestVamanaIndexParamSurface:
    """Verify the Python binding for ``VamanaIndexParam``."""

    def test_defaults(self):
        param = VamanaIndexParam()
        assert param.type == IndexType.VAMANA
        assert param.metric_type == MetricType.IP
        assert param.max_degree == DEFAULT_MAX_DEGREE
        assert param.search_list_size == DEFAULT_SEARCH_LIST_SIZE
        assert param.alpha == pytest.approx(DEFAULT_ALPHA)
        assert param.saturate_graph is DEFAULT_SATURATE_GRAPH
        assert param.use_contiguous_memory is False
        assert param.use_id_map is False
        assert param.quantize_type == QuantizeType.UNDEFINED

    def test_custom_construction(self):
        param = VamanaIndexParam(
            metric_type=MetricType.COSINE,
            max_degree=48,
            search_list_size=128,
            alpha=1.5,
            saturate_graph=True,
            use_contiguous_memory=True,
            use_id_map=False,
            quantize_type=QuantizeType.INT8,
        )
        assert param.type == IndexType.VAMANA
        assert param.metric_type == MetricType.COSINE
        assert param.max_degree == 48
        assert param.search_list_size == 128
        assert param.alpha == pytest.approx(1.5)
        assert param.saturate_graph is True
        assert param.use_contiguous_memory is True
        assert param.use_id_map is False
        assert param.quantize_type == QuantizeType.INT8

    def test_to_dict_includes_all_fields(self):
        param = VamanaIndexParam(
            metric_type=MetricType.L2,
            max_degree=32,
            search_list_size=80,
            alpha=1.3,
            saturate_graph=True,
            use_contiguous_memory=True,
            use_id_map=False,
            quantize_type=QuantizeType.FP16,
        )
        data = param.to_dict()
        assert data["type"] == "VAMANA"
        assert data["metric_type"] == "L2"
        assert data["max_degree"] == 32
        assert data["search_list_size"] == 80
        assert data["alpha"] == pytest.approx(1.3)
        assert data["saturate_graph"] is True
        assert data["use_contiguous_memory"] is True
        assert data["use_id_map"] is False
        assert data["quantize_type"] == "FP16"

    def test_repr_contains_key_fields(self):
        text = repr(
            VamanaIndexParam(
                metric_type=MetricType.COSINE,
                max_degree=24,
                search_list_size=72,
                alpha=1.4,
                saturate_graph=True,
                use_contiguous_memory=True,
            )
        )
        # Spot-check the most diagnostic fields are rendered.
        assert "VAMANA" in text
        assert "COSINE" in text
        assert "max_degree" in text and "24" in text
        assert "search_list_size" in text and "72" in text
        assert "alpha" in text
        assert "saturate_graph" in text and "true" in text
        assert "use_contiguous_memory" in text and "true" in text

    @pytest.mark.parametrize(
        "field, kwargs",
        [
            ("max_degree", dict(max_degree=99)),
            ("search_list_size", dict(search_list_size=99)),
            ("alpha", dict(alpha=1.7)),
            ("saturate_graph", dict(saturate_graph=True)),
            ("use_contiguous_memory", dict(use_contiguous_memory=True)),
            ("use_id_map", dict(use_id_map=True)),
        ],
    )
    def test_readonly_properties(self, field, kwargs):
        param = VamanaIndexParam(**kwargs)
        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            setattr(param, field, getattr(param, field))

    def test_pickle_roundtrip(self):
        original = VamanaIndexParam(
            metric_type=MetricType.COSINE,
            max_degree=48,
            search_list_size=120,
            alpha=1.4,
            saturate_graph=True,
            use_contiguous_memory=True,
            use_id_map=False,
            quantize_type=QuantizeType.INT8,
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.type == IndexType.VAMANA
        assert restored.metric_type == MetricType.COSINE
        assert restored.max_degree == 48
        assert restored.search_list_size == 120
        assert restored.alpha == pytest.approx(1.4)
        assert restored.saturate_graph is True
        assert restored.use_contiguous_memory is True
        assert restored.use_id_map is False
        assert restored.quantize_type == QuantizeType.INT8
        # to_dict equality is the strongest end-to-end equivalence we have.
        assert restored.to_dict() == original.to_dict()


class TestVamanaQueryParamSurface:
    """Verify the Python binding for ``VamanaQueryParam``."""

    def test_defaults(self):
        q = VamanaQueryParam()
        assert q.type == IndexType.VAMANA
        assert q.ef_search == DEFAULT_EF_SEARCH
        assert q.radius == pytest.approx(0.0)
        assert q.is_linear is False
        assert q.is_using_refiner is False

    def test_custom_construction(self):
        q = VamanaQueryParam(
            ef_search=300, radius=0.5, is_linear=True, is_using_refiner=True
        )
        assert q.type == IndexType.VAMANA
        assert q.ef_search == 300
        assert q.radius == pytest.approx(0.5)
        assert q.is_linear is True
        assert q.is_using_refiner is True

    def test_repr_contains_key_fields(self):
        text = repr(VamanaQueryParam(ef_search=128, radius=0.25))
        assert "VAMANA" in text
        assert "ef_search" in text and "128" in text
        assert "radius" in text

    def test_readonly_ef_search(self):
        q = VamanaQueryParam(ef_search=100)
        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            q.ef_search = 200  # type: ignore[misc]

    def test_pickle_roundtrip(self):
        original = VamanaQueryParam(
            ef_search=256, radius=0.3, is_linear=False, is_using_refiner=True
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.type == IndexType.VAMANA
        assert restored.ef_search == 256
        assert restored.radius == pytest.approx(0.3)
        assert restored.is_linear is False
        assert restored.is_using_refiner is True


class TestVamanaPublicNamespace:
    """The Vamana entry points must be importable from the top-level ``zvec``."""

    def test_top_level_exports(self):
        assert zvec.VamanaIndexParam is VamanaIndexParam
        assert zvec.VamanaQueryParam is VamanaQueryParam
        assert "VamanaIndexParam" in zvec.__all__
        assert "VamanaQueryParam" in zvec.__all__

    def test_index_type_enum_member(self):
        # Sanity: the IndexType enum exposes VAMANA and it is what the
        # bound params advertise.
        assert IndexType.VAMANA is not None
        assert VamanaIndexParam().type == IndexType.VAMANA
        assert VamanaQueryParam().type == IndexType.VAMANA


# ---------------------------------------------------------------------------
# 2) End-to-end: create collection, insert, query through the writer segment
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# Mirror the hnsw dense test fixture: only the mmap-backed variant is
# currently usable for vector index construction. BufferPool (enable_mmap=
# False) is intentionally omitted because the same write-path guard in
# ``SegmentImpl::merge_vector_indexer`` rejects that combination.
@pytest.fixture(params=[True], ids=["mmap_on"])
def collection_option(request) -> CollectionOption:
    return CollectionOption(read_only=False, enable_mmap=request.param)


class TestVamanaEndToEnd:
    """End-to-end: schema -> create_and_open -> insert -> query works."""

    def test_schema_round_trip(self, tmp_path_factory, collection_option):
        """The Vamana index params survive the schema persist path."""
        schema = _build_schema(
            "vamana_schema_rt",
            metric_type=MetricType.COSINE,
            max_degree=32,
            search_list_size=80,
            alpha=1.3,
            use_contiguous_memory=True,
        )
        path = tmp_path_factory.mktemp("zvec") / "vamana_schema_rt"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            vec_schema = coll.schema.vectors[0]
            ip = vec_schema.index_param
            assert ip.type == IndexType.VAMANA
            assert ip.metric_type == MetricType.COSINE
            assert ip.max_degree == 32
            assert ip.search_list_size == 80
            assert ip.alpha == pytest.approx(1.3)
            assert ip.use_contiguous_memory is True
        finally:
            coll.destroy()

    def test_insert_and_query_self_recall(
        self, tmp_path_factory, collection_option, rng
    ):
        """Top-1 of a query equal to an inserted vector must be that vector.

        Exercises the writer-segment Vamana streamer end-to-end through the
        Python entry point: ``VamanaIndexParam`` for build and
        ``VamanaQueryParam`` for search.
        """
        schema = _build_schema("vamana_e2e_recall")
        path = tmp_path_factory.mktemp("zvec") / "vamana_e2e_recall"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok(), f"insert failed: code={r.code()}"
            assert coll.stats.doc_count == NUM_DOCS

            # Self-recall: query with the i-th inserted vector, expect id i
            # to be the top result.
            for probe in (0, 7, 42, NUM_DOCS - 1):
                query_vec = docs[probe].vector("dense")
                ids = _query_topk(coll, query_vec)
                assert ids[0] == str(probe), (
                    f"expected self-recall at probe={probe}, got top-1 id={ids[0]} "
                    f"(top-{TOPK}={ids})"
                )
        finally:
            coll.destroy()

    def test_query_param_ef_search_affects_only_quality(
        self, tmp_path_factory, collection_option, rng
    ):
        """``ef_search`` is a search-time knob and must not crash for any
        sensible value. Larger ``ef_search`` should be at least as good as
        smaller for self-recall."""
        schema = _build_schema("vamana_e2e_ef")
        path = tmp_path_factory.mktemp("zvec") / "vamana_e2e_ef"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok()

            query_vec = docs[3].vector("dense")
            ids_small = _query_topk(coll, query_vec, ef_search=16)
            ids_large = _query_topk(coll, query_vec, ef_search=256)

            # Both should self-recall the probe vector at top-1.
            assert ids_small[0] == "3"
            assert ids_large[0] == "3"
            assert len(ids_small) == TOPK
            assert len(ids_large) == TOPK
        finally:
            coll.destroy()

    def test_optimize_then_query(self, tmp_path_factory, collection_option, rng):
        """The persisted Vamana segment built by ``optimize()`` must serve
        queries correctly.

        Until the cmake fix to force-load ``core_knn_vamana_static`` into the
        ``_zvec`` pybind module, this path failed at ``VamanaStreamer``
        creation because the global factory registration in
        ``vamana_streamer.cc`` was never linked in. This test pins down the
        regression.
        """
        schema = _build_schema("vamana_e2e_optimize")
        path = tmp_path_factory.mktemp("zvec") / "vamana_e2e_optimize"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok()
            assert coll.stats.doc_count == NUM_DOCS

            # Snapshot the writer-segment top-k for a probe vector.
            query_vec = docs[5].vector("dense")
            ids_pre = _query_topk(coll, query_vec)
            assert ids_pre[0] == "5"

            # Trigger persisted segment build. Pre-fix this raised
            # RuntimeError("Failed to create index").
            coll.optimize()

            # Persisted segment must still serve queries with the same
            # top-1 self-recall guarantee. We do not assert full top-k
            # equality with the writer segment because the persisted
            # streamer may visit nodes in a different order; top-1 self-
            # recall is the strong invariant.
            ids_post = _query_topk(coll, query_vec)
            assert ids_post[0] == "5", (
                f"post-optimize top-1 should still be probe id, got {ids_post}"
            )
            assert len(ids_post) == TOPK
        finally:
            coll.destroy()
