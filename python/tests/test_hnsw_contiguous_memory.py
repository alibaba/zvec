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
Tests for the ``use_contiguous_memory`` optimization on ``HnswIndexParam``.

The HNSW streamer supports two allocation strategies for graph nodes:

* ``use_contiguous_memory=False`` (default): each node allocates its own
  linked buffer. Lower peak memory usage, worse cache locality.
* ``use_contiguous_memory=True``: a single contiguous arena holds every
  node. Higher peak memory usage, better cache locality and search
  throughput.

These tests exercise the Python surface end-to-end and make sure that
when a collection is created / reopened with ``use_contiguous_memory=True``
the underlying HNSW streamer entity is constructed correctly and serves
search traffic.
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
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    VectorQuery,
    VectorSchema,
)
from zvec.typing import DataType, IndexType, MetricType, QuantizeType


DIMENSION = 32
NUM_DOCS = 128
TOPK = 5

# ---------------------------------------------------------------------------


def _debug_hnsw_storage_mode(coll: Collection, column: str = "dense") -> str:
    """Return the internal HNSW entity storage mode for ``column``.

    Exposes the debug-only introspection hook on the pybind11 ``_Collection``.
    Only meaningful after ``optimize()`` has built a persisted HNSW index; on
    a pure writing segment it will raise ``KeyError``.
    """
    underlying = coll._obj  # type: ignore[attr-defined]
    return underlying._debug_hnsw_storage_mode(column)


def _build_schema(name: str, *, use_contiguous_memory: bool) -> CollectionSchema:
    """Create a simple schema with a single FP32 HNSW vector column."""
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
                index_param=HnswIndexParam(
                    metric_type=MetricType.IP,
                    m=16,
                    ef_construction=100,
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


def _assert_query_matches(coll: Collection, query_vec: list[float]) -> list[str]:
    """Run a top-k vector query and return the returned ids in order."""
    vector_query = VectorQuery(
        field_name="dense",
        vector=query_vec,
        param=HnswQueryParam(ef=128),
    )
    hits = coll.query(vectors=vector_query, topk=TOPK)
    # Expect a single result group for the single vector query.
    assert hits is not None, "query returned None"
    assert len(hits) >= 1, f"expected at least one hit, got {hits!r}"
    return [doc.id for doc in hits]


# ---------------------------------------------------------------------------
# 1) Pure Python surface: construction / property / to_dict / repr / pickle
# ---------------------------------------------------------------------------


class TestHnswIndexParamContiguousMemorySurface:
    """Verify the Python binding exposes ``use_contiguous_memory`` correctly."""

    def test_default_is_false(self):
        param = HnswIndexParam()
        assert param.use_contiguous_memory is False

    def test_custom_true(self):
        param = HnswIndexParam(use_contiguous_memory=True)
        assert param.use_contiguous_memory is True
        assert param.type == IndexType.HNSW
        # other fields keep their default values
        assert param.m == 50
        assert param.ef_construction == 500

    def test_to_dict_includes_use_contiguous_memory(self):
        param = HnswIndexParam(
            metric_type=MetricType.L2,
            m=16,
            ef_construction=100,
            quantize_type=QuantizeType.FP16,
            use_contiguous_memory=True,
        )
        data = param.to_dict()
        assert data["use_contiguous_memory"] is True
        # Make sure existing fields are still present.
        assert data["metric_type"] == "L2"
        assert data["m"] == 16
        assert data["ef_construction"] == 100
        assert data["quantize_type"] == "FP16"

    def test_repr_contains_flag(self):
        on = repr(HnswIndexParam(use_contiguous_memory=True))
        off = repr(HnswIndexParam(use_contiguous_memory=False))
        assert "use_contiguous_memory" in on
        assert "use_contiguous_memory" in off
        assert "true" in on
        assert "false" in off

    def test_readonly_property(self):
        param = HnswIndexParam(use_contiguous_memory=True)
        if sys.version_info >= (3, 11):
            match_pattern = r"(can't set attribute|has no setter|readonly attribute)"
        else:
            match_pattern = r"can't set attribute"
        with pytest.raises(AttributeError, match=match_pattern):
            param.use_contiguous_memory = False  # type: ignore[misc]

    def test_pickle_roundtrip(self):
        original = HnswIndexParam(
            metric_type=MetricType.COSINE,
            m=24,
            ef_construction=150,
            quantize_type=QuantizeType.INT8,
            use_contiguous_memory=True,
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.use_contiguous_memory is True
        assert restored.metric_type == MetricType.COSINE
        assert restored.m == 24
        assert restored.ef_construction == 150
        assert restored.quantize_type == QuantizeType.INT8


# ---------------------------------------------------------------------------
# 2) End-to-end: create collection, insert, query with contiguous memory on
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# NOTE: the ``enable_mmap=False`` (BufferPool) variant is intentionally
# omitted from this fixture. Building a persisted HNSW index via
# ``optimize()`` / ``create_vector_index`` / ``drop_vector_index``
# currently requires mmap-backed storage, because the BufferPool backend
# has not implemented the ``create_new`` semantics yet and the guard in
# ``SegmentImpl::merge_vector_indexer`` rejects that combination. Once
# BufferPool gains write support, re-add ``False`` to ``params`` (and
# drop the guard in segment.cc) so these end-to-end tests cover both
# storage modes again.
@pytest.fixture(params=[True], ids=["mmap_on"])
def collection_option(request) -> CollectionOption:
    return CollectionOption(read_only=False, enable_mmap=request.param)


# Building a new persisted HNSW index currently requires mmap-backed storage
# because the BufferPool backend has not implemented `create_new` semantics
# yet. Collections opened with ``enable_mmap=False`` therefore cannot run
# optimize()/create_vector_index/drop_vector_index. Tests use this fixture
# to know which behaviour to assert, and once BufferPool gains write support
# the guard in segment.cc (and these branches) can be removed together.
@pytest.fixture
def build_index_supported(collection_option: CollectionOption) -> bool:
    return bool(collection_option.enable_mmap)


# Error message fragments emitted by the NotSupported guard in
# SegmentImpl::merge_vector_indexer / drop_vector_index. If the C++ message
# changes, update these together.
_BUILD_NOT_SUPPORTED_FRAGMENTS = ("not yet supported", "enable_mmap=false")


class TestHnswContiguousMemoryEndToEnd:
    """End-to-end: schema -> create_and_open -> insert -> query works."""

    def test_create_with_contiguous_memory_and_query(
        self,
        tmp_path_factory,
        collection_option,
        rng,
    ):
        """With the flag on, the schema round-trips and search works end-to-end.

        After ``optimize()`` the writing segment is compacted into a persisted
        segment backed by the configured HNSW entity. We assert both the
        user-observable behaviour (schema + search) and, via the debug hook,
        that the entity type actually honours ``use_contiguous_memory``.
        """
        schema = _build_schema("hnsw_contig_create", use_contiguous_memory=True)

        path = tmp_path_factory.mktemp("zvec") / "hnsw_contig_create"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            # Schema round-trips with the flag set.
            vec_schema = coll.schema.vectors[0]
            assert vec_schema.index_param.use_contiguous_memory is True

            docs = _generate_docs(rng)
            insert_result = coll.insert(docs=docs)
            for r in insert_result:
                assert r.ok(), f"insert failed: code={r.code()}"
            assert coll.stats.doc_count == NUM_DOCS

            # Build persisted HNSW index; this is where the contiguous entity
            # is actually instantiated.
            coll.optimize()
            assert _debug_hnsw_storage_mode(coll) == "contiguous", (
                "use_contiguous_memory=True should produce a contiguous entity"
            )

            # Pick an existing vector as the query; top-1 must be itself.
            query_vec = docs[0].vector("dense")
            ids = _assert_query_matches(coll, query_vec)
            assert ids[0] == "0", f"expected self-recall, got top-1 id={ids[0]}"
        finally:
            coll.destroy()

    def test_create_without_contiguous_memory_uses_mmap_entity(
        self,
        tmp_path_factory,
        collection_option,
        rng,
    ):
        """Baseline: when the flag is omitted the default (mmap) entity is used."""
        schema = _build_schema("hnsw_contig_default", use_contiguous_memory=False)
        path = tmp_path_factory.mktemp("zvec") / "hnsw_contig_default"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            vec_schema = coll.schema.vectors[0]
            assert vec_schema.index_param.use_contiguous_memory is False

            docs = _generate_docs(rng)
            for r in coll.insert(docs=docs):
                assert r.ok()
            assert coll.stats.doc_count == NUM_DOCS

            coll.optimize()
            # With the flag off and mmap on, the persisted entity must be the
            # default mmap layout — specifically, not the contiguous arena.
            assert _debug_hnsw_storage_mode(coll) == "mmap", (
                "use_contiguous_memory=False + enable_mmap=True should "
                "produce the mmap entity"
            )

            # Search still functions with the default entity backing.
            query_vec = docs[0].vector("dense")
            ids = _assert_query_matches(coll, query_vec)
            assert ids[0] == "0"
        finally:
            coll.destroy()

    def test_close_and_reopen_with_contiguous_memory(
        self,
        tmp_path_factory,
        collection_option,
        rng,
    ):
        """Reopening a collection must preserve the ``use_contiguous_memory`` flag.

        The core property: the flag survives the schema persist/reload
        round-trip so the HNSW streamer entity — constructed lazily on first
        persisted-segment build — honours the user's choice. We run
        ``optimize()`` after reopen and confirm the contiguous entity was
        materialized.
        """
        schema = _build_schema("hnsw_contig_reopen", use_contiguous_memory=True)
        path = tmp_path_factory.mktemp("zvec") / "hnsw_contig_reopen"
        path_str = str(path)

        created = zvec.create_and_open(
            path=path_str, schema=schema, option=collection_option
        )
        docs = _generate_docs(rng)
        for r in created.insert(docs=docs):
            assert r.ok()
        assert created.stats.doc_count == NUM_DOCS
        # Persist pending writes so that reopen reconstructs state from disk.
        created.flush()
        del created  # close the handle

        reopened = zvec.open(path=path_str, option=collection_option)
        try:
            assert reopened is not None
            assert reopened.stats.doc_count == NUM_DOCS

            # Schema persisted the flag across the reopen boundary.
            vec_schema = reopened.schema.vectors[0]
            assert vec_schema.index_param.use_contiguous_memory is True

            reopened.optimize()
            assert _debug_hnsw_storage_mode(reopened) == "contiguous"

            # Entity actually works: exact self-recall + fetch parity.
            query_vec = docs[7].vector("dense")
            ids = _assert_query_matches(reopened, query_vec)
            assert ids[0] == "7"

            fetched = reopened.fetch([d.id for d in docs[:10]])
            assert len(fetched) == 10
        finally:
            reopened.destroy()

    def test_result_parity_with_and_without_contiguous_memory(
        self,
        tmp_path_factory,
        rng,
    ):
        """
        Two collections built from the same documents must return the same
        top-k neighbors regardless of whether contiguous memory is enabled:
        the flag is a memory-layout optimization and must not alter recall
        for identical graph construction parameters on the same data.
        """
        docs = _generate_docs(rng)
        query_vec = docs[3].vector("dense")

        def _build_and_query(tag: str, flag: bool) -> list[str]:
            schema = _build_schema(f"hnsw_parity_{tag}", use_contiguous_memory=flag)
            option = CollectionOption(read_only=False, enable_mmap=True)
            path = tmp_path_factory.mktemp("zvec") / f"hnsw_parity_{tag}"
            coll = zvec.create_and_open(path=str(path), schema=schema, option=option)
            try:
                for r in coll.insert(docs=docs):
                    assert r.ok()
                coll.optimize()
                expected_mode = "contiguous" if flag else "mmap"
                assert _debug_hnsw_storage_mode(coll) == expected_mode, (
                    f"{tag}: unexpected entity type"
                )
                return _assert_query_matches(coll, query_vec)
            finally:
                coll.destroy()

        ids_off = _build_and_query("off", flag=False)
        ids_on = _build_and_query("on", flag=True)

        # The graph is built with the same (m, ef_construction, data, order),
        # so top-k results must match exactly.
        assert ids_on == ids_off, (
            f"top-{TOPK} results diverged between use_contiguous_memory modes: "
            f"on={ids_on}, off={ids_off}"
        )
        # Sanity: self-recall is still perfect.
        assert ids_on[0] == "3"
