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
"""HNSW contiguous storage is used only for read-only mmap opens.

Verify parameter persistence and query parity across writable/read-only opens.
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
    Query,
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
    vector_query = Query(
        field_name="dense",
        vector=query_vec,
        param=HnswQueryParam(ef=128),
    )
    hits = coll.query(vector_query, topk=TOPK)
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
            use_flat_contiguous_memory=True,
            flat_data_type=DataType.VECTOR_FP16,
        )
        data = param.to_dict()
        assert data["use_contiguous_memory"] is True
        # Make sure existing fields are still present.
        assert data["metric_type"] == "L2"
        assert data["m"] == 16
        assert data["ef_construction"] == 100
        assert data["quantize_type"] == "FP16"
        assert data["use_flat_contiguous_memory"] is True
        assert data["flat_data_type"] == "VECTOR_FP16"

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

    @pytest.mark.parametrize(
        "metric_type,quantize_type",
        [
            (MetricType.COSINE, QuantizeType.INT8),
            (MetricType.L2, QuantizeType.UNIFORM_UINT7),
            (MetricType.L2, QuantizeType.UNIFORM_UINT8),
            (MetricType.L2, QuantizeType.UNIFORM_UINT4),
        ],
    )
    def test_pickle_roundtrip(self, metric_type, quantize_type):
        original = HnswIndexParam(
            metric_type=metric_type,
            m=24,
            ef_construction=150,
            quantize_type=quantize_type,
            use_contiguous_memory=True,
            use_flat_contiguous_memory=True,
            flat_data_type=DataType.VECTOR_FP16,
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored.use_contiguous_memory is True
        assert restored.metric_type == metric_type
        assert restored.m == 24
        assert restored.ef_construction == 150
        assert restored.quantize_type == quantize_type
        assert restored.use_flat_contiguous_memory is True
        assert restored.flat_data_type == DataType.VECTOR_FP16
        assert restored.to_dict() == original.to_dict()


# ---------------------------------------------------------------------------
# 2) End-to-end: create collection, insert, query with contiguous memory on
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


# Persisted HNSW builds require mmap until BufferPool supports create_new.
@pytest.fixture(params=[True], ids=["mmap_on"])
def collection_option(request) -> CollectionOption:
    return CollectionOption(read_only=False, enable_mmap=request.param)


@pytest.fixture
def build_index_supported(collection_option: CollectionOption) -> bool:
    return bool(collection_option.enable_mmap)


_BUILD_NOT_SUPPORTED_FRAGMENTS = ("not yet supported", "enable_mmap=false")


class TestHnswContiguousMemoryEndToEnd:
    """End-to-end: schema -> create_and_open -> insert -> query works."""

    def test_create_with_contiguous_memory_and_query(
        self,
        tmp_path_factory,
        collection_option,
        rng,
    ):
        schema = _build_schema("hnsw_contig_create", use_contiguous_memory=True)

        path = tmp_path_factory.mktemp("zvec") / "hnsw_contig_create"
        coll = zvec.create_and_open(
            path=str(path), schema=schema, option=collection_option
        )
        try:
            vec_schema = coll.schema.vectors[0]
            assert vec_schema.index_param.use_contiguous_memory is True

            docs = _generate_docs(rng)
            insert_result = coll.insert(docs=docs)
            for r in insert_result:
                assert r.ok(), f"insert failed: code={r.code()}"
            assert coll.stats.doc_count == NUM_DOCS

            coll.optimize()
            assert _debug_hnsw_storage_mode(coll) == "mmap"

            query_vec = docs[0].vector("dense")
            ids = _assert_query_matches(coll, query_vec)
            assert ids[0] == "0", f"expected self-recall, got top-1 id={ids[0]}"

            coll.close()
            coll = zvec.open(path=str(path), option=CollectionOption(read_only=True))
            assert _debug_hnsw_storage_mode(coll) == "contiguous"
            assert _assert_query_matches(coll, query_vec) == ids
        finally:
            coll.close()

    def test_create_without_contiguous_memory_uses_mmap_entity(
        self,
        tmp_path_factory,
        collection_option,
        rng,
    ):
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
            assert _debug_hnsw_storage_mode(coll) == "mmap", (
                "use_contiguous_memory=False + enable_mmap=True should "
                "produce the mmap entity"
            )

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
        created.flush()
        created.close()

        reopened = zvec.open(path=path_str, option=collection_option)
        try:
            assert reopened is not None
            assert reopened.stats.doc_count == NUM_DOCS

            vec_schema = reopened.schema.vectors[0]
            assert vec_schema.index_param.use_contiguous_memory is True

            reopened.optimize()
            assert _debug_hnsw_storage_mode(reopened) == "mmap"

            query_vec = docs[7].vector("dense")
            ids = _assert_query_matches(reopened, query_vec)
            assert ids[0] == "7"

            fetched = reopened.fetch([d.id for d in docs[:10]])
            assert len(fetched) == 10

            reopened.close()
            reopened = zvec.open(path=path_str, option=CollectionOption(read_only=True))
            assert reopened.schema.vectors[0].index_param.use_contiguous_memory is True
            assert reopened.stats.doc_count == NUM_DOCS
            assert _debug_hnsw_storage_mode(reopened) == "contiguous"
            assert _assert_query_matches(reopened, query_vec) == ids
            assert len(reopened.fetch([d.id for d in docs[:10]])) == 10
        finally:
            reopened.close()

    def test_result_parity_with_and_without_contiguous_memory(
        self,
        tmp_path_factory,
        rng,
    ):
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
                assert _debug_hnsw_storage_mode(coll) == "mmap"
                writable_ids = _assert_query_matches(coll, query_vec)

                coll.close()
                coll = zvec.open(
                    path=str(path), option=CollectionOption(read_only=True)
                )
                expected_mode = "contiguous" if flag else "mmap"
                assert _debug_hnsw_storage_mode(coll) == expected_mode, (
                    f"{tag}: unexpected entity type"
                )
                readonly_ids = _assert_query_matches(coll, query_vec)
                assert readonly_ids == writable_ids
                return readonly_ids
            finally:
                coll.close()

        ids_off = _build_and_query("off", flag=False)
        ids_on = _build_and_query("on", flag=True)

        assert ids_on == ids_off, (
            f"top-{TOPK} results diverged between use_contiguous_memory modes: "
            f"on={ids_on}, off={ids_off}"
        )
        assert ids_on[0] == "3"
