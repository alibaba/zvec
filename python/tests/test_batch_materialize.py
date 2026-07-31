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
"""Equivalence tests for the batch-materialized query path.

`Collection.query` now goes through `_Collection.QueryMaterialized`, which
batch-materializes all hits into tuples in a single C++ call. These tests
verify that the results are identical to the legacy per-doc path
(`_Collection.Query` + `convert_to_py_doc`).
"""

from __future__ import annotations

import numpy as np
import pytest
import zvec
from zvec import (
    Collection,
    CollectionOption,
    DataType,
    Doc,
    FieldSchema,
    HnswIndexParam,
    HnswQueryParam,
    Query,
    RrfReRanker,
    VectorSchema,
)
from zvec.model.convert import convert_to_py_doc

DIM = 16
N_DOCS = 200


@pytest.fixture(scope="module")
def bm_collection(tmp_path_factory) -> Collection:
    schema = zvec.CollectionSchema(
        name="batch_mat_test",
        fields=[
            FieldSchema("num", DataType.INT64, nullable=False),
            FieldSchema("title", DataType.STRING, nullable=True),
        ],
        vectors=[
            VectorSchema(
                "vec",
                DataType.VECTOR_FP32,
                dimension=DIM,
                index_param=HnswIndexParam(),
            ),
        ],
    )
    path = tmp_path_factory.mktemp("zvec_batch_mat") / "coll"
    coll = zvec.create_and_open(
        path=str(path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )

    rng = np.random.default_rng(42)
    docs = [
        Doc(
            id=str(i),
            fields={"num": i, "title": f"doc-{i}"},
            vectors={"vec": rng.random(DIM, dtype=np.float32)},
        )
        for i in range(N_DOCS)
    ]
    for r in coll.insert(docs):
        assert r.ok()

    yield coll

    try:
        coll.destroy()
    except Exception:
        pass


def legacy_query(coll: Collection, query: Query, **kwargs) -> list[Doc]:
    """Old per-doc materialization path, kept as the reference behavior."""
    ctx_docs = coll._querier._build_queries(
        _make_ctx(coll, query, **kwargs), coll._obj
    )
    raw = coll._obj.Query(ctx_docs[0])
    return [convert_to_py_doc(d, coll._querier._schema) for d in raw]


def _make_ctx(coll: Collection, query: Query, **kwargs):
    from zvec.executor.query_executor import QueryContext

    return QueryContext(
        topk=kwargs.get("topk", 10),
        filter=kwargs.get("filter"),
        queries=[query],
        include_vector=kwargs.get("include_vector", False),
        output_fields=kwargs.get("output_fields"),
    )


def assert_docs_equal(new_docs: list[Doc], old_docs: list[Doc]):
    assert len(new_docs) == len(old_docs)
    for nd, od in zip(new_docs, old_docs):
        assert isinstance(nd, Doc)
        assert nd.id == od.id
        assert nd.score == pytest.approx(od.score)
        assert nd.fields == od.fields
        assert set(nd.vectors.keys()) == set(od.vectors.keys())
        for name, vec in nd.vectors.items():
            assert vec == pytest.approx(od.vectors[name])


class TestBatchMaterializeEquivalence:
    def _query_vec(self):
        return [0.5] * DIM

    def test_basic_equivalence(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        new_docs = bm_collection.query(q, topk=20)
        old_docs = legacy_query(bm_collection, q, topk=20)
        assert len(new_docs) == 20
        assert_docs_equal(new_docs, old_docs)
        # scalar fields fully materialized
        for d in new_docs:
            assert set(d.fields.keys()) == {"num", "title"}
            assert d.vectors == {}

    @pytest.mark.parametrize("include_vector", [False, True])
    def test_include_vector(self, bm_collection: Collection, include_vector: bool):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        new_docs = bm_collection.query(q, topk=5, include_vector=include_vector)
        old_docs = legacy_query(
            bm_collection, q, topk=5, include_vector=include_vector
        )
        assert_docs_equal(new_docs, old_docs)
        for d in new_docs:
            assert bool(d.vectors) is include_vector
            if include_vector:
                assert len(d.vectors["vec"]) == DIM

    def test_output_fields_subset(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        new_docs = bm_collection.query(q, topk=5, output_fields=["num"])
        old_docs = legacy_query(bm_collection, q, topk=5, output_fields=["num"])
        assert_docs_equal(new_docs, old_docs)
        for d in new_docs:
            assert set(d.fields.keys()) == {"num"}

    def test_filter_equivalence(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        new_docs = bm_collection.query(q, topk=10, filter="num < 50")
        old_docs = legacy_query(bm_collection, q, topk=10, filter="num < 50")
        assert_docs_equal(new_docs, old_docs)
        for d in new_docs:
            assert d.fields["num"] < 50

    def test_empty_result(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=10, filter="num < 0")
        assert docs == []

    def test_multi_query_rrf(self, bm_collection: Collection):
        q1 = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        q2 = Query(field_name="vec", vector=[0.1] * DIM, param=HnswQueryParam())
        docs = bm_collection.query([q1, q2], topk=10, reranker=RrfReRanker())
        assert len(docs) == 10
        for d in docs:
            assert isinstance(d, Doc)
            assert set(d.fields.keys()) == {"num", "title"}
