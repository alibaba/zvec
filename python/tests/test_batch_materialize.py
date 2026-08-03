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
"""Correctness tests for the batch-materialized query path.

`Collection.query` goes through `_Collection.Query`, which batch-materializes
all hits into tuples in a single C++ call. These tests validate the
materialized output against two independent references:

- a numpy brute-force ground truth over the inserted vectors (ids / scores);
- the `fetch` path, which materializes docs through a separate binding.
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
from zvec.typing import MetricType

DIM = 16
N_DOCS = 200


def _make_vectors() -> np.ndarray:
    """Same deterministic vectors as inserted by the fixture."""
    return np.random.default_rng(42).random((N_DOCS, DIM), dtype=np.float32)


def _brute_force_topk(
    query: np.ndarray, topk: int, mask: np.ndarray | None = None
) -> tuple[list[str], np.ndarray]:
    """Exact L2sq top-k ids and distances over the ground-truth vectors."""
    dists = ((_make_vectors() - query) ** 2).sum(axis=1)
    if mask is not None:
        dists = np.where(mask, dists, np.inf)
    idx = np.argsort(dists, kind="stable")[:topk]
    return [str(i) for i in idx], dists[idx]


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
                # explicit L2: score is the raw squared L2 distance (no
                # metric normalization), matching the brute-force ground truth
                index_param=HnswIndexParam(metric_type=MetricType.L2),
            ),
        ],
    )
    path = tmp_path_factory.mktemp("zvec_batch_mat") / "coll"
    coll = zvec.create_and_open(
        path=str(path),
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )

    vectors = _make_vectors()
    docs = [
        Doc(
            id=str(i),
            fields={"num": i, "title": f"doc-{i}"},
            vectors={"vec": vectors[i]},
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


class TestBatchMaterialize:
    def _query_vec(self) -> np.ndarray:
        return np.array([0.5] * DIM, dtype=np.float32)

    def test_matches_brute_force_ground_truth(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=20)
        assert len(docs) == 20

        exp_ids, exp_dists = _brute_force_topk(self._query_vec(), 20)
        assert [d.id for d in docs] == exp_ids
        scores = [d.score for d in docs]
        assert scores == sorted(scores)
        for d, dist in zip(docs, exp_dists):
            assert d.score == pytest.approx(float(dist), rel=1e-4)

        # scalar fields fully materialized, vectors excluded by default
        for d in docs:
            assert isinstance(d, Doc)
            assert set(d.fields.keys()) == {"num", "title"}
            assert d.fields["num"] == int(d.id)
            assert d.fields["title"] == f"doc-{d.id}"
            assert d.vectors == {}

    def test_fields_match_fetch_path(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=10)
        fetched = bm_collection.fetch([d.id for d in docs], include_vector=False)
        for d in docs:
            assert d.fields == fetched[d.id].fields

    @pytest.mark.parametrize("include_vector", [False, True])
    def test_include_vector(self, bm_collection: Collection, include_vector: bool):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=5, include_vector=include_vector)
        ground_truth = _make_vectors()
        for d in docs:
            assert bool(d.vectors) is include_vector
            if include_vector:
                assert np.allclose(d.vectors["vec"], ground_truth[int(d.id)])

    def test_output_fields_subset(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=5, output_fields=["num"])
        exp_ids, _ = _brute_force_topk(self._query_vec(), 5)
        assert [d.id for d in docs] == exp_ids
        for d in docs:
            assert set(d.fields.keys()) == {"num"}

    def test_filter(self, bm_collection: Collection):
        q = Query(field_name="vec", vector=self._query_vec(), param=HnswQueryParam())
        docs = bm_collection.query(q, topk=10, filter="num < 50")
        mask = np.arange(N_DOCS) < 50
        exp_ids, exp_dists = _brute_force_topk(self._query_vec(), 10, mask)
        assert [d.id for d in docs] == exp_ids
        for d, dist in zip(docs, exp_dists):
            assert d.fields["num"] < 50
            assert d.score == pytest.approx(float(dist), rel=1e-4)

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
