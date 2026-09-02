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

"""End-to-end tests for the FLAT TURBO_INT8 quantize path."""

from __future__ import annotations

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    FieldSchema,
    FlatIndexParam,
    Query,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType

DIMENSION = 32
DOC_COUNT = 200
TOPK = 10


def _random_vectors(count: int, dim: int, seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.tolist()


def _make_schema(name: str, metric: MetricType) -> CollectionSchema:
    return CollectionSchema(
        name=name,
        fields=[FieldSchema("ordinal", DataType.INT64, nullable=False)],
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=FlatIndexParam(
                    metric_type=metric,
                    quantize_type=QuantizeType.TURBO_INT8,
                    use_contiguous_memory=True,
                ),
            )
        ],
    )


def _insert(collection, vectors) -> None:
    docs = [
        Doc(id=str(i), fields={"ordinal": i}, vectors={"dense": vec})
        for i, vec in enumerate(vectors)
    ]
    for status in collection.insert(docs):
        assert status.ok()


def _query_rows(collection, vector):
    hits = collection.query(
        Query(field_name="dense", vector=vector), topk=TOPK
    )
    return [(hit.id, hit.score) for hit in hits]


@pytest.mark.parametrize(
    "metric", [MetricType.COSINE, MetricType.L2], ids=["cosine", "l2"]
)
def test_turbo_int8_flat_search_reopen(tmp_path, metric):
    vectors = _random_vectors(DOC_COUNT, DIMENSION, seed=2026)
    queries = _random_vectors(5, DIMENSION, seed=2027)
    path = str(tmp_path / f"turbo_int8_{metric.name.lower()}")

    schema = _make_schema("turbo_int8_flat", metric)
    collection = zvec.create_and_open(
        path=path, schema=schema, option=CollectionOption(read_only=False)
    )
    _insert(collection, vectors)
    collection.flush()

    before = [_query_rows(collection, q) for q in queries]
    collection.optimize()
    after_optimize = [_query_rows(collection, q) for q in queries]
    collection.close()

    # reopen: the turbo path must be rebuilt from the persisted manifest
    reopened = zvec.open(path, option=CollectionOption(read_only=True))
    after_reopen = [_query_rows(reopened, q) for q in queries]
    reopened.close()

    for pre, post in zip(before, after_optimize):
        assert [row[0] for row in pre] == [row[0] for row in post]
        for (p, q) in zip(pre, post):
            assert p[1] == pytest.approx(q[1], abs=1e-5)
    for pre, post in zip(before, after_reopen):
        assert [row[0] for row in pre] == [row[0] for row in post]
        for (p, q) in zip(pre, post):
            assert p[1] == pytest.approx(q[1], abs=1e-5)


def test_turbo_int8_flat_recall_vs_brute_force(tmp_path):
    metric = MetricType.COSINE
    vectors = _random_vectors(DOC_COUNT, DIMENSION, seed=2028)
    queries = _random_vectors(10, DIMENSION, seed=2029)
    path = str(tmp_path / "turbo_int8_recall")

    collection = zvec.create_and_open(
        path=path,
        schema=_make_schema("turbo_int8_recall", metric),
        option=CollectionOption(read_only=False),
    )
    _insert(collection, vectors)
    collection.optimize()

    matrix = np.asarray(vectors, dtype=np.float32)
    hits = 0
    total = 0
    for query in queries:
        query_vec = np.asarray(query, dtype=np.float32)
        scores = 1.0 - matrix @ query_vec
        truth = set(np.argsort(scores)[:TOPK].tolist())
        rows = _query_rows(collection, query)
        got = {int(row[0]) for row in rows}
        hits += len(truth & got)
        total += TOPK
        # scores must be 1 - cosine within int8 quantization error
        for doc_id, score in rows:
            expected = float(scores[int(doc_id)])
            assert score == pytest.approx(expected, abs=0.15)
    collection.close()
    assert hits / total >= 0.9


def test_turbo_int8_fetch_dequantizes(tmp_path):
    metric = MetricType.COSINE
    vectors = _random_vectors(DOC_COUNT, DIMENSION, seed=2030)
    path = str(tmp_path / "turbo_int8_fetch")

    collection = zvec.create_and_open(
        path=path,
        schema=_make_schema("turbo_int8_fetch", metric),
        option=CollectionOption(read_only=False),
    )
    _insert(collection, vectors)
    collection.optimize()

    fetched = collection.fetch(ids=["17"])["17"]
    stored = np.asarray(vectors[17], dtype=np.float32)
    got = np.asarray(fetched.vector("dense"), dtype=np.float32)
    np.testing.assert_allclose(got, stored, atol=1e-2)
    collection.close()


def test_turbo_int8_rejects_rotate_and_ip_metric(tmp_path):
    with pytest.raises(Exception):
        zvec.create_and_open(
            path=str(tmp_path / "turbo_rotate"),
            schema=CollectionSchema(
                name="turbo_rotate",
                fields=[FieldSchema("ordinal", DataType.INT64)],
                vectors=[
                    VectorSchema(
                        "dense",
                        DataType.VECTOR_FP32,
                        dimension=DIMENSION,
                        index_param=FlatIndexParam(
                            metric_type=MetricType.COSINE,
                            quantize_type=QuantizeType.TURBO_INT8,
                            quantizer_param=zvec.QuantizerParam(
                                enable_rotate=True
                            ),
                        ),
                    )
                ],
            ),
        )

    with pytest.raises(Exception):
        zvec.create_and_open(
            path=str(tmp_path / "turbo_ip"),
            schema=CollectionSchema(
                name="turbo_ip",
                fields=[FieldSchema("ordinal", DataType.INT64)],
                vectors=[
                    VectorSchema(
                        "dense",
                        DataType.VECTOR_FP32,
                        dimension=DIMENSION,
                        index_param=FlatIndexParam(
                            metric_type=MetricType.IP,
                            quantize_type=QuantizeType.TURBO_INT8,
                        ),
                    )
                ],
            ),
        )
