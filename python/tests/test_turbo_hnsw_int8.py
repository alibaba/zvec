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

"""Packaged-extension regression tests for the HNSW Turbo INT8 path."""

from __future__ import annotations

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionOption,
    CollectionSchema,
    Doc,
    HnswIndexParam,
    HnswQueryParam,
    Query,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType

DIMENSION = 32
DOC_COUNT = 64


def _normalized_vectors() -> list[list[float]]:
    rng = np.random.default_rng(2026)
    vectors = rng.standard_normal((DOC_COUNT, DIMENSION)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors.tolist()


@pytest.mark.parametrize(
    "metric", [MetricType.COSINE, MetricType.L2], ids=["cosine", "l2"]
)
def test_turbo_int8_hnsw_python_registration_roundtrip(tmp_path, metric):
    """Keep Turbo's factory registration reachable from the Python module."""
    path = str(tmp_path / f"turbo_int8_hnsw_{metric.name.lower()}")
    vectors = _normalized_vectors()
    schema = CollectionSchema(
        name="turbo_int8_hnsw",
        vectors=[
            VectorSchema(
                "dense",
                DataType.VECTOR_FP32,
                dimension=DIMENSION,
                index_param=HnswIndexParam(
                    metric_type=metric,
                    m=16,
                    ef_construction=100,
                    quantize_type=QuantizeType.INT8,
                ),
            )
        ],
    )

    collection = zvec.create_and_open(
        path=path,
        schema=schema,
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    try:
        docs = [
            Doc(id=str(i), vectors={"dense": vector})
            for i, vector in enumerate(vectors)
        ]
        for status in collection.insert(docs):
            assert status.ok()
        collection.optimize()

        query = Query(
            field_name="dense",
            vector=vectors[17],
            param=HnswQueryParam(ef=128),
        )
        hits = collection.query(query, topk=5)
        assert hits[0].id == "17"
    finally:
        collection.close()

    reopened = zvec.open(path, option=CollectionOption(read_only=True))
    try:
        hits = reopened.query(query, topk=5)
        assert hits[0].id == "17"
    finally:
        reopened.close()
