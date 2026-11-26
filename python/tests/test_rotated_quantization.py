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

"""Optimize must keep rotated INT8/INT4 indexes searchable (issue #886)."""

from __future__ import annotations

import numpy as np
import pytest

import zvec
from zvec import (
    CollectionSchema,
    Doc,
    FlatIndexParam,
    HnswIndexParam,
    Query,
    QuantizerParam,
    VectorSchema,
)
from zvec.typing import DataType, MetricType, QuantizeType

DIMENSION = 128
BATCH = 200
MIN_HITS = BATCH * 95 // 100


def _self_recall(collection, vectors, ids) -> int:
    hits = 0
    for i in ids:
        result = collection.query(Query(field_name="dense", vector=vectors[i]), topk=1)
        hits += bool(result) and result[0].id == str(i)
    return hits


@pytest.mark.parametrize(
    "metric", [MetricType.COSINE, MetricType.L2], ids=["cosine", "l2"]
)
@pytest.mark.parametrize(
    "quantize_type", [QuantizeType.INT8, QuantizeType.INT4], ids=["int8", "int4"]
)
@pytest.mark.parametrize("index_kind", ["hnsw", "flat"])
def test_rotated_quantizer_survives_optimize(
    tmp_path, index_kind, quantize_type, metric
):
    zvec.init()
    rng = np.random.default_rng(886)
    vectors = rng.standard_normal((BATCH * 2, DIMENSION)).astype(np.float32).tolist()

    index_cls = HnswIndexParam if index_kind == "hnsw" else FlatIndexParam
    index_param = index_cls(
        metric_type=metric,
        quantize_type=quantize_type,
        quantizer_param=QuantizerParam(enable_rotate=True),
    )
    schema = CollectionSchema(
        name="rotated_quantization",
        vectors=[
            VectorSchema(
                "dense", DataType.VECTOR_FP32, DIMENSION, index_param=index_param
            )
        ],
    )
    path = str(tmp_path / "collection")
    collection = zvec.create_and_open(path=path, schema=schema)

    def insert(ids):
        collection.insert([Doc(id=str(i), vectors={"dense": vectors[i]}) for i in ids])

    first, second = range(BATCH), range(BATCH, BATCH * 2)

    # The first optimize builds the index from one writing segment.
    insert(first)
    assert _self_recall(collection, vectors, first) >= MIN_HITS
    collection.optimize()
    assert _self_recall(collection, vectors, first) >= MIN_HITS

    # The second optimize merges the indexed segment with a new one.
    insert(second)
    collection.optimize()
    assert _self_recall(collection, vectors, first) >= MIN_HITS
    assert _self_recall(collection, vectors, second) >= MIN_HITS

    del collection
    collection = zvec.open(path)
    assert _self_recall(collection, vectors, range(BATCH * 2)) >= MIN_HITS * 2
