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

"""Compare read-only optimized Query with the writable general query engine."""

from __future__ import annotations

import numpy as np
import pytest

import zvec


@pytest.mark.parametrize("algorithm", ["hnsw", "vamana"])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.COSINE, zvec.MetricType.IP]
)
@pytest.mark.parametrize(
    "quantize",
    [zvec.QuantizeType.UNDEFINED, zvec.QuantizeType.INT8, zvec.QuantizeType.FP16],
)
def test_readonly_query_matches_general_engine(tmp_path, algorithm, metric, quantize):
    rng = np.random.default_rng(20261030)
    vectors = rng.normal(size=(256, 32)).astype(np.float32)
    if algorithm == "vamana":
        index = zvec.VamanaIndexParam(
            metric_type=metric,
            max_degree=16,
            search_list_size=64,
            use_contiguous_memory=True,
            quantize_type=quantize,
        )
        param = lambda refine: zvec.VamanaQueryParam(
            ef_search=64, is_using_refiner=refine
        )
    else:
        index = zvec.HnswIndexParam(
            metric_type=metric, m=16, ef_construction=64, quantize_type=quantize
        )
        param = lambda refine: zvec.HnswQueryParam(ef=64, is_using_refiner=refine)
    schema = zvec.CollectionSchema(
        "query_fast",
        fields=[zvec.FieldSchema("number", zvec.DataType.INT64)],
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, 32, index_param=index
            )
        ],
    )
    path = str(tmp_path / "index")
    collection = zvec.create_and_open(path, schema)
    expected = {}
    try:
        collection.insert(
            [
                zvec.Doc(
                    id=f"business-{i}", vectors={"vector": v}, fields={"number": i}
                )
                for i, v in enumerate(vectors)
            ]
        )
        collection.optimize()
        for refine in (False, True):
            if refine and quantize == zvec.QuantizeType.UNDEFINED:
                continue
            for topk in (1, 10, 300):
                for index in (0, 17):
                    docs = collection.query(
                        zvec.Query(
                            "vector", vector=vectors[index], param=param(refine)
                        ),
                        topk=topk,
                        output_fields=[],
                    )
                    expected[refine, topk, index] = [(d.id, d.score) for d in docs]
    finally:
        collection.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True))
    try:
        for (refine, topk, index), expected_docs in expected.items():
            docs = reader.query(
                zvec.Query("vector", vector=vectors[index], param=param(refine)),
                topk=topk,
                output_fields=[],
            )
            assert [d.id for d in docs] == [d[0] for d in expected_docs]
            np.testing.assert_allclose(
                [d.score for d in docs],
                [d[1] for d in expected_docs],
                rtol=1e-5,
                atol=1e-5,
            )
            assert all(d.fields == {} and d.vectors == {} for d in docs)
        # A request needing materialization must retain the general path.
        docs = reader.query(
            zvec.Query("vector", vector=vectors[0], param=param(False)),
            topk=3,
            include_vector=True,
            output_fields=["number"],
        )
        for doc in docs:
            i = int(doc.id.removeprefix("business-"))
            assert doc.field("number") == i
            np.testing.assert_allclose(doc.vector("vector"), vectors[i])
    finally:
        reader.close()
