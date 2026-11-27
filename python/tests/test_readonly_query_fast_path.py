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
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.COSINE, zvec.MetricType.IP]
)
@pytest.mark.parametrize(
    "quantize",
    [zvec.QuantizeType.UNDEFINED, zvec.QuantizeType.INT8, zvec.QuantizeType.FP16],
)
def test_readonly_query_matches_general_engine(
    tmp_path, algorithm, repeated, metric, quantize
):
    rng = np.random.default_rng(20261030)
    vectors = rng.normal(size=(256, 32)).astype(np.float32)
    if repeated:
        vectors[128:] = vectors[:128]
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
            # Cosine storage can normalize and reconstruct FP32 vectors. Allow
            # that round trip's rounding error on both x86 and ARM.
            np.testing.assert_allclose(
                doc.vector("vector"), vectors[i], rtol=1e-6, atol=1e-7
            )
    finally:
        reader.close()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize(
    "metric", [zvec.MetricType.L2, zvec.MetricType.COSINE, zvec.MetricType.IP]
)
def test_readonly_pq_query_matches_general_with_ties(tmp_path, bits, rotate, metric):
    rng = np.random.default_rng(903)
    # Duplicate vectors guarantee equal PQ scores, including after refinement.
    # More than 1000 rows exercise graph search rather than the small-index scan.
    vectors = np.repeat(
        rng.uniform(0.25, 1.5, size=(550, 16)).astype(np.float32), 2, axis=0
    )
    schema = zvec.CollectionSchema(
        "pq_query_fast",
        fields=[zvec.FieldSchema("number", zvec.DataType.INT64)],
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                16,
                index_param=zvec.HnswIndexParam(
                    metric_type=metric,
                    m=16,
                    ef_construction=100,
                    quantize_type=zvec.QuantizeType.PQ,
                    quantizer_param=zvec.QuantizerParam(
                        num_chunk=4,
                        num_bits=bits,
                        enable_rotate=rotate,
                        opq_iter=2,
                        opq_pq_iter=2,
                    ),
                ),
            )
        ],
    )
    path = str(tmp_path / "index")
    writer = zvec.create_and_open(path, schema)
    try:
        for start in range(0, len(vectors), 200):
            assert all(
                s.ok()
                for s in writer.insert(
                    [
                        zvec.Doc(
                            id=f"business-{i}",
                            vectors={"vector": vectors[i]},
                            fields={"number": i},
                        )
                        for i in range(start, min(start + 200, len(vectors)))
                    ]
                )
            )
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True))
    try:
        for refine in (False, True):
            for topk in (40, 80, 1103):
                query = zvec.Query(
                    "vector",
                    vector=vectors[0],
                    param=zvec.HnswQueryParam(
                        ef=2000, is_using_refiner=refine, scale_factor=4
                    ),
                )
                # Requesting a scalar forces the existing general query engine.
                expected = reader.query(query, topk=topk, output_fields=["number"])
                docs = reader.query(query, topk=topk, output_fields=[])
                assert [d.id for d in docs] == [d.id for d in expected]
                np.testing.assert_allclose(
                    [d.score for d in docs], [d.score for d in expected]
                )
                assert any(
                    expected[i].score == expected[i - 1].score
                    for i in range(1, len(expected))
                )
                ids, scores = reader.query_internal_ids(
                    query, topk=topk, return_scores=True
                )
                actual_by_id = dict(zip(ids[ids >= 0], scores[ids >= 0]))
                assert set(actual_by_id) == {d.field("number") for d in expected}
                for doc in expected:
                    assert actual_by_id[doc.field("number")] == pytest.approx(doc.score)
                np.testing.assert_array_equal(
                    ids, reader.query_internal_ids(query, topk=topk)
                )
                assert np.all(ids[len(expected) :] == -1)
                assert np.all(np.isnan(scores[len(expected) :]))
    finally:
        reader.close()


def test_readonly_uint8_refiner_preserves_tied_cutoffs(tmp_path):
    rng = np.random.default_rng(20261125)
    vectors = rng.integers(0, 8, size=(1100, 128)).astype(np.float32)
    queries = rng.integers(0, 8, size=(32, 128)).astype(np.float32)
    schema = zvec.CollectionSchema(
        "uint8_refiner_ties",
        fields=[zvec.FieldSchema("number", zvec.DataType.INT64)],
        vectors=[
            zvec.VectorSchema(
                "vector",
                zvec.DataType.VECTOR_FP32,
                128,
                index_param=zvec.VamanaIndexParam(
                    metric_type=zvec.MetricType.L2,
                    quantize_type=zvec.QuantizeType.UNIFORM_UINT4,
                    flat_data_type=zvec.DataType.VECTOR_UINT8,
                    use_flat_contiguous_memory=True,
                    use_contiguous_memory=True,
                    max_degree=16,
                    search_list_size=64,
                ),
            )
        ],
    )
    path = str(tmp_path / "index")
    writer = zvec.create_and_open(path, schema)
    try:
        for start in range(0, len(vectors), 200):
            assert all(
                status.ok()
                for status in writer.insert(
                    [
                        zvec.Doc(
                            id=str(i),
                            vectors={"vector": vectors[i]},
                            fields={"number": i},
                        )
                        for i in range(start, min(start + 200, len(vectors)))
                    ]
                )
            )
        writer.optimize()
    finally:
        writer.close()
    reader = zvec.open(path, zvec.CollectionOption(read_only=True, enable_mmap=True))
    saw_ties = False
    try:
        for topk in (1, 5, 10, 40):
            for vector in queries:
                query = zvec.Query(
                    "vector",
                    vector=vector,
                    param=zvec.VamanaQueryParam(
                        ef_search=64, is_using_refiner=True, scale_factor=1.7
                    ),
                )
                # Compare routes on the same read-only storage layout. A writer
                # may choose a different member of a tied group after reopen.
                expected = reader.query(query, topk=topk, output_fields=["number"])
                docs = reader.query(query, topk=topk, output_fields=[])
                assert [(d.id, d.score) for d in docs] == [
                    (d.id, d.score) for d in expected
                ]
                saw_ties |= any(
                    expected[i].score == expected[i - 1].score
                    for i in range(1, len(expected))
                )
    finally:
        reader.close()
    assert saw_ties
