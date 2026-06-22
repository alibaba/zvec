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
from __future__ import annotations

from typing import Dict, Union
from unittest.mock import MagicMock, patch

import numpy as np
import math
from _zvec.param import _SearchQuery

import pytest
from zvec.executor.query_executor import (
    QueryContext,
    QueryExecutor,
)
from zvec import (
    RrfReRanker,
    WeightedReRanker,
    HnswQueryParam,
    CollectionSchema,
    VectorSchema,
    DataType,
    MetricType,
    Query,
    VectorQuery,
)
from zvec.extension.multi_vector_reranker import CallbackReRanker


# ----------------------------
# Mock Collection Schema
# ----------------------------
class MockCollectionSchema(CollectionSchema):
    def __init__(self, vectors=Union[VectorSchema, Dict[str, VectorSchema]]):
        self._vectors = (
            [vectors] if not isinstance(vectors, Dict) else list(vectors.values())
        )

    @property
    def vectors(self):
        return self._vectors


# ----------------------------
# VectorQuery Test Case
# ----------------------------
class TestQuery:
    def test_init(self):
        query = Query(field_name="test_field")
        assert query.field_name == "test_field"
        assert query.id is None
        assert query.vector is None
        assert query.param is None

        param = HnswQueryParam()
        query = Query(
            field_name="test_field", id="test_id", vector=[1, 2, 3], param=param
        )
        assert query.field_name == "test_field"
        assert query.id == "test_id"
        assert query.vector == [1, 2, 3]
        assert query.param == param

    def test_has_id(self):
        query = Query(field_name="test_field")
        assert not query.has_id()

        query = Query(field_name="test_field", id="test_id")
        assert query.has_id()

    def test_has_vector(self):
        query = Query(field_name="test_field")
        assert not query.has_vector()

        query = Query(field_name="test_field", vector=[])
        assert not query.has_vector()

        query = Query(field_name="test_field", vector=[1, 2, 3])
        assert query.has_vector()

    def test_validate_dense_fp16_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP16)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float16)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_fp32_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP32)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_fp64_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP64)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float64)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_int8_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_INT8)
        vec = np.array([1, 2, 3], dtype=np.int8)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_sparse_fp32_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.SPARSE_VECTOR_FP32)
        vec = {1: 1.1, 2: 2.2, 3: 3.3}
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        for k in vec.keys():
            assert math.isclose(vec[k], ret[k], abs_tol=1e-6)

    def test_validate_sparse_fp16_convert(self):
        v = _SearchQuery()
        schema = VectorSchema(name="test", data_type=DataType.SPARSE_VECTOR_FP16)
        vec = {1: 1.1, 2: 2.2, 3: 3.3}
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        for k in vec.keys():
            assert math.isclose(np.float16(vec[k]), ret[k], abs_tol=1e-6)


class TestVectorQueryDeprecated:
    def test_deprecation_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vq = VectorQuery(field_name="test_field")
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Query" in str(w[0].message)

    def test_isinstance_compatibility(self):
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            vq = VectorQuery(field_name="test_field")
        assert isinstance(vq, Query)


class TestQueryContext:
    def test_init(self):
        ctx = QueryContext(topk=10)
        assert ctx.topk == 10
        assert ctx.queries == []
        assert ctx.filter is None
        assert ctx.reranker is None
        assert ctx.output_fields is None
        assert ctx.include_vector is False

    def test_properties(self):
        queries = [Query(field_name="test")]
        reranker = RrfReRanker()
        output_fields = ["field1", "field2"]

        ctx = QueryContext(
            topk=5,
            filter="test_filter",
            include_vector=True,
            queries=queries,
            output_fields=output_fields,
            reranker=reranker,
        )

        assert ctx.topk == 5
        assert ctx.queries == queries
        assert ctx.filter == "test_filter"
        assert ctx.reranker == reranker
        assert ctx.output_fields == output_fields
        assert ctx.include_vector is True

    def test_properties_with_weighted_reranker(self):
        queries = [Query(field_name="test")]
        reranker = WeightedReRanker(
            weights=[1.0],
        )

        ctx = QueryContext(
            topk=5,
            queries=queries,
            reranker=reranker,
        )

        assert ctx.reranker == reranker
        assert ctx.reranker.weights == [1.0]

    def test_properties_with_callback_reranker(self):
        queries = [Query(field_name="test")]
        cb = lambda query_results, topn: []
        reranker = CallbackReRanker(callback=cb)

        ctx = QueryContext(
            topk=5,
            queries=queries,
            reranker=reranker,
        )

        assert ctx.reranker == reranker


class TestQueryExecutor:
    def test_init(self):
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        assert isinstance(executor, QueryExecutor)

    def test_do_build_without_queries(self):
        # When no queries are given, build a single vector-less query.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        ctx = QueryContext(topk=5, filter="test_filter")

        result = executor._build_queries(ctx, MagicMock())
        assert len(result) == 1
        assert result[0].topk == 5
        assert result[0].filter == "test_filter"

    def test_do_build_query_wo_vector(self):
        # Vector-less core query should carry the context query params.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        ctx = QueryContext(topk=7, filter="f", include_vector=True)

        core_vector = executor._build_base_search_query(ctx)
        assert core_vector.topk == 7
        assert core_vector.filter == "f"
        assert core_vector.include_vector is True

    def test_do_merge_rerank_results_single_without_reranker(self):
        # A single result list without a reranker is returned as-is.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        ctx = QueryContext(topk=5)
        docs_list = [["doc1", "doc2"]]

        result = executor._merge_and_rerank(ctx, docs_list)
        assert result == ["doc1", "doc2"]

    def test_do_merge_rerank_results_empty(self):
        # Empty results should raise an error.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        ctx = QueryContext(topk=5)

        with pytest.raises(ValueError, match="Query results is empty"):
            executor._merge_and_rerank(ctx, [])

    def test_do_merge_rerank_results_with_reranker(self):
        # Multiple result lists are merged through the reranker.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        reranker = MagicMock()
        reranker.rerank.return_value = ["merged"]
        ctx = QueryContext(
            topk=5,
            queries=[Query(field_name="test1"), Query(field_name="test2")],
            reranker=reranker,
        )
        docs_list = [["d1"], ["d2"]]

        result = executor._merge_and_rerank(ctx, docs_list)
        assert result == ["merged"]
        reranker.rerank.assert_called_once_with(docs_list, ctx.topk)

    def test_execute_python_pipeline(self):
        # Each query is executed serially and converted into a result list.
        schema = MockCollectionSchema()
        executor = QueryExecutor(schema)
        collection = MagicMock()
        collection.Query.side_effect = [["raw1"], ["raw2"]]
        vectors = [MagicMock(), MagicMock()]

        with patch(
            "zvec.executor.query_executor.convert_to_py_doc",
            side_effect=lambda doc, schema: doc,
        ):
            results = executor._execute_python_pipeline(vectors, collection)
        assert results == [["raw1"], ["raw2"]]
        assert collection.Query.call_count == 2
