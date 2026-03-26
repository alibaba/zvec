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
from unittest.mock import MagicMock

import numpy as np
import math
from _zvec.param import _VectorQuery

import pytest
from zvec.executor.query_executor import (
    MultiVectorQueryExecutor,
    NoVectorQueryExecutor,
    QueryContext,
    QueryExecutor,
    QueryExecutorFactory,
    SingleVectorQueryExecutor,
)
from zvec import (
    RrfReRanker,
    HnswQueryParam,
    CollectionSchema,
    VectorSchema,
    DataType,
    Query,
    VectorQuery,
)


# ----------------------------
# Mock Vector Schema
# ----------------------------
class MockVectorSchema(VectorSchema):
    def __init__(self, name="test_vector"):
        self._name = name

    @property
    def name(self):
        return self._name

    def _get_object(self):
        return MagicMock()


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
        v = _VectorQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP16)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float16)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_fp32_convert(self):
        v = _VectorQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP32)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_fp64_convert(self):
        v = _VectorQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_FP64)
        vec = np.array([1.1, 2.1, 3.1], dtype=np.float64)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_dense_int8_convert(self):
        v = _VectorQuery()
        schema = VectorSchema(name="test", data_type=DataType.VECTOR_INT8)
        vec = np.array([1, 2, 3], dtype=np.int8)
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        assert np.array_equal(vec, ret)

    def test_validate_sparse_fp32_convert(self):
        v = _VectorQuery()
        schema = VectorSchema(name="test", data_type=DataType.SPARSE_VECTOR_FP32)
        vec = {1: 1.1, 2: 2.2, 3: 3.3}
        v.set_vector(schema._get_object(), vec)
        ret = v.get_vector(schema._get_object())
        for k in vec.keys():
            assert math.isclose(vec[k], ret[k], abs_tol=1e-6)

    def test_validate_sparse_fp16_convert(self):
        v = _VectorQuery()
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
        assert ctx.core_vectors == []

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

    def test_core_vectors_setter(self):
        ctx = QueryContext(topk=10)
        core_vectors = [MagicMock()]
        ctx.core_vectors = core_vectors
        assert ctx.core_vectors == core_vectors


class TestNoVectorQueryExecutor:
    def test_init(self):
        schema = MockCollectionSchema()
        executor = NoVectorQueryExecutor(schema)
        assert isinstance(executor, QueryExecutor)

    def test_do_validate_with_queries(self):
        schema = MockCollectionSchema()
        executor = NoVectorQueryExecutor(schema)
        ctx = QueryContext(topk=10, queries=[Query(field_name="test")])

        with pytest.raises(
            ValueError, match="Collection does not support query with vector or id"
        ):
            executor._do_validate(ctx)

    def test_do_validate_without_queries(self):
        schema = MockCollectionSchema()
        executor = NoVectorQueryExecutor(schema)
        ctx = QueryContext(topk=10)

        executor._do_validate(ctx)

    def test_do_build(self):
        schema = MockCollectionSchema()
        executor = NoVectorQueryExecutor(schema)
        ctx = QueryContext(topk=5, filter="test_filter")

        result = executor._do_build(ctx, MagicMock())
        assert len(result) == 1
        assert result[0].topk == 5
        assert result[0].filter == "test_filter"


class TestSingleVectorQueryExecutor:
    def test_init(self):
        schema = MockCollectionSchema()
        executor = SingleVectorQueryExecutor(schema)
        assert isinstance(executor, NoVectorQueryExecutor)

    def test_do_validate_multiple_queries(self):
        schema = MockCollectionSchema()
        executor = SingleVectorQueryExecutor(schema)
        queries = [Query(field_name="test1"), Query(field_name="test2")]
        ctx = QueryContext(topk=10, queries=queries)

        with pytest.raises(
            ValueError,
            match="Collection has only one vector field, cannot query with multiple vectors",
        ):
            executor._do_validate(ctx)

    def test_do_build_without_queries(self):
        schema = MockCollectionSchema()
        executor = SingleVectorQueryExecutor(schema)
        ctx = QueryContext(topk=5)

        result = executor._do_build(ctx, MagicMock())
        assert len(result) == 1
        assert result[0].topk == 5


class TestMultiVectorQueryExecutor:
    def test_init(self):
        schema = MockCollectionSchema()
        executor = MultiVectorQueryExecutor(schema)
        assert isinstance(executor, SingleVectorQueryExecutor)

    def test_do_validate_multiple_queries_without_reranker(self):
        schema = MockCollectionSchema()
        executor = MultiVectorQueryExecutor(schema)
        queries = [Query(field_name="test1"), Query(field_name="test2")]
        ctx = QueryContext(topk=10, queries=queries)

        with pytest.raises(
            ValueError, match="Reranker is required for multi-vector query"
        ):
            executor._do_validate(ctx)

    def test_do_validate_multiple_queries_with_reranker(self):
        schema = MockCollectionSchema()
        executor = MultiVectorQueryExecutor(schema)
        queries = [Query(field_name="test1"), Query(field_name="test2")]
        reranker = RrfReRanker()
        ctx = QueryContext(topk=10, queries=queries, reranker=reranker)

        executor._do_validate(ctx)


class TestQueryExecutorFactory:
    def test_create_no_vectors(self):
        schema = MockCollectionSchema()
        executor = QueryExecutorFactory.create(schema)
        assert isinstance(executor, NoVectorQueryExecutor)

    def test_create_single_vector(self):
        schema = MockCollectionSchema(vectors=MockVectorSchema())
        executor = QueryExecutorFactory.create(schema)
        assert isinstance(executor, SingleVectorQueryExecutor)

    def test_create_multiple_vectors(self):
        schema = MockCollectionSchema(
            vectors={"test1": MockVectorSchema(), "test2": MockVectorSchema()}
        )
        executor = QueryExecutorFactory.create(schema)
        assert isinstance(executor, MultiVectorQueryExecutor)
