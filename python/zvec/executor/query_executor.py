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

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, final

import numpy as np
from _zvec import _Collection
from _zvec.param import _VectorQuery

from ..extension import ReRanker, RrfReRanker, WeightedReRanker
from ..model.convert import convert_to_py_doc
from ..model.doc import Doc
from ..model.param.query import Query
from ..model.schema import CollectionSchema
from ..typing import DataType

__all__ = [
    "QueryContext",
    "QueryExecutor",
    "QueryExecutorFactory",
]

DTYPE_MAP = {
    DataType.VECTOR_FP16.value: np.float16,
    DataType.VECTOR_FP32.value: np.float32,
    DataType.VECTOR_FP64.value: np.float64,
    DataType.VECTOR_INT8.value: np.int8,
}


def convert_to_numpy(vec: Union[list, np.ndarray], dtype: np.dtype) -> np.ndarray:
    if isinstance(vec, np.ndarray):
        if vec.dtype == dtype and vec.ndim == 1:
            return vec
        return np.asarray(vec, dtype=dtype).flatten()

    try:
        arr = np.asarray(vec, dtype=dtype)
        if arr.ndim != 1:
            arr = arr.flatten()
        return arr
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot convert input to 1D numpy array with dtype={dtype}: {type(vec)}"
        ) from e


class QueryContext:
    def __init__(
        self,
        topk: int,
        filter: Optional[str] = None,
        include_vector: bool = False,
        queries: Optional[list[Query]] = None,
        output_fields: Optional[list[str]] = None,
        reranker: Optional[ReRanker] = None,
    ):
        # query param
        self._filter = filter
        self._queries = queries or []
        self._topk = topk
        self._include_vector = include_vector
        self._output_fields = output_fields

        # reranker
        self._reranker = reranker

        # core vectors
        self._core_vectors = []

    @property
    def topk(self):
        return self._topk

    @property
    def queries(self):
        return self._queries

    @property
    def filter(self):
        return self._filter

    @property
    def reranker(self):
        return self._reranker

    @property
    def output_fields(self):
        return self._output_fields

    @property
    def include_vector(self):
        return self._include_vector

    @property
    def core_vectors(self):
        return self._core_vectors

    @core_vectors.setter
    def core_vectors(self, core_vectors: list[_VectorQuery]):
        self._core_vectors = core_vectors


class QueryExecutor(ABC):
    def __init__(self, schema: CollectionSchema):
        self._schema = schema
        self._concurrency = max(1, int(os.getenv("ZVEC_QUERY_CONCURRENCY", "1")))

    @abstractmethod
    def _do_validate(self, ctx: QueryContext) -> None:
        pass

    @abstractmethod
    def _do_build(
        self, ctx: QueryContext, collection: _Collection
    ) -> list[_VectorQuery]:
        pass

    def _do_build_query_wo_vector(self, ctx: QueryContext) -> _VectorQuery:
        core_vector = _VectorQuery()
        core_vector.topk = ctx.topk
        core_vector.include_vector = ctx.include_vector
        if ctx.filter:
            core_vector.filter = ctx.filter
        if ctx.output_fields:
            core_vector.output_fields = ctx.output_fields
        return core_vector

    def _do_build_query_with_vector(
        self, ctx: QueryContext, query: Query, collection: _Collection
    ) -> _VectorQuery:
        core_vector = self._do_build_query_wo_vector(ctx)
        core_vector.field_name = query.field_name
        if query.param:
            core_vector.query_params = query.param

        vector_schema = (
            self._schema.vector(query.field_name) if query else self._schema.vectors[0]
        )

        if vector_schema is None:
            raise ValueError("No vector field found")

        # set output_fields
        core_vector.output_fields = ctx.output_fields

        # set vector
        if query.has_vector():
            vec_data = query.vector
        else:
            fetched = collection.Fetch([query.id])
            doc = next(iter(fetched.values()))
            if not doc:
                return core_vector
            vec_data = doc.get_any(vector_schema.name, vector_schema.data_type)

        target_dtype = DTYPE_MAP.get(vector_schema.data_type.value)
        core_vector.set_vector(
            vector_schema._get_object(),
            convert_to_numpy(vec_data, target_dtype) if target_dtype else vec_data,
        )
        return core_vector

    def _do_execute(
        self, vectors: list[_VectorQuery], collection: _Collection
    ) -> dict[str, list[Doc]]:
        query_cnt = len(vectors)
        if query_cnt == 0:
            raise ValueError("No query to execute")

        if len(vectors) == 1 or self._concurrency == 1:
            results = {}
            for query in vectors:
                docs = collection.Query(query)
                results[query.field_name] = [
                    convert_to_py_doc(doc, self._schema) for doc in docs
                ]
            return results

        results = {}
        with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
            future_to_query = {
                executor.submit(collection.Query, query): query.field_name
                for query in vectors
            }

            for future in as_completed(future_to_query):
                field_name = future_to_query[future]
                try:
                    docs = future.result()
                    results[field_name] = [
                        convert_to_py_doc(doc, self._schema) for doc in docs
                    ]
                except Exception as e:
                    raise e
        return results

    def _do_merge_rerank_results(
        self, ctx: QueryContext, docs_map: dict[str, list[Doc]]
    ) -> list[Doc]:
        query_result_cnt = len(docs_map) if docs_map else 0
        if query_result_cnt == 0:
            raise ValueError("Query results is none and dost not to rerank")
        if query_result_cnt == 1:
            if not ctx.reranker or isinstance(
                ctx.reranker, (RrfReRanker, WeightedReRanker)
            ):
                return next(iter(docs_map.values()))
            return ctx.reranker.rerank(docs_map)
        return ctx.reranker.rerank(docs_map)

    @final
    def execute(self, ctx: QueryContext, collection: _Collection) -> list[Doc]:
        # 1. validate query
        self._do_validate(ctx)
        # 2. build query vector
        query_vectors = self._do_build(ctx, collection)
        if not query_vectors:
            raise ValueError("No query to execute")
        # 3. execute query
        docs = self._do_execute(query_vectors, collection)
        # 4. merge and rerank result
        return self._do_merge_rerank_results(ctx, docs)


class NoVectorQueryExecutor(QueryExecutor):
    def __init__(self, schema: CollectionSchema):
        super().__init__(schema)

    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 0:
            raise ValueError("Collection does not support query with vector or id")

    def _do_build(
        self, ctx: QueryContext, _collection: _Collection
    ) -> list[_VectorQuery]:
        return [self._do_build_query_wo_vector(ctx)]


class SingleVectorQueryExecutor(NoVectorQueryExecutor):
    def __init__(self, schema: CollectionSchema) -> None:
        super().__init__(schema)

    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 1:
            raise ValueError(
                "Collection has only one vector field, cannot query with multiple vectors"
            )
        for query in ctx.queries:
            query._validate()

    def _do_build(
        self, ctx: QueryContext, collection: _Collection
    ) -> list[_VectorQuery]:
        if len(ctx.queries) == 0:
            return [self._do_build_query_wo_vector(ctx)]
        vectors = []
        for query in ctx.queries:
            vectors.append(self._do_build_query_with_vector(ctx, query, collection))
        return vectors


class MultiVectorQueryExecutor(SingleVectorQueryExecutor):
    def __init__(self, schema: CollectionSchema) -> None:
        super().__init__(schema)

    def _do_validate(self, ctx: QueryContext) -> None:
        if len(ctx.queries) > 1 and ctx.reranker is None:
            raise ValueError("Reranker is required for multi-vector query")
        seen_fields = set()
        for query in ctx.queries:
            query._validate()
            field = query.field_name
            if field in seen_fields:
                raise ValueError(f"Query field name '{field}' appears more than once")
            seen_fields.add(field)

    def _do_execute(
        self, vectors: list[_VectorQuery], collection: _Collection
    ) -> dict[str, list[Doc]]:
        return super()._do_execute(vectors, collection)


class QueryExecutorFactory:
    @staticmethod
    def create(schema: CollectionSchema) -> QueryExecutor:
        vectors = schema.vectors
        if len(vectors) == 0:
            return NoVectorQueryExecutor(schema)
        if len(vectors) == 1:
            return SingleVectorQueryExecutor(schema)
        return MultiVectorQueryExecutor(schema)
