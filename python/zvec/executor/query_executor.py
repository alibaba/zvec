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

from typing import Optional, Union

import numpy as np
from _zvec import _Collection, _MultiQuery
from _zvec.param import _Fts, _SearchQuery, _SubQuery

from ..extension import CallbackReRanker, ReRanker, RrfReRanker, WeightedReRanker
from ..model.convert import convert_to_py_doc
from ..model.doc import DocList
from ..model.param.query import Query
from ..model.schema import CollectionSchema
from ..typing import DataType

__all__ = [
    "QueryContext",
    "QueryExecutor",
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


class QueryExecutor:
    """Unified query executor that routes based on query count and reranker type."""

    def __init__(self, schema: CollectionSchema):
        self._schema = schema

    def _build_queries(
        self, ctx: QueryContext, collection: _Collection
    ) -> list[_SearchQuery]:
        """Build query vector list (no validation, conversion only)."""
        if not ctx.queries:
            return [self._build_base_search_query(ctx)]
        return [
            self._build_search_query(ctx, query, collection) for query in ctx.queries
        ]

    def execute(self, ctx: QueryContext, collection: _Collection) -> DocList:
        """Execute a query, routing by query count.

        A single (or vector-less) query is sent to C++ as a ``_SearchQuery``;
        multiple queries are assembled into a ``_MultiQuery``.
        """
        queries = self._build_queries(ctx, collection)
        if not queries:
            raise ValueError("No query to execute")

        if len(queries) == 1:
            return self._execute_single_query(queries[0], collection)
        return self._execute_multi_query(ctx, queries, collection)

    def _execute_single_query(
        self, query: _SearchQuery, collection: _Collection
    ) -> DocList:
        """Single/vector-less query: send a ``_SearchQuery`` to C++."""
        docs = collection.Query(query)
        return [convert_to_py_doc(doc, self._schema) for doc in docs]

    def _execute_multi_query(
        self, ctx: QueryContext, queries: list[_SearchQuery], collection: _Collection
    ) -> DocList:
        """Multiple queries: send a ``_MultiQuery`` to C++.

        A Python-only reranker (e.g. a model/API-based one) cannot run inside
        the C++ MultiQuery, so each route is executed individually and merged by
        the reranker in Python. The built-in RRF/Weighted/Callback rerankers use
        the C++ variant-based fast path.
        """
        reranker = ctx.reranker
        if reranker is None:
            raise ValueError(
                "A reranker is required to merge results from multiple queries; "
                "specify the 'reranker' argument."
            )
        if not isinstance(reranker, (RrfReRanker, WeightedReRanker, CallbackReRanker)):
            docs_list = self._execute_python_pipeline(queries, collection)
            return self._merge_and_rerank(ctx, docs_list)

        multi_query = self._build_multi_query(ctx, queries)
        docs = collection.Query(multi_query)
        return [convert_to_py_doc(doc, self._schema) for doc in docs]

    def _build_multi_query(
        self, ctx: QueryContext, queries: list[_SearchQuery]
    ) -> _MultiQuery:
        """Assemble a C++ ``_MultiQuery`` from per-route ``_SearchQuery`` objects."""
        multi_query = _MultiQuery()
        multi_query.queries = [_SubQuery.from_search_query(query) for query in queries]
        # num_candidates controls per-sub-query candidate count for reranking pool.
        # It must NOT be limited to the final output topk; use at least the C++
        # SubQuery default of 10 to ensure sufficient candidates for reranking.
        _DEFAULT_NUM_CANDIDATES = 10
        for sub in multi_query.queries:
            sub.num_candidates = max(ctx.topk, _DEFAULT_NUM_CANDIDATES)
        multi_query.topk = ctx.topk
        if ctx.filter:
            multi_query.filter = ctx.filter
        multi_query.include_vector = ctx.include_vector
        if ctx.output_fields is not None:
            multi_query.output_fields = ctx.output_fields
        # Set rerank strategy via the C++ variant-based API.
        reranker = ctx.reranker
        if isinstance(reranker, RrfReRanker):
            multi_query.set_rerank_rrf(reranker.rank_constant)
        elif isinstance(reranker, WeightedReRanker):
            multi_query.set_rerank_weighted(reranker.weights)
        elif isinstance(reranker, CallbackReRanker):
            multi_query.set_rerank_callback(reranker._callback)
        return multi_query

    def _execute_python_pipeline(
        self, vectors: list[_SearchQuery], collection: _Collection
    ) -> list[DocList]:
        """Execute queries serially for the Python-only reranker path."""
        return [self._execute_single_query(query, collection) for query in vectors]

    def _merge_and_rerank(self, ctx: QueryContext, docs_list: list[DocList]) -> DocList:
        """Merge and rerank results from the Python pipeline path."""
        if not docs_list:
            raise ValueError("Query results is empty")
        if len(docs_list) == 1 and not ctx.reranker:
            return docs_list[0]
        return ctx.reranker.rerank(docs_list, ctx.topk)

    def _build_base_search_query(self, ctx: QueryContext) -> _SearchQuery:
        search_query = _SearchQuery()
        search_query.topk = ctx.topk
        search_query.include_vector = ctx.include_vector
        if ctx.filter:
            search_query.filter = ctx.filter
        if ctx.output_fields is not None:
            search_query.output_fields = ctx.output_fields
        return search_query

    def _apply_fts(self, query: Query, search_query: _SearchQuery) -> None:
        """Set FTS query on search_query if the query has FTS parameters."""
        if query.has_fts():
            fts = _Fts()
            fts.query_string = query.fts.query_string or ""
            fts.match_string = query.fts.match_string or ""
            search_query.fts = fts

    def _build_search_query(
        self, ctx: QueryContext, query: Query, collection: _Collection
    ) -> _SearchQuery:
        search_query = self._build_base_search_query(ctx)
        search_query.field_name = query.field_name
        if query.param:
            search_query.query_params = query.param

        # set FTS query if provided
        self._apply_fts(query, search_query)

        vector_schema = None
        if query.has_vector() or query.has_id():
            vector_schema = (
                self._schema.vector(query.field_name)
                if query
                else self._schema.vectors[0]
            )

            if vector_schema is None:
                raise ValueError("No vector field found")

        # set vector
        if query.has_vector():
            vec_data = query.vector
        elif query.has_id():
            fetched = collection.Fetch([query.id])
            doc = next(iter(fetched.values()), None)
            if not doc:
                raise ValueError(f"Document with id '{query.id}' not found")
            vec_data = doc.get_any(vector_schema.name, vector_schema.data_type)
        else:
            return search_query

        target_dtype = DTYPE_MAP.get(vector_schema.data_type.value)
        search_query.set_vector(
            vector_schema._get_object(),
            convert_to_numpy(vec_data, target_dtype) if target_dtype else vec_data,
        )
        return search_query
