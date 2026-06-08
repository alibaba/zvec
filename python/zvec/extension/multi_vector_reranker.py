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

from collections.abc import Callable
from typing import Optional

from _zvec import _CallbackReranker, _RrfReranker, _WeightedReranker

from ..model.doc import DocList
from .rerank_function import RerankFunction


class RrfReRanker(RerankFunction):
    """Re-ranker using Reciprocal Rank Fusion (RRF) for multi-vector search.

    RRF combines results from multiple vector queries without requiring relevance scores.
    It assigns higher weight to documents that appear early in multiple result lists.

    The RRF score for a document at rank ``r`` is: ``1 / (k + r + 1)``,
    where ``k`` is the rank constant.

    Args:
        rank_constant (int, optional): Smoothing constant ``k`` in RRF formula.
            Larger values reduce the impact of early ranks. Defaults to 60.
    """

    def __init__(
        self,
        rank_constant: int = 60,
    ):
        self._rank_constant = rank_constant
        # Use C++ implementation for performance
        self._cpp_reranker = _RrfReranker(rank_constant)

    @property
    def rank_constant(self) -> int:
        return self._rank_constant

    def _get_object(self):
        """Return the underlying C++ RrfReranker instance."""
        return self._cpp_reranker

    def rerank(self, query_results: list[DocList], topn: int) -> DocList:
        """Re-rank using C++ RRF implementation.

        Args:
            query_results (list[DocList]): Multi-route recall results,
                positionally aligned with queries.
            topn (int): Number of top documents to return.

        Returns:
            DocList: Re-ranked documents.
        """
        return self._cpp_reranker.rerank(query_results, topn)


class WeightedReRanker(RerankFunction):
    """Re-ranker that combines scores from multiple vector fields using weights.

    Each vector field's relevance score is normalized based on its own metric
    type, then scaled by a user-provided weight. Final scores are summed across
    fields. The actual re-ranking logic lives in the C++ implementation.

    Args:
        weights (Optional[list[float]], optional): Weight per vector field,
            aligned by position with the queries supplied to ``collection.query()``.
            Defaults to None (treated as an empty list).
    """

    def __init__(
        self,
        weights: Optional[list[float]] = None,
    ):
        self._cpp_reranker = _WeightedReranker(weights or [])

    @property
    def weights(self) -> list[float]:
        """list[float]: Weight list for vector fields, aligned with queries."""
        return self._cpp_reranker.weights

    def _get_object(self):
        """Return the underlying C++ WeightedReranker instance."""
        return self._cpp_reranker

    def rerank(self, query_results: list[DocList], topn: int) -> DocList:
        """Re-rank using C++ Weighted implementation.

        Args:
            query_results (list[DocList]): Multi-route recall results,
                positionally aligned with queries.
            topn (int): Number of top documents to return.

        Returns:
            DocList: Re-ranked documents.
        """
        return self._cpp_reranker.rerank(query_results, topn)


class CallbackReRanker(RerankFunction):
    """Re-ranker that delegates to a user-provided Python callback.

    This bridges a Python callable into the C++ reranker interface, enabling
    custom re-ranking logic to be executed within the C++ MultiQuery path.

    The callback receives raw C++ ``_Doc`` objects grouped per query (as a
    ``list[list[_Doc]]``) and must return a ``list[_Doc]``.

    Args:
        callback: A callable with signature
            ``(query_results: list[list[_Doc]], topn: int) -> list[_Doc]``.
    """

    def __init__(
        self,
        callback: Callable,
    ):
        self._callback = callback
        self._cpp_reranker = _CallbackReranker(callback)

    def _get_object(self):
        """Return the underlying C++ CallbackReranker instance."""
        return self._cpp_reranker

    def rerank(self, query_results: list[DocList], topn: int) -> DocList:
        """Invoke the callback to re-rank documents.

        Args:
            query_results (list[DocList]): Multi-route recall results,
                positionally aligned with queries.
            topn (int): Number of top documents to return.

        Returns:
            DocList: Re-ranked documents.
        """
        return self._callback(query_results, topn)
