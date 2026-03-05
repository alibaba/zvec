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

import heapq
import math
from collections import defaultdict
from typing import Optional

from ..model.doc import Doc
from ..typing import MetricType
from .rerank_function import RerankFunction


class RrfReRanker(RerankFunction):
    """Re-ranker using Reciprocal Rank Fusion (RRF) for multi-vector search.

    RRF combines results from multiple vector queries without requiring relevance scores.
    It assigns higher weight to documents that appear early in multiple result lists.

    The RRF score for a document at rank ``r`` is: ``1 / (k + r + 1)``,
    where ``k`` is the rank constant.

    Note:
        This re-ranker is specifically designed for multi-vector scenarios where
        query results from multiple vector fields need to be combined.

    Args:
        topn (int, optional): Number of top documents to return. Defaults to 10.
        rerank_field (Optional[str], optional): Ignored by RRF. Defaults to None.
        rank_constant (int, optional): Smoothing constant ``k`` in RRF formula.
            Larger values reduce the impact of early ranks. Defaults to 60.
    """

    def __init__(
        self,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        rank_constant: int = 60,
    ):
        super().__init__(topn=topn, rerank_field=rerank_field)
        self._rank_constant = rank_constant

    @property
    def rank_constant(self) -> int:
        return self._rank_constant

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self._rank_constant + rank + 1)

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Apply Reciprocal Rank Fusion to combine multiple query results.

        Args:
            query_results (dict[str, list[Doc]]): Results from one or more vector queries.

        Returns:
            list[Doc]: Re-ranked documents with RRF scores in the ``score`` field.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for _, query_result in query_results.items():
            for rank, doc in enumerate(query_result):
                doc_id = doc.id
                rrf_score = self._rrf_score(rank)
                rrf_scores[doc_id] += rrf_score
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(self.topn, rrf_scores.items(), key=lambda x: x[1])
        results: list[Doc] = []
        for doc_id, rrf_score in top_docs:
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=rrf_score)
            results.append(new_doc)
        return results


class WeightedReRanker(RerankFunction):
    """Re-ranker that combines scores from multiple vector fields using weights.

    Each vector field's relevance score is normalized based on its metric type,
    then scaled by a user-provided weight. Final scores are summed across fields.

    Note:
        This re-ranker is specifically designed for multi-vector scenarios where
        query results from multiple vector fields need to be combined with
        configurable weights.

    Args:
        topn (int, optional): Number of top documents to return. Defaults to 10.
        rerank_field (Optional[str], optional): Ignored. Defaults to None.
        metric (MetricType, optional): Distance metric used for score normalization.
            Defaults to ``MetricType.L2``.
        weights (Optional[dict[str, float]], optional): Weight per vector field.
            Fields not listed use weight 1.0. Defaults to None.

    Note:
        Supported metrics: L2, IP, COSINE. Scores are normalized to [0, 1].
    """

    def __init__(
        self,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        metric: MetricType = MetricType.L2,
        weights: Optional[dict[str, float]] = None,
    ):
        super().__init__(topn=topn, rerank_field=rerank_field)
        self._weights = weights or {}
        self._metric = metric

    @property
    def weights(self) -> dict[str, float]:
        """dict[str, float]: Weight mapping for vector fields."""
        return self._weights

    @property
    def metric(self) -> MetricType:
        """MetricType: Distance metric used for score normalization."""
        return self._metric

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Combine scores from multiple vector fields using weighted sum.

        Args:
            query_results (dict[str, list[Doc]]): Results per vector field.

        Returns:
            list[Doc]: Re-ranked documents with combined scores in ``score`` field.
        """
        weighted_scores: dict[str, float] = defaultdict(float)
        id_to_doc: dict[str, Doc] = {}

        for vector_name, query_result in query_results.items():
            for _, doc in enumerate(query_result):
                doc_id = doc.id
                weighted_score = self._normalize_score(
                    doc.score, self.metric
                ) * self.weights.get(vector_name, 1.0)
                weighted_scores[doc_id] += weighted_score
                if doc_id not in id_to_doc:
                    id_to_doc[doc_id] = doc

        top_docs = heapq.nlargest(
            self.topn, weighted_scores.items(), key=lambda x: x[1]
        )
        results: list[Doc] = []
        for doc_id, weighted_score in top_docs:
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=weighted_score)
            results.append(new_doc)
        return results

    def _normalize_score(self, score: float, metric: MetricType) -> float:
        if metric == MetricType.L2:
            return 1.0 - 2 * math.atan(score) / math.pi
        if metric == MetricType.IP:
            return 0.5 + math.atan(score) / math.pi
        if metric == MetricType.COSINE:
            return 1.0 - score / 2.0
        raise ValueError("Unsupported metric type")
