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

from typing import Optional

from ..model.doc import Doc
from .qwen_function import QwenFunctionBase
from .rerank_function import RerankFunction


class QwenReRanker(QwenFunctionBase, RerankFunction):
    """Re-ranker using Qwen (DashScope) cross-encoder API for semantic re-ranking.

    This re-ranker leverages DashScope's TextReRank service to perform
    cross-encoder style re-ranking. It sends query and document pairs to the
    API and receives relevance scores based on deep semantic understanding.

    The re-ranker is suitable for single-vector or multi-vector search scenarios
    where semantic relevance to a specific query is required.

    Args:
        query (str): Query text for semantic re-ranking. **Required**.
        topn (int, optional): Maximum number of documents to return after re-ranking.
            Defaults to 10.
        rerank_field (str): Document field name to use as re-ranking input text.
            **Required** (e.g., "content", "title", "body").
        model (str, optional): DashScope re-ranking model identifier.
            Defaults to ``"gte-rerank-v2"``.
        api_key (Optional[str], optional): DashScope API authentication key.
            If not provided, reads from ``DASHSCOPE_API_KEY`` environment variable.

    Raises:
        ValueError: If ``query`` is empty/None, ``rerank_field`` is None,
            or API key is not available.

    Note:
        - Requires ``dashscope`` Python package installed
        - Documents without valid content in ``rerank_field`` are skipped
        - API rate limits and quotas apply per DashScope subscription

    Example:
        >>> reranker = QwenReRanker(
        ...     query="machine learning algorithms",
        ...     topn=5,
        ...     rerank_field="content",
        ...     model="gte-rerank-v2",
        ...     api_key="your-api-key"
        ... )
        >>> # Use in collection.query(reranker=reranker)
    """

    def __init__(
        self,
        query: Optional[str] = None,
        topn: int = 10,
        rerank_field: Optional[str] = None,
        model: str = "gte-rerank-v2",
        api_key: Optional[str] = None,
    ):
        """Initialize QwenReRanker with query and configuration.

        Args:
            query (Optional[str]): Query text for semantic matching. Required.
            topn (int): Number of top results to return.
            rerank_field (Optional[str]): Document field for re-ranking input.
            model (str): DashScope model name.
            api_key (Optional[str]): API key or None to use environment variable.

        Raises:
            ValueError: If query is empty or API key is unavailable.
        """
        QwenFunctionBase.__init__(self, model=model, api_key=api_key)
        RerankFunction.__init__(self, topn=topn, rerank_field=rerank_field)

        if not query:
            raise ValueError("Query is required for QwenReRanker")
        self._query = query

    @property
    def query(self) -> str:
        """str: Query text used for semantic re-ranking."""
        return self._query

    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-rank documents using Qwen's TextReRank API.

        Sends document texts to DashScope TextReRank service along with the query.
        Returns documents sorted by relevance scores from the cross-encoder model.

        Args:
            query_results (dict[str, list[Doc]]): Mapping from vector field names
                to lists of retrieved documents. Documents from all fields are
                deduplicated and re-ranked together.

        Returns:
            list[Doc]: Re-ranked documents (up to ``topn``) with updated ``score``
                fields containing relevance scores from the API.

        Raises:
            ValueError: If no valid documents are found or API call fails.

        Note:
            - Duplicate documents (same ID) across fields are processed once
            - Documents with empty/missing ``rerank_field`` content are skipped
            - Returned scores are relevance scores from the cross-encoder model
        """
        if not query_results:
            return []

        # Collect and deduplicate documents
        id_to_doc: dict[str, Doc] = {}
        doc_ids: list[str] = []
        contents: list[str] = []

        for _, query_result in query_results.items():
            for doc in query_result:
                doc_id = doc.id
                if doc_id in id_to_doc:
                    continue

                # Extract text content from specified field
                field_value = doc.field(self.rerank_field)
                rank_content = str(field_value).strip() if field_value else ""
                if not rank_content:
                    continue

                id_to_doc[doc_id] = doc
                doc_ids.append(doc_id)
                contents.append(rank_content)

        if not contents:
            raise ValueError("No documents to rerank")

        # Call DashScope TextReRank API
        output = self._call_rerank_api(
            query=self.query,
            documents=contents,
            top_n=self.topn,
        )

        # Build result list with updated scores
        results: list[Doc] = []
        for item in output["results"]:
            idx = item["index"]
            doc_id = doc_ids[idx]
            doc = id_to_doc[doc_id]
            new_doc = doc._replace(score=item["relevance_score"])
            results.append(new_doc)

        return results
