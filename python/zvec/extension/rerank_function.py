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

from abc import ABC, abstractmethod

from ..model.doc import DocList


class RerankFunction(ABC):
    """Abstract base class for re-ranking search results.

    Re-rankers refine the output of one or more vector queries by applying
    a secondary scoring strategy. They are used in the ``query()`` method of
    ``Collection`` via the ``reranker`` parameter.

    Note:
        Subclasses must implement the ``rerank()`` method.
    """

    @abstractmethod
    def rerank(self, query_results: list[DocList], topn: int) -> DocList:
        """Re-rank documents from multi-route recall results.

        Args:
            query_results (list[DocList]): List of query results from
                multi-route recall. Each element corresponds to a Query in the
                collection.query(queries=List[Query]) call, aligned by position.
            topn (int): Number of top documents to return after re-ranking.

        Returns:
            DocList: Re-ranked list of documents (length ≤ ``topn``),
                with updated ``score`` fields.
        """
        ...

    def _get_object(self):
        """Return the underlying C++ Reranker instance, if available.

        This is used internally by the query executor to pass the reranker
        to the C++ MultiQuery method. Subclasses that wrap a C++ reranker
        should override this method.

        Returns:
            The C++ Reranker shared pointer, or None if not available.
        """
        return None  # noqa: RET501
