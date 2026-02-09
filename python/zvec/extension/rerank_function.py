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
from typing import Optional

from ..model.doc import Doc


class RerankFunction(ABC):
    """Abstract base class for re-ranking search results.

    Re-rankers refine the output of one or more vector queries by applying
    a secondary scoring strategy. They are used in the ``query()`` method of
    ``Collection`` via the ``reranker`` parameter.

    Args:
        topn (int, optional): Number of top documents to return after re-ranking.
            Defaults to 10.
        rerank_field (Optional[str], optional): Field name used as input for
            re-ranking (e.g., document title or body). Defaults to None.

    Note:
        Subclasses must implement the ``rerank()`` method.
    """

    def __init__(
        self,
        topn: int = 10,
        rerank_field: Optional[str] = None,
    ):
        self._topn = topn
        self._rerank_field = rerank_field

    @property
    def topn(self) -> int:
        """int: Number of top documents to return after re-ranking."""
        return self._topn

    @property
    def rerank_field(self) -> Optional[str]:
        """Optional[str]: Field name used as re-ranking input."""
        return self._rerank_field

    @abstractmethod
    def rerank(self, query_results: dict[str, list[Doc]]) -> list[Doc]:
        """Re-rank documents from one or more vector queries.

        Args:
            query_results (dict[str, list[Doc]]): Mapping from vector field name
                to list of retrieved documents (sorted by relevance).

        Returns:
            list[Doc]: Re-ranked list of documents (length ≤ ``topn``),
                with updated ``score`` fields.
        """
        ...
