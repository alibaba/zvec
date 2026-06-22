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
from typing import TYPE_CHECKING

from ..model.doc import Doc, DocList

if TYPE_CHECKING:
    from ..model.schema import FieldSchema, VectorSchema


class RerankFunction(ABC):
    """Abstract base class for reranker parameter containers.

    Subclasses define rerank parameters and implement _to_cpp_params()
    for conversion to C++ parameter structs (used by collection fast path).
    Each subclass also provides a standalone rerank() implementation.
    """

    def _to_cpp_params(self):
        """Return C++ reranker params. Override in subclasses that use C++ path."""
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        query_results: list[list[Doc]],
        topn: int = 10,
        *,
        fields: list[FieldSchema | VectorSchema] | None = None,
    ) -> DocList:
        """Execute rerank on sub-query results.

        Args:
            query_results: List of per-sub-query document lists.
            topn: Maximum number of results to return.
            fields: Per-sub-query Python FieldSchema/VectorSchema objects
                (required for WeightedReRanker score normalization).

        Returns:
            Re-ranked document list.
        """
        ...
