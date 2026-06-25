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

import warnings
from dataclasses import dataclass
from typing import Optional, Union

from ...common import VectorType
from . import FtsQueryParam, HnswQueryParam, HnswRabitqQueryParam, IVFQueryParam

__all__ = ["Fts", "Query", "VectorQuery"]


@dataclass(frozen=True)
class Fts:
    """Full-text search query parameters.

    Attributes:
        query_string (Optional[str]): FTS query expression
            (e.g. '+vector -slow "exact phrase"'). Mutually exclusive with match_string.
        match_string (Optional[str]): Natural language match string,
            tokenized and combined using the default operator.
            Mutually exclusive with query_string.
    """

    query_string: Optional[str] = None
    match_string: Optional[str] = None


@dataclass(frozen=True)
class Query:
    """Represents a search query for a specific field in a collection.

    A `Query` can be constructed for either vector search or full-text search,
    but not both simultaneously.

    For vector search, provide `id` or `vector` (and optionally `param`).
    For FTS, provide `fts`.

    Attributes:
        field_name (str): Name of the field to query.
        id (Optional[str], optional): Document ID to fetch vector from. Default is None.
        vector (VectorType, optional): Explicit query vector. Default is None.
        param (Optional[Union[HnswQueryParam, HnswRabitqQueryParam, IVFQueryParam, FtsQueryParam]], optional):
            Index-specific query parameters. Default is None.
        fts (Optional[Fts], optional): Full-text search parameters. Default is None.

    Examples:
        >>> import zvec
        >>> # Query by ID
        >>> q1 = zvec.Query(field_name="embedding", id="doc123")
        >>> # Query by vector
        >>> q2 = zvec.Query(
        ...     field_name="embedding",
        ...     vector=[0.1, 0.2, 0.3],
        ...     param=HnswQueryParam(ef=300)
        ... )
        >>> # FTS query
        >>> q3 = zvec.Query(
        ...     field_name="content",
        ...     fts=Fts(match_string="machine learning")
        ... )
        >>> # FTS query with custom operator
        >>> q4 = zvec.Query(
        ...     field_name="content",
        ...     fts=Fts(match_string="machine learning"),
        ...     param=FtsQueryParam(default_operator="AND")
        ... )
    """

    field_name: str
    id: Optional[str] = None
    vector: VectorType = None
    param: Optional[
        Union[HnswQueryParam, HnswRabitqQueryParam, IVFQueryParam, FtsQueryParam]
    ] = None
    fts: Optional[Fts] = None

    def has_id(self) -> bool:
        """Check if the query is based on a document ID.

        Returns:
            bool: True if `id` is set, False otherwise.
        """
        return self.id is not None

    def has_vector(self) -> bool:
        """Check if the query contains an explicit vector.

        Returns:
            bool: True if `vector` is non-empty, False otherwise.
        """
        return self.vector is not None and len(self.vector) > 0

    def has_fts(self) -> bool:
        """Check if the query contains an FTS (full-text search) condition.

        Returns:
            bool: True if `fts` is set with a query_string or match_string.
        """
        if self.fts is not None:
            return bool(self.fts.query_string) or bool(self.fts.match_string)
        return False

    def _validate(self) -> None:
        if self.field_name is None:
            raise ValueError("Field name cannot be empty")
        if self.has_id() and self.has_vector():
            raise ValueError("Cannot provide both id and vector")
        if self.has_fts() and (self.has_vector() or self.has_id()):
            raise ValueError(
                "Cannot combine fts with vector search fields (id/vector) in a single Query"
            )
        if self.fts is not None and self.fts.query_string and self.fts.match_string:
            raise ValueError(
                "Cannot provide both query_string and match_string in Fts; "
                "they are mutually exclusive"
            )


class VectorQuery(Query):
    """Deprecated alias for Query. Use Query instead."""

    def __new__(cls, *args, **kwargs):  # noqa : ARG004
        warnings.warn(
            "VectorQuery is deprecated and will be removed in a future version. "
            "Use Query instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)
