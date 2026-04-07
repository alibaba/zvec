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

from dataclasses import dataclass
from typing import Optional, Union

from ...common import VectorType
from . import HnswQueryParam, HnswRabitqQueryParam, IVFQueryParam

__all__ = ["VectorQuery"]


@dataclass(frozen=True)
class VectorQuery:
    """Represents a vector search query for a specific field in a collection.

    A `VectorQuery` can be constructed using either a document ID (to look up
    its vector) or an explicit vector. It may optionally include index-specific
    query parameters to control search behavior (e.g., `ef` for HNSW, `nprobe` for IVF).

    Exactly one of `id` or `vector` should be provided. If both are given,
    behavior is implementation-defined (typically `id` takes precedence).

    Attributes:
        field_name (str): Name of the vector field to query.
        id (Optional[str], optional): Document ID to fetch vector from. Default is None.
        vector (VectorType, optional): Explicit query vector. Default is None.
        param (Optional[Union[HnswQueryParam, IVFQueryParam]], optional):
            Index-specific query parameters. Default is None.

    Examples:
        >>> import zvec
        >>> # Query by ID
        >>> q1 = zvec.VectorQuery(field_name="embedding", id="doc123")
        >>> # Query by vector
        >>> q2 = zvec.VectorQuery(
        ...     field_name="embedding",
        ...     vector=[0.1, 0.2, 0.3],
        ...     param=HnswQueryParam(ef=300)
        ... )
    """

    field_name: str
    id: Optional[str] = None
    vector: VectorType = None
    param: Optional[Union[HnswQueryParam, HnswRabitqQueryParam, IVFQueryParam]] = None

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

    def _validate(self) -> None:
        if self.field_name is None:
            raise ValueError("Field name cannot be empty")
        if self.id and self.vector:
            raise ValueError("Cannot provide both id and vector")
