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

import json
from typing import Any, Optional

from ..common import VectorType

__all__ = [
    "Doc",
    "DocList",
]


class Doc:
    """Represents a retrieved document with optional metadata, fields, and vectors.

    This immutable data class encapsulates the result of a search or retrieval
    operation. It includes the document ID, relevance score (if applicable),
    scalar fields, and vector embeddings.

    During initialization, any `numpy.ndarray` in `vectors` is automatically
    converted to a plain Python list for JSON serialization and immutability.

    Attributes:
        id (str): Unique identifier of the document.
        score (Optional[float], optional): Relevance score from search.
            Defaults to None.
        vectors (Optional[dict[str, VectorType]], optional): Named vector
            embeddings associated with the document. Values are converted to
            lists if originally `np.ndarray`. Defaults to None.
        fields (Optional[dict[str, Any]], optional): Scalar metadata fields
            (e.g., title, timestamp). Defaults to None.

    Examples:
        >>> import numpy as np
        >>> import zvec
        >>> doc = zvec.Doc(
        ...     id="doc1",
        ...     score=0.95,
        ...     vectors={"emb": np.array([0.1, 0.2, 0.3])},
        ...     fields={"title": "Hello World"}
        ... )
        >>> print(doc.vector("emb"))
        [0.1, 0.2, 0.3]
        >>> print(doc.has_field("title"))
        True
    """

    __slots__ = ("id", "score", "vectors", "fields")

    def __init__(
        self,
        id: str,
        score: Optional[float] = None,
        vectors: Optional[dict[str, VectorType]] = None,
        fields: Optional[dict[str, Any]] = None,
    ):
        self.id = id
        self.score = score
        self.vectors = vectors or {}
        self.fields = fields or {}

    def has_field(self, name: str) -> bool:
        """Check if the document contains a scalar field with the given name.

        Args:
            name (str): Name of the field to check.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        return name in self.fields

    def has_vector(self, name: str) -> bool:
        """Check if the document contains a vector with the given name.

        Args:
            name (str): Name of the vector to check.

        Returns:
            bool: True if the vector exists, False otherwise.
        """
        return name in self.vectors

    def vector(self, name: str):
        """Get a vector by name.

        Args:
            name (str): Name of the vector.

        Returns:
            Any: The vector (as a list) if it exists, otherwise None.
        """
        return self.vectors and self.vectors.get(name)

    def field(self, name: str):
        """Get a scalar field by name.

        Args:
            name (str): Name of the field.

        Returns:
            Any: The field value if it exists, otherwise None.
        """
        return self.fields and self.fields.get(name)

    def vector_names(self) -> list[str]:
        """Get the list of all vector names in this document.

        Returns:
            list[str]: A list of vector field names. Empty if no vectors.
        """
        return [] if not self.vectors else list(self.vectors.keys())

    def field_names(self) -> list[str]:
        """Get the list of all scalar field names in this document.

        Returns:
            list[str]: A list of field names. Empty if no fields.
        """
        return [] if not self.fields else list(self.fields.keys())

    def __repr__(self) -> str:
        try:
            schema = {
                "id": self.id,
                "score": self.score,
                "fields": self.fields,
                "vectors": self.vectors,
            }
            return json.dumps(schema, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<Doc error during repr: {e}>"

    def _replace(self, **changes):
        new_tuple = (
            changes.get("id", self.id),
            changes.get("score", self.score),
            changes.get("fields", self.fields.copy() if self.fields else None),
            changes.get("vectors", self.vectors.copy() if self.vectors else None),
        )
        return type(self)._from_tuple(new_tuple)

    @classmethod
    def _from_tuple(
        cls, data_tuple: tuple[str, float, dict[str, Any], dict[str, VectorType]]
    ):
        obj = object.__new__(cls)
        obj.id = data_tuple[0]
        obj.score = data_tuple[1]
        obj.fields = data_tuple[2] or {}

        vectors = data_tuple[3]
        if vectors is not None:
            obj.vectors = {
                name: (vec.tolist() if hasattr(vec, "tolist") else vec)
                for name, vec in vectors.items()
            }
        else:
            obj.vectors = {}
        return obj


#: Type alias for query results: a list of documents returned by a single query route.
DocList = list[Doc]
