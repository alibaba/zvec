"""
This module contains the schema of Zvec
"""

from __future__ import annotations

import collections.abc
import typing

import zvec._zvec.param
import zvec._zvec.typing

from .collection_schema import CollectionSchema
from .field_schema import FieldSchema, VectorSchema

__all__: list[str] = [
    "CollectionSchema",
    "CollectionStats",
    "FieldSchema",
    "VectorSchema",
]

class CollectionStats:
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def doc_count(self) -> int: ...
    @property
    def index_completeness(self) -> dict[str, float]: ...

class _CollectionSchema:
    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: _CollectionSchema) -> bool: ...
    def __init__(
        self, name: str, fields: collections.abc.Sequence[_FieldSchema]
    ) -> None:
        """
        Construct with name and list of fields
        """

    def __ne__(self, arg0: _CollectionSchema) -> bool: ...
    def fields(self) -> list[_FieldSchema]:
        """
        Return list of all field schemas.
        """

    def forward_fields(self) -> list[_FieldSchema]:
        """
        Return list of forward-indexed fields.
        """

    def get_field(self, field_name: str) -> _FieldSchema:
        """
        Get field by name (const pointer), returns None if not found.
        """

    def get_forward_field(self, field_name: str) -> _FieldSchema:
        """
        Get forward field (used for filtering).
        """

    def get_vector_field(self, field_name: str) -> _FieldSchema:
        """
        Get vector field by name.
        """

    def has_field(self, field_name: str) -> bool:
        """
        Check if a field exists.
        """

    def vector_fields(self) -> list[_FieldSchema]:
        """
        Return list of vector fields.
        """

    @property
    def name(self) -> str: ...

class _FieldSchema:
    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: _FieldSchema) -> bool: ...
    def __init__(
        self,
        name: str,
        data_type: zvec._zvec.typing.DataType,
        nullable: bool = False,
        dimension: typing.SupportsInt = 0,
        index_param: zvec._zvec.param.IndexParam = None,
    ) -> None: ...
    def __ne__(self, arg0: _FieldSchema) -> bool: ...
    @property
    def data_type(self) -> zvec._zvec.typing.DataType: ...
    @property
    def dimension(self) -> int: ...
    @property
    def index_param(self) -> typing.Any: ...
    @property
    def index_type(self) -> zvec._zvec.typing.IndexType: ...
    @property
    def is_dense_vector(self) -> bool: ...
    @property
    def is_sparse_vector(self) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def nullable(self) -> bool: ...
