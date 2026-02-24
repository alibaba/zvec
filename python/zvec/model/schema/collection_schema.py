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
from typing import Literal, Optional, Union

from _zvec.schema import _CollectionSchema, _FieldSchema

from .field_schema import FieldSchema, VectorSchema

__all__ = [
    "CollectionSchema",
]

# Compression methods
COMPRESSION_METHODS = Literal["zstd", "gzip", "lzma", "auto", "none"]


class CollectionSchema:
    """Defines the structure of a collection in Zvec.

    A collection schema specifies the name of the collection and its fields,
    including both scalar fields (e.g., int, string) and vector fields.
    Field names must be unique across both scalar and vector fields.

    Args:
        name (str): Name of the collection.
        fields (Optional[Union[FieldSchema, list[FieldSchema]]], optional):
            One or more scalar field definitions. Defaults to None.
        vectors (Optional[Union[VectorSchema, list[VectorSchema]]], optional):
            One or more vector field definitions. Defaults to None.
        compression (Optional[COMPRESSION_METHODS], optional):
            Compression method for vector storage. Defaults to "none".
            - "zstd": Zstandard compression (best, requires Python 3.14+)
            - "gzip": Gzip compression (good balance)
            - "lzma": LZMA compression (best ratio, slowest)
            - "auto": Automatic selection based on vector size
            - "none": No compression

    Raises:
        TypeError: If `fields` or `vectors` are of unsupported types.
        ValueError: If any field or vector name is duplicated.

    Examples:
        >>> from zvec import FieldSchema, VectorSchema, DataType, IndexType
        >>> id_field = FieldSchema("id", DataType.INT64, is_primary=True)
        >>> emb_field = VectorSchema("embedding", dim=128, data_type=DataType.VECTOR_FP32)
        >>> schema = CollectionSchema(
        ...     name="my_collection",
        ...     fields=id_field,
        ...     vectors=emb_field,
        ...     compression="gzip"
        ... )
        >>> print(schema.name)
        my_collection
    """

    def __init__(
        self,
        name: str,
        fields: Optional[Union[FieldSchema, list[FieldSchema]]] = None,
        vectors: Optional[Union[VectorSchema, list[VectorSchema]]] = None,
        compression: Optional[COMPRESSION_METHODS] = "none",
    ):
        if name is None or not isinstance(name, str):
            raise ValueError(
                f"schema validate failed: collection name must be str, got {type(name).__name__}"
            )

        # Validate compression method
        valid_compression = ["zstd", "gzip", "lzma", "auto", "none"]
        if compression is None:
            compression = "none"
        elif compression not in valid_compression:
            raise ValueError(
                f"schema validate failed: compression must be one of {valid_compression}, got {compression}"
            )

        self._compression = compression

        # handle fields
        _fields_name: list[str] = []
        _fields_list: list[_FieldSchema] = []

        self._check_fields(fields, _fields_name, _fields_list)
        self._check_vectors(vectors, _fields_name, _fields_list)

        # init
        self._cpp_obj = _CollectionSchema(
            name=name,
            fields=_fields_list,
        )

    def _check_fields(
        self,
        fields: Optional[Union[FieldSchema, list[FieldSchema]]],
        _fields_name: list[str],
        _fields_list: list[_FieldSchema],
    ) -> None:
        field_items = []

        if isinstance(fields, FieldSchema):
            field_items = [fields]
        elif isinstance(fields, list):
            field_items = fields
        elif fields is None:
            field_items = []
        else:
            raise TypeError(
                f"schema validate failed: invalid 'fields' type, expected FieldSchema or list[FieldSchema], "
                f"got {type(fields).__name__}"
            )

        for idx, field in enumerate(field_items):
            if not isinstance(field, FieldSchema):
                raise TypeError(
                    f"schema validate failed: invalid field type in 'fields' list, expected FieldSchema, "
                    f"got {type(field).__name__} at index {idx}"
                )

            if field.name in _fields_name:
                raise ValueError(
                    f"schema validate failed: duplicate field name '{field.name}': field names must be unique"
                )
            _fields_name.append(field.name)
            _fields_list.append(field._get_object())

    def _check_vectors(
        self,
        vectors: Optional[Union[VectorSchema, list[VectorSchema]]],
        _fields_name: list[str],
        _fields_list: list[_FieldSchema],
    ) -> None:
        # handle vector
        if isinstance(vectors, VectorSchema):
            vectors_items = [vectors]
        elif isinstance(vectors, list):
            vectors_items = vectors
        elif vectors is None:
            vectors_items = []
        else:
            raise TypeError(
                f"schema validate failed: invalid 'vectors' type, expected VectorSchema or list[VectorSchema], "
                f"got {type(vectors).__name__}"
            )

        for idx, vector in enumerate(vectors_items):
            if not isinstance(vector, VectorSchema):
                raise TypeError(
                    f"schema validate failed: invalid vector type in 'vectors' list, expected VectorSchema, "
                    f"got {type(vector).__name__} at index {idx}"
                )

            if vector.name in _fields_name:
                raise ValueError(
                    f"schema validate failed: duplicate vector name '{vector.name}', vector names must be unique "
                    f"(conflicts with existing field or vector)"
                )
            _fields_name.append(vector.name)
            _fields_list.append(vector._get_object())

    @classmethod
    def _from_core(cls, core_collection_schema: _CollectionSchema):
        inst = cls.__new__(cls)
        if not core_collection_schema:
            raise ValueError("schema validate failed: schema is null")
        inst._cpp_obj = core_collection_schema
        return inst

    @property
    def name(self) -> str:
        """str: The name of the collection."""
        return self._cpp_obj.name

    def field(self, name: str) -> Optional[FieldSchema]:
        """Retrieve a scalar field by name.

        Args:
            name (str): Name of the field.

        Returns:
            Optional[FieldSchema]: The field if found, otherwise None.
        """
        _field = self._cpp_obj.get_forward_field(name)
        return FieldSchema._from_core(_field) if _field else None

    def vector(self, name: str) -> Optional[VectorSchema]:
        """Retrieve a vector field by name.

        Args:
            name (str): Name of the vector field.

        Returns:
            Optional[VectorSchema]: The vector field if found, otherwise None.
        """
        _field = self._cpp_obj.get_vector_field(name)
        return VectorSchema._from_core(_field) if _field else None

    @property
    def fields(self) -> list[FieldSchema]:
        """list[FieldSchema]: All scalar (non-vector) fields in the schema."""
        _fields = self._cpp_obj.forward_fields()
        return [FieldSchema._from_core(_field) for _field in _fields]

    @property
    def vectors(self) -> list[VectorSchema]:
        """list[VectorSchema]: All vector fields in the schema."""
        _vectors = self._cpp_obj.vector_fields()
        return [VectorSchema._from_core(_vector) for _vector in _vectors]

    @property
    def compression(self) -> str:
        """str: Compression method for vector storage."""
        return self._compression

    def _get_object(self) -> _CollectionSchema:
        return self._cpp_obj

    def __repr__(self) -> str:
        try:
            schema = {
                "name": self.name,
                "compression": self.compression,
                "fields": {field.name: field.__dict__() for field in self.fields},
                "vectors": {vector.name: vector.__dict__() for vector in self.vectors},
            }
            return json.dumps(schema, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<CollectionSchema error during repr: {e}>"

    def __str__(self) -> str:
        return self.__repr__()
