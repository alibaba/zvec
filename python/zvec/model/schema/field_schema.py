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
from typing import Any, Optional, Union

from _zvec.schema import _FieldSchema

from zvec.model.param import (
    FlatIndexParam,
    HnswIndexParam,
    HnswRabitqIndexParam,
    InvertIndexParam,
    IVFIndexParam,
)
from zvec.typing import DataType

__all__ = [
    "FieldSchema",
    "VectorSchema",
]

SUPPORT_VECTOR_DATA_TYPE = [
    DataType.VECTOR_FP16,
    DataType.VECTOR_FP32,
    DataType.VECTOR_FP64,
    DataType.VECTOR_INT8,
    DataType.SPARSE_VECTOR_FP16,
    DataType.SPARSE_VECTOR_FP32,
]

SUPPORT_SCALAR_DATA_TYPE = [
    DataType.INT32,
    DataType.INT64,
    DataType.UINT32,
    DataType.UINT64,
    DataType.FLOAT,
    DataType.DOUBLE,
    DataType.STRING,
    DataType.BOOL,
    DataType.ARRAY_INT32,
    DataType.ARRAY_INT64,
    DataType.ARRAY_UINT32,
    DataType.ARRAY_UINT64,
    DataType.ARRAY_FLOAT,
    DataType.ARRAY_DOUBLE,
    DataType.ARRAY_STRING,
    DataType.ARRAY_BOOL,
]


class FieldSchema:
    """Represents a scalar (non-vector) field in a collection schema.

    A `FieldSchema` defines the name, data type, nullability, and optional
    inverted index configuration for a regular field (e.g., ID, timestamp, category).

    Args:
        name (str): Name of the field. Must be unique within the collection.
        data_type (DataType): Data type of the field (e.g., INT64, STRING).
        nullable (bool, optional): Whether the field can contain null values.
            Defaults to False.
        index_param (Optional[InvertIndexParam], optional): Inverted index
            parameters for this field. Only applicable to fields that support
            indexing (e.g., scalar fields used in filtering). Defaults to None.

    Examples:
        >>> from zvec.typing import DataType
        >>> from zvec.model.param import InvertIndexParam
        >>> id_field = FieldSchema(
        ...     name="id",
        ...     data_type=DataType.INT64,
        ...     nullable=False,
        ...     index_param=InvertIndexParam(enable_range_optimization=True)
        ... )
    """

    def __init__(
        self,
        name: str,
        data_type: DataType,
        nullable: bool = False,
        index_param: Optional[InvertIndexParam] = None,
    ):
        if name is None or not isinstance(name, str):
            raise ValueError(
                f"schema validate failed: field name must be str, got {type(name).__name__}"
            )

        if data_type not in SUPPORT_SCALAR_DATA_TYPE:
            raise ValueError(
                f"schema validate failed: scalar_field's data_type must be one of "
                f"{', '.join(str(dt) for dt in SUPPORT_SCALAR_DATA_TYPE)}, "
                f"but field[{name}]'s data_type is {data_type}"
            )

        self._cpp_obj = _FieldSchema(
            name=name,
            data_type=data_type,
            dimension=0,
            nullable=nullable,
            index_param=index_param,
        )

    @classmethod
    def _from_core(cls, core_field_schema: _FieldSchema):
        if core_field_schema is None:
            raise ValueError("schema validate failed: field schema is None")
        inst = cls.__new__(cls)
        inst._cpp_obj = core_field_schema
        return inst

    def _get_object(self) -> _FieldSchema:
        return self._cpp_obj

    @property
    def name(self) -> str:
        """str: The name of the field."""
        return self._cpp_obj.name

    @property
    def data_type(self) -> DataType:
        """DataType: The data type of the field (e.g., INT64, STRING)."""
        return self._cpp_obj.data_type

    @property
    def nullable(self) -> bool:
        """bool: Whether the field allows null values."""
        return self._cpp_obj.nullable

    @property
    def index_param(self) -> Optional[InvertIndexParam]:
        """Optional[InvertIndexParam]: Inverted index configuration, if any."""
        return self._cpp_obj.index_param

    def __dict__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": (
                self.data_type.name
                if hasattr(self.data_type, "name")
                else str(self.data_type)
            ),
            "nullable": self.nullable,
            "index_param": (
                self.index_param.to_dict() if self.index_param is not None else None
            ),
        }

    def __repr__(self) -> str:
        try:
            schema = self.__dict__()
            return json.dumps(schema, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<FieldSchema error during repr: {e}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldSchema):
            return False
        return self._cpp_obj == other._cpp_obj

    def __hash__(self) -> int:
        return hash((self.name, self.data_type, self.nullable))


class VectorSchema:
    """Represents a vector field in a collection schema.

    A `VectorSchema` defines the name, data type, dimensionality, and index
    configuration for a vector field used in similarity search.

    Args:
        name (str): Name of the vector field. Must be unique within the collection.
        data_type (DataType): Vector data type (e.g., VECTOR_FP32, VECTOR_INT8).
        dimension (int, optional): Dimensionality of the vector. Must be > 0 for dense vectors;
         may be `None` for sparse vectors.
        index_param (Union[HnswIndexParam, IVFIndexParam, FlatIndexParam], optional):
            Index configuration for this vector field. Defaults to
            ``HnswIndexParam()``.

    Examples:
        >>> from zvec.typing import DataType
        >>> from zvec.model.param import HnswIndexParam
        >>> emb_field = VectorSchema(
        ...     name="embedding",
        ...     data_type=DataType.VECTOR_FP32,
        ...     dimension=128,
        ...     index_param=HnswIndexParam(ef_construction=200, m=16)
        ... )
    """

    def __init__(
        self,
        name: str,
        data_type: DataType,
        dimension: Optional[int] = 0,
        index_param: Optional[
            Union[HnswIndexParam, HnswRabitqIndexParam, FlatIndexParam, IVFIndexParam]
        ] = None,
    ):
        if name is None or not isinstance(name, str):
            raise ValueError(
                f"schema validate failed: field name must be str, got {type(name).__name__}"
            )

        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError("schema validate failed: vector's dimension must be >= 0")

        if data_type not in SUPPORT_VECTOR_DATA_TYPE:
            raise ValueError(
                f"schema validate failed: vector's data_type must be one of "
                f"{', '.join(str(dt) for dt in SUPPORT_VECTOR_DATA_TYPE)}, "
                f"but field[{name}]'s data_type is {data_type}"
            )

        if index_param is None:
            index_param = FlatIndexParam()

        self._cpp_obj = _FieldSchema(
            name=name,
            data_type=data_type,
            dimension=dimension,
            nullable=False,
            index_param=index_param,
        )

    @classmethod
    def _from_core(cls, core_field_schema: _FieldSchema):
        inst = cls.__new__(cls)
        inst._cpp_obj = core_field_schema
        return inst

    def _get_object(self) -> _FieldSchema:
        return self._cpp_obj

    @property
    def name(self) -> str:
        """str: The name of the vector field."""
        return self._cpp_obj.name

    @property
    def data_type(self) -> DataType:
        """DataType: The vector data type (e.g., VECTOR_FP32)."""
        return self._cpp_obj.data_type

    @property
    def dimension(self) -> int:
        """int: The dimensionality of the vector."""
        return self._cpp_obj.dimension

    @property
    def index_param(
        self,
    ) -> Union[HnswIndexParam, HnswRabitqIndexParam, IVFIndexParam, FlatIndexParam]:
        """Union[HnswIndexParam, HnswRabitqIndexParam, IVFIndexParam, FlatIndexParam]: Index configuration for the vector."""
        return self._cpp_obj.index_param

    def __dict__(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": (
                self.data_type.name
                if hasattr(self.data_type, "name")
                else str(self.data_type)
            ),
            "dimension": self.dimension,
            "index_param": (
                self.index_param.to_dict() if self.index_param is not None else None
            ),
        }

    def __repr__(self) -> str:
        try:
            schema = self.__dict__()
            return json.dumps(schema, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"<FieldSchema error during repr: {e}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorSchema):
            return False
        return self._cpp_obj == other._cpp_obj

    def __hash__(self) -> int:
        return hash((self.name, self.data_type, self.dimension))
