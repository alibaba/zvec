"""
Zvec core module
"""

from __future__ import annotations

import collections

from . import typing
from .extension import ReRanker, RrfReRanker, WeightedReRanker
from .extension.embedding import DenseEmbeddingFunction
from .model import param, schema
from .model.collection import Collection
from .model.doc import Doc
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    IVFQueryParam,
    OptimizeOption,
)
from .model.param.vector_query import VectorQuery
from .model.schema import CollectionSchema, CollectionStats, FieldSchema, VectorSchema
from .tool import require_module
from .typing import (
    DataType,
    IndexType,
    MetricType,
    QuantizeType,
    Status,
    StatusCode,
)
from .typing.enum import LogLevel, LogType
from .zvec import create_and_open, init, open

__all__: list = [
    "AddColumnOption",
    "AlterColumnOption",
    "Collection",
    "CollectionOption",
    "CollectionSchema",
    "CollectionStats",
    "DataType",
    "DenseEmbeddingFunction",
    "DenseEmbeddingFunction",
    "Doc",
    "FieldSchema",
    "FlatIndexParam",
    "HnswIndexParam",
    "HnswQueryParam",
    "IVFIndexParam",
    "IVFQueryParam",
    "IndexOption",
    "IndexType",
    "InvertIndexParam",
    "LogLevel",
    "LogType",
    "MetricType",
    "OptimizeOption",
    "QuantizeType",
    "ReRanker",
    "ReRanker",
    "RrfReRanker",
    "Status",
    "StatusCode",
    "VectorQuery",
    "VectorSchema",
    "WeightedReRanker",
    "create_and_open",
    "init",
    "open",
    "require_module",
]

class _Collection:
    @staticmethod
    def CreateAndOpen(
        arg0: str, arg1: schema._CollectionSchema, arg2: param.CollectionOption
    ) -> _Collection: ...
    @staticmethod
    def Open(arg0: str, arg1: param.CollectionOption) -> _Collection: ...
    def AddColumn(
        self,
        arg0: schema._FieldSchema,
        arg1: str,
        arg2: param.AddColumnOption,
    ) -> None: ...
    def AlterColumn(
        self,
        arg0: str,
        arg1: str,
        arg2: schema._FieldSchema,
        arg3: param.AlterColumnOption,
    ) -> None: ...
    def CreateIndex(
        self, arg0: str, arg1: param.IndexParam, arg2: param.IndexOption
    ) -> None: ...
    def Delete(self, arg0: collections.abc.Sequence[str]) -> list[typing.Status]: ...
    def DeleteByFilter(self, arg0: str) -> None: ...
    def Destroy(self) -> None: ...
    def DropColumn(self, arg0: str) -> None: ...
    def DropIndex(self, arg0: str) -> None: ...
    def Fetch(self, arg0: collections.abc.Sequence[str]) -> dict[str, _Doc]: ...
    def Flush(self) -> None: ...
    def GroupByQuery(self, arg0: ...) -> list[...]: ...
    def Insert(self, arg0: collections.abc.Sequence[_Doc]) -> list[typing.Status]: ...
    def Optimize(self, arg0: param.OptimizeOption) -> None: ...
    def Options(self) -> param.CollectionOption: ...
    def Path(self) -> str: ...
    def Query(self, arg0: param._VectorQuery) -> list[_Doc]: ...
    def Schema(self) -> schema._CollectionSchema: ...
    def Stats(self) -> schema.CollectionStats: ...
    def Update(self, arg0: collections.abc.Sequence[_Doc]) -> list[typing.Status]: ...
    def Upsert(self, arg0: collections.abc.Sequence[_Doc]) -> list[typing.Status]: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...

class _Doc:
    def __getstate__(self) -> bytes: ...
    def __init__(self) -> None: ...
    def __setstate__(self, arg0: bytes) -> None: ...
    def field_names(self) -> list[str]: ...
    def get_any(self, arg0: str, arg1: typing.DataType) -> typing.Any: ...
    def has_field(self, arg0: str) -> bool: ...
    def pk(self) -> str: ...
    def score(self) -> float: ...
    def set_any(self, arg0: str, arg1: typing.DataType, arg2: typing.Any) -> bool: ...
    def set_pk(self, arg0: str) -> None: ...
    def set_score(self, arg0: typing.SupportsFloat) -> None: ...

class _DocOp:
    """
    Members:

      INSERT

      UPDATE

      DELETE

      UPSERT
    """

    DELETE: typing.ClassVar[_DocOp]  # value = <_DocOp.DELETE: 3>
    INSERT: typing.ClassVar[_DocOp]  # value = <_DocOp.INSERT: 0>
    UPDATE: typing.ClassVar[_DocOp]  # value = <_DocOp.UPDATE: 2>
    UPSERT: typing.ClassVar[_DocOp]  # value = <_DocOp.UPSERT: 1>
    __members__: typing.ClassVar[
        dict[str, _DocOp]
    ]  # value = {'INSERT': <_DocOp.INSERT: 0>, 'UPDATE': <_DocOp.UPDATE: 2>, 'DELETE': <_DocOp.DELETE: 3>, 'UPSERT': <_DocOp.UPSERT: 1>}

    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...
