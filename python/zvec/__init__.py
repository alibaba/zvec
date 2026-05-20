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

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.metadata import PackageNotFoundError


from . import model as model
from .extension import (
    BM25EmbeddingFunction,
    DefaultLocalDenseEmbedding,
    DefaultLocalReRanker,
    DefaultLocalSparseEmbedding,
    DenseEmbeddingFunction,
    OpenAIDenseEmbedding,
    OpenAIFunctionBase,
    QwenDenseEmbedding,
    QwenFunctionBase,
    QwenReRanker,
    QwenSparseEmbedding,
    ReRanker,
    RrfReRanker,
    SentenceTransformerFunctionBase,
    SparseEmbeddingFunction,
    WeightedReRanker,
)
from .model import param as param
from .model import schema as schema
from .model.collection import Collection
from .model.doc import Doc
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    FlatIndexParam,
    HnswIndexParam,
    HnswQueryParam,
    HnswRabitqIndexParam,
    HnswRabitqQueryParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    IVFQueryParam,
    OmegaIndexParam,
    OmegaQueryParam,
    OptimizeOption,
    VamanaIndexParam,
    VamanaQueryParam,
)
from .model.param.query import Query, VectorQuery
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

__all__ = [
    # Zvec functions
    "create_and_open",
    "init",
    "open",
    # Core classes
    "Collection",
    "Doc",
    # Schema
    "CollectionSchema",
    "FieldSchema",
    "VectorSchema",
    "CollectionStats",
    # Parameters
    "Query",
    "VectorQuery",
    "InvertIndexParam",
    "HnswIndexParam",
    "HnswRabitqIndexParam",
    "FlatIndexParam",
    "IVFIndexParam",
    "OmegaIndexParam",
    "CollectionOption",
    "IndexOption",
    "OptimizeOption",
    "AddColumnOption",
    "AlterColumnOption",
    "HnswQueryParam",
    "HnswRabitqQueryParam",
    "IVFQueryParam",
    "OmegaQueryParam",
    "VamanaIndexParam",
    "VamanaQueryParam",
    # Extensions
    "DenseEmbeddingFunction",
    "SparseEmbeddingFunction",
    "QwenFunctionBase",
    "OpenAIFunctionBase",
    "SentenceTransformerFunctionBase",
    "ReRanker",
    "DefaultLocalDenseEmbedding",
    "DefaultLocalSparseEmbedding",
    "BM25EmbeddingFunction",
    "OpenAIDenseEmbedding",
    "QwenDenseEmbedding",
    "QwenSparseEmbedding",
    "RrfReRanker",
    "WeightedReRanker",
    "DefaultLocalReRanker",
    "QwenReRanker",
    # Typing
    "DataType",
    "MetricType",
    "QuantizeType",
    "IndexType",
    "LogLevel",
    "LogType",
    "Status",
    "StatusCode",
    # Tools
    "require_module",
]

# ==============================
# Version handling
# ==============================
__version__: str

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # Python < 3.8

try:
    __version__ = version("zvec")
except Exception:
    __version__ = "unknown"
