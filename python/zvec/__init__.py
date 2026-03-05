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


# ==============================
# Public API — grouped by category
# ==============================

from . import model as model

# —— Extensions ——
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

# —— Typing ——
from .model import param as param
from .model import schema as schema

# —— Core data structures ——
from .model.collection import Collection
from .model.doc import Doc

# —— Query & index parameters ——
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

# —— Schema & field definitions ——
from .model.schema import CollectionSchema, CollectionStats, FieldSchema, VectorSchema

# —— tools ——
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

# —— lifecycle ——
from .zvec import create_and_open, init, open

# ==============================
# Public interface declaration
# ==============================
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
    "VectorQuery",
    "InvertIndexParam",
    "HnswIndexParam",
    "FlatIndexParam",
    "IVFIndexParam",
    "CollectionOption",
    "IndexOption",
    "OptimizeOption",
    "AddColumnOption",
    "AlterColumnOption",
    "HnswQueryParam",
    "IVFQueryParam",
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
