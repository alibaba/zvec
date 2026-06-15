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


# Register the wheel-bundled jieba dict dir so `import zvec` alone makes
# the jieba FTS tokenizer usable. Users can still override via
# zvec.init(jieba_dict_dir=...), zvec.set_default_jieba_dict_dir(...),
# ZVEC_JIEBA_DICT_DIR, or per-field FtsIndexParam.extra_params.
try:
    from importlib.resources import files as _resource_files

    from _zvec import (
        get_default_jieba_dict_dir,
        set_default_jieba_dict_dir,
    )

    set_default_jieba_dict_dir(str(_resource_files("zvec").joinpath("data/jieba_dict")))
except Exception:
    # Custom builds without bundled dict; users must configure explicitly.
    pass


# ==============================
# Public API — grouped by category
# ==============================

# —— DiskAnn runtime plugin ——
# Re-export the plugin management entry points defined by the C++ extension.
# DiskAnn normally auto-loads on first use; these APIs let tests and
# diagnostic tools preload the plugin and get a clear error if libaio is
# missing or the plugin shared object cannot be located.
from _zvec import (
    DISKANN_PLUGIN_DLOPEN_FAILED,
    DISKANN_PLUGIN_LIBAIO_MISSING,
    DISKANN_PLUGIN_OK,
    DISKANN_PLUGIN_UNSUPPORTED_PLATFORM,
    is_diskann_plugin_loaded,
    is_libaio_available,
    load_diskann_plugin,
)

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
from .model.doc import Doc, DocList

# —— Query & index parameters ——
# —— FTS params (C++ binding) ——
from .model.param import (
    AddColumnOption,
    AlterColumnOption,
    CollectionOption,
    DiskAnnIndexParam,
    DiskAnnQueryParam,
    FlatIndexParam,
    FtsIndexParam,
    FtsQueryParam,
    HnswIndexParam,
    HnswQueryParam,
    HnswRabitqIndexParam,
    HnswRabitqQueryParam,
    IndexOption,
    InvertIndexParam,
    IVFIndexParam,
    IVFQueryParam,
    OptimizeOption,
    VamanaIndexParam,
    VamanaQueryParam,
)
from .model.param.query import Fts, Query, VectorQuery

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
    "set_default_jieba_dict_dir",
    "get_default_jieba_dict_dir",
    # Core classes
    "Collection",
    "Doc",
    "DocList",
    # Schema
    "CollectionSchema",
    "FieldSchema",
    "VectorSchema",
    "CollectionStats",
    # Parameters
    "Query",
    "VectorQuery",
    "Fts",
    "FtsIndexParam",
    "FtsQueryParam",
    "InvertIndexParam",
    "HnswIndexParam",
    "HnswRabitqIndexParam",
    "FlatIndexParam",
    "IVFIndexParam",
    "DiskAnnIndexParam",
    "DiskAnnQueryParam",
    "CollectionOption",
    "IndexOption",
    "OptimizeOption",
    "AddColumnOption",
    "AlterColumnOption",
    "HnswQueryParam",
    "HnswRabitqQueryParam",
    "IVFQueryParam",
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
    # DiskAnn plugin
    "load_diskann_plugin",
    "is_diskann_plugin_loaded",
    "is_libaio_available",
    "DISKANN_PLUGIN_OK",
    "DISKANN_PLUGIN_UNSUPPORTED_PLATFORM",
    "DISKANN_PLUGIN_LIBAIO_MISSING",
    "DISKANN_PLUGIN_DLOPEN_FAILED",
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
