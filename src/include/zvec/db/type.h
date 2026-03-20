// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cstdint>

namespace zvec {

/*
 * Column index types
 */
enum class IndexType : uint32_t {
  UNDEFINED = 0,
  HNSW = 1,
  IVF = 2,
  FLAT = 3,
  HNSW_RABITQ = 4,
  INVERT = 10,
};

/*
 * Column data types
 */
enum class DataType : uint32_t {
  UNDEFINED = 0,

  BINARY = 1,
  STRING = 2,
  BOOL = 3,
  INT32 = 4,
  INT64 = 5,
  UINT32 = 6,
  UINT64 = 7,
  FLOAT = 8,
  DOUBLE = 9,

  VECTOR_BINARY32 = 20,
  VECTOR_BINARY64 = 21,
  VECTOR_FP16 = 22,
  VECTOR_FP32 = 23,
  VECTOR_FP64 = 24,
  VECTOR_INT4 = 25,
  VECTOR_INT8 = 26,
  VECTOR_INT16 = 27,

  SPARSE_VECTOR_FP16 = 30,
  SPARSE_VECTOR_FP32 = 31,

  ARRAY_BINARY = 40,
  ARRAY_STRING = 41,
  ARRAY_BOOL = 42,
  ARRAY_INT32 = 43,
  ARRAY_INT64 = 44,
  ARRAY_UINT32 = 45,
  ARRAY_UINT64 = 46,
  ARRAY_FLOAT = 47,
  ARRAY_DOUBLE = 48,
};

enum class QuantizeType : uint32_t {
  UNDEFINED = 0,
  FP16 = 1,
  INT8 = 2,
  INT4 = 3,
  RABITQ = 4,
};

enum class MetricType : uint32_t {
  UNDEFINED = 0,
  L2 = 1,
  IP = 2,
  COSINE = 3,
  MIPSL2 = 4,
};

enum class Operator : uint32_t {
  INSERT = 0,
  UPSERT = 1,
  UPDATE = 2,
  DELETE = 3,
};

enum class CompareOp : uint32_t {
  NONE = 0,
  EQ,
  NE,
  LT,
  LE,
  GT,
  GE,
  LIKE,
  CONTAIN_ALL,
  CONTAIN_ANY,
  NOT_CONTAIN_ALL,
  NOT_CONTAIN_ANY,
  IS_NULL,
  IS_NOT_NULL,
  HAS_PREFIX,
  HAS_SUFFIX,
};

enum RelationOp : uint32_t {
  NONE = 0,

  AND = 1,
  OR = 2
};

enum BlockType : uint32_t {
  UNDEFINED = 0,
  SCALAR = 1,
  SCALAR_INDEX = 2,
  VECTOR_INDEX = 3,
  VECTOR_INDEX_QUANTIZE = 4,
};


enum class FileFormat : uint32_t {
  UNKNOWN = 0,
  IPC = 1,
  PARQUET = 2,
};

enum class ColumnOp : uint32_t {
  UNDEFINED = 0,
  ADD,
  ALTER,
  DROP,
};

}  // namespace zvec
