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

//! On-disk enum values of the manifest format.
//!
//! These enum values are part of the persisted manifest format. The numeric
//! values must never change; only append new entries.

#pragma once

#include <cstdint>

namespace zvec {
namespace wire {

//! Mirrors proto enum DataType.
enum class DataType : int32_t {
  DT_UNDEFINED = 0,

  DT_BINARY = 1,
  DT_STRING = 2,
  DT_BOOL = 3,
  DT_INT32 = 4,
  DT_INT64 = 5,
  DT_UINT32 = 6,
  DT_UINT64 = 7,
  DT_FLOAT = 8,
  DT_DOUBLE = 9,

  DT_VECTOR_BINARY32 = 20,
  DT_VECTOR_BINARY64 = 21,
  DT_VECTOR_FP16 = 22,
  DT_VECTOR_FP32 = 23,
  DT_VECTOR_FP64 = 24,
  DT_VECTOR_INT4 = 25,
  DT_VECTOR_INT8 = 26,
  DT_VECTOR_INT16 = 27,

  DT_SPARSE_VECTOR_FP16 = 30,
  DT_SPARSE_VECTOR_FP32 = 31,

  DT_ARRAY_BINARY = 40,
  DT_ARRAY_STRING = 41,
  DT_ARRAY_BOOL = 42,
  DT_ARRAY_INT32 = 43,
  DT_ARRAY_INT64 = 44,
  DT_ARRAY_UINT32 = 45,
  DT_ARRAY_UINT64 = 46,
  DT_ARRAY_FLOAT = 47,
  DT_ARRAY_DOUBLE = 48,
};

//! Mirrors proto enum IndexType.
enum class IndexType : int32_t {
  IT_UNDEFINED = 0,
  IT_HNSW = 1,
  IT_IVF = 2,
  IT_FLAT = 3,
  IT_HNSW_RABITQ = 4,
  IT_VAMANA = 5,
  IT_DISKANN = 6,
  IT_IVF_RABITQ = 7,
  IT_INVERT = 10,
  IT_FTS = 11,
};

//! Mirrors proto enum QuantizeType.
enum class QuantizeType : int32_t {
  QT_UNDEFINED = 0,
  QT_FP16 = 1,
  QT_INT8 = 2,
  QT_INT4 = 3,
  QT_RABITQ = 4,
};

//! Mirrors proto enum MetricType.
enum class MetricType : int32_t {
  MT_UNDEFINED = 0,
  MT_L2 = 1,
  MT_IP = 2,
  MT_COSINE = 3,
};

//! Mirrors proto enum BlockType.
enum class BlockType : int32_t {
  BT_UNDEFINED = 0,
  BT_SCALAR = 1,
  BT_SCALAR_INDEX = 2,
  BT_VECTOR_INDEX = 3,
  BT_VECTOR_INDEX_QUANTIZE = 4,
  BT_FTS_INDEX = 5,
};

//! Converts a wire enum to its underlying numeric value for encoding.
template <typename E>
constexpr int32_t ToNumber(E value) {
  return static_cast<int32_t>(value);
}

//! Converts a numeric value read off the wire back to a wire enum. Unknown
//! values are preserved as-is; the CodeBook mapping turns them into
//! UNDEFINED, matching the previous protobuf-based behaviour.
template <typename E>
constexpr E FromNumber(int32_t value) {
  return static_cast<E>(value);
}

}  // namespace wire
}  // namespace zvec
