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

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <zvec/core/framework/index_meta.h>
#include <zvec/db/type.h>
#include "db/index/common/manifest_enum.h"

namespace zvec {

enum class SparseIndicesStatus {
  kOk,
  kNeedSort,
  kHasDuplicate,
};

//! Read-only check of sparse indices: sorted-unique, unsorted, or has
//! duplicates.  Uses adjacent_find so it is O(n) and non-mutating.
SparseIndicesStatus need_sanitize_sparse(const uint32_t *indices, size_t n);

//! Sort sparse (indices, values) pairs in place by index ascending and report
//! whether any duplicate index exists. value_byte_size is the per-value stride.
bool sort_and_find_duplicates(uint32_t *indices, char *values, size_t n,
                              size_t value_byte_size);

//! Index Type Codebook
struct IndexTypeCodeBook {
  //! Convert wire-format IndexType to C++ IndexType
  static IndexType Get(wire::IndexType type) {
    switch (type) {
      case wire::IndexType::IT_HNSW:
        return IndexType::HNSW;
      case wire::IndexType::IT_HNSW_RABITQ:
        return IndexType::HNSW_RABITQ;
      case wire::IndexType::IT_IVF_RABITQ:
        return IndexType::IVF_RABITQ;
      case wire::IndexType::IT_FLAT:
        return IndexType::FLAT;
      case wire::IndexType::IT_IVF:
        return IndexType::IVF;
      case wire::IndexType::IT_VAMANA:
        return IndexType::VAMANA;
      case wire::IndexType::IT_INVERT:
        return IndexType::INVERT;
      case wire::IndexType::IT_DISKANN:
        return IndexType::DISKANN;
      case wire::IndexType::IT_FTS:
        return IndexType::FTS;
      default:
        break;
    }
    return IndexType::UNDEFINED;
  }

  //! Convert C++ IndexType to wire-format IndexType
  static wire::IndexType Get(IndexType type) {
    switch (type) {
      case IndexType::HNSW:
        return wire::IndexType::IT_HNSW;
      case IndexType::HNSW_RABITQ:
        return wire::IndexType::IT_HNSW_RABITQ;
      case IndexType::IVF_RABITQ:
        return wire::IndexType::IT_IVF_RABITQ;
      case IndexType::FLAT:
        return wire::IndexType::IT_FLAT;
      case IndexType::IVF:
        return wire::IndexType::IT_IVF;
      case IndexType::VAMANA:
        return wire::IndexType::IT_VAMANA;
      case IndexType::INVERT:
        return wire::IndexType::IT_INVERT;
      case IndexType::DISKANN:
        return wire::IndexType::IT_DISKANN;
      case IndexType::FTS:
        return wire::IndexType::IT_FTS;
      default:
        break;
    }
    return wire::IndexType::IT_UNDEFINED;
  }

  //! Convert C++ IndexType to C++ String
  static std::string AsString(IndexType type) {
    switch (type) {
      case IndexType::HNSW:
        return "HNSW";
      case IndexType::HNSW_RABITQ:
        return "HNSW_RABITQ";
      case IndexType::IVF_RABITQ:
        return "IVF_RABITQ";
      case IndexType::FLAT:
        return "FLAT";
      case IndexType::IVF:
        return "IVF";
      case IndexType::DISKANN:
        return "DISKANN";
      case IndexType::VAMANA:
        return "VAMANA";
      case IndexType::INVERT:
        return "INVERT";
      case IndexType::FTS:
        return "FTS";
      default:
        break;
    }
    return "UNDEFINED";
  }
};

struct DataTypeCodeBook {
  static bool IsArrayType(wire::DataType type) {
    return wire::DataType::DT_ARRAY_BINARY <= type &&
           type <= wire::DataType::DT_ARRAY_DOUBLE;
  }

  static DataType Get(wire::DataType type) {
    DataType data_types = DataType::UNDEFINED;
    switch (type) {
      case wire::DataType::DT_BINARY:
        data_types = DataType::BINARY;
        break;
      case wire::DataType::DT_STRING:
        data_types = DataType::STRING;
        break;
      case wire::DataType::DT_BOOL:
        data_types = DataType::BOOL;
        break;
      case wire::DataType::DT_INT32:
        data_types = DataType::INT32;
        break;
      case wire::DataType::DT_INT64:
        data_types = DataType::INT64;
        break;
      case wire::DataType::DT_UINT32:
        data_types = DataType::UINT32;
        break;
      case wire::DataType::DT_UINT64:
        data_types = DataType::UINT64;
        break;
      case wire::DataType::DT_FLOAT:
        data_types = DataType::FLOAT;
        break;
      case wire::DataType::DT_DOUBLE:
        data_types = DataType::DOUBLE;
        break;
      case wire::DataType::DT_VECTOR_BINARY32:
        data_types = DataType::VECTOR_BINARY32;
        break;
      case wire::DataType::DT_VECTOR_BINARY64:
        data_types = DataType::VECTOR_BINARY64;
        break;
      case wire::DataType::DT_VECTOR_FP16:
        data_types = DataType::VECTOR_FP16;
        break;
      case wire::DataType::DT_VECTOR_FP32:
        data_types = DataType::VECTOR_FP32;
        break;
      case wire::DataType::DT_VECTOR_FP64:
        data_types = DataType::VECTOR_FP64;
        break;
      case wire::DataType::DT_VECTOR_INT4:
        data_types = DataType::VECTOR_INT4;
        break;
      case wire::DataType::DT_VECTOR_INT8:
        data_types = DataType::VECTOR_INT8;
        break;
      case wire::DataType::DT_VECTOR_INT16:
        data_types = DataType::VECTOR_INT16;
        break;
      case wire::DataType::DT_SPARSE_VECTOR_FP16:
        data_types = DataType::SPARSE_VECTOR_FP16;
        break;
      case wire::DataType::DT_SPARSE_VECTOR_FP32:
        data_types = DataType::SPARSE_VECTOR_FP32;
        break;
      case wire::DataType::DT_ARRAY_BINARY:
        data_types = DataType::ARRAY_BINARY;
        break;
      case wire::DataType::DT_ARRAY_STRING:
        data_types = DataType::ARRAY_STRING;
        break;
      case wire::DataType::DT_ARRAY_BOOL:
        data_types = DataType::ARRAY_BOOL;
        break;
      case wire::DataType::DT_ARRAY_INT32:
        data_types = DataType::ARRAY_INT32;
        break;
      case wire::DataType::DT_ARRAY_INT64:
        data_types = DataType::ARRAY_INT64;
        break;
      case wire::DataType::DT_ARRAY_UINT32:
        data_types = DataType::ARRAY_UINT32;
        break;
      case wire::DataType::DT_ARRAY_UINT64:
        data_types = DataType::ARRAY_UINT64;
        break;
      case wire::DataType::DT_ARRAY_FLOAT:
        data_types = DataType::ARRAY_FLOAT;
        break;
      case wire::DataType::DT_ARRAY_DOUBLE:
        data_types = DataType::ARRAY_DOUBLE;
        break;

      default:
        break;
    }
    return data_types;
  }

  static wire::DataType Get(const DataType type) {
    wire::DataType data_type = wire::DataType::DT_UNDEFINED;
    switch (type) {
      case DataType::BINARY:
        data_type = wire::DataType::DT_BINARY;
        break;
      case DataType::STRING:
        data_type = wire::DataType::DT_STRING;
        break;
      case DataType::BOOL:
        data_type = wire::DataType::DT_BOOL;
        break;
      case DataType::INT32:
        data_type = wire::DataType::DT_INT32;
        break;
      case DataType::INT64:
        data_type = wire::DataType::DT_INT64;
        break;
      case DataType::UINT32:
        data_type = wire::DataType::DT_UINT32;
        break;
      case DataType::UINT64:
        data_type = wire::DataType::DT_UINT64;
        break;
      case DataType::FLOAT:
        data_type = wire::DataType::DT_FLOAT;
        break;
      case DataType::DOUBLE:
        data_type = wire::DataType::DT_DOUBLE;
        break;
      case DataType::VECTOR_BINARY32:
        data_type = wire::DataType::DT_VECTOR_BINARY32;
        break;
      case DataType::VECTOR_BINARY64:
        data_type = wire::DataType::DT_VECTOR_BINARY64;
        break;
      case DataType::VECTOR_FP16:
        data_type = wire::DataType::DT_VECTOR_FP16;
        break;
      case DataType::VECTOR_FP32:
        data_type = wire::DataType::DT_VECTOR_FP32;
        break;
      case DataType::VECTOR_FP64:
        data_type = wire::DataType::DT_VECTOR_FP64;
        break;
      case DataType::VECTOR_INT4:
        data_type = wire::DataType::DT_VECTOR_INT4;
        break;
      case DataType::VECTOR_INT8:
        data_type = wire::DataType::DT_VECTOR_INT8;
        break;
      case DataType::VECTOR_INT16:
        data_type = wire::DataType::DT_VECTOR_INT16;
        break;
      case DataType::SPARSE_VECTOR_FP16:
        data_type = wire::DataType::DT_SPARSE_VECTOR_FP16;
        break;
      case DataType::SPARSE_VECTOR_FP32:
        data_type = wire::DataType::DT_SPARSE_VECTOR_FP32;
        break;
      case DataType::ARRAY_BINARY:
        data_type = wire::DataType::DT_ARRAY_BINARY;
        break;
      case DataType::ARRAY_BOOL:
        data_type = wire::DataType::DT_ARRAY_BOOL;
        break;
      case DataType::ARRAY_DOUBLE:
        data_type = wire::DataType::DT_ARRAY_DOUBLE;
        break;
      case DataType::ARRAY_FLOAT:
        data_type = wire::DataType::DT_ARRAY_FLOAT;
        break;
      case DataType::ARRAY_INT32:
        data_type = wire::DataType::DT_ARRAY_INT32;
        break;
      case DataType::ARRAY_INT64:
        data_type = wire::DataType::DT_ARRAY_INT64;
        break;
      case DataType::ARRAY_STRING:
        data_type = wire::DataType::DT_ARRAY_STRING;
        break;
      case DataType::ARRAY_UINT32:
        data_type = wire::DataType::DT_ARRAY_UINT32;
        break;
      case DataType::ARRAY_UINT64:
        data_type = wire::DataType::DT_ARRAY_UINT64;
        break;
      default:
        break;
    }

    return data_type;
  }

  static std::string AsString(DataType type) {
    std::string data_type;

    switch (type) {
      case DataType::BINARY:
        data_type = "BINARY";
        break;
      case DataType::STRING:
        data_type = "STRING";
        break;
      case DataType::BOOL:
        data_type = "BOOL";
        break;
      case DataType::INT32:
        data_type = "INT32";
        break;
      case DataType::INT64:
        data_type = "INT64";
        break;
      case DataType::UINT32:
        data_type = "UINT32";
        break;
      case DataType::UINT64:
        data_type = "UINT64";
        break;
      case DataType::FLOAT:
        data_type = "FLOAT";
        break;
      case DataType::DOUBLE:
        data_type = "DOUBLE";
        break;
      case DataType::VECTOR_BINARY32:
        data_type = "VECTOR_BINARY32";
        break;
      case DataType::VECTOR_BINARY64:
        data_type = "VECTOR_BINARY64";
        break;
      case DataType::VECTOR_FP16:
        data_type = "VECTOR_FP16";
        break;
      case DataType::VECTOR_FP32:
        data_type = "VECTOR_FP32";
        break;
      case DataType::VECTOR_FP64:
        data_type = "VECTOR_FP64";
        break;
      case DataType::VECTOR_INT4:
        data_type = "VECTOR_INT4";
        break;
      case DataType::VECTOR_INT8:
        data_type = "VECTOR_INT8";
        break;
      case DataType::VECTOR_INT16:
        data_type = "VECTOR_INT16";
        break;
      case DataType::SPARSE_VECTOR_FP16:
        data_type = "SPARSE_VECTOR_FP16";
        break;
      case DataType::SPARSE_VECTOR_FP32:
        data_type = "SPARSE_VECTOR_FP32";
        break;
      case DataType::ARRAY_BINARY:
        data_type = "ARRAY_BINARY";
        break;
      case DataType::ARRAY_BOOL:
        data_type = "ARRAY_BOOL";
        break;
      case DataType::ARRAY_DOUBLE:
        data_type = "ARRAY_DOUBLE";
        break;
      case DataType::ARRAY_FLOAT:
        data_type = "ARRAY_FLOAT";
        break;
      case DataType::ARRAY_INT32:
        data_type = "ARRAY_INT32";
        break;
      case DataType::ARRAY_INT64:
        data_type = "ARRAY_INT64";
        break;
      case DataType::ARRAY_STRING:
        data_type = "ARRAY_STRING";
        break;
      case DataType::ARRAY_UINT32:
        data_type = "ARRAY_UINT32";
        break;
      case DataType::ARRAY_UINT64:
        data_type = "ARRAY_UINT64";
        break;
      default:
        break;
    }

    return data_type;
  }

  static core::IndexMeta::DataType to_data_type(DataType type);
};

struct MetricTypeCodeBook {
  static MetricType Get(wire::MetricType type) {
    switch (type) {
      case wire::MetricType::MT_IP:
        return MetricType::IP;
      case wire::MetricType::MT_L2:
        return MetricType::L2;
      case wire::MetricType::MT_COSINE:
        return MetricType::COSINE;
      default:
        return MetricType::UNDEFINED;
    }
  }

  static wire::MetricType Get(MetricType type) {
    switch (type) {
      case MetricType::IP:
        return wire::MetricType::MT_IP;
      case MetricType::L2:
        return wire::MetricType::MT_L2;
      case MetricType::COSINE:
        return wire::MetricType::MT_COSINE;
      default:
        return wire::MetricType::MT_UNDEFINED;
    }
  }

  static std::string AsString(MetricType type) {
    switch (type) {
      case MetricType::IP:
        return "IP";
      case MetricType::L2:
        return "L2";
      case MetricType::COSINE:
        return "COSINE";
      default:
        return "UNDEFINED";
    }
  }
};

struct QuantizeTypeCodeBook {
  static QuantizeType Get(wire::QuantizeType type) {
    switch (type) {
      case wire::QuantizeType::QT_FP16:
        return QuantizeType::FP16;
      case wire::QuantizeType::QT_INT4:
        return QuantizeType::INT4;
      case wire::QuantizeType::QT_INT8:
        return QuantizeType::INT8;
      case wire::QuantizeType::QT_RABITQ:
        return QuantizeType::RABITQ;
      default:
        return QuantizeType::UNDEFINED;
    }
  }

  static wire::QuantizeType Get(QuantizeType type) {
    switch (type) {
      case QuantizeType::FP16:
        return wire::QuantizeType::QT_FP16;
      case QuantizeType::INT4:
        return wire::QuantizeType::QT_INT4;
      case QuantizeType::INT8:
        return wire::QuantizeType::QT_INT8;
      case QuantizeType::RABITQ:
        return wire::QuantizeType::QT_RABITQ;
      default:
        return wire::QuantizeType::QT_UNDEFINED;
    }
  }

  static std::string AsString(QuantizeType type) {
    switch (type) {
      case QuantizeType::FP16:
        return "FP16";
      case QuantizeType::INT4:
        return "INT4";
      case QuantizeType::INT8:
        return "INT8";
      case QuantizeType::RABITQ:
        return "RABITQ";
      default:
        return "UNDEFINED";
    }
  }

  static std::string AsString(std::set<QuantizeType> type) {
    std::string str;
    for (auto t : type) {
      str += QuantizeTypeCodeBook::AsString(t) + ",";
    }
    return str.substr(0, str.size() - 1);
  }
};

struct BlockTypeCodeBook {
  static BlockType Get(wire::BlockType type) {
    BlockType block_types = BlockType::UNDEFINED;
    switch (type) {
      case wire::BlockType::BT_SCALAR:
        block_types = BlockType::SCALAR;
        break;
      case wire::BlockType::BT_SCALAR_INDEX:
        block_types = BlockType::SCALAR_INDEX;
        break;
      case wire::BlockType::BT_VECTOR_INDEX:
        block_types = BlockType::VECTOR_INDEX;
        break;
      case wire::BlockType::BT_VECTOR_INDEX_QUANTIZE:
        block_types = BlockType::VECTOR_INDEX_QUANTIZE;
        break;
      case wire::BlockType::BT_FTS_INDEX:
        block_types = BlockType::FTS_INDEX;
        break;
      default:
        break;
    }
    return block_types;
  }

  static wire::BlockType Get(BlockType type) {
    wire::BlockType block_types = wire::BlockType::BT_UNDEFINED;
    switch (type) {
      case BlockType::SCALAR:
        block_types = wire::BlockType::BT_SCALAR;
        break;
      case BlockType::SCALAR_INDEX:
        block_types = wire::BlockType::BT_SCALAR_INDEX;
        break;
      case BlockType::VECTOR_INDEX:
        block_types = wire::BlockType::BT_VECTOR_INDEX;
        break;
      case BlockType::VECTOR_INDEX_QUANTIZE:
        block_types = wire::BlockType::BT_VECTOR_INDEX_QUANTIZE;
        break;
      case BlockType::FTS_INDEX:
        block_types = wire::BlockType::BT_FTS_INDEX;
        break;
      default:
        break;
    }

    return block_types;
  }

  static std::string AsString(BlockType type) {
    switch (type) {
      case BlockType::SCALAR:
        return "SCALAR";
      case BlockType::SCALAR_INDEX:
        return "SCALAR_INDEX";
      case BlockType::VECTOR_INDEX:
        return "VECTOR_INDEX";
      case BlockType::VECTOR_INDEX_QUANTIZE:
        return "VECTOR_INDEX_QUANTIZE";
      case BlockType::FTS_INDEX:
        return "FTS_INDEX";
      default:
        return "UNDEFINED";
    }
  }
};

}  // namespace zvec
