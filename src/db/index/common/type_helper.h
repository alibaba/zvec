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

#include <set>
#include <string>
#include <zvec/core/framework/index_meta.h>
#include <zvec/db/type.h>

namespace zvec {

struct IndexTypeCodeBook {
  static std::string AsString(IndexType type) {
    switch (type) {
      case IndexType::HNSW:
        return "HNSW";
      case IndexType::HNSW_RABITQ:
        return "HNSW_RABITQ";
      case IndexType::FLAT:
        return "FLAT";
      case IndexType::IVF:
        return "IVF";
      case IndexType::INVERT:
        return "INVERT";
      default:
        return "UNDEFINED";
    }
  }
};

struct DataTypeCodeBook {
  static bool IsArrayType(DataType type) {
    return type >= DataType::ARRAY_BINARY && type <= DataType::ARRAY_DOUBLE;
  }

  static std::string AsString(DataType type) {
    switch (type) {
      case DataType::BINARY:
        return "BINARY";
      case DataType::STRING:
        return "STRING";
      case DataType::BOOL:
        return "BOOL";
      case DataType::INT32:
        return "INT32";
      case DataType::INT64:
        return "INT64";
      case DataType::UINT32:
        return "UINT32";
      case DataType::UINT64:
        return "UINT64";
      case DataType::FLOAT:
        return "FLOAT";
      case DataType::DOUBLE:
        return "DOUBLE";
      case DataType::VECTOR_BINARY32:
        return "VECTOR_BINARY32";
      case DataType::VECTOR_BINARY64:
        return "VECTOR_BINARY64";
      case DataType::VECTOR_FP16:
        return "VECTOR_FP16";
      case DataType::VECTOR_FP32:
        return "VECTOR_FP32";
      case DataType::VECTOR_FP64:
        return "VECTOR_FP64";
      case DataType::VECTOR_INT4:
        return "VECTOR_INT4";
      case DataType::VECTOR_INT8:
        return "VECTOR_INT8";
      case DataType::VECTOR_INT16:
        return "VECTOR_INT16";
      case DataType::SPARSE_VECTOR_FP16:
        return "SPARSE_VECTOR_FP16";
      case DataType::SPARSE_VECTOR_FP32:
        return "SPARSE_VECTOR_FP32";
      case DataType::ARRAY_BINARY:
        return "ARRAY_BINARY";
      case DataType::ARRAY_STRING:
        return "ARRAY_STRING";
      case DataType::ARRAY_BOOL:
        return "ARRAY_BOOL";
      case DataType::ARRAY_INT32:
        return "ARRAY_INT32";
      case DataType::ARRAY_INT64:
        return "ARRAY_INT64";
      case DataType::ARRAY_UINT32:
        return "ARRAY_UINT32";
      case DataType::ARRAY_UINT64:
        return "ARRAY_UINT64";
      case DataType::ARRAY_FLOAT:
        return "ARRAY_FLOAT";
      case DataType::ARRAY_DOUBLE:
        return "ARRAY_DOUBLE";
      default:
        return "";
    }
  }

  static core::IndexMeta::DataType to_data_type(DataType type);
};

struct MetricTypeCodeBook {
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
      default:
        return "UNDEFINED";
    }
  }
};

}  // namespace zvec