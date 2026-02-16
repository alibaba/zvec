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

#include "schema_parser.hpp"
#include <zvec/db/index_params.h>
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

// Minimal JSON parsing using DuckDB's yyjson (bundled with DuckDB)
#include "yyjson.hpp"

using namespace duckdb_yyjson;  // NOLINT

namespace duckdb {

static zvec::DataType ParseDataType(const std::string &type_str) {
  auto upper = StringUtil::Upper(type_str);
  if (upper == "BOOL") return zvec::DataType::BOOL;
  if (upper == "INT32") return zvec::DataType::INT32;
  if (upper == "INT64") return zvec::DataType::INT64;
  if (upper == "UINT32") return zvec::DataType::UINT32;
  if (upper == "UINT64") return zvec::DataType::UINT64;
  if (upper == "FLOAT") return zvec::DataType::FLOAT;
  if (upper == "DOUBLE") return zvec::DataType::DOUBLE;
  if (upper == "STRING") return zvec::DataType::STRING;
  if (upper == "BINARY") return zvec::DataType::BINARY;
  if (upper == "VECTOR_FP32") return zvec::DataType::VECTOR_FP32;
  if (upper == "VECTOR_FP64") return zvec::DataType::VECTOR_FP64;
  if (upper == "VECTOR_FP16") return zvec::DataType::VECTOR_FP16;
  if (upper == "VECTOR_INT8") return zvec::DataType::VECTOR_INT8;
  if (upper == "VECTOR_INT16") return zvec::DataType::VECTOR_INT16;
  if (upper == "VECTOR_BINARY32") return zvec::DataType::VECTOR_BINARY32;
  if (upper == "VECTOR_BINARY64") return zvec::DataType::VECTOR_BINARY64;
  if (upper == "SPARSE_VECTOR_FP32") return zvec::DataType::SPARSE_VECTOR_FP32;
  if (upper == "SPARSE_VECTOR_FP16") return zvec::DataType::SPARSE_VECTOR_FP16;
  if (upper == "ARRAY_INT32") return zvec::DataType::ARRAY_INT32;
  if (upper == "ARRAY_INT64") return zvec::DataType::ARRAY_INT64;
  if (upper == "ARRAY_UINT32") return zvec::DataType::ARRAY_UINT32;
  if (upper == "ARRAY_UINT64") return zvec::DataType::ARRAY_UINT64;
  if (upper == "ARRAY_FLOAT") return zvec::DataType::ARRAY_FLOAT;
  if (upper == "ARRAY_DOUBLE") return zvec::DataType::ARRAY_DOUBLE;
  if (upper == "ARRAY_STRING") return zvec::DataType::ARRAY_STRING;
  if (upper == "ARRAY_BINARY") return zvec::DataType::ARRAY_BINARY;
  if (upper == "ARRAY_BOOL") return zvec::DataType::ARRAY_BOOL;
  throw InvalidInputException("zvec: unknown data type '%s'", type_str);
}

static zvec::MetricType ParseMetricType(const std::string &metric_str) {
  auto upper = StringUtil::Upper(metric_str);
  if (upper == "L2") return zvec::MetricType::L2;
  if (upper == "IP") return zvec::MetricType::IP;
  if (upper == "COSINE") return zvec::MetricType::COSINE;
  throw InvalidInputException("zvec: unknown metric type '%s'", metric_str);
}

static zvec::IndexParams::Ptr ParseIndexParams(yyjson_val *index_obj) {
  if (!index_obj || !yyjson_is_obj(index_obj)) {
    return nullptr;
  }

  auto *type_val = yyjson_obj_get(index_obj, "type");
  if (!type_val) {
    throw InvalidInputException("zvec: index object missing 'type' field");
  }
  auto type_str = StringUtil::Upper(std::string(yyjson_get_str(type_val)));

  if (type_str == "INVERT") {
    bool range_opt = true;
    auto *range_val = yyjson_obj_get(index_obj, "enable_range_optimization");
    if (range_val && yyjson_is_bool(range_val)) {
      range_opt = yyjson_get_bool(range_val);
    }
    return std::make_shared<zvec::InvertIndexParams>(range_opt);
  }

  // Vector index types need metric
  auto *metric_val = yyjson_obj_get(index_obj, "metric");
  zvec::MetricType metric = zvec::MetricType::COSINE;
  if (metric_val && yyjson_is_str(metric_val)) {
    metric = ParseMetricType(yyjson_get_str(metric_val));
  }

  if (type_str == "HNSW") {
    int m = 16;
    int ef_construction = 200;
    auto *m_val = yyjson_obj_get(index_obj, "m");
    if (m_val && yyjson_is_int(m_val)) {
      m = (int)yyjson_get_int(m_val);
    }
    auto *ef_val = yyjson_obj_get(index_obj, "ef_construction");
    if (ef_val && yyjson_is_int(ef_val)) {
      ef_construction = (int)yyjson_get_int(ef_val);
    }
    return std::make_shared<zvec::HnswIndexParams>(metric, m, ef_construction);
  }
  if (type_str == "FLAT") {
    return std::make_shared<zvec::FlatIndexParams>(metric);
  }
  if (type_str == "IVF") {
    int n_list = 1024;
    int n_iters = 10;
    auto *nl_val = yyjson_obj_get(index_obj, "n_list");
    if (nl_val && yyjson_is_int(nl_val)) {
      n_list = (int)yyjson_get_int(nl_val);
    }
    auto *ni_val = yyjson_obj_get(index_obj, "n_iters");
    if (ni_val && yyjson_is_int(ni_val)) {
      n_iters = (int)yyjson_get_int(ni_val);
    }
    return std::make_shared<zvec::IVFIndexParams>(metric, n_list, n_iters);
  }

  throw InvalidInputException("zvec: unknown index type '%s'", type_str);
}

std::shared_ptr<zvec::CollectionSchema> ParseSchemaJson(
    const std::string &json_str) {
  auto *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
  if (!doc) {
    throw InvalidInputException("zvec: invalid JSON schema string");
  }

  auto *root = yyjson_doc_get_root(doc);
  if (!yyjson_is_obj(root)) {
    yyjson_doc_free(doc);
    throw InvalidInputException("zvec: schema JSON must be an object");
  }

  // Parse collection name
  auto *name_val = yyjson_obj_get(root, "name");
  std::string name = "default";
  if (name_val && yyjson_is_str(name_val)) {
    name = yyjson_get_str(name_val);
  }

  auto schema = std::make_shared<zvec::CollectionSchema>(name);

  // Parse max_doc_count_per_segment if present
  auto *max_doc_val = yyjson_obj_get(root, "max_doc_count_per_segment");
  if (max_doc_val && yyjson_is_int(max_doc_val)) {
    schema->set_max_doc_count_per_segment(
        (uint32_t)yyjson_get_int(max_doc_val));
  }

  // Parse fields
  auto *fields_arr = yyjson_obj_get(root, "fields");
  if (!fields_arr || !yyjson_is_arr(fields_arr)) {
    yyjson_doc_free(doc);
    throw InvalidInputException("zvec: schema JSON must have a 'fields' array");
  }

  size_t idx, max;
  yyjson_val *field_obj;
  yyjson_arr_foreach(fields_arr, idx, max, field_obj) {
    auto *fname_val = yyjson_obj_get(field_obj, "name");
    if (!fname_val || !yyjson_is_str(fname_val)) {
      yyjson_doc_free(doc);
      throw InvalidInputException("zvec: field missing 'name'");
    }
    std::string fname = yyjson_get_str(fname_val);

    auto *ftype_val = yyjson_obj_get(field_obj, "type");
    if (!ftype_val || !yyjson_is_str(ftype_val)) {
      yyjson_doc_free(doc);
      throw InvalidInputException("zvec: field '%s' missing 'type'", fname);
    }
    auto data_type = ParseDataType(yyjson_get_str(ftype_val));

    bool nullable = true;
    auto *nullable_val = yyjson_obj_get(field_obj, "nullable");
    if (nullable_val && yyjson_is_bool(nullable_val)) {
      nullable = yyjson_get_bool(nullable_val);
    }

    auto *index_obj = yyjson_obj_get(field_obj, "index");
    auto index_params = ParseIndexParams(index_obj);

    zvec::FieldSchema::Ptr field_schema;
    if (zvec::FieldSchema::is_vector_field(data_type)) {
      uint32_t dimension = 0;
      auto *dim_val = yyjson_obj_get(field_obj, "dimension");
      if (dim_val && yyjson_is_int(dim_val)) {
        dimension = (uint32_t)yyjson_get_int(dim_val);
      }
      field_schema = std::make_shared<zvec::FieldSchema>(
          fname, data_type, dimension, nullable, index_params);
    } else {
      field_schema = std::make_shared<zvec::FieldSchema>(
          fname, data_type, nullable, index_params);
    }

    schema->add_field(field_schema);
  }

  yyjson_doc_free(doc);
  return schema;
}

}  // namespace duckdb
