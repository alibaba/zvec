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

#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "duckdb/main/extension_util.hpp"
#include "collection_registry.hpp"
#include "duckdb.hpp"
#include "type_conversion.hpp"

// JSON parsing via DuckDB's bundled yyjson
#include "yyjson.hpp"

using namespace duckdb_yyjson;  // NOLINT

namespace duckdb {

static void SetDocFieldFromJson(zvec::Doc &doc, const std::string &field_name,
                                zvec::DataType data_type, yyjson_val *val,
                                uint32_t dimension) {
  if (!val || yyjson_is_null(val)) {
    doc.set_null(field_name);
    return;
  }

  switch (data_type) {
    case zvec::DataType::BOOL:
      doc.set<bool>(field_name, yyjson_get_bool(val));
      break;
    case zvec::DataType::INT32:
      doc.set<int32_t>(field_name, (int32_t)yyjson_get_int(val));
      break;
    case zvec::DataType::INT64:
      doc.set<int64_t>(field_name, (int64_t)yyjson_get_int(val));
      break;
    case zvec::DataType::UINT32:
      doc.set<uint32_t>(field_name, (uint32_t)yyjson_get_uint(val));
      break;
    case zvec::DataType::UINT64:
      doc.set<uint64_t>(field_name, (uint64_t)yyjson_get_uint(val));
      break;
    case zvec::DataType::FLOAT:
      doc.set<float>(field_name, (float)yyjson_get_real(val));
      break;
    case zvec::DataType::DOUBLE:
      doc.set<double>(field_name, yyjson_get_real(val));
      break;
    case zvec::DataType::STRING:
    case zvec::DataType::BINARY:
      doc.set<std::string>(field_name, std::string(yyjson_get_str(val)));
      break;
    case zvec::DataType::VECTOR_FP32: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<float> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((float)yyjson_get_real(elem));
      }
      if (dimension > 0 && vec.size() != dimension) {
        throw InvalidInputException(
            "zvec: field '%s' dimension mismatch: expected %u, got %zu",
            field_name, dimension, vec.size());
      }
      doc.set<std::vector<float>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::VECTOR_FP64: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<double> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back(yyjson_get_real(elem));
      }
      doc.set<std::vector<double>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::VECTOR_INT8: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<int8_t> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((int8_t)yyjson_get_int(elem));
      }
      doc.set<std::vector<int8_t>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::VECTOR_INT16: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<int16_t> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((int16_t)yyjson_get_int(elem));
      }
      doc.set<std::vector<int16_t>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::ARRAY_INT32: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<int32_t> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((int32_t)yyjson_get_int(elem));
      }
      doc.set<std::vector<int32_t>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::ARRAY_INT64: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<int64_t> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((int64_t)yyjson_get_int(elem));
      }
      doc.set<std::vector<int64_t>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::ARRAY_FLOAT: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<float> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back((float)yyjson_get_real(elem));
      }
      doc.set<std::vector<float>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::ARRAY_DOUBLE: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<double> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back(yyjson_get_real(elem));
      }
      doc.set<std::vector<double>>(field_name, std::move(vec));
      break;
    }
    case zvec::DataType::ARRAY_STRING: {
      if (!yyjson_is_arr(val)) {
        throw InvalidInputException("zvec: field '%s' must be a JSON array",
                                    field_name);
      }
      std::vector<std::string> vec;
      size_t idx, max;
      yyjson_val *elem;
      yyjson_arr_foreach(val, idx, max, elem) {
        vec.push_back(std::string(yyjson_get_str(elem)));
      }
      doc.set<std::vector<std::string>>(field_name, std::move(vec));
      break;
    }
    default:
      throw InvalidInputException(
          "zvec: unsupported data type for field '%s' in JSON insert",
          field_name);
  }
}

static void ZvecInsertFunction(DataChunk &args, ExpressionState &state,
                               Vector &result) {
  auto &path_vec = args.data[0];
  auto &pk_vec = args.data[1];
  auto &doc_vec = args.data[2];
  auto count = args.size();

  auto paths = FlatVector::GetData<string_t>(path_vec);
  auto pks = FlatVector::GetData<string_t>(pk_vec);
  auto docs_json = FlatVector::GetData<string_t>(doc_vec);
  auto result_data = FlatVector::GetData<string_t>(result);

  for (idx_t i = 0; i < count; i++) {
    if (FlatVector::IsNull(path_vec, i) || FlatVector::IsNull(pk_vec, i) ||
        FlatVector::IsNull(doc_vec, i)) {
      FlatVector::SetNull(result, i, true);
      continue;
    }

    auto path = paths[i].GetString();
    auto pk = pks[i].GetString();
    auto json_str = docs_json[i].GetString();

    auto &registry = CollectionRegistry::Instance();
    auto collection = registry.GetOrOpen(path);
    auto schema = UnwrapOrThrow(collection->Schema(), "get schema");

    // Parse JSON document
    auto *json_doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!json_doc) {
      throw InvalidInputException("zvec: invalid JSON document for pk '%s'",
                                  pk);
    }
    auto *root = yyjson_doc_get_root(json_doc);
    if (!yyjson_is_obj(root)) {
      yyjson_doc_free(json_doc);
      throw InvalidInputException(
          "zvec: JSON document must be an object for pk '%s'", pk);
    }

    zvec::Doc doc;
    doc.set_pk(pk);

    for (auto &field : schema.fields()) {
      auto *field_val = yyjson_obj_get(root, field->name().c_str());
      if (field_val) {
        SetDocFieldFromJson(doc, field->name(), field->data_type(), field_val,
                            field->dimension());
      }
    }

    yyjson_doc_free(json_doc);

    std::vector<zvec::Doc> docs_to_insert{std::move(doc)};
    auto write_result =
        UnwrapOrThrow(collection->Insert(docs_to_insert), "insert");

    // Check per-doc status
    if (!write_result.empty() && !write_result[0].ok()) {
      throw IOException("zvec: insert failed for pk '%s': %s", pk,
                        write_result[0].message());
    }

    auto msg = "Inserted " + pk;
    result_data[i] = StringVector::AddString(result, msg);
  }
}

void RegisterZvecInsert(DatabaseInstance &db) {
  ScalarFunctionSet func_set("zvec_insert");
  func_set.AddFunction(ScalarFunction(
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LogicalType::VARCHAR, ZvecInsertFunction));
  ExtensionUtil::RegisterFunction(db, func_set);
}

}  // namespace duckdb
