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

#include "type_conversion.hpp"

namespace duckdb {

LogicalType ZvecTypeToDuckDB(zvec::DataType type) {
  switch (type) {
    case zvec::DataType::BOOL:
      return LogicalType::BOOLEAN;
    case zvec::DataType::INT32:
      return LogicalType::INTEGER;
    case zvec::DataType::INT64:
      return LogicalType::BIGINT;
    case zvec::DataType::UINT32:
      return LogicalType::UINTEGER;
    case zvec::DataType::UINT64:
      return LogicalType::UBIGINT;
    case zvec::DataType::FLOAT:
      return LogicalType::FLOAT;
    case zvec::DataType::DOUBLE:
      return LogicalType::DOUBLE;
    case zvec::DataType::STRING:
    case zvec::DataType::BINARY:
      return LogicalType::VARCHAR;
    case zvec::DataType::VECTOR_FP32:
      return LogicalType::LIST(LogicalType::FLOAT);
    case zvec::DataType::VECTOR_FP64:
      return LogicalType::LIST(LogicalType::DOUBLE);
    case zvec::DataType::VECTOR_FP16:
      return LogicalType::LIST(LogicalType::FLOAT);  // upcast fp16 to float
    case zvec::DataType::VECTOR_INT8:
      return LogicalType::LIST(LogicalType::TINYINT);
    case zvec::DataType::VECTOR_INT16:
      return LogicalType::LIST(LogicalType::SMALLINT);
    case zvec::DataType::VECTOR_BINARY32:
      return LogicalType::LIST(LogicalType::UINTEGER);
    case zvec::DataType::VECTOR_BINARY64:
      return LogicalType::LIST(LogicalType::UBIGINT);
    case zvec::DataType::SPARSE_VECTOR_FP32:
    case zvec::DataType::SPARSE_VECTOR_FP16:
      return LogicalType::VARCHAR;  // serialize sparse vectors as JSON string
    case zvec::DataType::ARRAY_INT32:
      return LogicalType::LIST(LogicalType::INTEGER);
    case zvec::DataType::ARRAY_INT64:
      return LogicalType::LIST(LogicalType::BIGINT);
    case zvec::DataType::ARRAY_UINT32:
      return LogicalType::LIST(LogicalType::UINTEGER);
    case zvec::DataType::ARRAY_UINT64:
      return LogicalType::LIST(LogicalType::UBIGINT);
    case zvec::DataType::ARRAY_FLOAT:
      return LogicalType::LIST(LogicalType::FLOAT);
    case zvec::DataType::ARRAY_DOUBLE:
      return LogicalType::LIST(LogicalType::DOUBLE);
    case zvec::DataType::ARRAY_STRING:
    case zvec::DataType::ARRAY_BINARY:
      return LogicalType::LIST(LogicalType::VARCHAR);
    case zvec::DataType::ARRAY_BOOL:
      return LogicalType::LIST(LogicalType::BOOLEAN);
    default:
      return LogicalType::VARCHAR;
  }
}

// Helper to set a DuckDB list from a std::vector
template <typename SRC_T, typename DST_T>
static void SetListFromVector(Vector &result, idx_t row_idx,
                              const std::vector<SRC_T> &vec) {
  auto list_size = vec.size();
  auto list_entry = ListVector::GetData(result);
  auto current_size = ListVector::GetListSize(result);
  list_entry[row_idx].offset = current_size;
  list_entry[row_idx].length = list_size;
  ListVector::Reserve(result, current_size + list_size);
  ListVector::SetListSize(result, current_size + list_size);
  auto &child = ListVector::GetEntry(result);
  auto child_data = FlatVector::GetData<DST_T>(child);
  for (idx_t i = 0; i < list_size; i++) {
    child_data[current_size + i] = static_cast<DST_T>(vec[i]);
  }
}

void DocFieldToDuckDB(const zvec::Doc &doc, const std::string &field_name,
                      zvec::DataType type, Vector &result, idx_t row_idx) {
  switch (type) {
    case zvec::DataType::BOOL: {
      auto val = doc.get<bool>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<bool>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::INT32: {
      auto val = doc.get<int32_t>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<int32_t>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::INT64: {
      auto val = doc.get<int64_t>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<int64_t>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::UINT32: {
      auto val = doc.get<uint32_t>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<uint32_t>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::UINT64: {
      auto val = doc.get<uint64_t>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<uint64_t>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::FLOAT: {
      auto val = doc.get<float>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<float>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::DOUBLE: {
      auto val = doc.get<double>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<double>(result)[row_idx] = val.value();
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::STRING:
    case zvec::DataType::BINARY: {
      auto val = doc.get<std::string>(field_name);
      if (val.has_value()) {
        FlatVector::GetData<string_t>(result)[row_idx] =
            StringVector::AddString(result, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_FP32: {
      auto val = doc.get<std::vector<float>>(field_name);
      if (val.has_value()) {
        SetListFromVector<float, float>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_FP64: {
      auto val = doc.get<std::vector<double>>(field_name);
      if (val.has_value()) {
        SetListFromVector<double, double>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_FP16: {
      auto val = doc.get<std::vector<zvec::float16_t>>(field_name);
      if (val.has_value()) {
        // Upcast fp16 to float for DuckDB
        auto &v = val.value();
        std::vector<float> floats(v.size());
        for (size_t i = 0; i < v.size(); i++) {
          floats[i] = static_cast<float>(v[i]);
        }
        SetListFromVector<float, float>(result, row_idx, floats);
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_INT8: {
      auto val = doc.get<std::vector<int8_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<int8_t, int8_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_INT16: {
      auto val = doc.get<std::vector<int16_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<int16_t, int16_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_BINARY32: {
      auto val = doc.get<std::vector<uint32_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<uint32_t, uint32_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::VECTOR_BINARY64: {
      auto val = doc.get<std::vector<uint64_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<uint64_t, uint64_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_INT32: {
      auto val = doc.get<std::vector<int32_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<int32_t, int32_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_INT64: {
      auto val = doc.get<std::vector<int64_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<int64_t, int64_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_UINT32: {
      auto val = doc.get<std::vector<uint32_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<uint32_t, uint32_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_UINT64: {
      auto val = doc.get<std::vector<uint64_t>>(field_name);
      if (val.has_value()) {
        SetListFromVector<uint64_t, uint64_t>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_FLOAT: {
      auto val = doc.get<std::vector<float>>(field_name);
      if (val.has_value()) {
        SetListFromVector<float, float>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_DOUBLE: {
      auto val = doc.get<std::vector<double>>(field_name);
      if (val.has_value()) {
        SetListFromVector<double, double>(result, row_idx, val.value());
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_STRING:
    case zvec::DataType::ARRAY_BINARY: {
      auto val = doc.get<std::vector<std::string>>(field_name);
      if (val.has_value()) {
        auto &vec = val.value();
        auto list_entry = ListVector::GetData(result);
        auto current_size = ListVector::GetListSize(result);
        list_entry[row_idx].offset = current_size;
        list_entry[row_idx].length = vec.size();
        ListVector::Reserve(result, current_size + vec.size());
        ListVector::SetListSize(result, current_size + vec.size());
        auto &child = ListVector::GetEntry(result);
        for (idx_t i = 0; i < vec.size(); i++) {
          FlatVector::GetData<string_t>(child)[current_size + i] =
              StringVector::AddString(child, vec[i]);
        }
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    case zvec::DataType::ARRAY_BOOL: {
      auto val = doc.get<std::vector<bool>>(field_name);
      if (val.has_value()) {
        auto &vec = val.value();
        auto list_entry = ListVector::GetData(result);
        auto current_size = ListVector::GetListSize(result);
        list_entry[row_idx].offset = current_size;
        list_entry[row_idx].length = vec.size();
        ListVector::Reserve(result, current_size + vec.size());
        ListVector::SetListSize(result, current_size + vec.size());
        auto &child = ListVector::GetEntry(result);
        auto child_data = FlatVector::GetData<bool>(child);
        for (idx_t i = 0; i < vec.size(); i++) {
          child_data[current_size + i] = vec[i];
        }
      } else {
        FlatVector::SetNull(result, row_idx, true);
      }
      break;
    }
    default:
      FlatVector::SetNull(result, row_idx, true);
      break;
  }
}

std::string FloatArrayToQueryVector(const vector<float> &floats) {
  std::string result;
  result.assign(reinterpret_cast<const char *>(floats.data()),
                floats.size() * sizeof(float));
  return result;
}

}  // namespace duckdb
