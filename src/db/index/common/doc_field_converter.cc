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

#include "doc_field_converter.h"
#include <string>
#include <vector>

namespace zvec {

namespace {

template <typename T>
Status DenseVectorDataConverter(
    const FieldSchema::Ptr &field,
    const vector_column_params::DenseVectorBuffer &buffer, Doc *doc) {
  const T *data_ptr = reinterpret_cast<const T *>(buffer.data.data());
  size_t data_size = buffer.data.size() / sizeof(T);
  std::vector<T> vector_data(data_ptr, data_ptr + data_size);
  doc->set(field->name(), std::move(vector_data));
  return Status::OK();
}

template <typename IndexType, typename ValueType>
Status SparseVectorDataConverter(
    const FieldSchema::Ptr &field,
    const vector_column_params::SparseVectorBuffer &buffer, Doc *doc) {
  const IndexType *indices_ptr =
      reinterpret_cast<const IndexType *>(buffer.indices.data());
  size_t indices_size = buffer.indices.size() / sizeof(IndexType);
  std::vector<IndexType> indices_vector(indices_ptr,
                                        indices_ptr + indices_size);

  const ValueType *values_ptr =
      reinterpret_cast<const ValueType *>(buffer.values.data());
  size_t values_size = buffer.values.size() / sizeof(ValueType);
  std::vector<ValueType> values_vector(values_ptr, values_ptr + values_size);

  std::pair<std::vector<IndexType>, std::vector<ValueType>> sparse_vector_pair(
      std::move(indices_vector), std::move(values_vector));
  doc->set(field->name(), std::move(sparse_vector_pair));
  return Status::OK();
}

//! Set a scalar Doc field from row `row` of a typed Arrow array.
//! Uses static_cast (type is guaranteed by the caller's switch dispatch).
template <typename ArrowArrayT>
Status SetScalarField(const arrow::Array *array, int64_t row,
                      const std::string &name, Doc *doc) {
  auto *typed_array = static_cast<const ArrowArrayT *>(array);
  if (typed_array->IsNull(row)) {
    return Status::OK();
  }
  if constexpr (std::is_same_v<ArrowArrayT, arrow::StringArray> ||
                std::is_same_v<ArrowArrayT, arrow::BinaryArray>) {
    doc->set(name, std::string(typed_array->GetView(row)));
  } else {
    doc->set(name, typed_array->Value(row));
  }
  return Status::OK();
}

//! Set an array Doc field from row `row` of an Arrow ListArray whose values
//! are of type ArrowArrayT. Null elements inside the list are skipped.
//! Uses Value() uniformly (returns string_view for string/binary types,
//! which emplace_back converts to std::string in place).
template <typename ArrowArrayT, typename T>
Status SetListField(const arrow::Array *array, int64_t row,
                    const std::string &name, Doc *doc) {
  auto *list_array = static_cast<const arrow::ListArray *>(array);
  auto values_slice = list_array->value_slice(row);
  auto *values = static_cast<const ArrowArrayT *>(values_slice.get());
  doc->set(name, ExtractTypedArrayValues<ArrowArrayT, T>(values));
  return Status::OK();
}

}  // namespace

Status ConvertVectorDataBufferToDocField(
    const FieldSchema::Ptr &field,
    const vector_column_params::VectorDataBuffer &buf, Doc *doc) {
  if (std::holds_alternative<vector_column_params::DenseVectorBuffer>(
          buf.vector_buffer)) {
    const auto &dense_buffer =
        std::get<vector_column_params::DenseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::VECTOR_BINARY32:
        return DenseVectorDataConverter<uint32_t>(field, dense_buffer, doc);
      case DataType::VECTOR_BINARY64:
        return DenseVectorDataConverter<uint64_t>(field, dense_buffer, doc);
      case DataType::VECTOR_FP16:
        return DenseVectorDataConverter<float16_t>(field, dense_buffer, doc);
      case DataType::VECTOR_FP32:
        return DenseVectorDataConverter<float>(field, dense_buffer, doc);
      case DataType::VECTOR_FP64:
        return DenseVectorDataConverter<double>(field, dense_buffer, doc);
      case DataType::VECTOR_INT8:
        return DenseVectorDataConverter<int8_t>(field, dense_buffer, doc);
      case DataType::VECTOR_INT16:
        return DenseVectorDataConverter<int16_t>(field, dense_buffer, doc);
      default:
        return Status::InvalidArgument(
            "Unsupported dense vector element type: ", field->data_type());
    }
  } else if (std::holds_alternative<vector_column_params::SparseVectorBuffer>(
                 buf.vector_buffer)) {
    const auto &sparse_buffer =
        std::get<vector_column_params::SparseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::SPARSE_VECTOR_FP16:
        return SparseVectorDataConverter<uint32_t, float16_t>(
            field, sparse_buffer, doc);
      case DataType::SPARSE_VECTOR_FP32:
        return SparseVectorDataConverter<uint32_t, float>(field, sparse_buffer,
                                                          doc);
      default:
        return Status::InvalidArgument(
            "Unsupported sparse vector element type: ", field->data_type());
    }
  }
  return Status::InvalidArgument("Unsupported vector buffer type");
}

Status ConvertArrowRowToDocField(const arrow::Array *array, int64_t row,
                                 const FieldSchema &field, Doc *doc) {
  if (!array || row < 0 || row >= array->length()) {
    return Status::InvalidArgument("Arrow row out of range for field: ",
                                   field.name());
  }
  if (array->IsNull(row)) {
    return Status::OK();  // null value: leave the field unset
  }

  const auto &name = field.name();
  switch (field.data_type()) {
    case DataType::BINARY:
      return SetScalarField<arrow::BinaryArray>(array, row, name, doc);
    case DataType::STRING:
      return SetScalarField<arrow::StringArray>(array, row, name, doc);
    case DataType::BOOL:
      return SetScalarField<arrow::BooleanArray>(array, row, name, doc);
    case DataType::INT32:
      return SetScalarField<arrow::Int32Array>(array, row, name, doc);
    case DataType::INT64:
      return SetScalarField<arrow::Int64Array>(array, row, name, doc);
    case DataType::UINT32:
      return SetScalarField<arrow::UInt32Array>(array, row, name, doc);
    case DataType::UINT64:
      return SetScalarField<arrow::UInt64Array>(array, row, name, doc);
    case DataType::FLOAT:
      return SetScalarField<arrow::FloatArray>(array, row, name, doc);
    case DataType::DOUBLE:
      return SetScalarField<arrow::DoubleArray>(array, row, name, doc);
    case DataType::ARRAY_BINARY:
      return SetListField<arrow::BinaryArray, std::string>(array, row, name,
                                                           doc);
    case DataType::ARRAY_STRING:
      return SetListField<arrow::StringArray, std::string>(array, row, name,
                                                           doc);
    case DataType::ARRAY_BOOL:
      return SetListField<arrow::BooleanArray, bool>(array, row, name, doc);
    case DataType::ARRAY_INT32:
      return SetListField<arrow::Int32Array, int32_t>(array, row, name, doc);
    case DataType::ARRAY_INT64:
      return SetListField<arrow::Int64Array, int64_t>(array, row, name, doc);
    case DataType::ARRAY_UINT32:
      return SetListField<arrow::UInt32Array, uint32_t>(array, row, name, doc);
    case DataType::ARRAY_UINT64:
      return SetListField<arrow::UInt64Array, uint64_t>(array, row, name, doc);
    case DataType::ARRAY_FLOAT:
      return SetListField<arrow::FloatArray, float>(array, row, name, doc);
    case DataType::ARRAY_DOUBLE:
      return SetListField<arrow::DoubleArray, double>(array, row, name, doc);
    default:
      return Status::InvalidArgument("Unsupported data type for field: ", name,
                                     ": ", field.data_type());
  }
}

}  // namespace zvec
