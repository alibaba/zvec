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

//! Set the value at `row` of a typed Arrow array into a Doc field. The row
//! must be valid (non-null); callers dispatch by field type, which guarantees
//! the array's dynamic type.
template <typename ArrowArrayT>
void SetScalarValue(const ArrowArrayT *typed_array, int64_t row,
                    const std::string &name, Doc *doc) {
  if constexpr (std::is_same_v<ArrowArrayT, arrow::StringArray> ||
                std::is_same_v<ArrowArrayT, arrow::BinaryArray>) {
    doc->set(name, std::string(typed_array->GetView(row)));
  } else {
    doc->set(name, typed_array->Value(row));
  }
}

//! Column-level scalar setter: checks null_count() once, so all-valid
//! columns are filled without per-row null checks.
template <typename ArrowArrayT>
Status SetScalarColumn(const arrow::Array *array, const std::string &name,
                       DocPtrList::iterator doc_it) {
  auto *typed_array = static_cast<const ArrowArrayT *>(array);
  const bool no_null = typed_array->null_count() == 0;
  for (int64_t i = 0; i < typed_array->length(); ++i, ++doc_it) {
    if (no_null || !typed_array->IsNull(i)) {
      SetScalarValue(typed_array, i, name, doc_it->get());
    }
  }
  return Status::OK();
}

//! Column-level list setter. null_count() refers to row-level nulls (whole
//! list null for a row); list elements never store nulls in zvec.
template <typename ArrowArrayT, typename T>
Status SetListColumn(const arrow::Array *array, const std::string &name,
                     DocPtrList::iterator doc_it) {
  auto *list_array = static_cast<const arrow::ListArray *>(array);
  const bool no_null = list_array->null_count() == 0;
  for (int64_t i = 0; i < list_array->length(); ++i, ++doc_it) {
    if (no_null || !list_array->IsNull(i)) {
      auto values_slice = list_array->value_slice(i);
      auto *values = static_cast<const ArrowArrayT *>(values_slice.get());
      doc_it->get()->set(name, ExtractTypedArrayValues<ArrowArrayT, T>(values));
    }
  }
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

Status ConvertArrowColumnToDocFields(const arrow::Array *array,
                                     const FieldSchema &field,
                                     DocPtrList::iterator doc_it) {
  if (!array) {
    return Status::InvalidArgument("Arrow array is null for field: ",
                                   field.name());
  }

  const auto &name = field.name();
  switch (field.data_type()) {
    case DataType::BINARY:
      return SetScalarColumn<arrow::BinaryArray>(array, name, doc_it);
    case DataType::STRING:
      return SetScalarColumn<arrow::StringArray>(array, name, doc_it);
    case DataType::BOOL:
      return SetScalarColumn<arrow::BooleanArray>(array, name, doc_it);
    case DataType::INT32:
      return SetScalarColumn<arrow::Int32Array>(array, name, doc_it);
    case DataType::INT64:
      return SetScalarColumn<arrow::Int64Array>(array, name, doc_it);
    case DataType::UINT32:
      return SetScalarColumn<arrow::UInt32Array>(array, name, doc_it);
    case DataType::UINT64:
      return SetScalarColumn<arrow::UInt64Array>(array, name, doc_it);
    case DataType::FLOAT:
      return SetScalarColumn<arrow::FloatArray>(array, name, doc_it);
    case DataType::DOUBLE:
      return SetScalarColumn<arrow::DoubleArray>(array, name, doc_it);
    case DataType::ARRAY_BINARY:
      return SetListColumn<arrow::BinaryArray, std::string>(array, name,
                                                            doc_it);
    case DataType::ARRAY_STRING:
      return SetListColumn<arrow::StringArray, std::string>(array, name,
                                                            doc_it);
    case DataType::ARRAY_BOOL:
      return SetListColumn<arrow::BooleanArray, bool>(array, name, doc_it);
    case DataType::ARRAY_INT32:
      return SetListColumn<arrow::Int32Array, int32_t>(array, name, doc_it);
    case DataType::ARRAY_INT64:
      return SetListColumn<arrow::Int64Array, int64_t>(array, name, doc_it);
    case DataType::ARRAY_UINT32:
      return SetListColumn<arrow::UInt32Array, uint32_t>(array, name, doc_it);
    case DataType::ARRAY_UINT64:
      return SetListColumn<arrow::UInt64Array, uint64_t>(array, name, doc_it);
    case DataType::ARRAY_FLOAT:
      return SetListColumn<arrow::FloatArray, float>(array, name, doc_it);
    case DataType::ARRAY_DOUBLE:
      return SetListColumn<arrow::DoubleArray, double>(array, name, doc_it);
    default:
      return Status::InvalidArgument("Unsupported data type for field: ", name,
                                     ": ", field.data_type());
  }
}

}  // namespace zvec
