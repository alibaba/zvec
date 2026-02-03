
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

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <zvec/db/doc.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"
#include "db/common/file_helper.h"
#include "db/index/common/meta.h"
#include "chunked_file_writer.h"


namespace zvec {

inline FileFormat InferFileFormat(const std::string &file_path) {
  std::string ext = std::filesystem::path(file_path).extension();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == ".parquet") {
    return FileFormat::PARQUET;
  } else if (ext == ".feather" || ext == ".ipc" || ext == ".arrow") {
    return FileFormat::IPC;
  } else {
    return FileFormat::UNKNOWN;
  }
}

inline arrow::Status ConvertFieldSchemaToArrowField(
    const FieldSchema *field, std::shared_ptr<arrow::Field> *out) {
  switch (field->data_type()) {
    case DataType::BINARY:
      *out = arrow::field(field->name(), arrow::binary(), field->nullable());
      break;
    case DataType::STRING:
      *out = arrow::field(field->name(), arrow::utf8(), field->nullable());
      break;
    case DataType::BOOL:
      *out = arrow::field(field->name(), arrow::boolean(), field->nullable());
      break;
    case DataType::INT32:
      *out = arrow::field(field->name(), arrow::int32(), field->nullable());
      break;
    case DataType::INT64:
      *out = arrow::field(field->name(), arrow::int64(), field->nullable());
      break;
    case DataType::UINT32:
      *out = arrow::field(field->name(), arrow::uint32(), field->nullable());
      break;
    case DataType::UINT64:
      *out = arrow::field(field->name(), arrow::uint64(), field->nullable());
      break;
    case DataType::FLOAT:
      *out = arrow::field(field->name(), arrow::float32(), field->nullable());
      break;
    case DataType::DOUBLE:
      *out = arrow::field(field->name(), arrow::float64(), field->nullable());
      break;
    case DataType::ARRAY_BINARY:
      *out = arrow::field(field->name(), arrow::list(arrow::binary()),
                          field->nullable());
      break;
    case DataType::ARRAY_STRING:
      *out = arrow::field(field->name(), arrow::list(arrow::utf8()),
                          field->nullable());
      break;
    case DataType::ARRAY_BOOL:
      *out = arrow::field(field->name(), arrow::list(arrow::boolean()),
                          field->nullable());
      break;
    case DataType::ARRAY_INT32:
      *out = arrow::field(field->name(), arrow::list(arrow::int32()),
                          field->nullable());
      break;
    case DataType::ARRAY_INT64:
      *out = arrow::field(field->name(), arrow::list(arrow::int64()),
                          field->nullable());
      break;
    case DataType::ARRAY_UINT32:
      *out = arrow::field(field->name(), arrow::list(arrow::uint32()),
                          field->nullable());
      break;
    case DataType::ARRAY_UINT64:
      *out = arrow::field(field->name(), arrow::list(arrow::uint64()),
                          field->nullable());
      break;
    case DataType::ARRAY_FLOAT:
      *out = arrow::field(field->name(), arrow::list(arrow::float32()),
                          field->nullable());
      break;
    case DataType::ARRAY_DOUBLE:
      *out = arrow::field(field->name(), arrow::list(arrow::float64()),
                          field->nullable());
      break;
    default:
      return arrow::Status::Invalid(
          "Unsupported data type ",
          DataTypeCodeBook::AsString(field->data_type()));
  }

  return arrow::Status::OK();
}

inline arrow::Status ConvertCollectionSchemaToArrowFields(
    const CollectionSchema::Ptr &schema, arrow::FieldVector *out) {
  arrow::FieldVector fields;
  fields.push_back(arrow::field(GLOBAL_DOC_ID, arrow::uint64(), false));
  fields.push_back(arrow::field(USER_ID, arrow::utf8(), false));
  for (auto &field : schema->forward_fields()) {
    std::shared_ptr<arrow::Field> arrow_field;
    ARROW_RETURN_NOT_OK(
        ConvertFieldSchemaToArrowField(field.get(), &arrow_field));
    fields.push_back(arrow_field);
  }
  *out = std::move(fields);
  return arrow::Status::OK();
}

template <typename BuilderType, typename ScalarType>
inline arrow::Status ConvertScalarVectorToArray(
    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars,
    std::shared_ptr<arrow::Array> *out) {
  std::shared_ptr<arrow::Array> arr;
  BuilderType builder;
  for (const auto &scalar : ordered_scalars) {
    if (scalar == nullptr || scalar->is_valid == false) {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
      continue;
    }
    auto status =
        builder.Append(dynamic_cast<const ScalarType &>(*scalar).value);
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&arr));
  *out = arr;
  return arrow::Status::OK();
}

template <>
inline arrow::Status
ConvertScalarVectorToArray<arrow::StringBuilder, arrow::StringScalar>(
    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars,
    std::shared_ptr<arrow::Array> *out) {
  std::shared_ptr<arrow::Array> arr;
  arrow::StringBuilder builder;
  for (const auto &scalar : ordered_scalars) {
    if (scalar == nullptr || scalar->is_valid == false) {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
      continue;
    }
    const auto &str = dynamic_cast<const arrow::StringScalar &>(*scalar).value;
    ARROW_RETURN_NOT_OK(
        builder.Append(str->data(), static_cast<int>(str->size())));
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&arr));
  *out = arr;
  return arrow::Status::OK();
}

template <>
inline arrow::Status
ConvertScalarVectorToArray<arrow::BinaryBuilder, arrow::BinaryScalar>(
    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars,
    std::shared_ptr<arrow::Array> *out) {
  std::shared_ptr<arrow::Array> arr;
  arrow::BinaryBuilder builder;
  for (const auto &scalar : ordered_scalars) {
    if (scalar == nullptr || scalar->is_valid == false) {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
      continue;
    }
    const auto &binary_scalar =
        dynamic_cast<const arrow::BinaryScalar &>(*scalar);
    if (binary_scalar.value) {
      ARROW_RETURN_NOT_OK(
          builder.Append(binary_scalar.value->data(),
                         static_cast<int>(binary_scalar.value->size())));
    } else {
      ARROW_RETURN_NOT_OK(builder.AppendEmptyValue());
    }
  }
  ARROW_RETURN_NOT_OK(builder.Finish(&arr));
  *out = arr;
  return arrow::Status::OK();
}

inline arrow::Status ConvertScalarVectorToArrayByType(
    std::vector<std::shared_ptr<arrow::Scalar>> ordered_scalars,
    std::shared_ptr<arrow::Array> *out) {
  auto type = ordered_scalars.empty() ? nullptr : ordered_scalars[0]->type;
  if (type == nullptr)
    return arrow::Status::Invalid("Cannot convert empty vector to array");
  arrow::Status status;
  switch (type->id()) {
    case arrow::Type::BINARY:
      status =
          ConvertScalarVectorToArray<arrow::BinaryBuilder, arrow::BinaryScalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::BOOL:
      status = ConvertScalarVectorToArray<arrow::BooleanBuilder,
                                          arrow::BooleanScalar>(ordered_scalars,
                                                                out);
      break;
    case arrow::Type::INT32:
      status =
          ConvertScalarVectorToArray<arrow::Int32Builder, arrow::Int32Scalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::UINT32:
      status =
          ConvertScalarVectorToArray<arrow::UInt32Builder, arrow::UInt32Scalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::INT64:
      status =
          ConvertScalarVectorToArray<arrow::Int64Builder, arrow::Int64Scalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::UINT64:
      status =
          ConvertScalarVectorToArray<arrow::UInt64Builder, arrow::UInt64Scalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::FLOAT:
      status =
          ConvertScalarVectorToArray<arrow::FloatBuilder, arrow::FloatScalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::DOUBLE:
      status =
          ConvertScalarVectorToArray<arrow::DoubleBuilder, arrow::DoubleScalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::STRING:
      status =
          ConvertScalarVectorToArray<arrow::StringBuilder, arrow::StringScalar>(
              ordered_scalars, out);
      break;
    case arrow::Type::LIST: {
      if (ordered_scalars.empty()) {
        return arrow::Status::Invalid(
            "Cannot convert empty vector to list array");
      }

      auto list_type = std::dynamic_pointer_cast<arrow::ListType>(type);
      if (!list_type) {
        return arrow::Status::TypeError("Expected ListType for LIST scalar");
      }

      std::unique_ptr<arrow::ArrayBuilder> value_builder;
      ARROW_RETURN_NOT_OK(arrow::MakeBuilder(arrow::default_memory_pool(),
                                             list_type->value_type(),
                                             &value_builder));

      arrow::ListBuilder builder(arrow::default_memory_pool(),
                                 std::move(value_builder), list_type);

      for (const auto &scalar : ordered_scalars) {
        if (scalar == nullptr || scalar->is_valid == false) {
          ARROW_RETURN_NOT_OK(builder.AppendNull());
          continue;
        }

        auto list_scalar = std::dynamic_pointer_cast<arrow::ListScalar>(scalar);
        if (!list_scalar) {
          return arrow::Status::TypeError("Expected ListScalar for LIST type");
        }

        ARROW_RETURN_NOT_OK(builder.Append());
        auto value_builder_ptr = builder.value_builder();
        ARROW_RETURN_NOT_OK(value_builder_ptr->AppendArraySlice(
            *list_scalar->value->data(), 0, list_scalar->value->length()));
      }

      std::shared_ptr<arrow::Array> arr;
      ARROW_RETURN_NOT_OK(builder.Finish(&arr));
      *out = arr;
      return arrow::Status::OK();
    }
    default:
      // TODO other type
      return arrow::Status::NotImplemented("Unsupported type");
  }

  return status;
}

template <typename ArrowBuilderType, typename ValueType>
inline arrow::Status AppendValue(ArrowBuilderType *builder, const Doc &doc,
                                 std::shared_ptr<arrow::Field> field) {
  auto value = doc.get<ValueType>(field->name());
  if (!value.has_value()) {
    return builder->AppendNull();
  }
  return builder->Append(value.value());
}

inline arrow::Status AppendFieldValueToBuilder(
    const Doc &doc, const std::shared_ptr<arrow::Field> &field,
    arrow::ArrayBuilder *builder) {
  auto type = field->type()->id();
  switch (type) {
    case arrow::Type::STRING: {
      auto string_builder = dynamic_cast<arrow::StringBuilder *>(builder);
      return AppendValue<arrow::StringBuilder, std::string>(string_builder, doc,
                                                            field);
    }
    case arrow::Type::INT32: {
      auto int32_builder = dynamic_cast<arrow::Int32Builder *>(builder);
      return AppendValue<arrow::Int32Builder, int32_t>(int32_builder, doc,
                                                       field);
    }
    case arrow::Type::INT64: {
      auto int64_builder = dynamic_cast<arrow::Int64Builder *>(builder);
      return AppendValue<arrow::Int64Builder, int64_t>(int64_builder, doc,
                                                       field);
    }
    case arrow::Type::UINT32: {
      auto uint32_builder = dynamic_cast<arrow::UInt32Builder *>(builder);
      return AppendValue<arrow::UInt32Builder, uint32_t>(uint32_builder, doc,
                                                         field);
    }
    case arrow::Type::UINT64: {
      auto uint64_builder = dynamic_cast<arrow::UInt64Builder *>(builder);
      return AppendValue<arrow::UInt64Builder, uint64_t>(uint64_builder, doc,
                                                         field);
    }
    case arrow::Type::DOUBLE: {
      auto double_builder = dynamic_cast<arrow::DoubleBuilder *>(builder);
      return AppendValue<arrow::DoubleBuilder, double>(double_builder, doc,
                                                       field);
    }
    case arrow::Type::FLOAT: {
      auto float_builder = dynamic_cast<arrow::FloatBuilder *>(builder);
      return AppendValue<arrow::FloatBuilder, float>(float_builder, doc, field);
    }
    case arrow::Type::BOOL: {
      auto bool_builder = dynamic_cast<arrow::BooleanBuilder *>(builder);
      return AppendValue<arrow::BooleanBuilder, bool>(bool_builder, doc, field);
    }
    case arrow::Type::BINARY: {
      auto binary_builder = dynamic_cast<arrow::BinaryBuilder *>(builder);
      return AppendValue<arrow::BinaryBuilder, std::string>(binary_builder, doc,
                                                            field);
    }
    case arrow::Type::LIST: {
      auto list_builder = dynamic_cast<arrow::ListBuilder *>(builder);
      auto list_type =
          std::dynamic_pointer_cast<arrow::ListType>(field->type());

      if (!list_type) {
        return arrow::Status::TypeError("Field type is not ListType");
      }

      auto value_type = list_type->value_type()->id();

      switch (value_type) {
        case arrow::Type::BINARY: {
          auto value = doc.get<std::vector<std::string>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto binary_builder = dynamic_cast<arrow::BinaryBuilder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(binary_builder->Append(item));
          }
          break;
        }

        case arrow::Type::BOOL: {
          auto value = doc.get<std::vector<bool>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto bool_builder = dynamic_cast<arrow::BooleanBuilder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(bool_builder->Append(item));
          }
          break;
        }

        case arrow::Type::INT32: {
          auto value = doc.get<std::vector<int32_t>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto int32_builder = dynamic_cast<arrow::Int32Builder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(int32_builder->Append(item));
          }
          break;
        }

        case arrow::Type::INT64: {
          auto value = doc.get<std::vector<int64_t>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto int64_builder = dynamic_cast<arrow::Int64Builder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(int64_builder->Append(item));
          }
          break;
        }

        case arrow::Type::UINT32: {
          auto value = doc.get<std::vector<uint32_t>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto uint32_builder = dynamic_cast<arrow::UInt32Builder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(uint32_builder->Append(item));
          }
          break;
        }

        case arrow::Type::UINT64: {
          auto value = doc.get<std::vector<uint64_t>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto uint64_builder = dynamic_cast<arrow::UInt64Builder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(uint64_builder->Append(item));
          }
          break;
        }

        case arrow::Type::FLOAT: {
          auto value = doc.get<std::vector<float>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto float_builder = dynamic_cast<arrow::FloatBuilder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(float_builder->Append(item));
          }
          break;
        }

        case arrow::Type::DOUBLE: {
          auto value = doc.get<std::vector<double>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto double_builder = dynamic_cast<arrow::DoubleBuilder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(double_builder->Append(item));
          }
          break;
        }

        case arrow::Type::STRING: {
          auto value = doc.get<std::vector<std::string>>(field->name());
          if (!value.has_value()) {
            return list_builder->AppendNull();
          }

          const auto &list_value = value.value();
          auto string_builder = dynamic_cast<arrow::StringBuilder *>(
              list_builder->value_builder());

          ARROW_RETURN_NOT_OK(list_builder->Append());
          for (const auto &item : list_value) {
            ARROW_RETURN_NOT_OK(string_builder->Append(item));
          }
          break;
        }

        default:
          return arrow::Status::NotImplemented(
              "unsupported list element type: ", value_type);
      }

      return arrow::Status::OK();
    }
    default:
      return arrow::Status::NotImplemented("unsupported type: ", type,
                                           ", field: ", field->name());
  }
}

template <typename ArrowArrayType, typename BuilderType>
inline arrow::Status BuildArrayFromIndices(
    const std::shared_ptr<arrow::ChunkedArray> &chunked_array,
    const std::vector<std::pair<int64_t, int64_t>> &indices_in_chunked_array,
    std::shared_ptr<arrow::Array> *out_array) {
  BuilderType builder;
  ARROW_RETURN_NOT_OK(
      builder.Reserve(static_cast<int64_t>(indices_in_chunked_array.size())));

  int64_t last_chunk_index = -1;
  const ArrowArrayType *cached_chunk{nullptr};

  bool no_null = chunked_array->null_count() == 0;
  for (const auto &pair : indices_in_chunked_array) {
    if (pair.first != last_chunk_index) {
      const auto &chunk = chunked_array->chunk(pair.first);
      cached_chunk = static_cast<const ArrowArrayType *>(chunk.get());
      last_chunk_index = pair.first;
    }

    if (no_null || !cached_chunk->IsNull(pair.second)) {
      ARROW_RETURN_NOT_OK(builder.Append(cached_chunk->Value(pair.second)));
    } else {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
  }

  return builder.Finish(out_array);
}

template <>
inline arrow::Status
BuildArrayFromIndices<arrow::StringArray, arrow::StringBuilder>(
    const std::shared_ptr<arrow::ChunkedArray> &chunked_array,
    const std::vector<std::pair<int64_t, int64_t>> &indices_in_chunked_array,
    std::shared_ptr<arrow::Array> *out_array) {
  arrow::StringBuilder builder;
  ARROW_RETURN_NOT_OK(
      builder.Reserve(static_cast<int64_t>(indices_in_chunked_array.size())));

  bool no_null = chunked_array->null_count() == 0;
  const arrow::StringArray *cached_chunk{nullptr};

  int64_t last_chunk_index = -1;
  int64_t data_size = 0;
  for (const auto &pair : indices_in_chunked_array) {
    if (pair.first != last_chunk_index) {
      const auto &chunk = chunked_array->chunk(pair.first);
      cached_chunk = static_cast<const arrow::StringArray *>(chunk.get());
      last_chunk_index = pair.first;
    }

    if (no_null || !cached_chunk->IsNull(pair.second)) {
      data_size += cached_chunk->Value(pair.second).size();
    }
  }
  ARROW_RETURN_NOT_OK(builder.ReserveData(data_size));


  last_chunk_index = -1;
  for (const auto &pair : indices_in_chunked_array) {
    if (pair.first != last_chunk_index) {
      const auto &chunk = chunked_array->chunk(pair.first);
      cached_chunk = static_cast<const arrow::StringArray *>(chunk.get());
      last_chunk_index = pair.first;
    }

    if (no_null || !cached_chunk->IsNull(pair.second)) {
      ARROW_RETURN_NOT_OK(builder.Append(cached_chunk->Value(pair.second)));
    } else {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
  }

  return builder.Finish(out_array);
}

inline arrow::Status BuildListArrayFromIndices(
    const std::shared_ptr<arrow::ChunkedArray> &chunked_array,
    const std::vector<std::pair<int64_t, int64_t>> &indices_in_chunked_array,
    const std::shared_ptr<arrow::ListType> &list_type,
    std::shared_ptr<arrow::Array> *out_array) {
  std::unique_ptr<arrow::ArrayBuilder> value_builder;
  ARROW_RETURN_NOT_OK(arrow::MakeBuilder(
      arrow::default_memory_pool(), list_type->value_type(), &value_builder));

  arrow::ListBuilder builder(arrow::default_memory_pool(),
                             std::move(value_builder), list_type);
  ARROW_RETURN_NOT_OK(
      builder.Reserve(static_cast<int64_t>(indices_in_chunked_array.size())));

  int64_t last_chunk_index = -1;
  const arrow::ListArray *cached_chunk{nullptr};

  for (const auto &pair : indices_in_chunked_array) {
    if (pair.first != last_chunk_index) {
      const auto &chunk = chunked_array->chunk(pair.first);
      cached_chunk = std::static_pointer_cast<arrow::ListArray>(chunk).get();
      last_chunk_index = pair.first;
    }

    if (cached_chunk->IsValid(pair.second)) {
      auto offset = cached_chunk->value_offset(pair.second);
      auto length = cached_chunk->value_length(pair.second);

      ARROW_RETURN_NOT_OK(builder.Append());
      auto value_builder_ptr = builder.value_builder();
      auto values = cached_chunk->values();
      ARROW_RETURN_NOT_OK(
          value_builder_ptr->AppendArraySlice(*values->data(), offset, length));
    } else {
      ARROW_RETURN_NOT_OK(builder.AppendNull());
    }
  }

  return builder.Finish(out_array);
}

inline arrow::Status BuildArrayFromIndicesWithType(
    const std::shared_ptr<arrow::ChunkedArray> &chunked_array,
    const std::vector<std::pair<int64_t, int64_t>> &indices_in_table,
    std::shared_ptr<arrow::Array> *out_array) {
  auto col_data_type = chunked_array->type();
  switch (col_data_type->id()) {
    case arrow::Type::STRING:
      return BuildArrayFromIndices<arrow::StringArray, arrow::StringBuilder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::INT32:
      return BuildArrayFromIndices<arrow::Int32Array, arrow::Int32Builder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::INT64:
      return BuildArrayFromIndices<arrow::Int64Array, arrow::Int64Builder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::UINT32:
      return BuildArrayFromIndices<arrow::UInt32Array, arrow::UInt32Builder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::UINT64:
      return BuildArrayFromIndices<arrow::UInt64Array, arrow::UInt64Builder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::DOUBLE:
      return BuildArrayFromIndices<arrow::DoubleArray, arrow::DoubleBuilder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::FLOAT:
      return BuildArrayFromIndices<arrow::FloatArray, arrow::FloatBuilder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::BOOL:
      return BuildArrayFromIndices<arrow::BooleanArray, arrow::BooleanBuilder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::BINARY:
      return BuildArrayFromIndices<arrow::BinaryArray, arrow::BinaryBuilder>(
          chunked_array, indices_in_table, out_array);
    case arrow::Type::LIST: {
      auto list_type =
          std::dynamic_pointer_cast<arrow::ListType>(col_data_type);
      return BuildListArrayFromIndices(chunked_array, indices_in_table,
                                       list_type, out_array);
    }
    default:
      return arrow::Status::NotImplemented("Unsupported element type: ",
                                           col_data_type->name().c_str());
  }
}

inline arrow::Status CreateRandomAccessFileByUri(
    const std::string &uri,
    std::shared_ptr<arrow::io::RandomAccessFile> *out_file,
    std::string *out_file_path) {
  std::string path_from_uri, file_path;
  std::shared_ptr<arrow::fs::FileSystem> fs;
  auto maybe_fs = arrow::fs::FileSystemFromUri(uri, &path_from_uri);

  if (maybe_fs.ok()) {
    fs = maybe_fs.ValueOrDie();
    *out_file_path = path_from_uri;
  } else {
    arrow::fs::LocalFileSystemOptions options;
    options.use_mmap = true;
    fs = std::make_shared<arrow::fs::LocalFileSystem>(options);
    if (uri.length() >= 2 && uri.substr(0, 2) == "./") {
      *out_file_path = uri.substr(2);
    } else {
      *out_file_path = uri;
    }
  }

  auto result = fs->OpenInputFile(*out_file_path);
  if (!result.ok()) {
    return result.status();
  }
  *out_file = result.ValueOrDie();
  return arrow::Status::OK();
}

inline std::vector<std::shared_ptr<arrow::Field>> SelectFields(
    const std::shared_ptr<arrow::Schema> &schema,
    const std::vector<std::string> &column_names) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (const auto &name : column_names) {
    if (name == LOCAL_ROW_ID) {
      fields.push_back(arrow::field(LOCAL_ROW_ID, arrow::uint64()));
    } else {
      fields.push_back(schema->field(schema->GetFieldIndex(name)));
    }
  }
  return fields;
}

inline arrow::Result<std::shared_ptr<arrow::Array>> SelectArrayByIndices(
    const std::shared_ptr<arrow::Array> &arr,
    const std::vector<int32_t> &indices) {
  arrow::Int32Builder builder;
  ARROW_RETURN_NOT_OK(builder.AppendValues(indices));
  std::shared_ptr<arrow::Array> indices_array;
  ARROW_RETURN_NOT_OK(builder.Finish(&indices_array));

  return arrow::compute::Take(*arr, *indices_array);
}

inline arrow::Result<std::shared_ptr<arrow::dataset::Dataset>>
ReadBlocksAsDataset(const std::vector<BlockMeta> &scalar_blocks,
                    const std::string &base_path, uint32_t collection_id,
                    bool use_parquet) {
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  auto pool = arrow::default_memory_pool();

  if (scalar_blocks.empty()) {
    return arrow::Status::Invalid("No block metadata provided");
  }

  using ColData = std::pair<std::shared_ptr<arrow::Field>,
                            std::shared_ptr<arrow::ChunkedArray>>;
  std::map<uint64_t, std::map<std::string, ColData>> segments;
  std::map<uint64_t, uint32_t> segment_doc_count;
  std::set<uint64_t> ordered_min_ids;

  for (const auto &block : scalar_blocks) {
    if (block.doc_count_ == 0 || block.columns_.empty()) continue;

    uint64_t start_row = block.min_doc_id_;
    uint32_t expected_count = block.doc_count_;

    std::string filepath = FileHelper::MakeForwardBlockPath(
        base_path, collection_id, block.id_, use_parquet);

    try {
      auto file_info = fs->GetFileInfo(filepath).ValueOrDie();
      auto file = fs->OpenInputFile(file_info.path()).ValueOrDie();

      std::shared_ptr<arrow::Table> table;

      if (use_parquet) {
        std::unique_ptr<parquet::arrow::FileReader> reader;
        reader = parquet::arrow::OpenFile(file, pool).ValueOrDie();
        ARROW_RETURN_NOT_OK(reader->ReadTable(&table));
      } else {
        auto reader =
            arrow::ipc::RecordBatchFileReader::Open(file).ValueOrDie();

        std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
        for (int i = 0; i < reader->num_record_batches(); ++i) {
          auto batch = reader->ReadRecordBatch(i).ValueOrDie();
          batches.push_back(batch);
        }

        table = arrow::Table::FromRecordBatches(batches).ValueOrDie();
      }

      if (segments.find(start_row) == segments.end()) {
        segments[start_row] = {};
        segment_doc_count[start_row] = expected_count;
        ordered_min_ids.insert(start_row);
      }

      for (int i = 0; i < table->num_columns(); ++i) {
        const auto &field = table->schema()->field(i);
        auto original_chunked_array = table->column(i);

        segments[start_row][field->name()] = {field, original_chunked_array};
      }
    } catch (const std::exception &e) {
      return arrow::Status::IOError("Failed to read block ",
                                    std::to_string(block.id_), ": ", e.what());
    }
  }

  if (segments.empty()) {
    return arrow::Status::Invalid("No valid data blocks found");
  }

  std::vector<uint64_t> sorted_starts(ordered_min_ids.begin(),
                                      ordered_min_ids.end());
  std::sort(sorted_starts.begin(), sorted_starts.end());

  std::vector<std::shared_ptr<arrow::Table>> segment_tables;
  for (uint64_t start_row : sorted_starts) {
    auto &col_map = segments[start_row];
    uint32_t count = segment_doc_count[start_row];

    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;

    for (const auto &kv : col_map) {
      fields.push_back(kv.second.first);
      columns.push_back(kv.second.second);
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    std::shared_ptr<arrow::Table> table =
        arrow::Table::Make(schema, columns, count);
    if (!table) {
      return arrow::Status::Invalid(
          "Failed to create table from schema and columns");
    }
    segment_tables.push_back(table);
  }

  ARROW_ASSIGN_OR_RAISE(auto final_table,
                        arrow::ConcatenateTables(segment_tables));
  auto dataset = std::make_shared<arrow::dataset::InMemoryDataset>(final_table);
  return dataset;
}

inline arrow::Result<std::shared_ptr<arrow::Table>>
EvaluateExpressionWithDataset(
    const std::shared_ptr<arrow::dataset::Dataset> &dataset,
    const std::string &new_column_name, const arrow::compute::Expression &expr,
    const std::shared_ptr<arrow::DataType> &expected_type) {
  auto new_scan_result = dataset->NewScan();
  if (!new_scan_result.ok()) {
    return arrow::Status::Invalid("Failed to create scanner builder");
  }
  auto scanner_builder = std::move(new_scan_result.ValueOrDie());

  arrow::compute::CastOptions cast_options;
  cast_options.to_type = expected_type;
  cast_options.allow_int_overflow = true;
  cast_options.allow_float_truncate = true;
  arrow::Expression cast_expr = call("cast", {expr}, cast_options);

  auto status = scanner_builder->Project({cast_expr}, {new_column_name});
  if (!status.ok()) {
    return arrow::Status::Invalid("Failed to project expression: ",
                                  status.ToString());
  }
  auto scanner_result = scanner_builder->Finish();
  if (!scanner_result.ok()) {
    return arrow::Status::Invalid("Failed to finish scanner builder: ",
                                  scanner_result.status().ToString());
  }
  auto scanner = std::move(scanner_result.ValueOrDie());

  auto to_table_result = scanner->ToTable();
  if (!to_table_result.ok()) {
    return arrow::Status::Invalid("Failed to convert scanner to table: ",
                                  to_table_result.status().ToString());
  }
  auto result_table = std::move(to_table_result.ValueOrDie());
  return result_table;
}

inline arrow::Status WriteColumnInBlocks(
    const std::string &column_name,
    const std::shared_ptr<arrow::ChunkedArray> &data,
    const std::vector<BlockMeta> &blocks, const std::string &base_path,
    uint32_t segment_id, std::function<BlockID()> allocate_block_id,
    bool use_parquet, std::vector<BlockMeta> *out) {
  int offset = 0;
  for (const auto &block : blocks) {
    auto slice = data->Slice(offset, block.doc_count_);
    auto field = arrow::field(column_name, slice->type());
    auto physic_schema = arrow::schema({field});
    auto table = arrow::Table::Make(arrow::schema({field}), {slice});

    BlockID block_id = allocate_block_id();
    std::string path = FileHelper::MakeForwardBlockPath(base_path, segment_id,
                                                        block_id, use_parquet);
    auto writer = ChunkedFileWriter::Open(
        path, physic_schema,
        use_parquet ? FileFormat::PARQUET : FileFormat::IPC);
    ARROW_RETURN_NOT_OK(writer->Write(*table));
    ARROW_RETURN_NOT_OK(writer->Close());

    BlockMeta new_block(block_id, BlockType::SCALAR, block.min_doc_id_,
                        block.max_doc_id_, block.doc_count_, {column_name});

    out->push_back(new_block);

    offset += block.doc_count_;
  }
  return arrow::Status::OK();
}

inline int64_t MemorySize(const arrow::RecordBatch &batch) {
  int64_t total = 0;
  for (int i = 0; i < batch.num_columns(); ++i) {
    const auto &array = batch.column(i);
    const auto &data = array->data();
    for (const auto &buffer : data->buffers) {
      if (buffer) {
        total += buffer->size();
      }
    }
  }
  return total;
}

}  // namespace zvec
