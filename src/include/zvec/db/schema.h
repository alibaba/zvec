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
#include <unordered_map>
#include <zvec/db/index_params.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>

namespace zvec {

const uint64_t MAX_DOC_COUNT_PER_SEGMENT = 10000000;
const uint64_t MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD = 1000;

/*
 * Field schema
 */
class FieldSchema {
 public:
  using Ptr = std::shared_ptr<FieldSchema>;

 public:
  FieldSchema() = default;
  FieldSchema(const std::string &name, DataType type)
      : name_(name),
        data_type_(type),
        nullable_(false),
        dimension_(0),
        index_params_(nullptr) {}
  FieldSchema(const std::string &name, DataType type, bool nullable,
              const IndexParams::Ptr &index_params = nullptr)
      : name_(name),
        data_type_(type),
        nullable_(nullable),
        dimension_(0),
        index_params_(index_params ? index_params->clone() : nullptr) {}
  FieldSchema(const std::string &name, DataType type, uint32_t dimension,
              bool nullable, const IndexParams::Ptr &index_params = nullptr)
      : name_(name),
        data_type_(type),
        nullable_(nullable),
        dimension_(dimension),
        index_params_(index_params ? index_params->clone() : nullptr) {}
  FieldSchema(const FieldSchema &other)
      : name_(other.name_),
        data_type_(other.data_type_),
        nullable_(other.nullable_),
        dimension_(other.dimension_),
        index_params_(other.index_params_ ? other.index_params_->clone()
                                          : nullptr) {}
  FieldSchema &operator=(const FieldSchema &other) {
    if (this != &other) {
      name_ = other.name_;
      data_type_ = other.data_type_;
      nullable_ = other.nullable_;
      dimension_ = other.dimension_;
      index_params_ =
          other.index_params_ ? other.index_params_->clone() : nullptr;
    }
    return *this;
  }
  FieldSchema(FieldSchema &&) = default;
  FieldSchema &operator=(FieldSchema &&) = default;
  ;
  ~FieldSchema() = default;

 public:
  bool operator==(const FieldSchema &other) const {
    bool index_params_equal = false;
    if (index_params_ == nullptr && other.index_params_ == nullptr) {
      index_params_equal = true;
    } else if (index_params_ != nullptr && other.index_params_ != nullptr) {
      index_params_equal = (*index_params_ == *(other.index_params_));
    } else {
      index_params_equal = false;
    }

    return name_ == other.name_ && data_type_ == other.data_type_ &&
           nullable_ == other.nullable_ && dimension_ == other.dimension_ &&
           index_params_equal;
  }

  bool operator!=(const FieldSchema &other) const {
    return !(*this == other);
  }

  std::string to_string() const;

  std::string to_string_formatted(int indent_level = 0) const;

 public:
  void set_name(const std::string &name) {
    name_ = name;
  }

  const std::string &name() const {
    return name_;
  }

  void set_data_type(DataType type) {
    data_type_ = type;
  }

  DataType data_type() const {
    return data_type_;
  }

  DataType element_data_type() const {
    return get_element_data_type(data_type_);
  }

  size_t element_data_size() const {
    return get_element_data_size(data_type_);
  }

  bool is_vector_field() const {
    return is_vector_field(data_type_);
  }

  bool is_dense_vector() const {
    return is_dense_vector_field(data_type_);
  }

  bool is_sparse_vector() const {
    return is_sparse_vector_field(data_type_);
  }

  bool nullable() const {
    return nullable_;
  }

  void set_nullable(bool nullable) {
    nullable_ = nullable;
  }

  bool has_invert_index() const {
    return !is_vector_field() && index_params_ != nullptr;
  }

  bool is_array_type() const {
    return data_type_ >= DataType::ARRAY_BINARY &&
           data_type_ <= DataType::ARRAY_DOUBLE;
  }

  void set_dimension(uint32_t dimension) {
    dimension_ = dimension;
  }

  uint32_t dimension() const {
    return dimension_;
  }

  IndexType index_type() const {
    if (index_params_) {
      return index_params_->type();
    }
    return IndexType::UNDEFINED;
  }

  IndexParams::Ptr index_params() const {
    return index_params_;
  }

  void set_index_params(const IndexParams::Ptr &index_params) {
    index_params_ = index_params;
  }

  void set_index_params(const IndexParams &index_params) {
    index_params_ = index_params.clone();
  }

  Status validate() const;

 public:
  static bool is_dense_vector_field(DataType type) {
    return type >= DataType::VECTOR_BINARY32 && type <= DataType::VECTOR_INT16;
  }

  static bool is_sparse_vector_field(DataType type) {
    return type >= DataType::SPARSE_VECTOR_FP16 &&
           type <= DataType::SPARSE_VECTOR_FP32;
  }

  static bool is_vector_field(DataType type) {
    return is_dense_vector_field(type) || is_sparse_vector_field(type);
  }

  static DataType get_element_data_type(DataType data_type) {
    switch (data_type) {
      case DataType::ARRAY_BINARY:
        return DataType::BINARY;
      case DataType::ARRAY_STRING:
        return DataType::STRING;
      case DataType::ARRAY_BOOL:
        return DataType::BOOL;
      case DataType::ARRAY_INT32:
        return DataType::INT32;
      case DataType::ARRAY_INT64:
        return DataType::INT64;
      case DataType::ARRAY_UINT32:
        return DataType::UINT32;
      case DataType::ARRAY_UINT64:
        return DataType::UINT64;
      case DataType::ARRAY_FLOAT:
        return DataType::FLOAT;
      case DataType::ARRAY_DOUBLE:
        return DataType::DOUBLE;
      default:
        return data_type;
    }
  }

  static size_t get_element_data_size(DataType data_type) {
    switch (data_type) {
      case DataType::ARRAY_BINARY:
        return 0;
      case DataType::ARRAY_STRING:
        return 0;
      case DataType::ARRAY_BOOL:
        return sizeof(bool);
      case DataType::ARRAY_INT32:
        return sizeof(int32_t);
      case DataType::ARRAY_INT64:
        return sizeof(int64_t);
      case DataType::ARRAY_UINT32:
        return sizeof(uint32_t);
      case DataType::ARRAY_UINT64:
        return sizeof(uint64_t);
      case DataType::ARRAY_FLOAT:
        return sizeof(float);
      case DataType::ARRAY_DOUBLE:
        return sizeof(double);
      case DataType::BINARY:
        return 0;
      case DataType::STRING:
        return 0;
      case DataType::BOOL:
        return sizeof(bool);
      case DataType::INT32:
        return sizeof(int32_t);
      case DataType::INT64:
        return sizeof(int64_t);
      case DataType::UINT32:
        return sizeof(uint32_t);
      case DataType::UINT64:
        return sizeof(uint64_t);
      case DataType::FLOAT:
        return sizeof(float);
      case DataType::DOUBLE:
        return sizeof(double);
      default:
        return 0;
    }
  }


 private:
  std::string name_;
  DataType data_type_{DataType::UNDEFINED};
  bool nullable_{false};
  uint32_t dimension_{0U};
  IndexParams::Ptr index_params_;
};

using FieldSchemaPtrList = std::vector<FieldSchema::Ptr>;
using FieldSchemaPtrMap = std::unordered_map<std::string, FieldSchema::Ptr>;

/*
 * Collection schema
 */
class CollectionSchema {
 public:
  using Ptr = std::shared_ptr<CollectionSchema>;

 public:
  CollectionSchema() = default;

  CollectionSchema(const std::string &name) : name_(name) {}

  CollectionSchema(const std::string &name, const FieldSchemaPtrList &fields)
      : name_(name) {
    copy_fields(fields);
  }

  CollectionSchema(const CollectionSchema &other) {
    name_ = other.name_;
    copy_fields(other.fields_);
    max_doc_count_per_segment_ = other.max_doc_count_per_segment_;
  }

 public:
  std::string to_string() const;


  std::string to_string_formatted(int indent_level = 0) const;

  std::string name() const {
    return name_;
  }

  void set_name(const std::string &name) {
    name_ = name;
  }

  Status add_field(FieldSchema::Ptr column_schema);

  Status alter_field(const std::string &column_name,
                     const FieldSchema::Ptr &new_column_options);

  Status drop_field(const std::string &column_name);

  bool has_field(const std::string &column) const;

  const FieldSchema *get_field(const std::string &column) const;
  FieldSchema *get_field(const std::string &column);
  const FieldSchema *get_forward_field(const std::string &column) const;
  FieldSchema *get_forward_field(const std::string &column);
  const FieldSchema *get_vector_field(const std::string &column) const;
  FieldSchema *get_vector_field(const std::string &column);

  FieldSchemaPtrList fields() const;

  FieldSchemaPtrList forward_fields() const;

  FieldSchemaPtrList forward_fields_with_index() const;

  std::vector<std::string> forward_field_names() const;

  std::vector<std::string> forward_field_names_with_index() const;

  std::vector<std::string> all_field_names() const;

  FieldSchemaPtrList vector_fields() const;

  uint64_t max_doc_count_per_segment() const;

  void set_max_doc_count_per_segment(uint64_t max_doc_count_per_segment);

  Status validate() const;

 public:
  Status add_index(const std::string &column,
                   const IndexParams::Ptr &index_options);

  Status drop_index(const std::string &column);

  bool has_index(const std::string &column) const;

 public:
  bool operator==(const CollectionSchema &other) const {
    if (name_ != other.name_ || fields_.size() != other.fields_.size()) {
      return false;
    }

    for (size_t i = 0; i < fields_.size(); ++i) {
      if (*fields_[i] != *other.fields_[i]) {
        return false;
      }
    }

    return true;
  }

  bool operator!=(const CollectionSchema &other) const {
    return !(*this == other);
  }

 private:
  void copy_fields(const FieldSchemaPtrList &fields) {
    for (auto &field : fields) {
      auto c = std::make_shared<FieldSchema>(*field);
      fields_.push_back(c);
      fields_map_[field->name()] = c;
    }
  }

 private:
  std::string name_{};
  FieldSchemaPtrList fields_{};
  FieldSchemaPtrMap fields_map_{};

  uint64_t max_doc_count_per_segment_{MAX_DOC_COUNT_PER_SEGMENT};
};

}  // namespace zvec