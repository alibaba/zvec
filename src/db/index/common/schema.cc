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

#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <zvec/db/index_params.h>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include <zvec/db/type.h>
#include "db/common/constants.h"
#include "db/common/typedef.h"
#include "db/common/utils.h"
#include "db/index/common/type_helper.h"

namespace zvec {

std::unordered_map<DataType, std::set<QuantizeType>> quantize_type_map = {
    {DataType::VECTOR_FP32,
     {QuantizeType::FP16, QuantizeType::INT4, QuantizeType::INT8}},
    // {DataType::VECTOR_FP64, {QuantizeType::FP16}},
    {DataType::SPARSE_VECTOR_FP32, {QuantizeType::FP16}},
};

std::unordered_set<DataType> support_dense_vector_type = {
    DataType::VECTOR_FP32,
    DataType::VECTOR_FP16,
    DataType::VECTOR_INT8,
};

std::unordered_set<DataType> support_sparse_vector_type = {
    DataType::SPARSE_VECTOR_FP32,
    DataType::SPARSE_VECTOR_FP16,
};

std::unordered_set<IndexType> support_dense_vector_index = {
    IndexType::FLAT, IndexType::HNSW, IndexType::IVF};

std::unordered_set<IndexType> support_sparse_vector_index = {IndexType::FLAT,
                                                             IndexType::HNSW};

Status FieldSchema::validate() const {
  if (data_type_ == DataType::UNDEFINED) {
    return Status::InvalidArgument("schema validate failed: field[", name_,
                                   "]'s data_type is not defined");
  }
  if (name_.empty()) {
    return Status::InvalidArgument("schema validate failed: field[", name_,
                                   "]'s name is empty");
  }
  if (!std::regex_match(name_, FIELD_NAME_REGEX)) {
    return Status::InvalidArgument(
        "schema validate failed: field[", name_,
        "]'s name cannot pass the regex verification");
  }
  if (is_vector_field()) {
    auto is_sparse = is_sparse_vector();
    if (!is_sparse && (dimension_ == 0 || dimension() > kMaxDenseDimSize)) {
      return Status::InvalidArgument("schema validate failed: field[", name_,
                                     "]'s dimension must be in (0,20000]");
    }

    if (!is_sparse) {
      if (support_dense_vector_type.find(data_type_) ==
          support_dense_vector_type.end()) {
        return Status::InvalidArgument(
            "schema validate failed: dense_vector's data type only "
            "support FP32, "
            "but field[",
            name_, "]'s data type is ", DataTypeCodeBook::AsString(data_type_));
      }
    } else {
      if (support_sparse_vector_type.find(data_type_) ==
          support_sparse_vector_type.end()) {
        return Status::InvalidArgument(
            "schema validate failed: sparse_vector's data type only "
            "support FP32, "
            "but field[",
            name_, "]'s data type is ", DataTypeCodeBook::AsString(data_type_));
      }
    }

    if (index_params_) {
      auto vector_index_params =
          std::dynamic_pointer_cast<VectorIndexParams>(index_params_);

      if (is_sparse) {
        if (support_sparse_vector_index.find(index_params_->type()) ==
            support_sparse_vector_index.end()) {
          return Status::InvalidArgument(
              "schema validate failed: sparse_vector's index_params only "
              "support FLAT|HNSW index, "
              "but field[",
              name_, "]'s index_type is ",
              IndexTypeCodeBook::AsString(index_params_->type()));
        }
        if (vector_index_params->metric_type() != MetricType::IP) {
          return Status::InvalidArgument(
              "schema validate failed: sparse_vector's index_params only "
              "support IP metric, but "
              "field[",
              name_, "]'s metric is ",
              MetricTypeCodeBook::AsString(vector_index_params->metric_type()));
        }

      } else {
        if (support_dense_vector_index.find(index_params_->type()) ==
            support_dense_vector_index.end()) {
          return Status::InvalidArgument(
              "schema validate failed: dense_vector's index_params only "
              "support FLAT|HNSW|IVF index, but field[",
              name_, "]'s index_type is ",
              IndexTypeCodeBook::AsString(index_params_->type()));
        }
      }

      if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
        auto iter = quantize_type_map.find(data_type_);
        if (iter == quantize_type_map.end()) {
          return Status::InvalidArgument(
              "schema validate failed: ",
              is_sparse ? "sparse_vector" : "dense_vector",
              "'s index_params of ", DataTypeCodeBook::AsString(data_type_),
              " do not support quantize, but field[", name_,
              "]'s quantize_type is ",
              QuantizeTypeCodeBook::AsString(
                  vector_index_params->quantize_type()));
        } else {
          if (iter->second.find(vector_index_params->quantize_type()) ==
              iter->second.end()) {
            return Status::InvalidArgument(
                "schema validate failed: ",
                is_sparse ? "sparse_vector" : "dense_vector",
                "'s index_params of ", DataTypeCodeBook::AsString(data_type_),
                " support ", QuantizeTypeCodeBook::AsString(iter->second),
                " quantize, but field[", name_, "]'s quantize_type is ",
                QuantizeTypeCodeBook::AsString(
                    vector_index_params->quantize_type()));
          }
        }
      }
      if (index_params_->type() == IndexType::IVF &&
          vector_index_params->metric_type() == MetricType::IP) {
        if (data_type_ != DataType::VECTOR_FP16 &&
            data_type_ != DataType::VECTOR_FP32) {
          return Status::InvalidArgument(
              "schema validate failed: IVF index only support FP32/FP16 data "
              "types according to the IP metric");
        }
      }
    }
  } else {
    if (index_params_) {
      if (index_params_->is_vector_index_type()) {
        return Status::InvalidArgument(
            "schema validate failed: scalar_field's index_params only support "
            "INVERT "
            "index, "
            "but field[",
            name_, "]'s index_type is ",
            IndexTypeCodeBook::AsString(index_params_->type()));
      }
    }
  }
  return Status::OK();
}

std::string FieldSchema::to_string() const {
  std::ostringstream oss;
  oss << "FieldSchema{"
      << "name:'" << name_ << "'"
      << ",data_type:" << DataTypeCodeBook::AsString(data_type_)
      << ",nullable:" << (nullable_ ? "true" : "false")
      << ",dimension:" << dimension_;

  if (index_params_) {
    oss << ",index_params:" << index_params_->to_string();
  } else {
    oss << ",index_params:null";
  }

  oss << "}";
  return oss.str();
}

std::string FieldSchema::to_string_formatted(int indent_level) const {
  std::ostringstream oss;
  if (is_vector_field()) {
    oss << indent(indent_level) << "FieldSchema[vector]{\n";
  } else {
    oss << indent(indent_level) << "FieldSchema[scalar]{\n";
  }

  oss << indent(indent_level + 1) << "name: '" << name_ << "',\n"
      << indent(indent_level + 1)
      << "data_type: " << DataTypeCodeBook::AsString(data_type_) << ",\n";

  if (is_vector_field()) {
    if (is_dense_vector()) {
      oss << indent(indent_level + 1) << "dimension: " << dimension_ << ",\n";
    }
  } else {
    oss << indent(indent_level + 1)
        << "nullable: " << (nullable_ ? "true" : "false") << ",\n";
  }

  if (index_params_) {
    oss << indent(indent_level + 1)
        << "index_params: " << index_params_->to_string() << "\n";
  } else {
    oss << indent(indent_level + 1) << "index_params: null\n";
  }

  oss << indent(indent_level) << "}";
  return oss.str();
}

Status CollectionSchema::validate() const {
  if (name_.empty()) {
    return Status::InvalidArgument("schema validate failed: name is empty");
  }
  if (!std::regex_match(name_, COLLECTION_NAME_REGEX)) {
    return Status::InvalidArgument(
        "schema validate failed: collection[", name_,
        "]'s name cannot pass the regex verification");
  }
  if (forward_fields().size() > kMaxScalarFieldSize) {
    return Status::InvalidArgument(
        "schema validate failed: collection[", name_,
        "]'s field size must <= ", kMaxScalarFieldSize);
  }
  if (max_doc_count_per_segment_ < MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD) {
    return Status::InvalidArgument(
        "schema validate failed: max_doc_count_per_segment must >= ",
        MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD);
  }
  auto v_fields = vector_fields();
  if (v_fields.empty()) {
    return Status::InvalidArgument(
        "schema validate failed: vector fields is empty");
  }
  if (v_fields.size() > kMaxVectorFieldSize) {
    return Status::InvalidArgument(
        "schema validate failed: collection[", name_,
        "]'s vector field size must <= ", kMaxVectorFieldSize);
  }
  for (auto &field : fields_) {
    auto s = field->validate();
    CHECK_RETURN_STATUS(s);
  }
  return Status::OK();
}

std::string CollectionSchema::to_string() const {
  std::ostringstream oss;
  oss << "CollectionSchema{"
      << "name:'" << name_ << "'"
      << ",max_doc_count_per_segment:" << max_doc_count_per_segment_
      << ",fields:[";

  for (size_t i = 0; i < fields_.size(); ++i) {
    if (i > 0) oss << ",";
    oss << fields_[i]->to_string();
  }

  oss << "]}";
  return oss.str();
}


std::string CollectionSchema::to_string_formatted(int indent_level) const {
  std::ostringstream oss;
  oss << indent(indent_level) << "CollectionSchema{\n"
      << indent(indent_level + 1) << "name: '" << name_ << "',\n"
      << indent(indent_level + 1)
      << "max_doc_count_per_segment: " << max_doc_count_per_segment_ << ",\n"
      << indent(indent_level + 1) << "fields: [\n";

  for (size_t i = 0; i < fields_.size(); ++i) {
    oss << fields_[i]->to_string_formatted(indent_level + 2);
    if (i < fields_.size() - 1) {
      oss << ",";
    }
    oss << "\n";
  }

  oss << indent(indent_level + 1) << "]\n" << indent(indent_level) << "}";
  return oss.str();
}

Status CollectionSchema::add_field(FieldSchema::Ptr column_schema) {
  // Check if field already exists
  if (has_field(column_schema->name())) {
    return Status::AlreadyExists("field[", column_schema->name(),
                                 "] already exists in schema");
  }

  // Add field to list and map
  if (column_schema->is_vector_field()) {
    if (column_schema->index_params() == nullptr) {
      column_schema->set_index_params(DefaultVectorIndexParams);
    }
  }

  fields_.push_back(column_schema);
  fields_map_[column_schema->name()] = column_schema;

  return Status::OK();
}

Status CollectionSchema::alter_field(
    const std::string &column_name,
    const FieldSchema::Ptr &new_column_options) {
  // Check if field exists
  if (!has_field(column_name)) {
    return Status::NotFound("field[", column_name, "] not found in schema");
  }

  std::string new_column_name = new_column_options->name();

  // If renaming to an existing field name (and it's not the same field)
  if (new_column_name != column_name && has_field(new_column_name)) {
    return Status::AlreadyExists("field[", new_column_name,
                                 "] already exists in schema");
  }

  // Update map: remove old entry if name changed, add new entry
  if (new_column_name != column_name) {
    fields_map_.erase(column_name);
  }
  fields_map_[new_column_name] = new_column_options;

  // Update list
  for (auto &field : fields_) {
    if (field->name() == column_name) {
      field = new_column_options;
      break;
    }
  }

  return Status::OK();
}

Status CollectionSchema::drop_field(const std::string &column_name) {
  // Check if field exists
  if (!has_field(column_name)) {
    return Status::NotFound("field[", column_name, "] not found in schema");
  }

  // Remove from map
  fields_map_.erase(column_name);

  // Remove from list
  fields_.erase(std::remove_if(fields_.begin(), fields_.end(),
                               [&column_name](const FieldSchema::Ptr &field) {
                                 return field->name() == column_name;
                               }),
                fields_.end());

  return Status::OK();
}

bool CollectionSchema::has_field(const std::string &column) const {
  return fields_map_.find(column) != fields_map_.end();
}

const FieldSchema *CollectionSchema::get_field(
    const std::string &column) const {
  auto it = fields_map_.find(column);
  if (it != fields_map_.end()) {
    return it->second.get();
  }
  return nullptr;
}

FieldSchema *CollectionSchema::get_field(const std::string &column) {
  auto it = fields_map_.find(column);
  if (it != fields_map_.end()) {
    return it->second.get();
  }
  return nullptr;
}

const FieldSchema *CollectionSchema::get_forward_field(
    const std::string &column) const {
  // Forward fields are typically non-vector fields
  auto field = get_field(column);
  if (field && !field->is_vector_field()) {
    return field;
  }
  return nullptr;
}

FieldSchema *CollectionSchema::get_forward_field(const std::string &column) {
  // Forward fields are typically non-vector fields
  auto field = get_field(column);
  if (field && !field->is_vector_field()) {
    return field;
  }
  return nullptr;
}

const FieldSchema *CollectionSchema::get_vector_field(
    const std::string &column) const {
  // Vector fields are fields with vector data types
  auto field = get_field(column);
  if (field && field->is_vector_field()) {
    return field;
  }
  return nullptr;
}

FieldSchema *CollectionSchema::get_vector_field(const std::string &column) {
  // Vector fields are fields with vector data types
  auto field = get_field(column);
  if (field && field->is_vector_field()) {
    return field;
  }
  return nullptr;
}

FieldSchemaPtrList CollectionSchema::fields() const {
  return fields_;
}

FieldSchemaPtrList CollectionSchema::forward_fields() const {
  FieldSchemaPtrList forward_fields;
  for (const auto &field : fields_) {
    if (!field->is_vector_field()) {
      forward_fields.push_back(field);
    }
  }
  return forward_fields;
}

FieldSchemaPtrList CollectionSchema::forward_fields_with_index() const {
  FieldSchemaPtrList forward_fields;
  for (const auto &field : fields_) {
    if (!field->is_vector_field() && field->index_params() != nullptr) {
      forward_fields.push_back(field);
    }
  }
  return forward_fields;
}

std::vector<std::string> CollectionSchema::forward_field_names() const {
  std::vector<std::string> names;
  for (const auto &field : fields_) {
    if (!field->is_vector_field()) {
      names.push_back(field->name());
    }
  }
  return names;
}

std::vector<std::string> CollectionSchema::forward_field_names_with_index()
    const {
  std::vector<std::string> names;
  for (const auto &field : fields_) {
    if (!field->is_vector_field() && field->index_params() != nullptr) {
      names.push_back(field->name());
    }
  }
  return names;
}

std::vector<std::string> CollectionSchema::all_field_names() const {
  std::vector<std::string> names;
  for (const auto &field : fields_) {
    names.push_back(field->name());
  }
  return names;
}

FieldSchemaPtrList CollectionSchema::vector_fields() const {
  FieldSchemaPtrList vector_fields;
  for (const auto &field : fields_) {
    if (field->is_vector_field()) {
      vector_fields.push_back(field);
    }
  }
  return vector_fields;
}

uint64_t CollectionSchema::max_doc_count_per_segment() const {
  return max_doc_count_per_segment_;
}

void CollectionSchema::set_max_doc_count_per_segment(
    uint64_t max_doc_count_per_segment) {
  max_doc_count_per_segment_ = max_doc_count_per_segment;
}

Status CollectionSchema::add_index(const std::string &column,
                                   const IndexParams::Ptr &index_params) {
  // Get field and set index params
  auto field = get_field(column);
  if (field) {
    field->set_index_params(index_params);
  } else {
    return Status::NotFound("field[", column, "] not found in schema");
  }

  return Status::OK();
}

Status CollectionSchema::drop_index(const std::string &column) {
  // Get field and clear index params
  auto field = get_field(column);
  if (field) {
    if (field->is_vector_field()) {
      field->set_index_params(DefaultVectorIndexParams);
    } else {
      field->set_index_params(nullptr);
    }
  } else {
    return Status::NotFound("field[", column, "] not found in schema");
  }

  return Status::OK();
}

bool CollectionSchema::has_index(const std::string &column) const {
  auto field = get_field(column);
  if (field) {
    if (field->is_vector_field()) {
      if (field->index_params() == nullptr) {
        return false;
      } else {
        return *field->index_params() != DefaultVectorIndexParams;
      }
    }
    return field->index_params() != nullptr;
  }
  return false;
}

}  // namespace zvec