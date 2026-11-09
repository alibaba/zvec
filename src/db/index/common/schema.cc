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
#include "db/index/common/identifier_validation.h"
#include "db/index/common/type_helper.h"

namespace zvec {

std::unordered_map<DataType, std::set<QuantizeType>> quantize_type_map = {
    {DataType::VECTOR_FP32,
     {QuantizeType::FP16, QuantizeType::INT4, QuantizeType::INT8,
      QuantizeType::RABITQ, QuantizeType::UNIFORM_UINT7,
      QuantizeType::UNIFORM_UINT8, QuantizeType::UNIFORM_UINT4,
      QuantizeType::PQ}},
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
    IndexType::FLAT,  IndexType::HNSW,       IndexType::HNSW_RABITQ,
    IndexType::IVF,   IndexType::IVF_RABITQ, IndexType::DISKANN,
    IndexType::VAMANA};

std::unordered_set<IndexType> support_sparse_vector_index = {IndexType::FLAT,
                                                             IndexType::HNSW};

static Status validate_index_params(const FieldSchema &field) {
  auto status = field.index_params()->validate();
  if (!status.ok()) {
    return Status(status.code(), "Invalid schema for field[" + field.name() +
                                     "]: " + status.message());
  }
  return Status::OK();
}

Status FieldSchema::validate() const {
  if (auto s = validate_field_name(name_); !s.ok()) {
    return s;
  }

  if (data_type_ == DataType::UNDEFINED) {
    return Status::InvalidArgument("Invalid schema for field[", name_,
                                   "]: data_type must not be UNDEFINED");
  }
  if (is_vector_field()) {
    auto is_sparse = is_sparse_vector();
    if (!is_sparse && (dimension_ == 0 || dimension() > kMaxDenseDimSize)) {
      return Status::InvalidArgument("Invalid schema for field[", name_,
                                     "]: dimension must be in [1, ",
                                     kMaxDenseDimSize, "]; got ", dimension_);
    }

    if (!is_sparse) {
      if (support_dense_vector_type.find(data_type_) ==
          support_dense_vector_type.end()) {
        return Status::InvalidArgument(
            "Invalid schema for field[", name_,
            "]: dense vectors support VECTOR_FP32/VECTOR_FP16/VECTOR_INT8 "
            "data_type; got ",
            DataTypeCodeBook::AsString(data_type_));
      }
    } else {
      if (support_sparse_vector_type.find(data_type_) ==
          support_sparse_vector_type.end()) {
        return Status::InvalidArgument(
            "Invalid schema for field[", name_,
            "]: sparse vectors support SPARSE_VECTOR_FP32/SPARSE_VECTOR_FP16 "
            "data_type; got ",
            DataTypeCodeBook::AsString(data_type_));
      }
    }

    if (index_params_) {
      const auto *vector_index_params =
          dynamic_cast<const VectorIndexParams *>(index_params_.get());
      if (!vector_index_params) {
        return Status::InvalidArgument(
            "Invalid schema for field[", name_,
            "]: vector fields require VectorIndexParams; got index_type=",
            IndexTypeCodeBook::AsString(index_params_->type()));
      }

      if (is_sparse) {
        if (support_sparse_vector_index.find(index_params_->type()) ==
            support_sparse_vector_index.end()) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: sparse vectors support FLAT/HNSW index_type; got ",
              IndexTypeCodeBook::AsString(index_params_->type()));
        }
        if (vector_index_params->metric_type() != MetricType::IP) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: sparse vectors require IP metric; got ",
              MetricTypeCodeBook::AsString(vector_index_params->metric_type()));
        }

      } else {
        if (support_dense_vector_index.find(index_params_->type()) ==
            support_dense_vector_index.end()) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: dense vectors support "
              "FLAT/HNSW/HNSW_RABITQ/IVF/IVF_RABITQ/DISKANN/VAMANA "
              "index_type; got ",
              IndexTypeCodeBook::AsString(index_params_->type()));
        }
      }

      if (index_params_->type() == IndexType::HNSW_RABITQ ||
          index_params_->type() == IndexType::IVF_RABITQ) {
        if (dimension_ < kMinRabitqDimSize || dimension_ > kMaxRabitqDimSize) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: RabitQ requires dimension in [", kMinRabitqDimSize, ", ",
              kMaxRabitqDimSize, "]; got ", dimension_);
        }
        if (data_type_ != DataType::VECTOR_FP32) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: RabitQ requires VECTOR_FP32 data_type; got ",
              DataTypeCodeBook::AsString(data_type_));
        }
      }

      if (index_params_->type() == IndexType::IVF_RABITQ) {
        if (!dynamic_cast<const IvfRabitqIndexParams *>(index_params_.get())) {
          return Status::InvalidArgument("Invalid schema for field[", name_,
                                         "]: IVF_RABITQ index requires "
                                         "IvfRabitqIndexParams");
        }
      }

      auto params_status = validate_index_params(*this);
      CHECK_RETURN_STATUS(params_status);

      const auto flat_data_type = vector_index_params->flat_data_type();
      if (is_sparse && flat_data_type != DataType::UNDEFINED &&
          flat_data_type != DataType::VECTOR_FP32) {
        return Status::InvalidArgument(
            "Invalid schema for field[", name_,
            "]: sparse vectors require VECTOR_FP32 flat_data_type; got ",
            DataTypeCodeBook::AsString(flat_data_type));
      }

      if (vector_index_params->quantize_type() != QuantizeType::UNDEFINED) {
        const auto quantize_type = vector_index_params->quantize_type();
        if (quantize_type == QuantizeType::PQ) {
          const auto &pq = vector_index_params->quantizer_param();
          if (data_type_ != DataType::VECTOR_FP32) {
            return Status::InvalidArgument(
                "Invalid schema for field[", name_,
                "]: PQ requires VECTOR_FP32 data_type; got ",
                DataTypeCodeBook::AsString(data_type_));
          }
          if (static_cast<uint32_t>(pq.num_chunk()) > dimension_) {
            return Status::InvalidArgument(
                "Invalid schema for field[", name_,
                "]: PQ num_chunk must not exceed dimension; got num_chunk=",
                pq.num_chunk(), ", dimension=", dimension_);
          }
          if (pq.num_bits() == 4 && dimension_ % pq.num_chunk() != 0) {
            return Status::InvalidArgument(
                "Invalid schema for field[", name_,
                "]: PQ4 requires dimension divisible by num_chunk; got "
                "dimension=",
                dimension_, ", num_chunk=", pq.num_chunk());
          }
        }
        auto iter = quantize_type_map.find(data_type_);
        if (iter == quantize_type_map.end()) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: data_type=", DataTypeCodeBook::AsString(data_type_),
              " does not support quantization; got quantize_type=",
              QuantizeTypeCodeBook::AsString(quantize_type));
        }
        if (iter->second.find(quantize_type) == iter->second.end()) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: data_type=", DataTypeCodeBook::AsString(data_type_),
              " supports quantize_type in ",
              QuantizeTypeCodeBook::AsString(iter->second), "; got ",
              QuantizeTypeCodeBook::AsString(quantize_type));
        }
      }
      if (index_params_->type() == IndexType::IVF &&
          vector_index_params->metric_type() == MetricType::IP) {
        if (data_type_ != DataType::VECTOR_FP16 &&
            data_type_ != DataType::VECTOR_FP32) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: IVF with IP metric requires VECTOR_FP32/VECTOR_FP16 "
              "data_type; got ",
              DataTypeCodeBook::AsString(data_type_));
        }
      }
      if (vector_index_params->metric_type() == MetricType::COSINE) {
        if (data_type_ != DataType::VECTOR_FP16 &&
            data_type_ != DataType::VECTOR_FP32) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: COSINE metric requires VECTOR_FP32/VECTOR_FP16 data_type; "
              "got ",
              DataTypeCodeBook::AsString(data_type_));
        }
      }
    }
  } else {
    if (index_params_) {
      if (index_params_->is_vector_index_type()) {
        return Status::InvalidArgument(
            "Invalid schema for field[", name_,
            "]: scalar fields do not support vector index params; got "
            "index_type=",
            IndexTypeCodeBook::AsString(index_params_->type()));
      }
      if (index_params_->type() == IndexType::FTS &&
          data_type_ != DataType::STRING) {
        return Status::InvalidArgument("Invalid schema for field[", name_,
                                       "]: FTS requires STRING data_type; got ",
                                       DataTypeCodeBook::AsString(data_type_));
      }
      if (index_params_->type() == IndexType::FTS) {
        if (!dynamic_cast<const FtsIndexParams *>(index_params_.get())) {
          return Status::InvalidArgument(
              "Invalid schema for field[", name_,
              "]: FTS index requires FtsIndexParams");
        }
      }
      auto params_status = validate_index_params(*this);
      CHECK_RETURN_STATUS(params_status);
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
  if (auto s = validate_collection_name(name_); !s.ok()) {
    return s;
  }
  std::unordered_set<std::string> names;
  for (const auto &field : fields_) {
    if (!field) {
      return Status::InvalidArgument("Invalid schema for collection[",
                                     format_name(name_),
                                     "]: field schema must not be null");
    }
    if (!names.insert(field->name()).second) {
      return Status::InvalidArgument("Invalid schema: duplicate field name [",
                                     format_name(field->name()),
                                     "]; field names must be unique");
    }
  }
  if (forward_fields().size() > kMaxScalarFieldSize) {
    return Status::InvalidArgument(
        "Invalid schema for collection[", format_name(name_),
        "]: scalar field count must be <= ", kMaxScalarFieldSize, "; got ",
        forward_fields().size());
  }
  if (max_doc_count_per_segment_ < MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD) {
    return Status::InvalidArgument("Invalid schema for collection[",
                                   format_name(name_),
                                   "]: max_doc_count_per_segment must be >= ",
                                   MAX_DOC_COUNT_PER_SEGMENT_MIN_THRESHOLD,
                                   "; got ", max_doc_count_per_segment_);
  }
  if (fields_.empty()) {
    return Status::InvalidArgument("Invalid schema for collection[",
                                   format_name(name_), "]: has no fields");
  }
  auto v_fields = vector_fields();
  if (v_fields.size() > kMaxVectorFieldSize) {
    return Status::InvalidArgument(
        "Invalid schema for collection[", format_name(name_),
        "]: vector field count must be <= ", kMaxVectorFieldSize, "; got ",
        v_fields.size());
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
  if (!column_schema) {
    return Status::InvalidArgument(
        "Invalid schema: field schema must not be null");
  }
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
  if (!new_column_options) {
    return Status::InvalidArgument(
        "Invalid schema: field schema must not be null");
  }
  // Check if field exists
  if (!has_field(column_name)) {
    return Status::NotFound("field[", format_name(column_name),
                            "] not found in schema");
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
    return Status::NotFound("field[", format_name(column_name),
                            "] not found in schema");
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

FieldSchemaPtrList CollectionSchema::invert_fields() const {
  FieldSchemaPtrList invert;
  for (const auto &field : fields_) {
    if (field->index_type() == IndexType::INVERT) {
      invert.push_back(field);
    }
  }
  return invert;
}

bool CollectionSchema::has_fts_field() const {
  for (const auto &field : fields_) {
    if (field->index_type() == IndexType::FTS) {
      return true;
    }
  }
  return false;
}

FieldSchemaPtrList CollectionSchema::fts_fields() const {
  FieldSchemaPtrList fts;
  for (const auto &field : fields_) {
    if (field->index_type() == IndexType::FTS) {
      fts.push_back(field);
    }
  }
  return fts;
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
  auto field = get_field(column);
  if (field) {
    field->set_index_params(index_params);
  } else {
    return Status::NotFound("field[", format_name(column),
                            "] not found in schema");
  }

  return Status::OK();
}

Status CollectionSchema::drop_index(const std::string &column) {
  auto field = get_field(column);
  if (field) {
    if (field->is_vector_field()) {
      field->set_index_params(DefaultVectorIndexParams);
    } else {
      field->set_index_params(nullptr);
    }
  } else {
    return Status::NotFound("field[", format_name(column),
                            "] not found in schema");
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
