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

#include <cstdint>
#include <zvec/db/query.h>
#include <zvec/db/schema.h>
#include "db/common/constants.h"
#include "db/index/common/type_helper.h"

namespace zvec {

Status QueryTarget::validate(const FieldSchema *schema,
                             bool *need_sanitize) const {
  // schema describes the field selected by field_name_.
  // Fields referenced by scalar filters are validated separately.
  auto opt_view = get_vector_view();
  auto *fc = get_fts_clause();
  auto &field_name = field_name_;
  auto &query_params = query_params_;
  bool no_vector_payload = !opt_view || (opt_view->query_vector_.empty() &&
                                         opt_view->sparse_indices_.empty());

  // No field schema was found for the query's target field name.
  if (schema == nullptr) {
    if (fc != nullptr) {
      // FTS queries require an existing target field.
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: field does not exist in the collection");
    }
    if (no_vector_payload) {
      // A scalar-only filter and does not require a target field.
      return Status::OK();
    } else {
      // A vector query requires an existing target field.
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: field does not exist in the collection");
    }
  }

  // FTS query: field must be an FTS-indexed field.
  if (fc != nullptr) {
    if (schema->index_type() != IndexType::FTS) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: FTS query requires an FTS index; got ",
          IndexTypeCodeBook::AsString(schema->index_type()));
    }
    if (query_params && query_params->type() != schema->index_type()) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: query params type mismatch; expected ",
          IndexTypeCodeBook::AsString(schema->index_type()), " but got ",
          IndexTypeCodeBook::AsString(query_params->type()));
    }
    return Status::OK();
  }

  // FTS is handled above; the remaining targets require a vector payload.
  if (no_vector_payload) {
    return Status::InvalidArgument("Invalid query for field[", field_name,
                                   "]: query vector must not be empty");
  }

  auto &query_vector = opt_view->query_vector_;
  auto &query_sparse_indices = opt_view->sparse_indices_;
  auto &query_sparse_values = opt_view->sparse_values_;

  if (schema->is_dense_vector()) {
    size_t value_byte_size = 0;
    switch (schema->data_type()) {
      case DataType::VECTOR_FP16:
        value_byte_size = sizeof(float16_t);
        break;
      case DataType::VECTOR_FP32:
        value_byte_size = sizeof(float);
        break;
      case DataType::VECTOR_FP64:
        value_byte_size = sizeof(double);
        break;
      case DataType::VECTOR_INT8:
        value_byte_size = sizeof(int8_t);
        break;
      case DataType::VECTOR_INT16:
      case DataType::VECTOR_INT4:
      case DataType::VECTOR_BINARY32:
      case DataType::VECTOR_BINARY64:
        return Status::NotSupported(
            "Invalid query for field[", field_name,
            "]: unsupported dense vector data type ",
            DataTypeCodeBook::AsString(schema->data_type()));
      default:
        return Status::InvalidArgument(
            "Invalid query for field[", field_name,
            "]: expected a dense vector field; got ",
            DataTypeCodeBook::AsString(schema->data_type()));
    }
    auto dim = schema->dimension();
    if (dim * value_byte_size != query_vector.size()) {
      if (query_vector.size() % value_byte_size != 0) {
        return Status::InvalidArgument(
            "Invalid query for field[", field_name,
            "]: vector data size must be a multiple of ", value_byte_size,
            " bytes for ", DataTypeCodeBook::AsString(schema->data_type()),
            "; got ", query_vector.size(), " bytes");
      }
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: dimension mismatch; expected ", dim, " but got ",
          query_vector.size() / value_byte_size, " (",
          DataTypeCodeBook::AsString(schema->data_type()), ")");
    }
  } else if (schema->is_sparse_vector()) {
    size_t value_byte_size = 0;
    switch (schema->data_type()) {
      case DataType::SPARSE_VECTOR_FP32:
        value_byte_size = sizeof(float);
        break;
      case DataType::SPARSE_VECTOR_FP16:
        value_byte_size = sizeof(float16_t);
        break;
      default:
        return Status::InvalidArgument(
            "Invalid query for field[", field_name,
            "]: unsupported sparse vector data type ",
            DataTypeCodeBook::AsString(schema->data_type()));
    }
    if (query_sparse_indices.size() % sizeof(uint32_t) != 0) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: sparse index data size must be a multiple of ", sizeof(uint32_t),
          " bytes; got ", query_sparse_indices.size(), " bytes");
    }
    if (query_sparse_values.size() % value_byte_size != 0) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: sparse value data size must be a multiple of ", value_byte_size,
          " bytes for ", DataTypeCodeBook::AsString(schema->data_type()),
          "; got ", query_sparse_values.size(), " bytes");
    }
    size_t n_indices = query_sparse_indices.size() / sizeof(uint32_t);
    size_t n_values = query_sparse_values.size() / value_byte_size;
    if (n_indices != n_values) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: sparse indices and values must have the same element count; "
          "got indices=",
          n_indices, ", values=", n_values);
    }
    if (n_indices > kSparseMaxDimSize) {
      return Status::InvalidArgument("Invalid query for field[", field_name,
                                     "]: sparse index count must not exceed ",
                                     kSparseMaxDimSize, "; got ", n_indices);
    }
    // The caller initializes *need_sanitize to false. This read-only scan may
    // defer duplicate detection for unsorted indices until sanitization.
    if (n_indices > 1 && need_sanitize) {
      const auto *idx =
          reinterpret_cast<const uint32_t *>(query_sparse_indices.data());
      auto status = need_sanitize_sparse(idx, n_indices);
      if (status == SparseIndicesStatus::kHasDuplicate) {
        return Status::InvalidArgument(
            "Invalid query for field[", field_name,
            "]: sparse vector contains duplicate indices");
      }
      if (status == SparseIndicesStatus::kNeedSort) {
        *need_sanitize = true;
      }
    }
  } else {
    return Status::InvalidArgument(
        "Invalid query for field[", field_name,
        "]: vector search requires a vector field; got ",
        DataTypeCodeBook::AsString(schema->data_type()));
  }
  // Validate query_params type
  if (query_params && query_params->type() != schema->index_type()) {
    return Status::InvalidArgument(
        "Invalid query for field[", field_name,
        "]: query params type mismatch; expected ",
        IndexTypeCodeBook::AsString(schema->index_type()), " but got ",
        IndexTypeCodeBook::AsString(query_params->type()));
  }
  // Validate refinement support for the field and configured index.
  if (query_params && query_params->is_using_refiner()) {
    if (schema->is_sparse_vector()) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: refinement is not supported for sparse vectors");
    }
    const auto index_type = schema->index_type();
    if (index_type == IndexType::HNSW || index_type == IndexType::IVF ||
        index_type == IndexType::DISKANN || index_type == IndexType::VAMANA) {
      const auto *index_params =
          dynamic_cast<const VectorIndexParams *>(schema->index_params().get());
      if (index_params &&
          index_params->quantize_type() == QuantizeType::UNDEFINED) {
        return Status::InvalidArgument(
            "Invalid query for field[", field_name,
            "]: refinement is not supported for this ",
            IndexTypeCodeBook::AsString(index_type), " index configuration");
      }
    }
  }
  if (query_params && query_params->type() == IndexType::IVF_RABITQ) {
    auto ivf_rabitq_params =
        std::dynamic_pointer_cast<IvfRabitqQueryParams>(query_params);
    if (!ivf_rabitq_params) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: IVF_RABITQ index requires IvfRabitqQueryParams");
    }
    if (ivf_rabitq_params->nprobe() <= 0) {
      return Status::InvalidArgument(
          "Invalid query for field[", field_name,
          "]: IVF_RABITQ nprobe must be greater than 0; got ",
          ivf_rabitq_params->nprobe());
    }
  }
  return Status::OK();
}

Status validate_topk_and_output_fields(
    int topk, const std::optional<std::vector<std::string>> &output_fields) {
  if (static_cast<uint32_t>(topk) > kMaxQueryTopk) {
    return Status::InvalidArgument("Invalid query: topk must be in [0, ",
                                   kMaxQueryTopk, "]; got ", topk);
  }
  if (output_fields.has_value() &&
      output_fields->size() > kMaxOutputFieldSize) {
    return Status::InvalidArgument(
        "Invalid query: output field count must not exceed ",
        kMaxOutputFieldSize, "; got ", output_fields->size());
  }
  return Status::OK();
}

Status SearchQuery::validate(const FieldSchema *schema,
                             bool *need_sanitize) const {
  if (need_sanitize) {
    *need_sanitize = false;
  }
  auto s = validate_topk_and_output_fields(topk_, output_fields_);
  if (!s.ok()) {
    return s;
  }
  return target_.validate(schema, need_sanitize);
}

Status sanitize_sparse_vector(VectorClause &vc, const FieldSchema *schema) {
  if (!schema || !schema->is_sparse_vector()) {
    return Status::OK();
  }
  size_t value_byte_size = 0;
  switch (schema->data_type()) {
    case DataType::SPARSE_VECTOR_FP32:
      value_byte_size = sizeof(float);
      break;
    case DataType::SPARSE_VECTOR_FP16:
      value_byte_size = sizeof(float16_t);
      break;
    default:
      return Status::OK();
  }
  size_t n_indices = vc.sparse_indices_.size() / sizeof(uint32_t);
  if (n_indices <= 1) {
    return Status::OK();
  }
  if (sort_and_find_duplicates(
          reinterpret_cast<uint32_t *>(vc.sparse_indices_.data()),
          vc.sparse_values_.data(), n_indices, value_byte_size)) {
    return Status::InvalidArgument(
        "Invalid query for field[", schema->name(),
        "]: sparse vector contains duplicate indices");
  }
  return Status::OK();
}

Status sanitize_sparse_vector(QueryTarget &target, const FieldSchema *schema) {
  if (auto *vvc = target.get_vector_view_clause()) {
    // Own the buffers before sorting so borrowed input remains unchanged.
    target.clause_ = VectorClause{std::string(vvc->query_vector_),
                                  std::string(vvc->sparse_indices_),
                                  std::string(vvc->sparse_values_)};
  }
  return sanitize_sparse_vector(*target.get_vector_clause(), schema);
}

}  // namespace zvec
