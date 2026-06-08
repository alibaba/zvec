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

Status SearchQuery::validate_and_sanitize(const FieldSchema *schema) {
  if ((uint32_t)topk_ > kMaxQueryTopk) {
    return Status::InvalidArgument("Invalid query: topk[", topk_,
                                   "] exceeds the maximum allowed value of ",
                                   kMaxQueryTopk);
  }
  if (output_fields_.has_value() &&
      output_fields_->size() > kMaxOutputFieldSize) {
    return Status::InvalidArgument(
        "Invalid query: too many output fields, the maximum allowed is ",
        kMaxOutputFieldSize);
  }

  auto *vc = target_.get_vector_clause();
  auto *fc = target_.get_fts_clause();
  auto &field_name = target_.field_name_;
  auto &query_params = target_.query_params_;
  // A "scalar-only filter" query has no vector payload — either the clause
  // is not a VectorClause (e.g., FtsClause) or its fields are all empty.
  bool no_vector_payload = (vc == nullptr) || (vc->query_vector_.empty() &&
                                               vc->sparse_indices_.empty());

  if (schema == nullptr) {
    if (fc != nullptr) {
      // FTS query requires a valid field_name_ that resolves to an FTS field.
      return Status::InvalidArgument(
          "Invalid query: fts requires a valid FTS field, but field[",
          field_name, "] does not exist in the collection");
    }
    if (no_vector_payload) {
      // Scalar-only filter query
      return Status::OK();
    } else {
      // If a query vector was provided, the field must exist as a vector field
      // since we are performing a vector similarity search.
      return Status::InvalidArgument(
          "Invalid query: query vector is provided, but query field[",
          field_name,
          "] does not exist or is not a vector field in the collection");
    }
  }

  // FTS query: field must be an FTS-indexed field.
  if (fc != nullptr) {
    if (schema->index_type() != IndexType::FTS) {
      return Status::InvalidArgument(
          "Invalid query: fts requires an FTS-indexed field, but field[",
          field_name, "] has index type ",
          IndexTypeCodeBook::AsString(schema->index_type()));
    }
    if (query_params && query_params->type() != schema->index_type()) {
      return Status::InvalidArgument(
          "Invalid query: query params type does not match the index type of "
          "FTS field[",
          field_name, "], expected ",
          IndexTypeCodeBook::AsString(schema->index_type()), " but got ",
          IndexTypeCodeBook::AsString(query_params->type()));
    }
    return Status::OK();
  }

  // Schema is non-null from here on: a vector payload is required.
  if (no_vector_payload) {
    return Status::InvalidArgument(
        "Invalid query: missing query clause for field[", field_name, "]");
  }

  auto &query_vector = vc->query_vector_;
  auto &query_sparse_indices = vc->sparse_indices_;
  auto &query_sparse_values = vc->sparse_values_;

  // Vector query
  if (schema->is_dense_vector()) {
    // Validate dimension
    auto dim = schema->dimension();
    switch (schema->data_type()) {
      case DataType::VECTOR_FP16:
        if (dim * sizeof(float16_t) != query_vector.size()) {
          return Status::InvalidArgument(
              "Invalid query: dimension mismatch, expected ", dim, " but got ",
              query_vector.size() / sizeof(float16_t), " (FP16)");
        }
        break;
      case DataType::VECTOR_FP32:
        if (dim * sizeof(float) != query_vector.size()) {
          return Status::InvalidArgument(
              "Invalid query: dimension mismatch, expected ", dim, " but got ",
              query_vector.size() / sizeof(float), " (FP32)");
        }
        break;
      case DataType::VECTOR_FP64:
        if (dim * sizeof(double) != query_vector.size()) {
          return Status::InvalidArgument(
              "Invalid query: dimension mismatch, expected ", dim, " but got ",
              query_vector.size() / sizeof(double), " (FP64)");
        }
        break;
      case DataType::VECTOR_INT8:
        if (dim * sizeof(int8_t) != query_vector.size()) {
          return Status::InvalidArgument(
              "Invalid query: dimension mismatch, expected ", dim, " but got ",
              query_vector.size() / sizeof(int8_t), " (INT8)");
        }
        break;
      case DataType::VECTOR_INT16:
      case DataType::VECTOR_INT4:
      case DataType::VECTOR_BINARY32:
      case DataType::VECTOR_BINARY64:
        return Status::NotSupported(
            "Invalid query: dense vector type of field[", field_name,
            "] is not supported");
      default:
        return Status::InvalidArgument("Invalid query: field[", field_name,
                                       "] is not a dense vector field");
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
            "Invalid query: sparse vector type of field[", field_name,
            "] is not supported");
    }
    if (query_sparse_indices.size() % sizeof(uint32_t) != 0 ||
        query_sparse_values.size() % value_byte_size != 0 ||
        query_sparse_indices.size() / sizeof(uint32_t) !=
            query_sparse_values.size() / value_byte_size) {
      return Status::InvalidArgument(
          "Invalid query: sparse vector query for field[", field_name,
          "] has mismatched indices and values sizes");
    }
    size_t n_indices = query_sparse_indices.size() / sizeof(uint32_t);
    if (n_indices > kSparseMaxDimSize) {
      return Status::InvalidArgument(
          "Invalid query: too many sparse indices, the maximum allowed is ",
          kSparseMaxDimSize);
    }
    if (sort_and_find_duplicates(
            reinterpret_cast<uint32_t *>(query_sparse_indices.data()),
            query_sparse_values.data(), n_indices, value_byte_size)) {
      return Status::InvalidArgument(
          "Invalid query: sparse vector query for field[", field_name,
          "] contains duplicate indices");
    }
  } else {
    return Status::InvalidArgument("Invalid query: field[", field_name,
                                   "] is not a vector field");
  }
  // Validate query_params type
  if (query_params && query_params->type() != schema->index_type()) {
    return Status::InvalidArgument(
        "Invalid query: query params type does not match the index type of "
        "vector field[",
        field_name, "], expected ",
        IndexTypeCodeBook::AsString(schema->index_type()), " but got ",
        IndexTypeCodeBook::AsString(query_params->type()));
  }
  return Status::OK();
}

}  // namespace zvec
