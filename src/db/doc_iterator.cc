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

#include <arrow/api.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/doc_iterator.h>
#include "db/common/constants.h"
#include "db/doc_iterator_internal.h"

namespace zvec {

// ── VectorDataBuffer → Doc field conversion ──
// Same logic as SegmentImpl::ConvertVectorDataBufferToDocField
// (segment.cc:1024-1093) but as a free function to avoid modifying Segment's
// interface.
static Status SetVectorFieldFromBuffer(
    const FieldSchema::Ptr &field,
    const vector_column_params::VectorDataBuffer &buf, Doc *doc) {
  if (std::holds_alternative<vector_column_params::DenseVectorBuffer>(
          buf.vector_buffer)) {
    const auto &dense =
        std::get<vector_column_params::DenseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::VECTOR_FP32: {
        const auto *p = reinterpret_cast<const float *>(dense.data.data());
        size_t n = dense.data.size() / sizeof(float);
        doc->set(field->name(), std::vector<float>(p, p + n));
        break;
      }
      case DataType::VECTOR_FP16: {
        const auto *p = reinterpret_cast<const float16_t *>(dense.data.data());
        size_t n = dense.data.size() / sizeof(float16_t);
        doc->set(field->name(), std::vector<float16_t>(p, p + n));
        break;
      }
      case DataType::VECTOR_INT8: {
        const auto *p = reinterpret_cast<const int8_t *>(dense.data.data());
        size_t n = dense.data.size() / sizeof(int8_t);
        doc->set(field->name(), std::vector<int8_t>(p, p + n));
        break;
      }
      case DataType::VECTOR_FP64: {
        const auto *p = reinterpret_cast<const double *>(dense.data.data());
        size_t n = dense.data.size() / sizeof(double);
        doc->set(field->name(), std::vector<double>(p, p + n));
        break;
      }
      case DataType::VECTOR_INT16: {
        const auto *p = reinterpret_cast<const int16_t *>(dense.data.data());
        size_t n = dense.data.size() / sizeof(int16_t);
        doc->set(field->name(), std::vector<int16_t>(p, p + n));
        break;
      }
      default:
        return Status::InvalidArgument("Unsupported dense vector type: ",
                                       field->data_type());
    }
  } else if (std::holds_alternative<vector_column_params::SparseVectorBuffer>(
                 buf.vector_buffer)) {
    const auto &sparse =
        std::get<vector_column_params::SparseVectorBuffer>(buf.vector_buffer);
    switch (field->data_type()) {
      case DataType::SPARSE_VECTOR_FP32: {
        const auto *idx =
            reinterpret_cast<const uint32_t *>(sparse.indices.data());
        size_t idx_n = sparse.indices.size() / sizeof(uint32_t);
        const auto *val = reinterpret_cast<const float *>(sparse.values.data());
        size_t val_n = sparse.values.size() / sizeof(float);
        doc->set(field->name(),
                 std::make_pair(std::vector<uint32_t>(idx, idx + idx_n),
                                std::vector<float>(val, val + val_n)));
        break;
      }
      case DataType::SPARSE_VECTOR_FP16: {
        const auto *idx =
            reinterpret_cast<const uint32_t *>(sparse.indices.data());
        size_t idx_n = sparse.indices.size() / sizeof(uint32_t);
        const auto *val =
            reinterpret_cast<const float16_t *>(sparse.values.data());
        size_t val_n = sparse.values.size() / sizeof(float16_t);
        doc->set(field->name(),
                 std::make_pair(std::vector<uint32_t>(idx, idx + idx_n),
                                std::vector<float16_t>(val, val + val_n)));
        break;
      }
      default:
        return Status::InvalidArgument("Unsupported sparse vector type: ",
                                       field->data_type());
    }
  }
  return Status::OK();
}

// ── Extract a numeric list field
// (ARRAY_INT32/INT64/UINT32/UINT64/FLOAT/DOUBLE) from row `row` of a ListArray.
// Numeric arrays are contiguous, so raw_values() (already offset-adjusted for
// the sliced sub-array) can be copied directly.
template <typename ArrowArrayT, typename T>
static void SetNumericListField(
    const std::shared_ptr<arrow::ListArray> &list_array, int64_t row,
    const std::string &name, Doc *doc) {
  auto values =
      std::dynamic_pointer_cast<ArrowArrayT>(list_array->value_slice(row));
  if (values) {
    doc->set(name, std::vector<T>(values->raw_values(),
                                  values->raw_values() + values->length()));
  }
}

// ── DocIterator implementation ──

DocIterator::DocIterator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DocIterator::~DocIterator() {
  Close();
}

void DocIterator::Close() {
  if (impl_) {
    impl_->closed = true;
    impl_->reader.reset();
    impl_->current_batch.reset();
    impl_->vector_cache_.clear();
  }
}

Result<Doc::Ptr> DocIterator::Next() {
  if (!impl_ || impl_->closed) {
    return tl::make_unexpected(Status::InternalError("Iterator is closed"));
  }

  // If current batch exhausted or not loaded, read next batch
  if (!impl_->current_batch ||
      impl_->current_row >= impl_->current_batch->num_rows()) {
    auto status = impl_->reader->ReadNext(&impl_->current_batch);
    if (!status.ok()) {
      return tl::make_unexpected(
          Status::InternalError("ReadNext failed: ", status.ToString()));
    }
    impl_->current_row = 0;
    if (!impl_->current_batch) {
      return Doc::Ptr(nullptr);  // EOF
    }
    // Pre-fetch vectors for the new batch (batch-level, not per-doc)
    impl_->vector_cache_.clear();
    if (impl_->include_vector && impl_->schema) {
      auto &batch = *impl_->current_batch;
      int gc = batch.schema()->GetFieldIndex(GLOBAL_DOC_ID);
      if (gc >= 0) {
        auto ga =
            std::dynamic_pointer_cast<arrow::UInt64Array>(batch.column(gc));
        if (ga && batch.num_rows() > 0) {
          uint64_t first_gdoc = ga->Value(0);
          for (const auto &seg : impl_->segments) {
            if (first_gdoc >= seg->meta()->min_doc_id() &&
                first_gdoc <= seg->meta()->max_doc_id()) {
              uint64_t min_doc_id = seg->meta()->min_doc_id();
              for (const auto &field : impl_->schema->vector_fields()) {
                auto indexer = seg->get_combined_vector_indexer(field->name());
                if (!indexer) continue;
                std::vector<vector_column_params::VectorDataBuffer> bufs;
                bufs.reserve(batch.num_rows());
                for (int64_t i = 0; i < batch.num_rows(); i++) {
                  uint32_t seg_doc_id =
                      static_cast<uint32_t>(ga->Value(i) - min_doc_id);
                  auto fr = indexer->Fetch(seg_doc_id);
                  if (fr.has_value()) {
                    bufs.push_back(std::move(fr.value()));
                  } else {
                    bufs.emplace_back();
                  }
                }
                impl_->vector_cache_[field->name()] = std::move(bufs);
              }
              break;
            }
          }
        }
      }
    }
  }

  auto &batch = *impl_->current_batch;
  int64_t row = impl_->current_row;
  auto doc = std::make_shared<Doc>();

  // 1. Extract PK from _zvec_uid_ column
  int uid_col = batch.schema()->GetFieldIndex(USER_ID);
  if (uid_col >= 0) {
    auto uid_array =
        std::dynamic_pointer_cast<arrow::StringArray>(batch.column(uid_col));
    if (uid_array) {
      doc->set_pk(uid_array->GetScalar(row).ValueOrDie()->ToString());
    }
  }

  // 2. Extract doc_id from _zvec_g_doc_id_ column
  int gdoc_col = batch.schema()->GetFieldIndex(GLOBAL_DOC_ID);
  if (gdoc_col >= 0) {
    auto gdoc_array =
        std::dynamic_pointer_cast<arrow::UInt64Array>(batch.column(gdoc_col));
    if (gdoc_array) {
      doc->set_doc_id(gdoc_array->Value(row));
    }
  }

  // 3. Extract scalar fields from Arrow batch
  if (impl_->schema) {
    for (const auto &field : impl_->schema->forward_fields()) {
      int col = batch.schema()->GetFieldIndex(field->name());
      if (col < 0) continue;

      auto array = batch.column(col);
      if (array->IsNull(row)) continue;

      switch (field->data_type()) {
        case DataType::STRING: {
          auto str_array = std::dynamic_pointer_cast<arrow::StringArray>(array);
          if (str_array) {
            doc->set(field->name(),
                     str_array->GetScalar(row).ValueOrDie()->ToString());
          }
          break;
        }
        case DataType::INT32: {
          auto a = std::dynamic_pointer_cast<arrow::Int32Array>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::INT64: {
          auto a = std::dynamic_pointer_cast<arrow::Int64Array>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::UINT32: {
          auto a = std::dynamic_pointer_cast<arrow::UInt32Array>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::UINT64: {
          auto a = std::dynamic_pointer_cast<arrow::UInt64Array>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::FLOAT: {
          auto a = std::dynamic_pointer_cast<arrow::FloatArray>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::DOUBLE: {
          auto a = std::dynamic_pointer_cast<arrow::DoubleArray>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::BOOL: {
          auto a = std::dynamic_pointer_cast<arrow::BooleanArray>(array);
          if (a) doc->set(field->name(), a->Value(row));
          break;
        }
        case DataType::ARRAY_INT32: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::Int32Array, int32_t>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_INT64: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::Int64Array, int64_t>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_UINT32: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::UInt32Array, uint32_t>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_UINT64: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::UInt64Array, uint64_t>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_FLOAT: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::FloatArray, float>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_DOUBLE: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la)
            SetNumericListField<arrow::DoubleArray, double>(
                la, row, field->name(), doc.get());
          break;
        }
        case DataType::ARRAY_BOOL: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la) {
            auto vals = std::dynamic_pointer_cast<arrow::BooleanArray>(
                la->value_slice(row));
            if (vals) {
              std::vector<bool> vec;
              vec.reserve(vals->length());
              for (int64_t i = 0; i < vals->length(); ++i) {
                vec.push_back(vals->Value(i));
              }
              doc->set(field->name(), vec);
            }
          }
          break;
        }
        case DataType::ARRAY_STRING: {
          auto la = std::dynamic_pointer_cast<arrow::ListArray>(array);
          if (la) {
            auto vals = std::dynamic_pointer_cast<arrow::StringArray>(
                la->value_slice(row));
            if (vals) {
              std::vector<std::string> vec;
              vec.reserve(vals->length());
              for (int64_t i = 0; i < vals->length(); ++i) {
                vec.push_back(vals->GetString(i));
              }
              doc->set(field->name(), vec);
            }
          }
          break;
        }
        default:
          break;
      }
    }
  }

  // 4. Extract vector fields from pre-fetched cache
  if (impl_->include_vector && impl_->schema) {
    for (const auto &field : impl_->schema->vector_fields()) {
      auto it = impl_->vector_cache_.find(field->name());
      if (it == impl_->vector_cache_.end()) continue;
      if (row >= static_cast<int64_t>(it->second.size())) continue;

      auto s = SetVectorFieldFromBuffer(field, it->second[row], doc.get());
      if (!s.ok()) {
        LOG_ERROR("SetVectorFieldFromBuffer failed for %s: %s",
                  field->name().c_str(), s.message().c_str());
      }
    }
  }

  impl_->current_row++;
  return doc;
}

}  // namespace zvec
