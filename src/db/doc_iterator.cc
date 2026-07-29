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
#include <zvec/db/doc_iterator.h>
#include "db/common/constants.h"
#include "db/doc_iterator_internal.h"

namespace zvec {

// ── VectorDataBuffer → Doc field conversion ──
// Same logic as SegmentImpl::ConvertVectorDataBufferToDocField, but as a free
// function to avoid modifying Segment's interface.
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
    // Release the Arrow readers/batch first (they reference segment files),
    // then release the kept-alive snapshot resources so a closed iterator
    // retains nothing.
    impl_->readers.clear();
    impl_->current_batch.reset();
    impl_->vector_cache_.clear();
    impl_->segments.clear();
    impl_->delete_store.reset();
    impl_->schema.reset();
  }
}

Result<Doc::Ptr> DocIterator::Next() {
  if (!impl_ || impl_->closed) {
    return tl::make_unexpected(Status::InternalError("Iterator is closed"));
  }

  // Load a new batch if the current one is exhausted, advancing across
  // segments. Each segment has its own reader, so the batch's owning segment
  // is always known directly via current_segment_index (no doc-id search).
  if (!impl_->current_batch ||
      impl_->current_row >= impl_->current_batch->num_rows()) {
    bool loaded = false;
    while (impl_->current_segment_index < impl_->readers.size()) {
      auto &reader = impl_->readers[impl_->current_segment_index];
      auto status = reader->ReadNext(&impl_->current_batch);
      if (!status.ok()) {
        return tl::make_unexpected(
            Status::InternalError("ReadNext failed: ", status.ToString()));
      }
      if (impl_->current_batch) {
        loaded = true;
        break;
      }
      // Current segment exhausted; move to the next one.
      impl_->current_segment_index++;
    }
    if (!loaded) {
      return Doc::Ptr(nullptr);  // EOF: all segments consumed
    }
    impl_->current_row = 0;

    // Pre-fetch vectors for the new batch from its owning segment.
    impl_->vector_cache_.clear();
    if (impl_->include_vector && impl_->schema) {
      auto &batch = *impl_->current_batch;
      // Segment-local row ids emitted by Segment::scan (requested in
      // build_scan_columns). Unlike g_doc_id arithmetic, they stay valid for
      // compacted segments whose g_doc_ids are non-contiguous.
      int row_id_col = batch.schema()->GetFieldIndex(LOCAL_ROW_ID);
      auto row_ids = row_id_col >= 0
                         ? std::dynamic_pointer_cast<arrow::UInt64Array>(
                               batch.column(row_id_col))
                         : nullptr;
      if (!row_ids) {
        return tl::make_unexpected(Status::InternalError(
            "Iterator batch is missing the segment row-id column"));
      }
      const auto &seg = impl_->segments[impl_->current_segment_index];
      for (const auto &field : impl_->schema->vector_fields()) {
        auto indexer = seg->get_combined_vector_indexer(field->name());
        if (!indexer) {
          // A declared vector field must have an indexer; report the internal
          // inconsistency instead of silently omitting the vector.
          return tl::make_unexpected(Status::InternalError(
              "vector indexer missing for field: ", field->name()));
        }
        std::vector<vector_column_params::VectorDataBuffer> bufs;
        bufs.reserve(batch.num_rows());
        for (int64_t i = 0; i < batch.num_rows(); i++) {
          uint32_t seg_doc_id = static_cast<uint32_t>(row_ids->Value(i));
          auto fr = indexer->Fetch(seg_doc_id);
          if (!fr.has_value()) {
            // Propagate the failure instead of returning a doc with a
            // silently missing vector.
            return tl::make_unexpected(Status::InternalError(
                "vector fetch failed, field: ", field->name(), ": ",
                fr.error().message()));
          }
          bufs.push_back(std::move(fr.value()));
        }
        impl_->vector_cache_[field->name()] = std::move(bufs);
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
      // GetView avoids the per-row Scalar allocation of GetScalar()->ToString()
      doc->set_pk(std::string(uid_array->GetView(row)));
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
            // GetView avoids a per-row Scalar allocation in the hot path
            doc->set(field->name(), std::string(str_array->GetView(row)));
          }
          break;
        }
        case DataType::INT32: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::Int32Array>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::INT64: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::Int64Array>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::UINT32: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::UInt32Array>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::UINT64: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::UInt64Array>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::FLOAT: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::FloatArray>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::DOUBLE: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::DoubleArray>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
          break;
        }
        case DataType::BOOL: {
          auto typed_array =
              std::dynamic_pointer_cast<arrow::BooleanArray>(array);
          if (typed_array) doc->set(field->name(), typed_array->Value(row));
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

  // 4. Extract vector fields from the pre-fetched cache
  if (impl_->include_vector && impl_->schema) {
    for (const auto &field : impl_->schema->vector_fields()) {
      auto it = impl_->vector_cache_.find(field->name());
      if (it == impl_->vector_cache_.end()) continue;
      if (row >= static_cast<int64_t>(it->second.size())) continue;

      auto s = SetVectorFieldFromBuffer(field, it->second[row], doc.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    }
  }

  impl_->current_row++;
  return doc;
}

}  // namespace zvec
