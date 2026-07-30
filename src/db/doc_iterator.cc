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
#include "db/index/common/doc_field_converter.h"

namespace zvec {

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

  // 3. Extract scalar/array fields from the Arrow batch via the shared
  //    row-level converter (same type coverage and null semantics as the
  //    SQL engine).
  if (impl_->schema) {
    for (const auto &field : impl_->schema->forward_fields()) {
      int col = batch.schema()->GetFieldIndex(field->name());
      if (col < 0) continue;

      auto s =
          ConvertArrowRowToDocField(batch.column(col), row, *field, doc.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    }
  }

  // 4. Extract vector fields from the pre-fetched cache
  if (impl_->include_vector && impl_->schema) {
    for (const auto &field : impl_->schema->vector_fields()) {
      auto it = impl_->vector_cache_.find(field->name());
      if (it == impl_->vector_cache_.end()) continue;
      if (row >= static_cast<int64_t>(it->second.size())) continue;

      auto s =
          ConvertVectorDataBufferToDocField(field, it->second[row], doc.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    }
  }

  impl_->current_row++;
  return doc;
}

}  // namespace zvec
