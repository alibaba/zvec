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

#include <algorithm>
#include <arrow/api.h>
#include <zvec/db/doc_iterator.h>
#include "db/common/constants.h"
#include "db/doc_iterator_internal.h"
#include "db/index/common/doc_field_converter.h"
#include "db/index/segment/filtering_reader.h"

namespace zvec {

namespace {

// Prefetch vectors for batch rows [begin, end) of the current segment into
// impl->vector_cache_. The bounded window keeps peak memory O(window)
// regardless of the batch size produced by the underlying store.
Status PrefetchVectorWindow(DocIterator::Impl *impl, int64_t begin,
                            int64_t end) {
  auto &batch = *impl->current_batch;
  // Segment-local row ids emitted by Segment::scan (requested in
  // build_scan_columns); valid for compacted segments too.
  int row_id_col = batch.schema()->GetFieldIndex(LOCAL_ROW_ID);
  const arrow::UInt64Array *row_ids = nullptr;
  if (row_id_col >= 0) {
    const auto &col = batch.columns()[row_id_col];
    if (col->type_id() == arrow::Type::UINT64) {
      row_ids = static_cast<const arrow::UInt64Array *>(col.get());
    }
  }
  if (!row_ids) {
    return Status::InternalError(
        "Iterator batch is missing the segment row-id column");
  }
  const auto &seg = impl->segments[impl->current_segment_index];
  impl->vector_cache_.clear();
  for (const auto &field : impl->schema->vector_fields()) {
    auto indexer = seg->get_combined_vector_indexer(field->name());
    if (!indexer) {
      // A declared vector field must have an indexer; report the internal
      // inconsistency instead of silently omitting the vector.
      return Status::InternalError("vector indexer missing for field: ",
                                   field->name());
    }
    std::vector<vector_column_params::VectorDataBuffer> bufs;
    bufs.reserve(end - begin);
    for (int64_t i = begin; i < end; i++) {
      uint32_t seg_doc_id = static_cast<uint32_t>(row_ids->Value(i));
      auto fr = indexer->Fetch(seg_doc_id);
      if (!fr.has_value()) {
        // Propagate the failure instead of returning a doc with a
        // silently missing vector.
        return Status::InternalError("vector fetch failed, field: ",
                                     field->name(), ": ", fr.error().message());
      }
      bufs.push_back(std::move(fr.value()));
    }
    impl->vector_cache_[field->name()] = std::move(bufs);
  }
  impl->vector_window_start = begin;
  return Status::OK();
}

}  // namespace

// ── DocIterator implementation ──

DocIterator::DocIterator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DocIterator::~DocIterator() {
  Close();
}

void DocIterator::Close() {
  if (impl_) {
    impl_->closed = true;
    impl_->current_reader.reset();
    impl_->current_batch.reset();
    impl_->vector_cache_.clear();
    impl_->segments.clear();
    impl_->delete_store.reset();
    impl_->schema.reset();
    impl_.reset();
  }
}

Result<Doc::Ptr> DocIterator::Next() {
  if (!impl_ || impl_->closed) {
    return tl::make_unexpected(Status::InternalError("Iterator is closed"));
  }

  // Load a new batch if the current one is exhausted, advancing across
  // segments. Readers are opened lazily — at most one segment's reader is
  // open at any time, and it is released as soon as that segment is done.
  if (!impl_->current_batch ||
      impl_->current_row >= impl_->current_batch->num_rows()) {
    bool loaded = false;
    while (impl_->current_segment_index < impl_->segments.size()) {
      if (!impl_->current_reader) {
        auto scalar_reader =
            impl_->segments[impl_->current_segment_index]->scan(
                impl_->scan_columns);
        if (!scalar_reader) {
          return tl::make_unexpected(
              Status::InternalError("Segment::scan failed during iteration"));
        }
        impl_->current_reader =
            FilteringReader::Make(std::move(scalar_reader), impl_->filter);
      }
      auto status = impl_->current_reader->ReadNext(&impl_->current_batch);
      if (!status.ok()) {
        return tl::make_unexpected(
            Status::InternalError("ReadNext failed: ", status.ToString()));
      }
      if (impl_->current_batch && impl_->current_batch->num_rows() > 0) {
        loaded = true;
        break;
      }
      if (!impl_->current_batch) {
        // Reader signaled EOF: release it and move to the next segment.
        impl_->current_reader.reset();
        impl_->current_segment_index++;
      }
      // A non-null empty batch is simply skipped on the next ReadNext.
    }
    if (!loaded) {
      return Doc::Ptr(nullptr);  // EOF: all segments consumed
    }
    impl_->current_row = 0;
    impl_->vector_cache_.clear();
    impl_->vector_window_start = 0;

    // Resolve column indices once per loaded batch (the scan schema is stable
    // across batches), so per-row extraction avoids GetFieldIndex lookups.
    const auto &bschema = *impl_->current_batch->schema();
    impl_->uid_col = bschema.GetFieldIndex(USER_ID);
    impl_->gdoc_col = bschema.GetFieldIndex(GLOBAL_DOC_ID);
    impl_->forward_cols.clear();
    for (const auto &field : impl_->schema->forward_fields()) {
      int col = bschema.GetFieldIndex(field->name());
      if (col >= 0) {
        impl_->forward_cols.emplace_back(field.get(), col);
      }
    }
  }

  auto &batch = *impl_->current_batch;
  int64_t row = impl_->current_row;
  auto doc = std::make_shared<Doc>();

  // 1. Extract PK from _zvec_uid_ column.
  if (impl_->uid_col < 0) {
    return tl::make_unexpected(
        Status::InternalError("Iterator batch is missing the uid column"));
  }
  {
    const auto &col = batch.columns()[impl_->uid_col];
    if (col->type_id() != arrow::Type::STRING) {
      return tl::make_unexpected(Status::InternalError(
          "Iterator batch uid column is not a string array"));
    }
    doc->set_pk(std::string(
        static_cast<const arrow::StringArray *>(col.get())->GetView(row)));
  }

  // 2. Extract doc_id from _zvec_g_doc_id_ column
  if (impl_->gdoc_col < 0) {
    return tl::make_unexpected(Status::InternalError(
        "Iterator batch is missing the global doc id column"));
  }
  {
    const auto &col = batch.columns()[impl_->gdoc_col];
    if (col->type_id() != arrow::Type::UINT64) {
      return tl::make_unexpected(Status::InternalError(
          "Iterator batch global doc id column is not a UInt64 array"));
    }
    doc->set_doc_id(
        static_cast<const arrow::UInt64Array *>(col.get())->Value(row));
  }

  // 3. Extract scalar/array fields from the Arrow batch via the shared
  //    row-level converter (same type coverage and null semantics as the
  //    SQL engine).
  for (const auto &[field, col] : impl_->forward_cols) {
    auto s = ConvertArrowRowToDocField(batch.columns()[col].get(), row, *field,
                                       doc.get());
    if (!s.ok()) {
      return tl::make_unexpected(s);
    }
  }

  // 4. Extract vector fields from the prefetch cache. Rows are consumed in
  //    ascending order, so refilling the window when the row passes its end
  //    prefetches each vector exactly once while keeping the cache bounded.
  if (impl_->include_vector && impl_->schema &&
      !impl_->schema->vector_fields().empty()) {
    int64_t window_len =
        impl_->vector_cache_.empty()
            ? 0
            : static_cast<int64_t>(impl_->vector_cache_.begin()->second.size());
    if (row >= impl_->vector_window_start + window_len) {
      int64_t end =
          std::min(row + kIteratorVectorPrefetchWindow, batch.num_rows());
      auto ws = PrefetchVectorWindow(impl_.get(), row, end);
      if (!ws.ok()) {
        return tl::make_unexpected(ws);
      }
    }
    int64_t cache_row = row - impl_->vector_window_start;
    for (const auto &field : impl_->schema->vector_fields()) {
      auto it = impl_->vector_cache_.find(field->name());
      if (it == impl_->vector_cache_.end()) {
        return tl::make_unexpected(Status::InternalError(
            "vector cache missing for field: ", field->name()));
      }
      if (cache_row >= static_cast<int64_t>(it->second.size())) {
        return tl::make_unexpected(Status::InternalError(
            "vector cache row out of range for field: ", field->name()));
      }

      auto s = ConvertVectorDataBufferToDocField(field, it->second[cache_row],
                                                 doc.get());
      if (!s.ok()) {
        return tl::make_unexpected(s);
      }
    }
  }

  impl_->current_row++;
  return doc;
}

}  // namespace zvec
