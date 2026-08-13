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
#include "db/index/segment/filtering_reader.h"

namespace zvec {

namespace {

// Resolve and validate the column indices from the reader's schema, once per
// segment reader (the schema is stable across all its batches).
Status ResolveReaderColumns(DocIterator::Impl *impl,
                            const arrow::Schema &schema) {
  impl->uid_col = schema.GetFieldIndex(USER_ID);
  if (impl->uid_col < 0 ||
      schema.field(impl->uid_col)->type()->id() != arrow::Type::STRING) {
    return Status::InternalError(
        "Iterator reader is missing the string uid column");
  }
  impl->gdoc_col = schema.GetFieldIndex(GLOBAL_DOC_ID);
  if (impl->gdoc_col < 0 ||
      schema.field(impl->gdoc_col)->type()->id() != arrow::Type::UINT64) {
    return Status::InternalError(
        "Iterator reader is missing the uint64 global doc id column");
  }
  impl->row_id_col = -1;
  if (impl->include_vector && !impl->schema->vector_fields().empty()) {
    // Segment-local row ids emitted by Segment::scan (requested in
    // build_scan_columns); valid for compacted segments too.
    impl->row_id_col = schema.GetFieldIndex(LOCAL_ROW_ID);
    if (impl->row_id_col < 0 ||
        schema.field(impl->row_id_col)->type()->id() != arrow::Type::UINT64) {
      return Status::InternalError(
          "Iterator reader is missing the uint64 segment row-id column");
    }
  }
  impl->forward_cols.clear();
  for (const auto &field : impl->schema->forward_fields()) {
    int col = schema.GetFieldIndex(field->name());
    if (col >= 0) {
      impl->forward_cols.emplace_back(field.get(), col);
    }
  }
  return Status::OK();
}

// Materialize every row of `batch` into impl->batch_docs, column by column:
// the shared column-level converter dispatches the field type and checks
// null_count() once per column, and each vector field is fetched in one pass.
// Peak memory stays bounded by one batch of docs (the stores cap batches at
// kMaxRecordBatchNumRows rows).
Status MaterializeBatch(DocIterator::Impl *impl,
                        const std::shared_ptr<arrow::RecordBatch> &batch) {
  int64_t num_rows = batch->num_rows();
  impl->batch_docs.clear();
  impl->batch_docs.reserve(num_rows);
  for (int64_t i = 0; i < num_rows; i++) {
    impl->batch_docs.push_back(std::make_shared<Doc>());
  }

  // System columns (types validated in ResolveReaderColumns).
  const auto *uids = static_cast<const arrow::StringArray *>(
      batch->columns()[impl->uid_col].get());
  const auto *gdocs = static_cast<const arrow::UInt64Array *>(
      batch->columns()[impl->gdoc_col].get());
  for (int64_t i = 0; i < num_rows; i++) {
    impl->batch_docs[i]->set_pk(std::string(uids->GetView(i)));
    impl->batch_docs[i]->set_doc_id(gdocs->Value(i));
  }

  for (const auto &[field, col] : impl->forward_cols) {
    auto s = ConvertArrowColumnToDocFields(batch->columns()[col].get(), *field,
                                           impl->batch_docs.begin());
    if (!s.ok()) {
      return s;
    }
  }

  if (impl->row_id_col >= 0) {
    const auto *row_ids = static_cast<const arrow::UInt64Array *>(
        batch->columns()[impl->row_id_col].get());
    const auto &seg = impl->segments[impl->current_segment_index];
    for (const auto &field : impl->schema->vector_fields()) {
      auto indexer = seg->get_combined_vector_indexer(field->name());
      if (!indexer) {
        // A declared vector field must have an indexer; report the internal
        // inconsistency instead of silently omitting the vector.
        return Status::InternalError("vector indexer missing for field: ",
                                     field->name());
      }
      for (int64_t i = 0; i < num_rows; i++) {
        auto fetched = indexer->Fetch(static_cast<uint32_t>(row_ids->Value(i)));
        if (!fetched.has_value()) {
          // Propagate the failure instead of returning a doc with a
          // silently missing vector.
          return Status::InternalError(
              "vector fetch failed, field: ", field->name(), ": ",
              fetched.error().message());
        }
        auto s = ConvertVectorDataBufferToDocField(field, fetched.value(),
                                                   impl->batch_docs[i].get());
        if (!s.ok()) {
          return s;
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace

// ── DocIterator implementation ──

DocIterator::DocIterator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DocIterator::~DocIterator() {
  Close();
}

void DocIterator::Close() {
  // Impl's member declaration order alone guarantees a safe teardown
  // (current_reader is declared last, so it releases its file handles
  // before the kept-alive segments are destroyed).
  impl_.reset();
}

Result<Doc::Ptr> DocIterator::Next() {
  if (!impl_) {
    return tl::make_unexpected(Status::InternalError("Iterator is closed"));
  }
  if (!impl_->error.ok()) {
    return tl::make_unexpected(impl_->error);
  }

  // Materialize the next batch if the current one is exhausted, advancing
  // across segments. Readers are opened lazily — at most one segment's
  // reader is open at any time, and it is released as soon as that segment
  // is done.
  if (impl_->current_row >= impl_->batch_docs.size()) {
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
        auto rs =
            ResolveReaderColumns(impl_.get(), *impl_->current_reader->schema());
        if (!rs.ok()) {
          return tl::make_unexpected(rs);
        }
      }
      std::shared_ptr<arrow::RecordBatch> batch;
      auto status = impl_->current_reader->ReadNext(&batch);
      if (!status.ok()) {
        return tl::make_unexpected(
            Status::InternalError("ReadNext failed: ", status.ToString()));
      }
      if (batch && batch->num_rows() > 0) {
        auto ms = MaterializeBatch(impl_.get(), batch);
        if (!ms.ok()) {
          // Sticky failure: drop the partially filled batch and keep
          // returning the error instead of handing out incomplete docs.
          impl_->batch_docs.clear();
          impl_->error = ms;
          return tl::make_unexpected(ms);
        }
        loaded = true;
        break;
      }
      if (!batch) {
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
  }

  return impl_->batch_docs[impl_->current_row++];
}

}  // namespace zvec
