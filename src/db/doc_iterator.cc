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

// Resolve and validate column indices from the reader's schema, once per
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

// Materialize rows [begin, end) of the current batch column by column. The
// window (not the batch) bounds doc memory: a Parquet scan returns a whole
// row group per ReadNext (up to ~1M rows), so materializing a full batch at
// once would spike memory.
Status MaterializeWindow(DocIterator::Impl *impl, int64_t begin, int64_t end) {
  const auto &batch = *impl->current_batch;
  int64_t num_rows = end - begin;
  impl->batch_docs.clear();
  impl->batch_docs.reserve(num_rows);
  for (int64_t i = 0; i < num_rows; i++) {
    impl->batch_docs.push_back(std::make_shared<Doc>());
  }

  const auto *uids = static_cast<const arrow::StringArray *>(
      batch.columns()[impl->uid_col].get());
  const auto *gdocs = static_cast<const arrow::UInt64Array *>(
      batch.columns()[impl->gdoc_col].get());
  for (int64_t i = 0; i < num_rows; i++) {
    impl->batch_docs[i]->set_pk(std::string(uids->GetView(begin + i)));
    impl->batch_docs[i]->set_doc_id(gdocs->Value(begin + i));
  }

  for (const auto &[field, col] : impl->forward_cols) {
    auto s = ConvertArrowColumnToDocFields(
        batch.columns()[col]->Slice(begin, num_rows).get(), *field,
        impl->batch_docs.begin());
    if (!s.ok()) {
      return s;
    }
  }

  if (impl->row_id_col >= 0) {
    const auto *row_ids = static_cast<const arrow::UInt64Array *>(
        batch.columns()[impl->row_id_col].get());
    const auto &seg = impl->segments[impl->current_segment_index];
    for (const auto &field : impl->schema->vector_fields()) {
      auto indexer = seg->get_combined_vector_indexer(field->name());
      if (!indexer) {
        return Status::InternalError("vector indexer missing for field: ",
                                     field->name());
      }
      for (int64_t i = 0; i < num_rows; i++) {
        auto fetched =
            indexer->Fetch(static_cast<uint32_t>(row_ids->Value(begin + i)));
        if (!fetched.has_value()) {
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

DocIterator::DocIterator(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DocIterator::~DocIterator() {
  Close();
}

void DocIterator::Close() {
  if (!impl_) {
    return;
  }
  // Release the active-iterator slot first (takes the collection's
  // exclusive schema lock), then tear down resources. Member declaration
  // order guarantees safe teardown of the rest: current_reader is declared
  // last, so Arrow file handles are released before the kept-alive segments
  // are destroyed.
  if (impl_->release_slot) {
    impl_->release_slot();
  }
  impl_.reset();
}

Result<Doc::Ptr> DocIterator::Next() {
  if (!impl_) {
    return tl::make_unexpected(Status::InternalError("Iterator is closed"));
  }
  if (!impl_->error.ok()) {
    return tl::make_unexpected(impl_->error);
  }

  // Materialize a new window when the current one is exhausted. Readers are
  // opened lazily, at most one segment's reader at a time.
  while (impl_->current_row >= impl_->batch_docs.size()) {
    if (!impl_->current_batch ||
        impl_->batch_offset >= impl_->current_batch->num_rows()) {
      impl_->current_batch.reset();
      bool loaded = false;
      while (impl_->current_segment_index < impl_->segments.size()) {
        if (!impl_->current_reader) {
          auto scalar_reader =
              impl_->segments[impl_->current_segment_index]->scan(
                  impl_->iterator_columns);
          if (!scalar_reader) {
            impl_->error =
                Status::InternalError("Segment::scan failed during iteration");
            return tl::make_unexpected(impl_->error);
          }
          impl_->current_reader =
              FilteringReader::Make(std::move(scalar_reader), impl_->filter);
          auto rs = ResolveReaderColumns(impl_.get(),
                                         *impl_->current_reader->schema());
          if (!rs.ok()) {
            return tl::make_unexpected(rs);
          }
        }
        std::shared_ptr<arrow::RecordBatch> batch;
        auto status = impl_->current_reader->ReadNext(&batch);
        if (!status.ok()) {
          impl_->error =
              Status::InternalError("ReadNext failed: ", status.ToString());
          return tl::make_unexpected(impl_->error);
        }
        if (batch && batch->num_rows() > 0) {
          impl_->current_batch = std::move(batch);
          impl_->batch_offset = 0;
          loaded = true;
          break;
        }
        if (!batch) {
          // EOF for this segment: release the reader and advance.
          impl_->current_reader.reset();
          impl_->current_segment_index++;
        }
      }
      if (!loaded) {
        return Doc::Ptr(nullptr);  // all segments consumed
      }
    }

    int64_t begin = impl_->batch_offset;
    int64_t end = std::min(begin + kMaxRecordBatchNumRows,
                           impl_->current_batch->num_rows());
    auto ms = MaterializeWindow(impl_.get(), begin, end);
    if (!ms.ok()) {
      // Sticky failure: drop the partial window and keep returning the error
      // instead of handing out incomplete docs.
      impl_->batch_docs.clear();
      impl_->error = ms;
      return tl::make_unexpected(ms);
    }
    impl_->batch_offset = end;
    if (end >= impl_->current_batch->num_rows()) {
      impl_->current_batch.reset();
    }
    impl_->current_row = 0;
  }

  return impl_->batch_docs[impl_->current_row++];
}

}  // namespace zvec
