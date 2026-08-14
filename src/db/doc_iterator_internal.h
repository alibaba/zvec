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

// Internal header — NOT in public includes (src/include/)
// Shared by collection.cc and doc_iterator.cc
#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>
#include <zvec/db/collection.h>
#include <zvec/db/doc_iterator.h>
#include <zvec/db/schema.h>
#include "db/index/common/delete_store.h"
#include "db/index/common/index_filter.h"
#include "db/index/segment/segment.h"
#include "db/index/storage/base_forward_store.h"

namespace zvec {

// RAII guard that decrements the collection's active-iterator count when
// the iterator releases its snapshot.
struct ActiveIteratorGuard {
  std::shared_ptr<std::atomic<int>> count;

  ActiveIteratorGuard() = default;
  ActiveIteratorGuard(const ActiveIteratorGuard &) = delete;
  ActiveIteratorGuard &operator=(const ActiveIteratorGuard &) = delete;
  ~ActiveIteratorGuard() {
    if (count) {
      count->fetch_sub(1, std::memory_order_release);
    }
  }
};

struct DocIterator::Impl {
  // Declaration order controls destruction order (reverse of declaration).
  // collection must be declared FIRST: it owns the mutex behind schema_lock,
  // so it must be destroyed last. current_reader must be declared LAST so
  // Arrow file handles are released before Segment::cleanup() deletes files
  // from disk (important on Windows).
  //
  // The iterator keeps the collection alive and holds its schema lock
  // (shared) for its whole lifetime: schema changes (create/drop index,
  // add/alter/drop column), Optimize and Close/Destroy are rejected while
  // any iterator is open, so the snapshot below stays valid until Close().
  std::shared_ptr<Collection> collection;
  ActiveIteratorGuard active_guard;
  std::shared_lock<std::shared_mutex> schema_lock;

  std::vector<Segment::Ptr> segments;  // keep Segment alive
  DeleteStore::Ptr delete_store;       // keep delete bitmap alive
  CollectionSchema::Ptr schema;
  // Columns to scan and the delete filter, kept so readers can be opened
  // lazily (one segment at a time) instead of all upfront.
  std::vector<std::string> scan_columns;
  IndexFilter::Ptr filter;
  // Index of the segment currently being read.
  size_t current_segment_index{0};
  bool include_vector{false};  // whether to fetch vector fields
  // Column indices resolved once per segment reader (the reader schema is
  // stable across its batches), so materialization skips per-batch lookups.
  int uid_col{-1};
  int gdoc_col{-1};
  int row_id_col{-1};
  std::vector<std::pair<const FieldSchema *, int>> forward_cols;
  // Arrow batch currently being materialized. Memory/IPC stores emit small
  // batches, but a Parquet scan returns a whole row group per ReadNext (up
  // to ~1M rows), so docs are materialized from it in bounded windows.
  std::shared_ptr<arrow::RecordBatch> current_batch;
  // First row of the next materialization window within current_batch.
  int64_t batch_offset{0};
  // Docs of the current window, materialized column by column; Next() hands
  // them out one at a time. Peak doc memory stays bounded by one window
  // (at most kMaxRecordBatchNumRows rows), regardless of the batch size the
  // underlying store produces.
  std::vector<Doc::Ptr> batch_docs;
  size_t current_row{0};
  // First materialization failure; sticky so a caller that ignores the
  // error cannot keep iterating over a partially filled batch.
  Status error{Status::OK()};
  // Reader for the current segment only (opened lazily, released when the
  // segment is exhausted, so at most one reader is open at any time).
  RecordBatchReaderPtr current_reader;
};

}  // namespace zvec
