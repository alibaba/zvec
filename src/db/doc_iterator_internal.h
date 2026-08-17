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

// RAII guard that decrements the collection's active-iterator count when the
// iterator releases its snapshot.
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
  // Declaration order controls destruction order (reverse of declaration):
  // schema_lock is declared FIRST so it is released last, and current_reader
  // LAST so Arrow file handles are released before the kept-alive segments
  // are destroyed (important on Windows). The collection must outlive the
  // iterator (it owns schema_lock's mutex).
  ActiveIteratorGuard active_guard;
  std::shared_lock<std::shared_mutex> schema_lock;

  std::vector<Segment::Ptr> segments;  // keep Segment alive
  DeleteStore::Ptr delete_store;       // keep delete bitmap alive
  CollectionSchema::Ptr schema;
  // Kept so segment readers can be opened lazily, one segment at a time.
  std::vector<std::string> iterator_columns;
  IndexFilter::Ptr filter;
  size_t current_segment_index{0};
  bool include_vector{false};
  // Column indices resolved once per segment reader by ResolveReaderColumns.
  int uid_col{-1};
  int gdoc_col{-1};
  int row_id_col{-1};
  std::vector<std::pair<const FieldSchema *, int>> forward_cols;
  // Batch currently being materialized. A Parquet scan returns a whole row
  // group per ReadNext (up to ~1M rows), so docs are materialized from it in
  // bounded windows of at most kMaxRecordBatchNumRows rows.
  std::shared_ptr<arrow::RecordBatch> current_batch;
  int64_t batch_offset{0};  // first row of the next window in current_batch
  std::vector<Doc::Ptr> batch_docs;
  size_t current_row{0};
  // First failure; sticky so a caller ignoring it cannot keep iterating over
  // a partially filled window.
  Status error{Status::OK()};
  // Reader for the current segment only (opened lazily, at most one open).
  RecordBatchReaderPtr current_reader;
};

}  // namespace zvec
