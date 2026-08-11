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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <zvec/db/doc_iterator.h>
#include <zvec/db/schema.h>
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/index/common/delete_store.h"
#include "db/index/common/index_filter.h"
#include "db/index/segment/segment.h"
#include "db/index/storage/base_forward_store.h"

namespace zvec {

struct DocIterator::Impl {
  // Declaration order controls destruction order (reverse of declaration).
  // segments must be declared FIRST → destroyed LAST.
  // current_reader must be declared LAST → destroyed FIRST.
  // This ensures Arrow file handles are released before Segment::cleanup()
  // deletes files from disk (important on Windows).
  std::vector<Segment::Ptr> segments;  // keep Segment alive
  DeleteStore::Ptr delete_store;       // keep delete bitmap alive
  CollectionSchema::Ptr schema;
  // Columns to scan and the delete filter, kept so readers can be opened
  // lazily (one segment at a time) instead of all upfront.
  std::vector<std::string> scan_columns;
  IndexFilter::Ptr filter;
  // Index of the segment currently being read.
  size_t current_segment_index{0};
  int64_t current_row{0};
  bool include_vector{false};  // whether to fetch vector fields
  std::shared_ptr<arrow::RecordBatch> current_batch;
  // Column indices resolved once per loaded batch (the batch schema is stable
  // for the whole scan), so Next() avoids per-row GetFieldIndex lookups.
  int uid_col{-1};
  int gdoc_col{-1};
  std::vector<std::pair<const FieldSchema *, int>> forward_cols;
  // First batch row covered by vector_cache_ (prefetched in bounded windows).
  int64_t vector_window_start{0};
  // Pre-fetched vector data for the current window.
  // Key: field_name, Value: one VectorDataBuffer per row in the window.
  std::unordered_map<std::string,
                     std::vector<vector_column_params::VectorDataBuffer>>
      vector_cache_;
  // Reader for the current segment only (opened lazily, released when the
  // segment is exhausted, so at most one reader is open at any time).
  RecordBatchReaderPtr current_reader;
};

}  // namespace zvec
