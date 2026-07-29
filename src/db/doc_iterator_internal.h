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
#include <vector>
#include <zvec/db/doc_iterator.h>
#include "db/index/column/vector_column/combined_vector_column_indexer.h"
#include "db/index/column/vector_column/vector_column_params.h"
#include "db/index/common/delete_store.h"
#include "db/index/segment/segment.h"
#include "db/index/storage/base_forward_store.h"

namespace zvec {

struct DocIterator::Impl {
  // Declaration order controls destruction order (reverse of declaration).
  // segments must be declared FIRST → destroyed LAST.
  // readers must be declared LAST → destroyed FIRST.
  // This ensures Arrow file handles are released before Segment::cleanup()
  // deletes files from disk (important on Windows).
  std::vector<Segment::Ptr> segments;  // keep Segment alive
  DeleteStore::Ptr delete_store;       // keep delete bitmap alive
  CollectionSchema::Ptr schema;
  // Index of the segment currently being read (segments/readers are parallel:
  // readers[i] was built from segments[i]).
  size_t current_segment_index{0};
  int64_t current_row{0};
  bool closed{false};
  bool include_vector{false};  // whether to fetch vector fields
  std::shared_ptr<arrow::RecordBatch> current_batch;
  // Pre-fetched vector data for the current batch.
  // Key: field_name, Value: one VectorDataBuffer per row in current_batch.
  std::unordered_map<std::string,
                     std::vector<vector_column_params::VectorDataBuffer>>
      vector_cache_;
  // One reader per segment (parallel to `segments`); destroyed before segments
  std::vector<RecordBatchReaderPtr> readers;
};

}  // namespace zvec
