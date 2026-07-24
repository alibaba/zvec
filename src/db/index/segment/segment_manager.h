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
#pragma once

#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include "segment.h"

namespace zvec {
class SegmentManager {
 public:
  using Ptr = std::shared_ptr<SegmentManager>;

  SegmentManager() = default;
  ~SegmentManager() = default;

 public:
  uint32_t segment_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return segments_map_.size();
  }

  Status add_segment(Segment::Ptr segment);

  Status remove_segment(SegmentID segment_id);

  //! Atomically add (or replace by id) segments and destroy others, so that
  //! concurrent readers never observe a partially committed segment set.
  //! Segments replaced by id are dropped without destroy(): their files are
  //! shared with the replacing instance and in-flight readers keep the old
  //! instance alive via shared_ptr until they finish.
  Status replace_segments(const std::vector<Segment::Ptr> &segments_to_add,
                          const std::vector<SegmentID> &segment_ids_to_destroy);

  std::vector<Segment::Ptr> get_segments() const;

  std::vector<SegmentMeta::Ptr> get_segments_meta() const;

  Status add_column(const FieldSchema::Ptr &column_schema,
                    const std::string &expression, int concurrency);

  Status alter_column(const std::string &column_name,
                      const FieldSchema::Ptr &new_column_schema,
                      int concurrency);

  Status drop_column(const std::string &column_name);

 private:
  // protects segments_map_ against concurrent readers (e.g. queries
  // collecting segments) while segments are added/removed by writes,
  // optimize or DDL operations
  mutable std::shared_mutex mutex_;
  std::unordered_map<SegmentID, Segment::Ptr> segments_map_;
};
}  // namespace zvec