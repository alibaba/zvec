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
    return segments_map_.size();
  }

  Status add_segment(Segment::Ptr segment);

  Status remove_segment(SegmentID segment_id);

  Status destroy_segment(SegmentID segment_id);

  std::vector<Segment::Ptr> get_segments() const;

  std::vector<SegmentMeta::Ptr> get_segments_meta() const;

  Status add_column(const FieldSchema::Ptr &column_schema,
                    const std::string &expression, int concurrency);

  Status alter_column(const std::string &column_name,
                      const FieldSchema::Ptr &new_column_schema,
                      int concurrency);

  Status drop_column(const std::string &column_name);

 private:
  std::unordered_map<SegmentID, Segment::Ptr> segments_map_;
};
}  // namespace zvec