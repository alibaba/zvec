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

#include <functional>
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

  struct FtsSuccessor {
    Segment::Ptr segment;
    SegmentID next_segment_id;
  };
  using PrepareFtsSuccessor =
      std::function<Result<FtsSuccessor>(uint64_t min_doc_id)>;

  // Caller holds collection write/schema exclusion for this whole operation.
  // Prepares every resource, commits one independent version, then publishes
  // the sealed reader, manager contents and writing pointer without allocation.
  Status switch_fts_writer(Segment::Ptr *writing,
                           const VersionManager::Ptr &versions,
                           const PrepareFtsSuccessor &prepare_successor,
                           const DurableFile::Hook &hook = {});

  class PreparedUpdate {
    friend class SegmentManager;
    std::unordered_map<SegmentID, Segment::Ptr> segments_;
    std::vector<Segment::Ptr> retired_;
    std::vector<fts::FtsFileOwner::Pointer> retired_files_;
  };

  // Collection write/schema exclusion spans preparation through publication.
  // Replacements with the same ID share a directory; only removed IDs retire
  // their directories. All map allocation and validation happens up front.
  Result<std::unique_ptr<PreparedUpdate>> prepare_update(
      const std::vector<Segment::Ptr> &additions,
      const std::vector<SegmentID> &removals) const;
  void publish_update(std::unique_ptr<PreparedUpdate> update);

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
  // protects segments_map_ against concurrent readers (e.g. queries
  // collecting segments) while segments are added/removed by writes,
  // optimize or DDL operations
  mutable std::shared_mutex mutex_;
  std::unordered_map<SegmentID, Segment::Ptr> segments_map_;
};
}  // namespace zvec
