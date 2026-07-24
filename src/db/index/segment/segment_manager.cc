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

// segment_manager.cpp
#include "db/index/segment/segment_manager.h"
#include <algorithm>
#include <future>
#include <thread>
#include <vector>
#include <zvec/db/status.h>
#include "db/common/typedef.h"

namespace zvec {

Status SegmentManager::add_segment(Segment::Ptr segment) {
  if (!segment) {
    return Status::InvalidArgument("Segment is null");
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  segments_map_[segment->id()] = segment;
  return Status::OK();
}

Status SegmentManager::remove_segment(SegmentID segment_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto iter = segments_map_.find(segment_id);
  if (iter == segments_map_.end()) {
    return Status::NotFound("Segment not found");
  }

  segments_map_.erase(segment_id);
  return Status::OK();
}

Status SegmentManager::replace_segments(
    const std::vector<Segment::Ptr> &segments_to_add,
    const std::vector<SegmentID> &segment_ids_to_destroy) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // Phase 1: validate everything up front, so that a failure never leaves
  // a partially committed segment set behind.
  for (auto &segment : segments_to_add) {
    if (!segment) {
      return Status::InvalidArgument("Segment is null");
    }
  }
  for (auto segment_id : segment_ids_to_destroy) {
    auto iter = segments_map_.find(segment_id);
    if (iter == segments_map_.end()) {
      return Status::NotFound("Segment not found");
    }
  }

  // Phase 2: apply the whole swap under the single exclusive lock, so
  // concurrent get_segments() observe either the complete old set or the
  // complete new set, never a mix (e.g. merge inputs and their merged
  // output visible at the same time). destroy() cannot fail here: a
  // segment is marked and erased atomically under this lock, so a
  // map-resident segment is never already marked.
  for (auto segment_id : segment_ids_to_destroy) {
    auto iter = segments_map_.find(segment_id);
    auto s = iter->second->destroy();
    CHECK_RETURN_STATUS(s);
    segments_map_.erase(iter);
  }

  for (auto &segment : segments_to_add) {
    // overwrites any existing entry with the same id: the replaced
    // instance is dropped without destroy() so its shared files stay on
    // disk; in-flight readers keep it alive until they finish
    segments_map_[segment->id()] = segment;
  }

  return Status::OK();
}

std::vector<Segment::Ptr> SegmentManager::get_segments() const {
  std::vector<Segment::Ptr> segments;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto &pair : segments_map_) {
      segments.push_back(pair.second);
    }
  }
  std::sort(segments.begin(), segments.end(),
            [](Segment::Ptr a, Segment::Ptr b) {
              return a->meta()->min_doc_id() < b->meta()->min_doc_id();
            });
  return segments;
}

std::vector<SegmentMeta::Ptr> SegmentManager::get_segments_meta() const {
  std::vector<SegmentMeta::Ptr> segments_meta;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto &pair : segments_map_) {
      segments_meta.push_back(pair.second->meta());
    }
  }

  std::sort(segments_meta.begin(), segments_meta.end(),
            [](SegmentMeta::Ptr a, SegmentMeta::Ptr b) {
              return a->min_doc_id() < b->min_doc_id();
            });

  return segments_meta;
}

Status SegmentManager::add_column(const FieldSchema::Ptr &column_schema,
                                  const std::string &expression,
                                  int concurrency) {
  if (concurrency <= 0) {
    concurrency = static_cast<int>(std::thread::hardware_concurrency());
  }

  std::vector<std::future<Status>> futures;
  std::vector<std::pair<SegmentID, Segment::Ptr>> segments;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    segments.assign(segments_map_.begin(), segments_map_.end());
  }

  for (size_t i = 0; i < segments.size(); i += concurrency) {
    size_t end = std::min(i + concurrency, segments.size());
    for (size_t j = i; j < end; ++j) {
      auto &segment = segments[j].second;
      futures.emplace_back(std::async(std::launch::async, [&]() -> Status {
        return segment->add_column(column_schema, expression,
                                   AddColumnOptions{concurrency});
      }));
    }

    for (auto it = futures.begin(); it != futures.end(); ++it) {
      Status status = it->get();
      if (!status.ok()) {
        return status;
      }
    }
    futures.clear();
  }

  return Status::OK();
}

Status SegmentManager::alter_column(const std::string &column_name,
                                    const FieldSchema::Ptr &new_column_schema,
                                    int concurrency) {
  if (concurrency <= 0) {
    concurrency = static_cast<int>(std::thread::hardware_concurrency());
  }

  std::vector<std::future<Status>> futures;
  std::vector<std::pair<SegmentID, Segment::Ptr>> segments;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    segments.assign(segments_map_.begin(), segments_map_.end());
  }

  for (size_t i = 0; i < segments.size(); i += concurrency) {
    size_t end = std::min(i + concurrency, segments.size());
    for (size_t j = i; j < end; ++j) {
      auto &segment = segments[j].second;
      futures.emplace_back(std::async(std::launch::async, [&]() -> Status {
        return segment->alter_column(column_name, new_column_schema,
                                     AlterColumnOptions{concurrency});
      }));
    }

    for (auto it = futures.begin(); it != futures.end(); ++it) {
      Status status = it->get();
      if (!status.ok()) {
        return status;
      }
    }
    futures.clear();
  }

  return Status::OK();
}

Status SegmentManager::drop_column(const std::string &column_name) {
  std::vector<Segment::Ptr> segments;
  {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (auto &[segment_id, segment] : segments_map_) {
      segments.push_back(segment);
    }
  }

  for (auto &segment : segments) {
    auto s = segment->drop_column(column_name);
    CHECK_RETURN_STATUS(s);
  }

  return Status::OK();
}

}  // namespace zvec