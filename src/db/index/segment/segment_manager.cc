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
#include <set>
#include <thread>
#include <vector>
#include <zvec/db/status.h>
#include "db/common/typedef.h"

namespace zvec {

Status SegmentManager::switch_fts_writer(
    Segment::Ptr *writing, const VersionManager::Ptr &versions,
    const PrepareFtsSuccessor &prepare_successor,
    const DurableFile::Hook &hook) {
  if (!writing || !*writing || !versions || !prepare_successor) {
    return Status::InvalidArgument("Missing FTS switch resource");
  }
  if (versions->requires_recovery()) {
    return Status::FailedPrecondition("Uncertain manifest; reopen required");
  }
  // Allocate the new manager contents before the durable commit. Collection
  // write/schema exclusion keeps this snapshot current until publication.
  decltype(segments_map_) prepared_segments;
  {
    std::shared_lock lock(mutex_);
    prepared_segments = segments_map_;
  }
  if (prepared_segments.count((*writing)->id()) != 0) {
    return Status::FailedPrecondition("Writing segment is already persisted");
  }
  // The first authority record also references older, legacy segments. Make
  // those files durable before upgrading the collection's commit protocol.
  for (const auto &entry : prepared_segments) {
    CHECK_RETURN_STATUS(entry.second->checkpoint_for_fts());
  }
  auto prepared = (*writing)->prepare_fts_seal(hook);
  if (!prepared) {
    return prepared.error();
  }
  if (prepared->metadata->max_doc_id() == UINT64_MAX) {
    return Status{StatusCode::RESOURCE_EXHAUSTED, "Document ID exhausted"};
  }
  prepared_segments.emplace((*writing)->id(), *writing);
  auto successor = prepare_successor(prepared->metadata->max_doc_id() + 1);
  if (!successor) {
    return successor.error();
  }
  auto next = std::move(successor->segment);
  if (!next || next == *writing || prepared_segments.count(next->id()) != 0 ||
      next->doc_count_snapshot() != 0 ||
      successor->next_segment_id <= next->id()) {
    return Status::InvalidArgument("Invalid private FTS successor");
  }
  auto discard_successor = [&]() {
    auto status = next->destroy();
    if (!status.ok()) {
      LOG_WARN("Failed to discard FTS successor: %s", status.message().c_str());
    }
  };
  auto status = next->checkpoint_for_fts();
  if (!status.ok()) {
    discard_successor();
    return status;
  }
  auto candidate = versions->get_current_version();
  status = candidate.add_persisted_segment_meta(prepared->metadata);
  if (!status.ok()) {
    discard_successor();
    return status;
  }
  candidate.reset_writing_segment_meta(next->meta());
  candidate.set_next_segment_id(successor->next_segment_id);
  status = versions->commit(candidate, hook);
  if (!status.ok()) {
    if (versions->requires_recovery()) {
      // Recovery may select this candidate: retain both generations and the
      // empty successor, with the existing writer frozen by VersionManager.
      if (prepared->indexer) {
        prepared->indexer->preserve_files();
      }
    } else {
      discard_successor();
    }
    return status;
  }
  (*writing)->publish_fts_seal(std::move(*prepared));
  {
    std::unique_lock lock(mutex_);
    segments_map_.swap(prepared_segments);
  }
  writing->swap(next);
  return Status::OK();
}

Status SegmentManager::add_segment(Segment::Ptr segment) {
  if (!segment) {
    return Status::InvalidArgument("Segment is null");
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  segments_map_[segment->id()] = segment;
  return Status::OK();
}

Result<std::unique_ptr<SegmentManager::PreparedUpdate>>
SegmentManager::prepare_update(const std::vector<Segment::Ptr> &additions,
                               const std::vector<SegmentID> &removals) const {
  auto prepared = std::make_unique<PreparedUpdate>();
  {
    std::shared_lock lock(mutex_);
    prepared->segments_ = segments_map_;
  }
  for (auto id : removals) {
    auto found = prepared->segments_.find(id);
    if (found == prepared->segments_.end()) {
      return tl::make_unexpected(Status::NotFound("Missing retired segment"));
    }
    prepared->retired_.push_back(found->second);
    prepared->segments_.erase(found);
  }
  std::set<SegmentID> added;
  for (const auto &segment : additions) {
    if (!segment || !added.insert(segment->id()).second ||
        std::find(removals.begin(), removals.end(), segment->id()) !=
            removals.end()) {
      return tl::make_unexpected(
          Status::InvalidArgument("Invalid segment update"));
    }
    auto previous = prepared->segments_.find(segment->id());
    if (previous != prepared->segments_.end() && previous->second != segment) {
      auto files =
          previous->second->prepare_vector_retirement(*segment->meta());
      if (!files) {
        return tl::make_unexpected(files.error());
      }
      prepared->retired_files_.insert(prepared->retired_files_.end(),
                                      files->begin(), files->end());
    }
    prepared->segments_[segment->id()] = segment;
  }
  return prepared;
}

void SegmentManager::publish_update(std::unique_ptr<PreparedUpdate> update) {
  {
    std::unique_lock lock(mutex_);
    segments_map_.swap(update->segments_);
  }
  for (const auto &segment : update->retired_) {
    auto status = segment->destroy();
    if (!status.ok()) {
      // The committed map remains authoritative; failed cleanup never rolls
      // it back. The segment owner retains the directory until its last user.
      LOG_WARN("Failed to retire committed segment: %s",
               status.message().c_str());
    }
  }
  for (const auto &file : update->retired_files_) {
    file->retire();
  }
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

Status SegmentManager::destroy_segment(SegmentID segment_id) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto iter = segments_map_.find(segment_id);
  if (iter == segments_map_.end()) {
    return Status::NotFound("Segment not found");
  }

  auto s = iter->second->destroy();
  CHECK_RETURN_STATUS(s);

  segments_map_.erase(iter);
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
      futures.emplace_back(
          std::async(std::launch::async, [&, segment]() -> Status {
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
      futures.emplace_back(
          std::async(std::launch::async, [&, segment]() -> Status {
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
    for (auto &pair : segments_map_) {
      segments.push_back(pair.second);
    }
  }

  for (auto &segment : segments) {
    auto s = segment->drop_column(column_name);
    CHECK_RETURN_STATUS(s);
  }

  return Status::OK();
}

}  // namespace zvec
