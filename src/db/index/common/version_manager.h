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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <zvec/db/schema.h>
#include <zvec/db/status.h>
#include "db/index/common/meta.h"

namespace zvec {

class Version {
 public:
  using Ptr = std::shared_ptr<Version>;

  Version() = default;

  static Status Load(const std::string &path, Version *version);

  static Status Save(const std::string &path, const Version &version);

 public:
  void set_schema(const CollectionSchema &schema) {
    schema_ = std::make_shared<CollectionSchema>(schema);
  }

  const CollectionSchema &schema() const {
    return *schema_;
  }

  void set_enable_mmap(bool enable_mmap) {
    enable_mmap_ = enable_mmap;
  }

  bool enable_mmap() const {
    return enable_mmap_;
  }

  Status add_persisted_segment_meta(const SegmentMeta::Ptr &meta) {
    if (meta == nullptr) {
      return Status::InvalidArgument("Segment meta is null");
    }
    auto iter = persisted_segment_metas_map_.find(meta->id());
    if (iter != persisted_segment_metas_map_.end()) {
      return Status::InvalidArgument("Segment meta already exists");
    }
    persisted_segment_metas_map_[meta->id()] = meta;
    return Status::OK();
  }

  Status remove_persisted_segment_meta(SegmentID segment_id) {
    auto iter = persisted_segment_metas_map_.find(segment_id);
    if (iter == persisted_segment_metas_map_.end()) {
      return Status::NotFound("Segment meta not found");
    }
    persisted_segment_metas_map_.erase(segment_id);
    return Status::OK();
  }

  Status update_persisted_segment_meta(SegmentMeta::Ptr meta) {
    if (meta == nullptr) {
      return Status::InvalidArgument("Segment meta is null");
    }
    auto iter = persisted_segment_metas_map_.find(meta->id());
    if (iter == persisted_segment_metas_map_.end()) {
      return Status::NotFound("Segment meta not found");
    }
    persisted_segment_metas_map_[meta->id()] =
        std::make_shared<SegmentMeta>(*meta);
    return Status::OK();
  }

  void set_persisted_segment_metas(const std::vector<SegmentMeta::Ptr> &metas) {
    for (auto &meta : metas) {
      persisted_segment_metas_map_[meta->id()] = meta;
    }
  }

  std::vector<SegmentMeta::Ptr> persisted_segment_metas() const {
    std::vector<SegmentMeta::Ptr> segment_metas;
    segment_metas.reserve(persisted_segment_metas_map_.size());
    for (auto &segment_meta : persisted_segment_metas_map_) {
      segment_metas.push_back(segment_meta.second);
    }

    std::sort(segment_metas.begin(), segment_metas.end(),
              [](const SegmentMeta::Ptr &lhs, const SegmentMeta::Ptr &rhs) {
                return lhs->min_doc_id() < rhs->min_doc_id();
              });

    return segment_metas;
  }

  void reset_writing_segment_meta(SegmentMeta::Ptr segment_meta) {
    writing_segment_meta_ = segment_meta;
  }

  SegmentMeta::Ptr writing_segment_meta() const {
    return writing_segment_meta_;
  }

  void set_id_map_path_suffix(uint32_t suffix) {
    id_map_path_suffix_ = suffix;
  }

  uint32_t id_map_path_suffix() const {
    return id_map_path_suffix_;
  }

  void set_delete_snapshot_path_suffix(uint32_t suffix) {
    delete_snapshot_path_suffix_ = suffix;
  }

  uint32_t delete_snapshot_path_suffix() const {
    return delete_snapshot_path_suffix_;
  }

  void set_next_segment_id(SegmentID id) {
    next_segment_id_ = id;
  }

  SegmentID next_segment_id() const {
    return next_segment_id_;
  }

 public:
  bool operator==(const Version &other) const {
    if (*schema_ != *other.schema_ ||
        persisted_segment_metas_map_.size() !=
            other.persisted_segment_metas_map_.size()) {
      return false;
    }

    for (const auto &item : persisted_segment_metas_map_) {
      auto it = other.persisted_segment_metas_map_.find(item.first);
      if (it == other.persisted_segment_metas_map_.end() ||
          *item.second != *it->second) {
        return false;
      }
    }

    return true;
  }

  std::string to_string() const;

  std::string to_string_formatted(int indent_level = 0) const;

 private:
  CollectionSchema::Ptr schema_;
  bool enable_mmap_;

  std::unordered_map<SegmentID, SegmentMeta::Ptr> persisted_segment_metas_map_;

  SegmentMeta::Ptr writing_segment_meta_;

  uint32_t id_map_path_suffix_{0};
  uint32_t delete_snapshot_path_suffix_{0};

  SegmentID next_segment_id_{0};
};

// Wrapper of Current Version
class VersionManager {
 public:
  using Ptr = std::shared_ptr<VersionManager>;

  static Result<VersionManager::Ptr> Recovery(const std::string &path);

  static Result<VersionManager::Ptr> Create(const std::string &path,
                                            const Version &initial_version);

 private:
  VersionManager(const std::string &path, const Version &initial_version,
                 uint64_t version_id = 0);

 public:
  Version get_current_version();

  // overwrite the current version
  Status apply(const Version &version);

  Status reset_writing_segment_meta(SegmentMeta::Ptr meta);

  Status add_persisted_segment_meta(SegmentMeta::Ptr meta);

  Status remove_persisted_segment_meta(SegmentID id);

  Status flush();

  void set_id_map_path_suffix(uint32_t suffix) {
    std::lock_guard lock(mtx_);
    current_version_.set_id_map_path_suffix(suffix);
  }

  void set_delete_snapshot_path_suffix(uint32_t suffix) {
    std::lock_guard lock(mtx_);
    current_version_.set_delete_snapshot_path_suffix(suffix);
  }

  uint32_t delete_snapshot_path_suffix() const {
    std::lock_guard lock(mtx_);
    return current_version_.delete_snapshot_path_suffix();
  }

  void set_next_segment_id(SegmentID id) {
    std::lock_guard lock(mtx_);
    current_version_.set_next_segment_id(id);
  }

  void set_enable_mmap(bool enable_mmap) {
    std::lock_guard lock(mtx_);
    current_version_.set_enable_mmap(enable_mmap);
  }

 private:
  const std::string path_;
  Version current_version_;
  mutable std::mutex mtx_;

  uint64_t version_id_ = 0;
};

}  // namespace zvec
