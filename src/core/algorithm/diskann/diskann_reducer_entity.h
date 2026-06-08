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

#include <mutex>
#include <set>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_holder.h>
#include "diskann_entity.h"
#include "diskann_file_reader.h"
#include "diskann_pq_table.h"
#include "diskann_util.h"

namespace zvec {
namespace core {

class DiskAnnReducerEntity : public DiskAnnEntity {
 public:
  using Pointer = std::shared_ptr<DiskAnnReducerEntity>;
  using SegmentPointer = IndexStorage::Segment::Pointer;

 public:
  DiskAnnReducerEntity() = default;
  virtual ~DiskAnnReducerEntity() = default;

  int load(const IndexStorage::Pointer &container, bool check_crc);
  int load_segments(bool check_crc);
  int load_header_segment();
  int load_vector_segment();
  int load_key_segment();
  int load_key_mapping_segment();

  bool do_crc_check(std::vector<SegmentPointer> &segments) const;

  diskann_id_t get_id(diskann_key_t key) const override;
  diskann_key_t get_key(diskann_id_t id) const override;
  const void *get_vector(diskann_id_t id) const override;

 private:
  IndexStorage::Pointer container_{};
  IndexStorage::Segment::Pointer meta_segment_{};
  IndexStorage::Segment::Pointer vector_segment_{};
  IndexStorage::Segment::Pointer key_segment_{};
  IndexStorage::Segment::Pointer key_mapping_segment_{};

  std::string key_buffer_;
  std::string key_mapping_buffer_;

  size_t sector_num_per_node_{0};

  mutable size_t sector_id_{-1U};
  mutable std::string sector_buffer_;

  bool loaded_{false};
};

}  // namespace core
}  // namespace zvec