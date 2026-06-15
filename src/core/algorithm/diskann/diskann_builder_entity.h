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

#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_holder.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

// wrapper class aligned with diskann
class DiskAnnBuilderEntity : public DiskAnnEntity {
 public:
  using Pointer = std::shared_ptr<DiskAnnBuilderEntity>;

  DiskAnnBuilderEntity() = default;
  virtual ~DiskAnnBuilderEntity() = default;

 public:
  int add_vector(diskann_key_t key, const void *vec) override;

  std::pair<uint32_t, const diskann_id_t *> get_neighbors(
      diskann_id_t id) const override;

  int set_neighbors(diskann_id_t id,
                    const std::vector<diskann_id_t> &neighbor_ids) override;

  int add_neighbor(diskann_id_t id, diskann_id_t neighbor_id) override;

  diskann_id_t get_id(diskann_key_t key) const override;
  diskann_key_t get_key(diskann_id_t id) const override;
  const void *get_vector(diskann_id_t id) const override;

 public:
  int init(const IndexMeta &meta, uint32_t max_degree, uint32_t list_size,
           double memory_limit, uint32_t build_threads);

  int dump(IndexHolder::Pointer holder, IndexMeta &meta,
           const IndexDumper::Pointer &dumper);

  int64_t dump_segment(const IndexDumper::Pointer &dumper,
                       const std::string &segment_id, const void *data,
                       size_t size) const;
  int dump_dummy_segment(const IndexDumper::Pointer &dumper) const;
  int dump_pq_meta_segment(const IndexDumper::Pointer &dumper) const;
  int dump_pq_data_segment(const IndexDumper::Pointer &dumper) const;
  int dump_key_mapping_segment(const IndexDumper::Pointer &dumper) const;
  int dump_entrypoint_segment(const IndexDumper::Pointer &dumper) const;
  int dump_key_segment(const IndexDumper::Pointer &dumper) const;

  int reserve_space(uint32_t docs);

  std::vector<uint8_t> &pq_full_pivot_data() {
    return pq_full_pivot_data_;
  }

  std::vector<uint8_t> &pq_centroid() {
    return pq_centroid_;
  }

  std::vector<uint32_t> &pq_chunk_offsets() {
    return pq_chunk_offsets_;
  }

  std::vector<uint8_t> &block_compressed_data() {
    return block_compressed_data_;
  }

 private:
  uint32_t max_degree_{0};
  uint32_t list_size_{0};
  double memory_limit_{0};
  uint32_t num_threads_{0};
  uint32_t max_build_degree_{0};
  uint32_t max_observed_degree_{0};
  uint32_t neighbor_size_{0};

  std::string mem_index_file_{""};
  std::string index_path_prefix_{""};

  std::string vectors_buffer_{};
  std::string keys_buffer_{};
  std::string neighbors_buffer_{};
  std::vector<diskann_id_t> entrypoints_{};

  IndexMeta meta_;

  std::vector<uint8_t> pq_full_pivot_data_;
  std::vector<uint8_t> pq_centroid_;
  std::vector<uint32_t> pq_chunk_offsets_;
  std::vector<uint8_t> block_compressed_data_;
};

}  // namespace core
}  // namespace zvec