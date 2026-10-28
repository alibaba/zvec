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
  ~DiskAnnBuilderEntity() override = default;

 public:
  void clear();

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

  // Keeps the storage behind borrowed vector pointers alive until
  // release_vectors(), even after the holder or iterator is destroyed.
  // The owner may be mmap storage, a holder, or an owned snapshot; the
  // shared_ptr retains its deleter without exposing its type here.
  int reserve_space(uint32_t docs, std::shared_ptr<const void> lifetime);

  // Graph construction is the only consumer of these vectors. PQ encoding
  // and dump read the retained source holder instead.
  void release_vectors();

  std::string &pq_quantizer_meta_buffer() {
    return pq_quantizer_meta_buffer_;
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

  std::shared_ptr<const void> vector_lifetime_{};
  std::vector<const void *> vector_ptrs_{};
  std::string keys_buffer_{};
  std::string neighbors_buffer_{};
  std::vector<diskann_id_t> entrypoints_{};

  IndexMeta meta_;

  std::string pq_quantizer_meta_buffer_{};
  std::vector<uint8_t> block_compressed_data_;
};

}  // namespace core
}  // namespace zvec
