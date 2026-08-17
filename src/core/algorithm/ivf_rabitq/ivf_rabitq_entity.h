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

#include <cstdint>
#include <memory>
#include <vector>
#include <rabitqlib/utils/space.hpp>
#include "zvec/core/framework/index_document.h"
#include "zvec/core/framework/index_filter.h"
#include "zvec/core/framework/index_storage.h"
#include "ivf_rabitq_context.h"
#include "ivf_rabitq_index_format.h"

namespace zvec {
namespace core {

/*! IVF RaBitQ Entity
 * Stores quantized data (batch-packed binary codes + extra-bit codes + keys)
 * for all clusters. Provides cluster scanning with lower-bound pruning.
 */
class IvfRabitqEntity {
 public:
  typedef std::shared_ptr<IvfRabitqEntity> Pointer;

  IvfRabitqEntity() = default;
  ~IvfRabitqEntity() = default;

  //! Load entity data from storage
  int load(IndexStorage::Pointer storage);

  //! Scan one cluster and insert qualifying candidates into heap.
  //! Uses 1-bit lower bound to prune before expensive extra-bit boosting.
  //! Same interface as IVFEntity::search for consistency.
  int search_cluster(uint32_t cluster_id,
                     const IvfRabitqQueryState &query_state, size_t padded_dim,
                     size_t ex_bits, IndexDocumentHeap *heap) const;

  int search_cluster(uint32_t cluster_id,
                     const IvfRabitqQueryState &query_state, size_t padded_dim,
                     size_t ex_bits, const IndexFilter &filter,
                     IndexDocumentHeap *heap) const;

  int search_cluster_group_by(uint32_t cluster_id,
                              const IvfRabitqQueryState &query_state,
                              size_t padded_dim, size_t ex_bits,
                              const IndexGroupBy &group_by, uint32_t group_topk,
                              float threshold,
                              IvfRabitqContext::GroupTopkHeaps *heaps) const;

  int search_cluster_group_by(uint32_t cluster_id,
                              const IvfRabitqQueryState &query_state,
                              size_t padded_dim, size_t ex_bits,
                              const IndexFilter &filter,
                              const IndexGroupBy &group_by, uint32_t group_topk,
                              float threshold,
                              IvfRabitqContext::GroupTopkHeaps *heaps) const;

  int compute_distances(uint32_t cluster_id, const std::vector<uint32_t> &ids,
                        const IvfRabitqQueryState &query_state,
                        size_t padded_dim, size_t ex_bits,
                        IndexDocumentList *documents) const;

  uint32_t cluster_count() const {
    return header_.cluster_count;
  }
  uint32_t total_vector_count() const {
    return header_.total_vector_count;
  }
  uint32_t padded_dim() const {
    return header_.padded_dim;
  }
  uint32_t dimension() const {
    return header_.dimension;
  }
  uint8_t ex_bits() const {
    return header_.ex_bits;
  }

  const IvfRabitqClusterMeta &cluster_meta(uint32_t cluster_id) const {
    return cluster_metas_[cluster_id];
  }

  uint64_t get_key(size_t id) const;

  const void *get_vector_by_key(uint64_t key) const;

  int get_vector_by_key(uint64_t key, IndexStorage::MemoryBlock &block) const;

  uint32_t key_to_id(uint64_t key) const;

  uint32_t get_cluster_id(uint32_t id) const;

  const uint32_t *get_key_order_mapping() const {
    if (!mapping_) {
      return nullptr;
    }
    const void *data = nullptr;
    size_t size = total_vector_count() * sizeof(uint32_t);
    if (mapping_->read(0, &data, size) != size) {
      return nullptr;
    }
    return static_cast<const uint32_t *>(data);
  }

 private:
  IvfRabitqHeader header_;
  std::vector<IvfRabitqClusterMeta> cluster_metas_;

  IndexStorage::MemoryBlock batch_data_block_;
  IndexStorage::MemoryBlock ex_data_block_;
  IndexStorage::MemoryBlock keys_block_;
  IndexStorage::Segment::Pointer mapping_;

  const char *batch_data_{nullptr};
  const char *ex_data_{nullptr};
  const uint64_t *keys_{nullptr};
  size_t batch_data_size_{0};
  size_t ex_data_size_{0};

  // Resolved once at load time to avoid per-cluster function-pointer lookup
  rabitqlib::ex_ipfunc ip_func_{nullptr};

  template <bool HasFilter>
  int search_cluster_impl(uint32_t cluster_id,
                          const IvfRabitqQueryState &query_state,
                          size_t padded_dim, size_t ex_bits,
                          const IndexFilter *filter,
                          IndexDocumentHeap *heap) const;

  template <bool HasFilter>
  int search_cluster_group_by_impl(
      uint32_t cluster_id, const IvfRabitqQueryState &query_state,
      size_t padded_dim, size_t ex_bits, const IndexFilter *filter,
      const IndexGroupBy &group_by, uint32_t group_topk, float threshold,
      IvfRabitqContext::GroupTopkHeaps *heaps) const;

  const IvfRabitqClusterMeta *find_cluster_meta(size_t id) const;

  int init_data_layout();

  int validate_header() const;

  int validate_cluster_metas() const;

  int load_key_order_mapping(IndexStorage::Pointer storage);
};

}  // namespace core
}  // namespace zvec
