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

#include <chrono>
#include <mutex>
#include <vector>
#include <ailego/parallel/lock.h>
#include "vamana_context.h"
#include "vamana_dist_calculator.h"
#include "vamana_streamer_entity.h"

namespace zvec {
namespace core {

// Non-template base class providing a type-erased interface so that
// VamanaStreamer can hold a pointer without knowing the EntityType.
class VamanaAlgorithmBase {
 public:
  typedef std::unique_ptr<VamanaAlgorithmBase> UPointer;

  virtual ~VamanaAlgorithmBase() = default;

  virtual int cleanup() = 0;

  // Insert a new node into the Vamana graph.
  // The node's vector must already be stored in the entity.
  virtual int add_node(node_id_t id, VamanaContext *ctx) = 0;

  // Greedy search: find approximate nearest neighbors.
  virtual int search(VamanaContext *ctx) const = 0;

  virtual int init() = 0;
};

// Vamana graph algorithm, templated on EntityType for hot-path optimization.
// EntityType should be VamanaMmapStreamerEntity,
// VamanaBufferPoolStreamerEntity, or VamanaContiguousStreamerEntity.
//
// Core operations:
//   - GreedySearch: beam search from entry point, expanding best candidates
//   - RobustPrune: select diverse neighbors using alpha-based pruning
//   - add_node: insert + prune + reverse-link update
template <typename EntityType>
class VamanaAlgorithm : public VamanaAlgorithmBase {
 public:
  using MemBlockType = typename EntityType::MemoryBlock;

  explicit VamanaAlgorithm(EntityType &entity)
      : entity_(entity), lock_pool_(kLockCnt) {}

  ~VamanaAlgorithm() override = default;

  int cleanup() override {
    return 0;
  }

  int init() override {
    return 0;
  }

  // Insert node `id` into the graph. Its vector must already be in the entity.
  int add_node(node_id_t id, VamanaContext *ctx) override;

  // Greedy search from entry point. Results are stored in ctx->topk_heap().
  int search(VamanaContext *ctx) const override;

 private:
  // GreedySearch: starting from entry_point, greedily expand the closest
  // unvisited candidate until the search list is exhausted or scan limit
  // is reached. Results accumulate in topk_heap.
  void greedy_search(node_id_t entry_point, VamanaContext *ctx) const;

  // RobustPrune: given a candidate set (topk_heap), select up to max_degree
  // diverse neighbors using alpha-based distance comparison.
  // Result is stored in ctx->prune_result().
  void robust_prune(node_id_t id, TopkHeap &candidates, float alpha,
                    uint32_t max_degree, VamanaContext *ctx) const;

  // Update node's neighbors and handle reverse links.
  void update_neighbors_and_reverse_links(
      node_id_t id,
      const std::vector<std::pair<node_id_t, dist_t>> &new_neighbors,
      VamanaContext *ctx);

  // Check if adding `id` as a reverse neighbor of `neighbor_id` requires
  // pruning, and if so, prune neighbor_id's neighbor list.
  void reverse_update_neighbor(node_id_t id, node_id_t neighbor_id, dist_t dist,
                               VamanaContext *ctx);

 private:
  VamanaAlgorithm(const VamanaAlgorithm &) = delete;
  VamanaAlgorithm &operator=(const VamanaAlgorithm &) = delete;

  static constexpr uint32_t kLockCnt{1U << 8};
  static constexpr uint32_t kLockMask{kLockCnt - 1U};

  EntityType &entity_;
  mutable ailego::SpinMutex spin_lock_{};
  std::vector<std::mutex> lock_pool_{};
};

}  // namespace core
}  // namespace zvec
