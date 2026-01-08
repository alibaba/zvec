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

#include <stdint.h>
#include <ailego/parallel/lock.h>
#include "hnsw_context.h"
#include "hnsw_dist_calculator.h"
#include "hnsw_entity.h"

namespace zvec {
namespace core {

//! hnsw graph algorithm implement
class HnswAlgorithm {
 public:
  typedef std::unique_ptr<HnswAlgorithm> UPointer;

 public:
  //! Constructor
  explicit HnswAlgorithm(HnswEntity &entity);

  //! Destructor
  ~HnswAlgorithm() = default;

  //! Cleanup HnswAlgorithm
  int cleanup();

  //! Add a node to hnsw graph
  //! @id:     the node unique id
  //! @level:  a node will be add to graph in each level [0, level]
  //! return 0 on success, or errCode in failure
  int add_node(node_id_t id, level_t level, HnswContext *ctx);

  //! do knn search in graph
  //! return 0 on success, or errCode in failure. results saved in ctx
  int search(HnswContext *ctx) const;

  //! Initiate HnswAlgorithm
  int init() {
    level_probas_.clear();
    double level_mult =
        1 / std::log(static_cast<double>(entity_.scaling_factor()));
    for (int level = 0;; level++) {
      // refers faiss get_random_level alg
      double proba =
          std::exp(-level / level_mult) * (1 - std::exp(-1 / level_mult));
      if (proba < 1e-9) {
        break;
      }
      level_probas_.push_back(proba);
    }

    return 0;
  }

  //! Generate a random level
  //! return graph level
  uint32_t get_random_level() const {
    // gen rand float (0, 1)
    double f = mt_() / static_cast<float>(mt_.max());
    for (size_t level = 0; level < level_probas_.size(); level++) {
      if (f < level_probas_[level]) {
        return level;
      }
      f -= level_probas_[level];
    }
    return level_probas_.size() - 1;
  }

 private:
  //! Select in upper layer to get entry point for next layer search
  void select_entry_point(level_t level, node_id_t *entry_point, dist_t *dist,
                          HnswContext *ctx) const;

  //! update node id neighbors from topkHeap, and reverse link is also updated
  void add_neighbors(node_id_t id, level_t level, TopkHeap &topk_heap,
                     HnswContext *ctx);

  //! Given a node id and level, search the nearest neighbors in graph
  //! Note: the nearest neighbors result keeps in topk, and entry_point and
  //! dist will be updated to current level nearest node id and distance
  void search_neighbors(level_t level, node_id_t *entry_point, dist_t *dist,
                        TopkHeap &topk, HnswContext *ctx) const;

  //! Update the node's neighbors
  void update_neighbors(HnswDistCalculator &dc, node_id_t id, level_t level,
                        TopkHeap &topk_heap);

  //! Checking linkId could be id's new neighbor, and add as neighbor if true
  //! @dc         distance calculator
  //! @updateHeap temporary heap in updating neighbors
  void reverse_update_neighbors(HnswDistCalculator &dc, node_id_t id,
                                level_t level, node_id_t link_id, dist_t dist,
                                TopkHeap &update_heap);

  //! expand neighbors until group nums are reached
  void expand_neighbors_by_group(TopkHeap &topk, HnswContext *ctx) const;

 private:
  HnswAlgorithm(const HnswAlgorithm &) = delete;
  HnswAlgorithm &operator=(const HnswAlgorithm &) = delete;

 private:
  static constexpr uint32_t kLockCnt{1U << 8};
  static constexpr uint32_t kLockMask{kLockCnt - 1U};

  HnswEntity &entity_;
  mutable std::mt19937 mt_{};
  std::vector<double> level_probas_{};

  mutable ailego::SpinMutex spin_lock_{};  // global spin lock
  std::mutex mutex_{};                     // global mutex
  // TODO: spin lock?
  std::vector<std::mutex> lock_pool_{};
};

}  // namespace core
}  // namespace zvec