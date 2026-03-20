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
#include "hnsw_rabitq_context.h"
#include "hnsw_rabitq_dist_calculator.h"
#include "hnsw_rabitq_entity.h"

namespace zvec {
namespace core {

//! hnsw graph algorithm implement
class HnswRabitqAlgorithm {
 public:
  typedef std::unique_ptr<HnswRabitqAlgorithm> UPointer;

 public:
  //! Constructor
  explicit HnswRabitqAlgorithm(HnswRabitqEntity &entity);

  //! Destructor
  ~HnswRabitqAlgorithm() = default;

  //! Cleanup HnswRabitqAlgorithm
  int cleanup();

  //! Add a node to hnsw graph
  //! @id:     the node unique id
  //! @level:  a node will be add to graph in each level [0, level]
  //! return 0 on success, or errCode in failure
  int add_node(node_id_t id, level_t level, HnswRabitqContext *ctx);

  //! Initiate HnswRabitqAlgorithm
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
  void select_entry_point(level_t level, node_id_t *entry_point,
                          ResultRecord *dist, HnswRabitqContext *ctx) const;

  //! update node id neighbors from topkHeap, and reverse link is also updated
  void add_neighbors(node_id_t id, level_t level, TopkHeap &topk_heap,
                     HnswRabitqContext *ctx);

  //! Given a node id and level, search the nearest neighbors in graph
  //! Note: the nearest neighbors result keeps in topk, and entry_point and
  //! dist will be updated to current level nearest node id and distance
  void search_neighbors(level_t level, node_id_t *entry_point,
                        ResultRecord *dist, TopkHeap &topk,
                        HnswRabitqContext *ctx) const;

  //! Update the node's neighbors
  void update_neighbors(HnswRabitqAddDistCalculator &dc, node_id_t id,
                        level_t level, TopkHeap &topk_heap);

  //! Checking linkId could be id's new neighbor, and add as neighbor if true
  //! @dc         distance calculator
  //! @updateHeap temporary heap in updating neighbors
  void reverse_update_neighbors(HnswRabitqAddDistCalculator &dc, node_id_t id,
                                level_t level, node_id_t link_id,
                                ResultRecord dist, TopkHeap &update_heap);

 private:
  HnswRabitqAlgorithm(const HnswRabitqAlgorithm &) = delete;
  HnswRabitqAlgorithm &operator=(const HnswRabitqAlgorithm &) = delete;

 private:
  static constexpr uint32_t kLockCnt{1U << 8};
  static constexpr uint32_t kLockMask{kLockCnt - 1U};

  HnswRabitqEntity &entity_;
  mutable std::mt19937 mt_{};
  std::vector<double> level_probas_{};

  mutable ailego::SpinMutex spin_lock_{};  // global spin lock
  std::mutex mutex_{};                     // global mutex
  // TODO: spin lock?
  std::vector<std::mutex> lock_pool_{};
};

}  // namespace core
}  // namespace zvec