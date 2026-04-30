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
#include <chrono>
#include <vector>
#include <ailego/internal/cpu_features.h>
#include <ailego/parallel/lock.h>
#include "hnsw_context.h"
#include "hnsw_dist_calculator.h"
#include "hnsw_entity.h"
#include "hnsw_streamer_entity.h"

namespace zvec {
namespace core {

//! Non-template base class for HnswAlgorithm
class HnswAlgorithmBase {
 public:
  typedef std::unique_ptr<HnswAlgorithmBase> UPointer;

  virtual ~HnswAlgorithmBase() = default;

  virtual int cleanup() = 0;
  virtual int add_node(node_id_t id, level_t level, HnswContext *ctx) = 0;
  virtual int search(HnswContext *ctx) const = 0;
  virtual int init() = 0;
  virtual uint32_t get_random_level() const = 0;
};

//! hnsw graph algorithm implement, templated on EntityType
template <typename EntityType>
class HnswAlgorithm : public HnswAlgorithmBase {
 public:
  using MemBlockType = typename EntityType::MemoryBlock;

  //! Constructor
  explicit HnswAlgorithm(EntityType &entity)
      : entity_(entity),
        mt_(std::chrono::system_clock::now().time_since_epoch().count()),
        lock_pool_(kLockCnt) {}

  //! Destructor
  ~HnswAlgorithm() override = default;

  //! Cleanup HnswAlgorithm
  int cleanup() override {
    return 0;
  }

  //! Add a node to hnsw graph
  //! @id:     the node unique id
  //! @level:  a node will be add to graph in each level [0, level]
  //! return 0 on success, or errCode in failure
  int add_node(node_id_t id, level_t level, HnswContext *ctx) override;

  //! do knn search in graph
  //! return 0 on success, or errCode in failure. results saved in ctx
  int search(HnswContext *ctx) const override;

  //! Initiate HnswAlgorithm
  int init() override {
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
  uint32_t get_random_level() const override {
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

  EntityType &entity_;
  mutable std::mt19937 mt_{};
  std::vector<double> level_probas_{};

  mutable ailego::SpinMutex spin_lock_{};  // global spin lock
  std::mutex mutex_{};                     // global mutex
  // TODO: spin lock?
  std::vector<std::mutex> lock_pool_{};
};

}  // namespace core
}  // namespace zvec