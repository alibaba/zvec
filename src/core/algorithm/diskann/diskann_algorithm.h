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
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_meta.h>
#include "diskann_context.h"

namespace zvec {
namespace core {

class DiskAnnAlgorithm {
 public:
  typedef std::unique_ptr<DiskAnnAlgorithm> UPointer;

 public:
  DiskAnnAlgorithm(DiskAnnEntity &entity, uint32_t max_degree);

 public:
  int add_node(diskann_id_t id, DiskAnnContext *ctx);
  int prune_node(diskann_id_t id, DiskAnnContext *ctx);

 private:
  int search_neighbor_and_prune(diskann_id_t id,
                                std::vector<diskann_id_t> &pruned_list,
                                DiskAnnContext *ctx);
  int iterate_to_fixed_point(const std::vector<diskann_id_t> &init_ids,
                             DiskAnnContext *ctx);
  int prune_neighbors(diskann_id_t id, std::vector<Neighbor> &pool,
                      std::vector<diskann_id_t> &pruned_list,
                      DiskAnnContext *ctx);
  int inter_insert(diskann_id_t id, std::vector<diskann_id_t> &pruned_list,
                   DiskAnnContext *ctx);
  int occlude_list(diskann_id_t id, std::vector<Neighbor> &pool,
                   std::vector<diskann_id_t> &result, DiskAnnContext *ctx);

  std::vector<diskann_id_t> get_init_ids(DiskAnnContext *ctx);

 private:
  static constexpr uint32_t kLockCnt{1U << 16};
  static constexpr uint32_t kLockMask{kLockCnt - 1U};

  DiskAnnEntity &entity_;

  uint32_t max_degree_{DiskAnnEntity::kDefaultMaxDegree};
  uint32_t max_candidate_size_{DiskAnnEntity::kDefaultMaxOcclusionSize};

  std::vector<std::mutex> lock_pool_{};

  //! Acquire the striped lock guarding the given id (RAII).
  [[nodiscard]] std::unique_lock<std::mutex> lock_for(diskann_id_t id) {
    return std::unique_lock<std::mutex>(lock_pool_[id & kLockMask]);
  }

  float alpha_{DiskAnnEntity::kDefaultAlpha};
  bool saturate_graph_{true};
};

}  // namespace core
}  // namespace zvec