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

#include "diskann_algorithm.h"
#include <algorithm>
#include <set>
#include <zvec/core/framework/index_holder.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

DiskAnnAlgorithm::DiskAnnAlgorithm(DiskAnnEntity &entity, uint32_t max_degree)
    : entity_(entity), max_degree_(max_degree), lock_pool_(kLockCnt) {}

std::vector<diskann_id_t> DiskAnnAlgorithm::get_init_ids(DiskAnnContext *ctx) {
  const auto &entity = ctx->get_entity();

  std::vector<diskann_id_t> init_ids;

  init_ids.emplace_back(entity.medoid());

  return init_ids;
}

int DiskAnnAlgorithm::add_node(diskann_id_t id, DiskAnnContext *ctx) {
  const void *vec = entity_.get_vector(id);

  ctx->reset_query(vec);

  std::vector<diskann_id_t> pruned_list;

  int ret = search_neighbor_and_prune(id, pruned_list, ctx);
  if (ret != 0) {
    return ret;
  }

  {
    auto lock = lock_for(id);
    entity_.set_neighbors(id, pruned_list);
  }

  return inter_insert(id, pruned_list, ctx);
}

int DiskAnnAlgorithm::prune_node(diskann_id_t id, DiskAnnContext *ctx) {
  DistCalculator &dc = ctx->dist_calculator();

  auto neighbors = entity_.get_neighbors(id);

  if (neighbors.first > max_degree_) {
    std::set<diskann_id_t> dummy_visited;
    std::vector<Neighbor> dummy_pool(0);
    std::vector<diskann_id_t> new_out_neighbors;

    for (size_t i = 0; i < neighbors.first; ++i) {
      diskann_id_t node_id = (neighbors.second)[i];

      auto itr = dummy_visited.find(node_id);
      if (itr == dummy_visited.end() && node_id != id) {
        float dist = dc.dist(id, node_id);

        dummy_pool.emplace_back(Neighbor(node_id, dist));
        dummy_visited.insert(node_id);
      }
    }

    prune_neighbors(id, dummy_pool, new_out_neighbors, ctx);

    {
      auto lock = lock_for(id);
      entity_.set_neighbors(id, new_out_neighbors);
    }
  }

  return 0;
}

int DiskAnnAlgorithm::inter_insert(diskann_id_t id,
                                   std::vector<diskann_id_t> &pruned_list,
                                   DiskAnnContext *ctx) {
  DistCalculator &dc = ctx->dist_calculator();

  for (auto &des : pruned_list) {
    std::vector<diskann_id_t> new_neighbors;
    bool need_prune = false;

    {
      auto lock = lock_for(des);

      auto neighbors = entity_.get_neighbors(des);

      bool found = false;
      for (size_t i = 0; i < neighbors.first; ++i) {
        if ((neighbors.second)[i] == id) {
          found = true;
          break;
        }
      }

      if (!found) {
        if (neighbors.first <
            static_cast<uint64_t>(DiskAnnEntity::kDefaultGraphSlackFactor *
                                  max_degree_)) {
          entity_.add_neighbor(des, id);
          need_prune = false;
        } else {
          new_neighbors.resize(neighbors.first + 1);
          memcpy(&new_neighbors[0], neighbors.second,
                 sizeof(diskann_id_t) * neighbors.first);

          new_neighbors[neighbors.first] = id;

          need_prune = true;
        }
      }
    }

    if (need_prune) {
      std::set<diskann_id_t> new_visited;
      std::vector<Neighbor> new_pool(0);

      size_t reserve_size = static_cast<size_t>(std::ceil(
          1.05 * DiskAnnEntity::kDefaultGraphSlackFactor * max_degree_));

      new_pool.reserve(reserve_size);

      for (auto node_id : new_neighbors) {
        if (new_visited.find(node_id) == new_visited.end() && node_id != des) {
          float dist = dc.dist(des, node_id);
          new_pool.emplace_back(Neighbor(node_id, dist));
          new_visited.insert(node_id);
        }
      }

      std::vector<diskann_id_t> new_pruned_neighbors;
      prune_neighbors(des, new_pool, new_pruned_neighbors, ctx);

      {
        auto lock = lock_for(des);
        entity_.set_neighbors(des, new_pruned_neighbors);
      }
    }
  }

  return 0;
}

int DiskAnnAlgorithm::iterate_to_fixed_point(
    const std::vector<diskann_id_t> &init_ids, DiskAnnContext *ctx) {
  DistCalculator &dc = ctx->dist_calculator();
  std::vector<Neighbor> &expanded_nodes = ctx->expanded_nodes();
  NeighborPriorityQueue &best_list_nodes = ctx->best_list_nodes();
  VisitFilter &visit = ctx->visit_filter();

  best_list_nodes.reserve(ctx->list_size());

  for (auto id : init_ids) {
    const void *vec = entity_.get_vector(id);

    float distance = dc.dist(vec);

    Neighbor nn = Neighbor(id, distance);
    best_list_nodes.insert(nn);
  }

  while (best_list_nodes.has_unexpanded_node()) {
    auto neighbor = best_list_nodes.closest_unexpanded();
    auto node_id = neighbor.id;

    expanded_nodes.emplace_back(neighbor);

    std::vector<diskann_id_t> id_scratch;
    {
      auto lock = lock_for(node_id);
      auto neighbors = entity_.get_neighbors(node_id);

      for (size_t i = 0; i < neighbors.first; ++i) {
        diskann_id_t neighbor_id = (neighbors.second)[i];

        if (!visit.visited(neighbor_id)) {
          id_scratch.push_back(neighbor_id);

          visit.set_visited(neighbor_id);
        }
      }
    }

    for (size_t i = 0; i < id_scratch.size(); ++i) {
      diskann_id_t id = id_scratch[i];

      const void *vec = entity_.get_vector(id);
      float dist = dc.dist(vec);

      best_list_nodes.insert(Neighbor(id, dist));
    }
  }

  return 0;
}

int DiskAnnAlgorithm::occlude_list(diskann_id_t id, std::vector<Neighbor> &pool,
                                   std::vector<diskann_id_t> &result,
                                   DiskAnnContext *ctx) {
  if (pool.size() == 0) return 0;

  DistCalculator &dc = ctx->dist_calculator();

  ailego_assert(std::is_sorted(pool.begin(), pool.end()));
  ailego_assert(result.size() == 0);

  if (pool.size() > max_candidate_size_) {
    pool.resize(max_candidate_size_);
  }

  std::vector<float> &occlude_factor = ctx->occlude_factor();

  occlude_factor.clear();
  occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

  float cur_alpha = 1;
  while (cur_alpha <= alpha_ && result.size() < max_degree_) {
    for (auto iter = pool.begin();
         result.size() < max_degree_ && iter != pool.end(); ++iter) {
      if (occlude_factor[iter - pool.begin()] > cur_alpha) {
        continue;
      }

      occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();

      if (iter->id != id) {
        result.push_back(iter->id);
      }

      for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
        auto t = iter2 - pool.begin();
        if (occlude_factor[t] > alpha_) {
          continue;
        }

        float djk = dc.dist(iter2->id, iter->id);

        occlude_factor[t] =
            (djk == 0) ? std::numeric_limits<float>::max()
                       : std::max(occlude_factor[t], iter2->distance / djk);
      }
    }
    cur_alpha *= 1.2f;
  }

  return 0;
}

int DiskAnnAlgorithm::prune_neighbors(diskann_id_t id,
                                      std::vector<Neighbor> &pool,
                                      std::vector<diskann_id_t> &pruned_list,
                                      DiskAnnContext *ctx) {
  if (pool.size() == 0) {
    pruned_list.clear();
    return 0;
  }

  std::sort(pool.begin(), pool.end());

  pruned_list.clear();
  pruned_list.reserve(max_degree_);

  occlude_list(id, pool, pruned_list, ctx);

  ailego_assert(pruned_list.size() <= max_degree_);

  if (saturate_graph_ && alpha_ > 1) {
    for (const auto &node : pool) {
      if (pruned_list.size() >= max_degree_) {
        break;
      }

      if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) ==
           pruned_list.end()) &&
          node.id != id) {
        pruned_list.push_back(node.id);
      }
    }
  }

  return 0;
}

int DiskAnnAlgorithm::search_neighbor_and_prune(
    diskann_id_t id, std::vector<diskann_id_t> &pruned_list,
    DiskAnnContext *ctx) {
  const std::vector<diskann_id_t> init_ids = get_init_ids(ctx);

  int ret = iterate_to_fixed_point(init_ids, ctx);
  if (ret != 0) {
    return ret;
  }

  auto &pool = ctx->expanded_nodes();
  pool.erase(std::remove_if(pool.begin(), pool.end(),
                            [id](const auto &node) { return node.id == id; }),
             pool.end());

  ret = prune_neighbors(id, pool, pruned_list, ctx);
  if (ret != 0) {
    return ret;
  }

  return 0;
}

}  // namespace core
}  // namespace zvec