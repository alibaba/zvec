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
#include "hnsw_rabitq_query_algorithm.h"
#include <chrono>
#include <iostream>
#include <ailego/internal/cpu_features.h>
#include <rabitqlib/index/estimator.hpp>
#include "zvec/ailego/internal/platform.h"
#include "hnsw_rabitq_entity.h"
#include "hnsw_rabitq_query_entity.h"

namespace zvec {
namespace core {

HnswRabitqQueryAlgorithm::HnswRabitqQueryAlgorithm(HnswRabitqEntity &entity,
                                                   size_t num_clusters,
                                                   RabitqMetricType metric_type)
    : entity_(entity),
      mt_(std::chrono::system_clock::now().time_since_epoch().count()),
      lock_pool_(kLockCnt),
      num_clusters_(num_clusters),
      metric_type_(metric_type) {
  ex_bits_ = entity_.ex_bits();
  padded_dim_ = entity_.padded_dim();
  ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);
  LOG_INFO(
      "Create query algorithm. num_clusters=%zu ex_bits=%zu padded_dim=%zu",
      num_clusters_, ex_bits_, padded_dim_);
}

int HnswRabitqQueryAlgorithm::cleanup() {
  return 0;
}

int HnswRabitqQueryAlgorithm::search(HnswRabitqQueryEntity *entity,
                                     HnswRabitqContext *ctx) const {
  spin_lock_.lock();
  auto maxLevel = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  EstimateRecord curest;
  get_bin_est(entity_.get_vector(entry_point), curest, *entity);

  for (level_t cur_level = maxLevel; cur_level >= 1; --cur_level) {
    select_entry_point(cur_level, &entry_point, &curest, ctx, entity);
  }

  auto &topk_heap = ctx->topk_heap();
  topk_heap.clear();
  search_neighbors(0, &entry_point, &curest, topk_heap, ctx, entity);

  if (ctx->group_by_search()) {
    expand_neighbors_by_group(topk_heap, ctx, entity);
  }

  return 0;
}


//! select_entry_point on hnsw level, ef = 1
void HnswRabitqQueryAlgorithm::select_entry_point(
    level_t level, node_id_t *entry_point, EstimateRecord *curest,
    HnswRabitqContext *ctx, HnswRabitqQueryEntity *query_entity) const {
  auto &entity = ctx->get_entity();
  while (true) {
    const Neighbors neighbors = entity.get_neighbors(level, *entry_point);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }
    ailego_prefetch(neighbors.data);
    uint32_t size = neighbors.size();
    if (size == 0) {
      break;
    }

    bool find_closer = false;
    for (uint32_t i = 0; i < size; ++i) {
      EstimateRecord candest;
      get_bin_est(entity_.get_vector(neighbors[i]), candest, *query_entity);

      if (candest.est_dist < curest->est_dist) {
        *curest = candest;
        *entry_point = neighbors[i];
        find_closer = true;
      }
    }

    if (!find_closer) {
      break;
    }
  }

  return;
}

void HnswRabitqQueryAlgorithm::search_neighbors(
    level_t level, node_id_t *entry_point, EstimateRecord *dist, TopkHeap &topk,
    HnswRabitqContext *ctx, HnswRabitqQueryEntity *query_entity) const {
  const auto &entity = ctx->get_entity();
  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();
  std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
  if (ctx->filter().is_valid()) {
    filter = [&](node_id_t id) { return ctx->filter()(entity.get_key(id)); };
  }

  candidates.clear();
  visit.clear();
  visit.set_visited(*entry_point);
  if (!filter(*entry_point)) {
    topk.emplace(*entry_point, ResultRecord(*dist));
  }

  candidates.emplace(*entry_point, ResultRecord(*dist));
  while (!candidates.empty() && !ctx->reach_scan_limit()) {
    auto top = candidates.begin();
    node_id_t main_node = top->first;
    auto main_dist = top->second;

    if (topk.full() && main_dist.est_dist > topk[0].second.est_dist) {
      break;
    }

    candidates.pop();
    const Neighbors neighbors = entity.get_neighbors(level, main_node);
    ailego_prefetch(neighbors.data);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }

    std::vector<node_id_t> neighbor_ids(neighbors.size());
    uint32_t size = 0;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) {
        if (ailego_unlikely(ctx->debugging())) {
          (*ctx->mutable_stats_visit_dup_cnt())++;
        }
        continue;
      }
      visit.set_visited(node);
      neighbor_ids[size++] = node;
    }
    if (size == 0) {
      continue;
    }

    for (uint32_t i = 0; i < size; ++i) {
      node_id_t node = neighbor_ids[i];
      EstimateRecord candest;
      auto *cand_vector = entity_.get_vector(node);
      ailego_prefetch(cand_vector);
      get_bin_est(cand_vector, candest, *query_entity);

      if (ex_bits_ > 0) {
        // Check preliminary score against current worst full estimate.
        bool flag_update_KNNs =
            (!topk.full()) || candest.low_dist < topk[0].second.est_dist;

        if (flag_update_KNNs) {
          // Compute the full estimate if promising.
          get_full_est(cand_vector, candest, *query_entity);
        } else {
          continue;
        }
      } else {
        // ex_bits_ == 0: est_dist is already the best estimate
        if (topk.full() && candest.est_dist >= topk[0].second.est_dist) {
          continue;
        }
      }
      candidates.emplace(node, ResultRecord(candest));
      // update entry_point for next level scan
      if (candest < *dist) {
        *entry_point = node;
        *dist = candest;
      }
      if (!filter(node)) {
        topk.emplace(node, ResultRecord(candest));
      }
    }  // end for
  }  // while

  return;
}

void HnswRabitqQueryAlgorithm::expand_neighbors_by_group(
    TopkHeap &topk, HnswRabitqContext *ctx,
    HnswRabitqQueryEntity *query_entity) const {
  if (!ctx->group_by().is_valid()) {
    return;
  }

  const auto &entity = ctx->get_entity();
  std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
    return ctx->group_by()(entity.get_key(id));
  };

  // devide into groups
  std::map<std::string, TopkHeap> &group_topk_heaps = ctx->group_topk_heaps();
  for (uint32_t i = 0; i < topk.size(); ++i) {
    node_id_t id = topk[i].first;
    auto score = topk[i].second;

    std::string group_id = group_by(id);

    auto &topk_heap = group_topk_heaps[group_id];
    if (topk_heap.empty()) {
      topk_heap.limit(ctx->group_topk());
    }
    topk_heap.emplace_back(id, score);
  }

  // stage 2, expand to reach group num as possible
  if (group_topk_heaps.size() < ctx->group_num()) {
    VisitFilter &visit = ctx->visit_filter();
    CandidateHeap &candidates = ctx->candidates();

    std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
    if (ctx->filter().is_valid()) {
      filter = [&](node_id_t id) { return ctx->filter()(entity.get_key(id)); };
    }

    // refill to get enough groups
    candidates.clear();
    visit.clear();
    for (uint32_t i = 0; i < topk.size(); ++i) {
      node_id_t id = topk[i].first;
      auto score = topk[i].second;

      visit.set_visited(id);
      candidates.emplace_back(id, score);
    }

    // do expand
    while (!candidates.empty() && !ctx->reach_scan_limit()) {
      auto top = candidates.begin();
      node_id_t main_node = top->first;

      candidates.pop();
      const Neighbors neighbors = entity.get_neighbors(0, main_node);
      ailego_prefetch(neighbors.data);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_neighbors())++;
      }

      std::vector<node_id_t> neighbor_ids(neighbors.size());
      uint32_t size = 0;
      for (uint32_t i = 0; i < neighbors.size(); ++i) {
        node_id_t node = neighbors[i];
        if (visit.visited(node)) {
          if (ailego_unlikely(ctx->debugging())) {
            (*ctx->mutable_stats_visit_dup_cnt())++;
          }
          continue;
        }
        visit.set_visited(node);
        neighbor_ids[size++] = node;
      }
      if (size == 0) {
        continue;
      }

      for (uint32_t i = 0; i < size; ++i) {
        node_id_t node = neighbor_ids[i];
        EstimateRecord candest;
        auto *cand_vector = entity_.get_vector(node);
        ailego_prefetch(cand_vector);
        get_full_est(cand_vector, candest, *query_entity);

        if (!filter(node)) {
          std::string group_id = group_by(node);

          auto &topk_heap = group_topk_heaps[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace_back(node, ResultRecord(candest));

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }
        candidates.emplace(node, ResultRecord(candest));
      }  // end for
    }  // end while
  }  // end if
}

void HnswRabitqQueryAlgorithm::get_bin_est(
    const void *vector, EstimateRecord &res,
    HnswRabitqQueryEntity &entity) const {
  const auto &q_to_centroids = entity.q_to_centroids;
  auto &query_wrapper = *entity.query_wrapper;
  uint32_t cluster_id = entity_.get_cluster_id(vector);
  const char *bin_data = entity_.get_bin_data(vector);
  if (metric_type_ == RabitqMetricType::kIP) {
    float norm = q_to_centroids[cluster_id];
    float error = q_to_centroids[cluster_id + num_clusters_];
    rabitqlib::split_single_estdist(bin_data, query_wrapper, padded_dim_,
                                    res.ip_x0_qr, res.est_dist, res.low_dist,
                                    -norm, error);
  } else {
    // L2 distance
    float norm = q_to_centroids[cluster_id];
    rabitqlib::split_single_estdist(bin_data, query_wrapper, padded_dim_,
                                    res.ip_x0_qr, res.est_dist, res.low_dist,
                                    norm * norm, norm);
  }
}

void HnswRabitqQueryAlgorithm::get_full_est(
    const void *vector, EstimateRecord &res,
    HnswRabitqQueryEntity &entity) const {
  const auto &q_to_centroids = entity.q_to_centroids;
  auto &query_wrapper = *entity.query_wrapper;
  uint32_t cluster_id = entity_.get_cluster_id(vector);
  const char *bin_data = entity_.get_bin_data(vector);
  const char *ex_data = entity_.get_ex_data(vector);

  if (metric_type_ == RabitqMetricType::kIP) {
    float norm = q_to_centroids[cluster_id];
    float error = q_to_centroids[cluster_id + num_clusters_];
    rabitqlib::split_single_fulldist(bin_data, ex_data, ip_func_, query_wrapper,
                                     padded_dim_, ex_bits_, res.est_dist,
                                     res.low_dist, res.ip_x0_qr, -norm, error);
  } else {
    // L2 distance
    float norm = q_to_centroids[cluster_id];
    rabitqlib::split_single_fulldist(
        bin_data, ex_data, ip_func_, query_wrapper, padded_dim_, ex_bits_,
        res.est_dist, res.low_dist, res.ip_x0_qr, norm * norm, norm);
  }
}


}  // namespace core
}  // namespace zvec
