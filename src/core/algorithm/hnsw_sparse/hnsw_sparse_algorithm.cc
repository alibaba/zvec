
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
#include "hnsw_sparse_algorithm.h"
#include <chrono>
#include <iostream>
#include <ailego/internal/cpu_features.h>

namespace zvec {
namespace core {

HnswSparseAlgorithm::HnswSparseAlgorithm(HnswSparseEntity &entity)
    : entity_(entity),
      mt_(std::chrono::system_clock::now().time_since_epoch().count()),
      lock_pool_(kLockCnt) {}

int HnswSparseAlgorithm::cleanup() {
  return 0;
}

int HnswSparseAlgorithm::add_node(node_id_t id, level_t level,
                                  HnswSparseContext *ctx) {
  spin_lock_.lock();

  // std::cout << "id: " << id << ", level: " << level << std::endl;

  auto cur_max_level = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    entity_.update_ep_and_level(id, level);
    spin_lock_.unlock();
    return 0;
  }
  spin_lock_.unlock();

  if (ailego_unlikely(level > cur_max_level)) {
    mutex_.lock();
    // re-check max level
    cur_max_level = entity_.cur_max_level();
    entry_point = entity_.entry_point();
    if (level <= cur_max_level) {
      mutex_.unlock();
    }
  }

  level_t cur_level = cur_max_level;
  dist_t dist = ctx->dist_calculator()(entry_point);
  for (; cur_level > level; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  for (; cur_level >= 0; --cur_level) {
    search_neighbors(cur_level, &entry_point, &dist, ctx->level_topk(cur_level),
                     ctx);
  }

  // add neighbors from down level to top level, to avoid upper level visible
  // to knn_search but the under layer level not ready
  for (cur_level = 0; cur_level <= level; ++cur_level) {
    add_neighbors(id, cur_level, ctx->level_topk(cur_level), ctx);
    ctx->level_topk(cur_level).clear();
  }

  if (ailego_unlikely(level > cur_max_level)) {
    spin_lock_.lock();
    entity_.update_ep_and_level(id, level);
    spin_lock_.unlock();
    mutex_.unlock();
  }

  return 0;
}

int HnswSparseAlgorithm::search(HnswSparseContext *ctx) const {
  spin_lock_.lock();
  auto maxLevel = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  dist_t dist = ctx->dist_calculator().dist(entry_point);
  for (level_t cur_level = maxLevel; cur_level >= 1; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  auto &topk_heap = ctx->topk_heap();
  topk_heap.clear();
  search_neighbors(0, &entry_point, &dist, topk_heap, ctx);

  if (ctx->group_by_search()) {
    expand_neighbors_by_group(topk_heap, ctx);
  }

  return 0;
}

//! select_entry_point on hnsw level, ef = 1
void HnswSparseAlgorithm::select_entry_point(level_t level,
                                             node_id_t *entry_point,
                                             dist_t *dist,
                                             HnswSparseContext *ctx) const {
  auto &entity = ctx->get_entity();
  HnswSparseDistCalculator &dc = ctx->dist_calculator();
  while (true) {
    const Neighbors neighbors = entity.get_neighbors(level, *entry_point);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }
    uint32_t size = neighbors.size();
    if (size == 0) {
      break;
    }

    std::vector<IndexStorage::MemoryBlock> neighbor_block_vecs;
    int ret = entity.get_vector_metas(&neighbors[0], size, neighbor_block_vecs);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }
    bool find_closer = false;
    for (uint32_t i = 0; i < size; ++i) {
      dist_t cur_dist = dc.dist(neighbor_block_vecs[i].data());
      if (cur_dist < *dist) {
        *entry_point = neighbors[i];
        *dist = cur_dist;
        find_closer = true;
      }
    }

    if (!find_closer) {
      break;
    }
  }

  return;
}

void HnswSparseAlgorithm::add_neighbors(node_id_t id, level_t level,
                                        TopkHeap &topk_heap,
                                        HnswSparseContext *ctx) {
  if (ailego_unlikely(topk_heap.size() == 0)) {
    return;
  }

  HnswSparseDistCalculator &dc = ctx->dist_calculator();

  update_neighbors(dc, id, level, topk_heap);

  // reverse update neighbors
  for (size_t i = 0; i < topk_heap.size(); ++i) {
    reverse_update_neighbors(dc, topk_heap[i].first, level, id,
                             topk_heap[i].second, ctx->update_heap());
  }

  return;
}

void HnswSparseAlgorithm::search_neighbors(level_t level,
                                           node_id_t *entry_point, dist_t *dist,
                                           TopkHeap &topk,
                                           HnswSparseContext *ctx) const {
  const auto &entity = ctx->get_entity();
  HnswSparseDistCalculator &dc = ctx->dist_calculator();
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
    topk.emplace(*entry_point, *dist);
  }

  candidates.emplace(*entry_point, *dist);
  while (!candidates.empty() && !ctx->reach_scan_limit()) {
    auto top = candidates.begin();
    node_id_t main_node = top->first;
    dist_t main_dist = top->second;

    if (topk.full() && main_dist > topk[0].second) {
      break;
    }

    candidates.pop();
    const Neighbors neighbors = entity.get_neighbors(level, main_node);
    ailego_prefetch(neighbors.data);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }

    node_id_t neighbor_ids[neighbors.size()];
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

    std::vector<IndexStorage::MemoryBlock> neighbor_block_vecs;
    int ret = entity.get_vector_metas(neighbor_ids, size, neighbor_block_vecs);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    static constexpr node_id_t PREFETCH_STEP = 2;
    static constexpr node_id_t SPARSE_PREFETCH_STEP = 1;

    for (uint32_t i = 0; i < std::min(PREFETCH_STEP, size); ++i) {
      ailego_prefetch(neighbor_block_vecs[i].data());
    }
    for (uint32_t i = 0; i < size; ++i) {
      node_id_t node = neighbor_ids[i];
      node_id_t prefetch_id = i + PREFETCH_STEP;
      if (prefetch_id < size) {
        ailego_prefetch(neighbor_block_vecs[prefetch_id].data());
      }

      node_id_t sparse_prefetch_id = i + SPARSE_PREFETCH_STEP;
      if (sparse_prefetch_id < size) {
        IndexStorage::MemoryBlock sparse_block;
        int sparse_length = 0;
        entity.get_sparse_data_from_vector(
            neighbor_block_vecs[sparse_prefetch_id].data(), sparse_block,
            sparse_length);
        auto sparse_data = std::make_pair(sparse_block.data(), sparse_length);
        if (sparse_data.first != nullptr) {
          ailego_prefetch(sparse_data.first);
        }
      }

      dist_t cur_dist = dc.dist(neighbor_block_vecs[i].data());
      if ((!topk.full()) || cur_dist < topk[0].second) {
        candidates.emplace(node, cur_dist);
        // update entry_point for next level scan
        if (cur_dist < *dist) {
          *entry_point = node;
          *dist = cur_dist;
        }
        if (!filter(node)) {
          topk.emplace(node, cur_dist);
        }
      }  // end if
    }  // end for
  }  // while

  return;
}

void HnswSparseAlgorithm::expand_neighbors_by_group(
    TopkHeap &topk, HnswSparseContext *ctx) const {
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
    HnswSparseDistCalculator &dc = ctx->dist_calculator();

    std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
    if (ctx->filter().is_valid()) {
      filter = [&](node_id_t id) { return ctx->filter()(entity.get_key(id)); };
    }

    // refill to get enough groups
    candidates.clear();
    visit.clear();
    for (uint32_t i = 0; i < topk.size(); ++i) {
      node_id_t id = topk[i].first;
      float score = topk[i].second;

      visit.set_visited(id);
      candidates.emplace_back(id, score);
    }

    // do expand
    while (!candidates.empty() && !ctx->reach_scan_limit()) {
      auto top = candidates.begin();
      node_id_t main_node = top->first;

      candidates.pop();
      const Neighbors neighbors = entity.get_neighbors(0, main_node);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_neighbors())++;
      }

      node_id_t neighbor_ids[neighbors.size()];
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

      std::vector<IndexStorage::MemoryBlock> neighbor_block_vecs;
      int ret =
          entity.get_vector_metas(neighbor_ids, size, neighbor_block_vecs);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_vector())++;
      }
      if (ailego_unlikely(ret != 0)) {
        break;
      }

      static constexpr node_id_t PREFETCH_STEP = 2;
      for (uint32_t i = 0; i < size; ++i) {
        node_id_t node = neighbor_ids[i];
        node_id_t prefetch_id = i + PREFETCH_STEP;
        if (prefetch_id < size) {
          ailego_prefetch(neighbor_block_vecs[prefetch_id].data());
        }
        dist_t cur_dist = dc.dist(neighbor_block_vecs[i].data());

        if (!filter(node)) {
          std::string group_id = group_by(node);

          auto &topk_heap = group_topk_heaps[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace_back(node, cur_dist);

          if (group_topk_heaps.size() >= ctx->group_num()) {
            break;
          }
        }

        candidates.emplace(node, cur_dist);
      }  // end for
    }  // end while
  }  // end if
}

void HnswSparseAlgorithm::update_neighbors(HnswSparseDistCalculator &dc,
                                           node_id_t id, level_t level,
                                           TopkHeap &topk_heap) {
  topk_heap.sort();

  uint32_t max_neighbor_cnt = entity_.neighbor_cnt(level);
  if (topk_heap.size() <= static_cast<size_t>(entity_.prune_cnt())) {
    if (topk_heap.size() <= static_cast<size_t>(max_neighbor_cnt)) {
      entity_.update_neighbors(level, id, topk_heap);
      return;
    }
  }

  uint32_t cur_size = 0;
  for (size_t i = 0; i < topk_heap.size(); ++i) {
    node_id_t cur_node = topk_heap[i].first;
    dist_t cur_node_dist = topk_heap[i].second;
    bool good = true;
    for (uint32_t j = 0; j < cur_size; ++j) {
      dist_t tmp_dist = dc.dist(cur_node, topk_heap[j].first);
      if (tmp_dist <= cur_node_dist) {
        good = false;
        break;
      }
    }

    if (good) {
      topk_heap[cur_size].first = cur_node;
      topk_heap[cur_size].second = cur_node_dist;
      cur_size++;
      if (cur_size >= max_neighbor_cnt) {
        break;
      }
    }
  }

  // when after-prune neighbor count is too seldom,
  // we use this strategy to make-up enough edges
  // not only just make-up out-degrees
  // we also make-up enough in-degrees
  uint32_t min_neighbors = entity_.min_neighbor_cnt();
  for (size_t k = cur_size; cur_size < min_neighbors && k < topk_heap.size();
       ++k) {
    bool exist = false;
    for (size_t j = 0; j < cur_size; ++j) {
      if (topk_heap[j].first == topk_heap[k].first) {
        exist = true;
        break;
      }
    }
    if (!exist) {
      topk_heap[cur_size].first = topk_heap[k].first;
      topk_heap[cur_size].second = topk_heap[k].second;
      cur_size++;
    }
  }

  topk_heap.resize(cur_size);
  entity_.update_neighbors(level, id, topk_heap);

  return;
}

void HnswSparseAlgorithm::reverse_update_neighbors(HnswSparseDistCalculator &dc,
                                                   node_id_t id, level_t level,
                                                   node_id_t link_id,
                                                   dist_t dist,
                                                   TopkHeap &update_heap) {
  const size_t max_neighbor_cnt = entity_.neighbor_cnt(level);

  uint32_t lock_idx = id & kLockMask;
  lock_pool_[lock_idx].lock();
  const Neighbors neighbors = entity_.get_neighbors(level, id);
  size_t size = neighbors.size();
  ailego_assert_with(size <= max_neighbor_cnt, "invalid neighbor size");
  if (size < max_neighbor_cnt) {
    entity_.add_neighbor(level, id, size, link_id);
    lock_pool_[lock_idx].unlock();
    return;
  }

  update_heap.emplace(link_id, dist);

  for (size_t i = 0; i < size; ++i) {
    node_id_t node = neighbors[i];
    dist_t cur_dist = dc.dist(id, node);
    update_heap.emplace(node, cur_dist);
  }

  //! TODO: optimize prune
  //! prune edges
  update_heap.sort();
  size_t cur_size = 0;
  for (size_t i = 0; i < update_heap.size(); ++i) {
    node_id_t cur_node = update_heap[i].first;
    dist_t cur_node_dist = update_heap[i].second;
    bool good = true;
    for (size_t j = 0; j < cur_size; ++j) {
      dist_t tmp_dist = dc.dist(cur_node, update_heap[j].first);
      if (tmp_dist <= cur_node_dist) {
        good = false;
        break;
      }
    }

    if (good) {
      update_heap[cur_size].first = cur_node;
      update_heap[cur_size].second = cur_node_dist;
      cur_size++;
      if (cur_size >= max_neighbor_cnt) {
        break;
      }
    }
  }

  update_heap.resize(cur_size);
  entity_.update_neighbors(level, id, update_heap);

  lock_pool_[lock_idx].unlock();

  update_heap.clear();

  return;
}

}  // namespace core
}  // namespace zvec