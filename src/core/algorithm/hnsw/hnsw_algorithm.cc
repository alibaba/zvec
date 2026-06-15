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
#include "hnsw_algorithm.h"
#include <type_traits>

namespace zvec {
namespace core {

template <typename EntityType>
int HnswAlgorithm<EntityType>::add_node(node_id_t id, level_t level,
                                        HnswContext *ctx) {
  spin_lock_.lock();

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
  dist_t dist = ctx->dist_calculator().batch_dist(entry_point);
  for (; cur_level > level; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  for (; cur_level >= 0; --cur_level) {
    search_neighbors(cur_level, &entry_point, &dist, ctx->level_topk(cur_level),
                     ctx, /*use_pool=*/false);
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

template <typename EntityType>
int HnswAlgorithm<EntityType>::search(HnswContext *ctx) const {
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
  search_neighbors(0, &entry_point, &dist, topk_heap, ctx, /*use_pool=*/true);

  if (ctx->group_by_search()) {
    expand_neighbors_by_group(topk_heap, ctx);
  }

  return 0;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::select_entry_point(level_t level,
                                                   node_id_t *entry_point,
                                                   dist_t *dist,
                                                   HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  HnswDistCalculator &dc = ctx->dist_calculator();
  while (true) {
    const auto neighbors = entity.get_neighbors_typed(level, *entry_point);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }
    uint32_t size = neighbors.size();
    if (size == 0) {
      break;
    }

    std::vector<MemBlockType> neighbor_vec_blocks;
    int ret = entity.get_vector_typed(&neighbors[0], size, neighbor_vec_blocks);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    bool find_closer = false;

    std::vector<float> dists(size);
    std::vector<const void *> neighbor_vecs(size);
    for (uint32_t i = 0; i < size; ++i) {
      neighbor_vecs[i] = neighbor_vec_blocks[i].data();
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data());

    for (uint32_t i = 0; i < size; ++i) {
      dist_t cur_dist = dists[i];

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

template <typename EntityType>
void HnswAlgorithm<EntityType>::add_neighbors(node_id_t id, level_t level,
                                              TopkHeap &topk_heap,
                                              HnswContext *ctx) {
  if (ailego_unlikely(topk_heap.size() == 0)) {
    return;
  }

  HnswDistCalculator &dc = ctx->dist_calculator();

  update_neighbors(dc, id, level, topk_heap);

  // reverse update neighbors
  for (size_t i = 0; i < topk_heap.size(); ++i) {
    reverse_update_neighbors(dc, topk_heap[i].first, level, id,
                             topk_heap[i].second, ctx->update_heap());
  }

  return;
}

// ============================================================================
// search_neighbors helper templates
//
// Two specialized inner loops, dispatched from search_neighbors():
//
//   fast_search_neighbors:       mmap/contiguous with direct vector pointers.
//                                Uses BlockHeap (AVX2) or LinearPool (scalar)
//                                for visited tracking and top-k maintenance.
//   dual_heap_search_neighbors:  CandidateHeap + TopkHeap + VisitFilter.
//                                Used for add_node (use_pool=false), filtered
//                                search, upper levels, and BufferPool fallback.
// ============================================================================

// mmap/contiguous variant: resolve vectors via get_vector_ptr and use
// LinearPool or BlockHeap for visited tracking + top-k maintenance.
// HeapType must expose reset/set_visited/check_visited/push_block/has_next/pop.
template <typename EntityType, typename HeapType>
void fast_search_neighbors(const EntityType &entity, HeapType &pool,
                           VisitFilter &visit, HnswDistCalculator &dc,
                           uint32_t topk, uint32_t ef, node_id_t entry_point,
                           dist_t entry_dist, uint32_t prefetch_lines,
                           uint32_t prefetch_offset) {
  const uint32_t max_deg = entity.max_degree(0);  // level 0 only
  const uint32_t cap = std::max(topk, ef);
  pool.reset(static_cast<int32_t>(cap), static_cast<int32_t>(max_deg));
  visit.clear();

  visit.set_visited(entry_point);
  pool.push_block(&entry_dist, &entry_point, 1);

  uint32_t buf_capacity = max_deg;
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);

  while (pool.has_next()) {
    auto current_node = pool.pop();

    const auto neighbors = entity.get_neighbors_typed(0, current_node);
    ailego_prefetch(neighbors.data);

    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
    }

    const uint32_t po =
        std::min(static_cast<uint32_t>(neighbors.size()), prefetch_offset);
    uint32_t unvisited_count = 0;
    uint32_t i = 0;

    // Phase 1: scan first `po` neighbors with prefetch.
    for (; i < po; ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      const void *vec_ptr = entity.get_vector_ptr(node);
      const char *p = reinterpret_cast<const char *>(vec_ptr);
      for (uint32_t cl = 0; cl < prefetch_lines; ++cl) {
        ailego_prefetch(p + cl * 64);
      }
      neighbor_ids[unvisited_count] = node;
      neighbor_vecs[unvisited_count] = vec_ptr;
      unvisited_count++;
    }

    // Phase 2: scan remaining neighbors.
    for (; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      neighbor_ids[unvisited_count] = node;
      neighbor_vecs[unvisited_count] = entity.get_vector_ptr(node);
      unvisited_count++;
    }

    if (unvisited_count == 0) continue;
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data());

    pool.push_block(dists.data(), neighbor_ids.data(),
                    static_cast<int32_t>(unvisited_count));
  }
}

// ============================================================================
// dual_heap_search_neighbors: shared core for the fallback dual-heap path.
//
// Maintains a candidate min-heap + topk heap + VisitFilter.  Supports
// arbitrary levels, filters, and MemoryBlock types (BufferPool/Mmap).
// Also updates entry_point/dist for next-level continuation.
// ============================================================================
template <typename EntityType, typename MemBlockType, typename FilterFn>
void dual_heap_search_neighbors(const EntityType &entity, level_t level,
                                node_id_t *entry_point, dist_t *dist,
                                TopkHeap &topk, HnswContext *ctx,
                                HnswDistCalculator &dc, FilterFn &&filter) {
  const uint32_t prefetch_offset = ctx->po();
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;

  uint32_t buf_capacity = entity.max_degree(level);
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<MemBlockType> neighbor_vec_blocks;
  neighbor_vec_blocks.reserve(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);

  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();

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
    const auto neighbors = entity.get_neighbors_typed(level, main_node);
    ailego_prefetch(neighbors.data);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_neighbors())++;
    }

    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      neighbor_vec_blocks.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
    }

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

    neighbor_vec_blocks.clear();
    int ret =
        entity.get_vector_typed(neighbor_ids.data(), size, neighbor_vec_blocks);
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    // do prefetch
    for (uint32_t i = 0; i < std::min(prefetch_offset, size); ++i) {
      const char *base =
          static_cast<const char *>(neighbor_vec_blocks[i].data());
      for (uint32_t cl = 0; cl < prefetch_lines; ++cl) {
        ailego_prefetch(base + cl * 64);
      }
    }

    for (uint32_t i = 0; i < size; ++i) {
      neighbor_vecs[i] = neighbor_vec_blocks[i].data();
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data());

    for (uint32_t i = 0; i < size; ++i) {
      node_id_t node = neighbor_ids[i];
      dist_t cur_dist = dists[i];

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
      }
    }
  }
}

// ============================================================================
// search_neighbors: Dispatch to fast or dual-heap path.
//
// - add_node / filtered / upper levels  →  dual_heap_search_neighbors
// - level-0 unfiltered search:
//     MmapMemoryBlock  →  fast_search_neighbors (BlockHeap/LinearPool)
//     BufferPool       →  dual_heap_search_neighbors (fallback)
// ============================================================================
template <typename EntityType>
void HnswAlgorithm<EntityType>::search_neighbors(level_t level,
                                                 node_id_t *entry_point,
                                                 dist_t *dist, TopkHeap &topk,
                                                 HnswContext *ctx,
                                                 bool use_pool) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  HnswDistCalculator &dc = ctx->dist_calculator();

  if (!use_pool || ctx->filter().is_valid() || level != 0) {
    // Dual-heap path: add_node, filtered search, or upper-level scan.
    auto run_with_filter = [&](auto &&filter) {
      dual_heap_search_neighbors<EntityType, MemBlockType>(
          entity, level, entry_point, dist, topk, ctx, dc,
          std::forward<decltype(filter)>(filter));
    };

    if (ctx->filter().is_valid()) {
      auto filter = [&](node_id_t id) {
        return ctx->filter()(entity.get_key_typed(id));
      };
      run_with_filter(filter);
    } else {
      auto filter = [](node_id_t) { return false; };
      run_with_filter(filter);
    }
  } else {
    // Pool-based path for level-0 unfiltered search.
    if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
      const uint32_t prefetch_lines =
          ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;

      // Fast path: direct pointer access via get_vector_ptr.
      // BlockHeap (AVX2) or LinearPool (scalar) for top-k tracking.
      const uint32_t topk_v = static_cast<uint32_t>(ctx->topk());
      const uint32_t ef_v = ctx->ef();
      const bool avx2_ok =
          zvec::ailego::internal::CpuFeatures::static_flags_.AVX2;

      auto &visit = ctx->visit_filter();

      if (avx2_ok) {
        auto &bpool = ctx->block_pool();
        fast_search_neighbors(entity, bpool, visit, dc, topk_v, ef_v,
                              *entry_point, *dist, prefetch_lines, ctx->po());
        copy_pool_to_topk(bpool, topk);
      } else {
        auto &lpool = ctx->pool();
        fast_search_neighbors(entity, lpool, visit, dc, topk_v, ef_v,
                              *entry_point, *dist, prefetch_lines, ctx->po());
        copy_pool_to_topk(lpool, topk);
      }
    } else {
      // BufferPool entities: fallback to dual-heap path.
      auto filter = [](node_id_t) { return false; };
      dual_heap_search_neighbors<EntityType, MemBlockType>(
          entity, level, entry_point, dist, topk, ctx, dc, filter);
    }
  }
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::expand_neighbors_by_group(
    TopkHeap &topk, HnswContext *ctx) const {
  if (!ctx->group_by().is_valid()) {
    return;
  }

  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
    return ctx->group_by()(entity.get_key_typed(id));
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
    HnswDistCalculator &dc = ctx->dist_calculator();

    std::function<bool(node_id_t)> filter = [](node_id_t) { return false; };
    if (ctx->filter().is_valid()) {
      filter = [&](node_id_t id) {
        return ctx->filter()(entity.get_key_typed(id));
      };
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
      const auto neighbors = entity.get_neighbors_typed(0, main_node);
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

      std::vector<MemBlockType> neighbor_vec_blocks;
      int ret = entity.get_vector_typed(neighbor_ids.data(), size,
                                        neighbor_vec_blocks);
      if (ailego_unlikely(ctx->debugging())) {
        (*ctx->mutable_stats_get_vector())++;
      }
      if (ailego_unlikely(ret != 0)) {
        break;
      }

      std::vector<float> dists(size);
      std::vector<const void *> neighbor_vecs(size);
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
      }
      dc.batch_dist(neighbor_vecs.data(), size, dists.data());

      for (uint32_t i = 0; i < size; ++i) {
        node_id_t node = neighbor_ids[i];
        dist_t cur_dist = dists[i];

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
      }
    }
  }
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::update_neighbors(HnswDistCalculator &dc,
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

template <typename EntityType>
void HnswAlgorithm<EntityType>::reverse_update_neighbors(
    HnswDistCalculator &dc, node_id_t id, level_t level, node_id_t link_id,
    dist_t dist, TopkHeap &update_heap) {
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

// Explicit template instantiation
template class HnswAlgorithm<HnswMmapStreamerEntity>;
template class HnswAlgorithm<HnswBufferPoolStreamerEntity>;
template class HnswAlgorithm<HnswContiguousStreamerEntity>;

}  // namespace core
}  // namespace zvec
