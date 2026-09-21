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
  dist_t dist = ctx->batch_dist(entry_point);
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

template <typename EntityType>
int HnswAlgorithm<EntityType>::search(HnswContext *ctx) const {
  spin_lock_.lock();
  auto max_level = entity_.cur_max_level();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  dist_t dist = ctx->dist_calculator().dist(entry_point);
  for (level_t cur_level = max_level; cur_level >= 1; --cur_level) {
    select_entry_point(cur_level, &entry_point, &dist, ctx);
  }

  const uint32_t capacity = std::max(ctx->topk(), ctx->ef());
  if (!ctx->filter().is_valid()) {
    ctx->search_heap().reset_pool(capacity, entity_.max_degree(0));
    const int ret = dispatch_search_neighbors(entry_point, dist, ctx);
    if (ailego_unlikely(ret != 0)) return ret;
  } else {
    auto &topk = ctx->search_heap().reset<TopkHeap>(capacity);
    search_neighbors(0, &entry_point, &dist, topk, ctx);
  }

  if (ctx->group_by_search()) {
    expand_neighbors_by_group(ctx);
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
  const bool use_provider = dc.has_provider();
  const bool has_extra_values = ctx->has_extra_values();
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
    std::vector<IndexStorage::MemoryBlock> provider_vec_blocks;
    int ret;
    if (ailego_unlikely(use_provider)) {
      ret = dc.get_vector(&neighbors[0], size, provider_vec_blocks);
    } else {
      ret = entity.get_vector_typed(&neighbors[0], size, neighbor_vec_blocks);
    }
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    bool find_closer = false;

    std::vector<float> dists(size);
    std::vector<const void *> neighbor_vecs(size);
    std::vector<const void *> neighbor_extra_values(has_extra_values ? size
                                                                     : 0);
    if (ailego_unlikely(use_provider)) {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = provider_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    } else {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);

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

  update_neighbors(dc, id, level, topk_heap, ctx);

  // reverse update neighbors
  for (size_t i = 0; i < topk_heap.size(); ++i) {
    reverse_update_neighbors(dc, topk_heap[i].first, level, id,
                             topk_heap[i].second, ctx->update_heap(), ctx);
  }

  return;
}

// ============================================================================
// Search helper templates
//
// The query boundary selects a specialized inner loop:
//
//   fast_search_neighbors:       mmap/contiguous with direct vector pointers.
//   fast_search_neighbors_buffer: BufferStorage with page-backed MemoryBlocks.
//                                Both use BlockHeap (AVX2) or LinearPool
//                                (scalar) plus a concrete VisitFilterView.
//   dual_heap_search_neighbors:  CandidateHeap + TopkHeap + VisitFilter.
//                                Used for add_node, filtered
//                                search and upper levels for every backend.
// ============================================================================

// mmap/contiguous variant: resolve vectors via get_vector_ptr and use
// LinearPool or BlockHeap for top-k maintenance, with a concrete visit view.
// HeapType must expose push_block/has_next/pop; callers reset it before search.
template <typename EntityType, typename HeapType, typename Visit>
void fast_search_neighbors(const EntityType &entity, HeapType &pool,
                           Visit visit, HnswDistCalculator &dc,
                           HnswContext *ctx, node_id_t entry_point,
                           dist_t entry_dist, uint32_t prefetch_lines,
                           uint32_t prefetch_offset) {
  const uint32_t max_deg = entity.max_degree(0);  // level 0 only

  visit.set_visited(entry_point);
  pool.push_block(&entry_dist, &entry_point, 1);

  uint32_t buf_capacity = max_deg;
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);
  const bool has_extra_values = ctx->has_extra_values();
  std::vector<const void *> neighbor_extra_values(
      has_extra_values ? buf_capacity : 0);

  while (pool.has_next()) {
    auto current_node = pool.pop();

    const auto neighbors = entity.get_neighbors_typed(0, current_node);
    ailego_prefetch(neighbors.data);

    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
      if (has_extra_values) {
        neighbor_extra_values.resize(buf_capacity);
      }
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
      if (has_extra_values) {
        neighbor_extra_values[unvisited_count] = ctx->get_extra_values(vec_ptr);
      }
      unvisited_count++;
    }

    // Phase 2: scan remaining neighbors.
    for (; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      neighbor_ids[unvisited_count] = node;
      neighbor_vecs[unvisited_count] = entity.get_vector_ptr(node);
      if (has_extra_values) {
        neighbor_extra_values[unvisited_count] =
            ctx->get_extra_values(neighbor_vecs[unvisited_count]);
      }
      unvisited_count++;
    }

    if (unvisited_count == 0) continue;
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);

    pool.push_block(dists.data(), neighbor_ids.data(),
                    static_cast<int32_t>(unvisited_count));
  }
}

// BufferStorage variant of the level-0 fast path.  It intentionally keeps the
// MemoryBlocks alive through batch_dist(): a buffer-pool page may be evicted as
// soon as its last block is released.  Apart from vector resolution, this is
// the same graph traversal used by mmap, so selecting BufferStorage does not
// silently switch HNSW to the slower dual-heap algorithm.
template <typename EntityType, typename HeapType, typename Visit>
void fast_search_neighbors_buffer(const EntityType &entity, HeapType &pool,
                                  Visit visit, HnswDistCalculator &dc,
                                  HnswContext *ctx, node_id_t entry_point,
                                  dist_t entry_dist, uint32_t prefetch_lines,
                                  uint32_t prefetch_offset) {
  using MemBlockType = typename EntityType::MemoryBlock;

  const uint32_t max_deg = entity.max_degree(0);
  visit.set_visited(entry_point);
  pool.push_block(&entry_dist, &entry_point, 1);

  uint32_t buf_capacity = max_deg;
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);
  std::vector<MemBlockType> neighbor_vec_blocks;
  neighbor_vec_blocks.reserve(buf_capacity);
  const bool has_extra_values = ctx->has_extra_values();
  std::vector<const void *> neighbor_extra_values(
      has_extra_values ? buf_capacity : 0);

  while (pool.has_next()) {
    const auto current_node = pool.pop();
    const auto neighbors = entity.get_neighbors_typed(0, current_node);
    ailego_prefetch(neighbors.data);

    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
      neighbor_vec_blocks.reserve(buf_capacity);
      if (has_extra_values) {
        neighbor_extra_values.resize(buf_capacity);
      }
    }

    uint32_t unvisited_count = 0;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      const node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      neighbor_ids[unvisited_count++] = node;
    }
    if (unvisited_count == 0) continue;

    neighbor_vec_blocks.clear();
    if (ailego_unlikely(entity.get_vector_typed(neighbor_ids.data(),
                                                unvisited_count,
                                                neighbor_vec_blocks) != 0)) {
      break;
    }
    for (uint32_t i = 0; i < unvisited_count; ++i) {
      neighbor_vecs[i] = neighbor_vec_blocks[i].data();
      if (has_extra_values) {
        neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
      }
    }
    const uint32_t po = std::min(prefetch_offset, unvisited_count);
    for (uint32_t i = 0; i < po; ++i) {
      const char *p = static_cast<const char *>(neighbor_vecs[i]);
      for (uint32_t cl = 0; cl < prefetch_lines; ++cl) {
        ailego_prefetch(p + cl * 64);
      }
    }

    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);
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
  std::vector<IndexStorage::MemoryBlock> provider_vec_blocks;
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);
  const bool has_extra_values = ctx->has_extra_values();
  std::vector<const void *> neighbor_extra_values(
      has_extra_values ? buf_capacity : 0);

  const bool use_provider = dc.has_provider();
  if (ailego_unlikely(use_provider)) {
    provider_vec_blocks.reserve(buf_capacity);
  }

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
      if (has_extra_values) {
        neighbor_extra_values.resize(buf_capacity);
      }
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
    provider_vec_blocks.clear();
    int ret;
    if (ailego_unlikely(use_provider)) {
      ret = dc.get_vector(neighbor_ids.data(), size, provider_vec_blocks);
    } else {
      ret = entity.get_vector_typed(neighbor_ids.data(), size,
                                    neighbor_vec_blocks);
    }
    if (ailego_unlikely(ctx->debugging())) {
      (*ctx->mutable_stats_get_vector())++;
    }
    if (ailego_unlikely(ret != 0)) {
      break;
    }

    if (ailego_unlikely(use_provider)) {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = provider_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    } else {
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
    }

    // do prefetch
    for (uint32_t i = 0; i < std::min(prefetch_offset, size); ++i) {
      const char *base = static_cast<const char *>(neighbor_vecs[i]);
      for (uint32_t cl = 0; cl < prefetch_lines; ++cl) {
        ailego_prefetch(base + cl * 64);
      }
    }

    dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                  has_extra_values ? neighbor_extra_values.data() : nullptr);

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
// search_neighbors: fallback dual-heap path used for construction, filtered
// queries, and upper levels for every backend.
// ============================================================================
template <typename EntityType>
void HnswAlgorithm<EntityType>::search_neighbors(level_t level,
                                                 node_id_t *entry_point,
                                                 dist_t *dist, TopkHeap &topk,
                                                 HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  HnswDistCalculator &dc = ctx->dist_calculator();

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
    run_with_filter([](node_id_t) { return false; });
  }
}

template <typename EntityType>
int HnswAlgorithm<EntityType>::dispatch_search_neighbors(
    node_id_t entry_point, dist_t entry_dist, HnswContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;
  const bool dispatched =
      dispatch_visit_filter(ctx->visit_filter(), [&](auto visit) {
        visit.clear();
        ctx->search_heap().dispatch([&](auto &pool) {
          using Heap = std::decay_t<decltype(pool)>;
          if constexpr (!std::is_same_v<Heap, TopkHeap>) {
            if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
              fast_search_neighbors(entity, pool, visit, ctx->dist_calculator(),
                                    ctx, entry_point, entry_dist,
                                    prefetch_lines, ctx->po());
            } else {
              fast_search_neighbors_buffer(
                  entity, pool, visit, ctx->dist_calculator(), ctx, entry_point,
                  entry_dist, prefetch_lines, ctx->po());
            }
          }
        });
      });
  if (ailego_unlikely(!dispatched)) {
    LOG_ERROR("Failed to dispatch HNSW visit filter, mode %d",
              ctx->visit_filter().get_mode());
    return IndexError_Runtime;
  }
  return 0;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::expand_neighbors_by_group(
    HnswContext *ctx) const {
  if (!ctx->group_by().is_valid()) {
    return;
  }

  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  std::function<std::string(node_id_t)> group_by = [&](node_id_t id) {
    return ctx->group_by()(entity.get_key_typed(id));
  };

  // devide into groups
  std::map<std::string, TopkHeap> &group_topk_heaps = ctx->group_topk_heaps();
  ctx->search_heap().for_each([&](node_id_t id, dist_t score) {
    std::string group_id = group_by(id);

    auto &topk_heap = group_topk_heaps[group_id];
    if (topk_heap.empty()) {
      topk_heap.limit(ctx->group_topk());
    }
    topk_heap.emplace(id, score);
    return true;
  });

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
    ctx->search_heap().for_each([&](node_id_t id, dist_t score) {
      visit.set_visited(id);
      candidates.emplace(id, score);
      return true;
    });

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
      const bool has_extra_values = ctx->has_extra_values();
      std::vector<const void *> neighbor_extra_values(has_extra_values ? size
                                                                       : 0);
      for (uint32_t i = 0; i < size; ++i) {
        neighbor_vecs[i] = neighbor_vec_blocks[i].data();
        if (has_extra_values) {
          neighbor_extra_values[i] = ctx->get_extra_values(neighbor_vecs[i]);
        }
      }
      dc.batch_dist(neighbor_vecs.data(), size, dists.data(),
                    has_extra_values ? neighbor_extra_values.data() : nullptr);

      for (uint32_t i = 0; i < size; ++i) {
        node_id_t node = neighbor_ids[i];
        dist_t cur_dist = dists[i];

        if (!filter(node)) {
          std::string group_id = group_by(node);

          auto &topk_heap = group_topk_heaps[group_id];
          if (topk_heap.empty()) {
            topk_heap.limit(ctx->group_topk());
          }
          topk_heap.emplace(node, cur_dist);

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
                                                 TopkHeap &topk_heap,
                                                 HnswContext *ctx) {
  topk_heap.sort();

  uint32_t max_neighbor_cnt = entity_.neighbor_cnt(level);
  if (topk_heap.size() <= static_cast<size_t>(entity_.prune_cnt())) {
    if (topk_heap.size() <= static_cast<size_t>(max_neighbor_cnt)) {
      entity_.update_neighbors(level, id, topk_heap.container());
      return;
    }
  }

  uint32_t cur_size = 0;
  if constexpr (std::is_same_v<EntityType, HnswBufferPoolStreamerEntity>) {
    const auto &read_entity =
        static_cast<const EntityType &>(ctx->get_entity());
    auto &prune_ids = ctx->prune_ids();
    auto &prune_blocks = ctx->prune_blocks();
    auto &selected_indices = ctx->prune_selected_indices();

    prune_ids.clear();
    prune_ids.reserve(topk_heap.size());
    for (const auto &candidate : topk_heap) {
      prune_ids.emplace_back(candidate.first);
    }
    prune_blocks.clear();
    // The active build metric may describe the provider's original vectors,
    // not the representation stored in the BufferPool entity.
    const int ret =
        dc.has_provider()
            ? dc.get_vector(prune_ids.data(), prune_ids.size(), prune_blocks)
            : read_entity.get_vector_for_prune(prune_ids.data(),
                                               prune_ids.size(), prune_blocks);
    if (ailego_unlikely(ret != 0)) {
      dc.set_error();
      return;
    }
    selected_indices.clear();
    selected_indices.reserve(max_neighbor_cnt);

    for (size_t i = 0; i < topk_heap.size(); ++i) {
      node_id_t cur_node = topk_heap[i].first;
      dist_t cur_node_dist = topk_heap[i].second;
      bool good = true;
      for (const size_t selected : selected_indices) {
        const dist_t pair_distance =
            dc.dist(prune_blocks[i].data(), prune_blocks[selected].data());
        if (pair_distance <= cur_node_dist) {
          good = false;
          break;
        }
      }

      if (good) {
        topk_heap.mutable_at(cur_size).first = cur_node;
        topk_heap.mutable_at(cur_size).second = cur_node_dist;
        selected_indices.emplace_back(i);
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
      }
    }
    prune_blocks.clear();
  } else {
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
        topk_heap.mutable_at(cur_size).first = cur_node;
        topk_heap.mutable_at(cur_size).second = cur_node_dist;
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
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
      topk_heap.mutable_at(cur_size).first = topk_heap[k].first;
      topk_heap.mutable_at(cur_size).second = topk_heap[k].second;
      cur_size++;
    }
  }

  topk_heap.truncate(cur_size);
  entity_.update_neighbors(level, id, topk_heap.container());

  return;
}

template <typename EntityType>
void HnswAlgorithm<EntityType>::reverse_update_neighbors(
    HnswDistCalculator &dc, node_id_t id, level_t level, node_id_t link_id,
    dist_t dist, TopkHeap &update_heap, HnswContext *ctx) {
  const size_t max_neighbor_cnt = entity_.neighbor_cnt(level);

  const uint32_t lock_idx = id & kLockMask;
  std::unique_lock<std::mutex> node_lock(lock_pool_[lock_idx]);
  const Neighbors neighbors = entity_.get_neighbors(level, id);
  const size_t size = neighbors.size();
  ailego_assert_with(size <= max_neighbor_cnt, "invalid neighbor size");
  if (size < max_neighbor_cnt) {
    entity_.add_neighbor(level, id, size, link_id);
    return;
  }

  if constexpr (std::is_same_v<EntityType, HnswBufferPoolStreamerEntity>) {
    const auto &read_entity =
        static_cast<const EntityType &>(ctx->get_entity());
    // A wide level-0 list can require O(M^2) pairwise comparisons.  Resolving
    // each pair through HnswDistCalculator::dist(node_id, node_id) repeatedly
    // pins and releases the same BufferStorage pages and dominates high-M
    // construction.  Resolve the center and every candidate once, keep those
    // blocks alive for the prune, and preserve the existing scalar pruning
    // order exactly.
    auto &candidate_ids = ctx->prune_ids();
    auto &candidate_blocks = ctx->prune_blocks();

    candidate_ids.clear();
    candidate_ids.reserve(size + 2);
    candidate_ids.emplace_back(id);
    candidate_ids.emplace_back(link_id);
    for (size_t i = 0; i < size; ++i) {
      candidate_ids.emplace_back(neighbors[i]);
    }

    candidate_blocks.clear();
    const int ret =
        dc.has_provider()
            ? dc.get_vector(candidate_ids.data(), candidate_ids.size(),
                            candidate_blocks)
            : read_entity.get_vector_for_prune(
                  candidate_ids.data(), candidate_ids.size(), candidate_blocks);
    if (ailego_unlikely(ret != 0)) {
      dc.set_error();
      return;
    }

    update_heap.clear();
    // Use block indices until pruning is complete.  This keeps sorted heap
    // entries directly associated with their pinned vectors without a hash
    // map.
    update_heap.emplace(1, dist);
    for (size_t i = 0; i < size; ++i) {
      const dist_t candidate_distance =
          dc.dist(candidate_blocks[0].data(), candidate_blocks[i + 2].data());
      update_heap.emplace(static_cast<node_id_t>(i + 2), candidate_distance);
    }
    update_heap.sort();

    size_t selected_count = 0;
    for (size_t i = 0; i < update_heap.size(); ++i) {
      const node_id_t candidate_block_index = update_heap[i].first;
      const dist_t candidate_distance = update_heap[i].second;
      bool good = true;
      if (selected_count != 0) {
        for (size_t j = 0; j < selected_count; ++j) {
          const dist_t pair_distance =
              dc.dist(candidate_blocks[candidate_block_index].data(),
                      candidate_blocks[update_heap[j].first].data());
          if (pair_distance <= candidate_distance) {
            good = false;
            break;
          }
        }
      }

      if (good) {
        update_heap.mutable_at(selected_count) = {candidate_block_index,
                                                  candidate_distance};
        ++selected_count;
        if (selected_count >= max_neighbor_cnt) {
          break;
        }
      }
    }

    update_heap.truncate(selected_count);
    for (auto &selected : update_heap.mutable_container()) {
      selected.first = candidate_ids[selected.first];
    }
    entity_.update_neighbors(level, id, update_heap.container());

    node_lock.unlock();
    update_heap.clear();
    // Release BufferStorage pins before this worker consumes its next node.
    candidate_blocks.clear();
  } else {
    update_heap.emplace(link_id, dist);
    for (size_t i = 0; i < size; ++i) {
      node_id_t node = neighbors[i];
      dist_t cur_dist = dc.dist(id, node);
      update_heap.emplace(node, cur_dist);
    }

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
        update_heap.mutable_at(cur_size).first = cur_node;
        update_heap.mutable_at(cur_size).second = cur_node_dist;
        cur_size++;
        if (cur_size >= max_neighbor_cnt) {
          break;
        }
      }
    }

    update_heap.truncate(cur_size);
    entity_.update_neighbors(level, id, update_heap.container());
    update_heap.clear();
  }

  return;
}

// Explicit template instantiation
template class HnswAlgorithm<HnswMmapStreamerEntity>;
template class HnswAlgorithm<HnswBufferPoolStreamerEntity>;
template class HnswAlgorithm<HnswContiguousStreamerEntity>;
template class HnswAlgorithm<HnswExternalStreamerEntity>;

}  // namespace core
}  // namespace zvec
