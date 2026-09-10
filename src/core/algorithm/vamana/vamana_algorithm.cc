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

#include "vamana_algorithm.h"
#include <type_traits>
#include <ailego/internal/cpu_features.h>

namespace zvec {
namespace core {

// ============================================================================
// add_node: Insert a new node into the Vamana graph.
//
// Algorithm (from DiskANN / Vamana paper):
//   1. GreedySearch from entry_point to find candidate neighbors
//   2. RobustPrune to select diverse neighbors for the new node
//   3. Update the new node's neighbor list
//   4. For each new neighbor, add reverse link; if over-degree, RobustPrune
//   5. If this is the first node, set it as entry point
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::add_node(node_id_t id, VamanaContext *ctx) {
  // Lazily initialize distance storage on first insert
  entity_.ensure_dist_storage();

  spin_lock_.lock();
  auto entry_point = entity_.entry_point();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    entity_.update_entry_point(id);
    spin_lock_.unlock();
    return 0;
  }
  spin_lock_.unlock();

  // Step 1: GreedySearch to find candidate neighbors
  uint32_t search_list_size = entity_.search_list_size();
  auto &build_heap = ctx->search_heap().reset<TopkHeap>(search_list_size);
  ctx->visit_filter().clear();
  ctx->candidates().clear();
  ctx->dist_calculator().clear_compare_cnt();

  // Set query to the new node's vector. Use reset_query (same as search path)
  // so that greedy_search works with the search-optimized distance kernel.
  const void *query_vec = entity_.get_vector(id);
  if (ailego_unlikely(query_vec == nullptr)) {
    LOG_ERROR("Failed to get vector for node %u", id);
    return IndexError_ReadData;
  }
  ctx->reset_query(query_vec);

  int ret = greedy_search(entry_point, ctx);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  // Step 2: RobustPrune to select diverse neighbors
  robust_prune(id, build_heap, entity_.alpha(), entity_.max_degree(), ctx);
  // Copy result before reverse updates (which also call robust_prune)
  auto pruned_neighbors = ctx->prune_result();

  // Step 3: Update the new node's neighbor list and distances
  entity_.update_neighbors(id, pruned_neighbors);
  entity_.update_neighbor_dists(id, pruned_neighbors);

  // Step 4: Reverse-link updates
  update_neighbors_and_reverse_links(id, pruned_neighbors, ctx);

  return 0;
}

// ============================================================================
// refine_graph: Full-graph Vamana second pass.
//
// Standard two-pass construction first builds the graph with alpha=1.0. Once
// every point is visible, this pass searches from the final entry point,
// combines the search result with each node's existing outgoing edges, and
// applies RobustPrune with the configured target alpha.
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::refine_graph(VamanaContext *ctx, float alpha) {
  if (ctx == nullptr) return IndexError_InvalidArgument;
  int ret = entity_.ensure_dist_storage();
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  const uint32_t n = entity_.doc_cnt();
  if (n <= 1 || entity_.entry_point() == kInvalidNodeId) {
    return 0;
  }

  ctx->check_need_adjuct_ctx(n);
  for (node_id_t id = 0; id < n; ++id) {
    if (entity_.get_key(id) == kInvalidKey) continue;
    if (entity_.get_vector(id) == nullptr) continue;
    ret = refine_node(id, alpha, ctx);
    if (ailego_unlikely(ret != 0)) {
      return ret;
    }
    if (ailego_unlikely(ctx->error())) {
      return IndexError_Runtime;
    }
  }
  return 0;
}

// ============================================================================
// search: Greedy search for approximate nearest neighbors.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::prepare_search(VamanaContext *ctx) const {
  // Queries use ef, independently of the construction-time search_list_size.
  const uint32_t capacity = std::max(ctx->topk(), ctx->ef());
  ctx->visit_filter().clear();
  ctx->candidates().clear();
  if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
    if (!ctx->filter().is_valid()) {
      const bool avx2_ok =
          zvec::ailego::internal::CpuFeatures::static_flags_.AVX2;
      ctx->search_heap().reset_pool(avx2_ok, capacity, entity_.max_degree());
      return;
    }
  }
  ctx->search_heap().reset<TopkHeap>(capacity);
}

template <typename EntityType>
int VamanaAlgorithm<EntityType>::search(VamanaContext *ctx) const {
  prepare_search(ctx);
  spin_lock_.lock();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  return greedy_search(entry_point, ctx);
}

// ============================================================================
// greedy_search helper templates
//
// Two specialized inner loops, dispatched from greedy_search():
//
//   fast_greedy_search:       mmap/contiguous with direct vector pointers.
//                             Uses batch_dist on a pointer array.
//   dual_heap_greedy_search:  Construction, filtered queries and BufferPool.
//                             Uses get_vector_typed with MemBlock wrappers.
//
// The fast path accepts either BlockHeap or LinearPool as `HeapType`.
// Both expose reset(capacity, block_size) / push_block(dists, ids, n);
// LinearPool adapts via push_block and ignores the block_size hint.
// ============================================================================

// Prefetch all cache lines covering a packed contiguous graph row.
ailego_force_inline void prefetch_graph_row(
    const VamanaContiguousStreamerEntity &entity, node_id_t node) {
  constexpr uintptr_t kCacheLineMask = ~uintptr_t{63};
  const char *data = entity.graph_prefetch_data(node);
  if (ailego_unlikely(data == nullptr)) return;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data) & kCacheLineMask;
  const uintptr_t end =
      reinterpret_cast<uintptr_t>(data) + entity.graph_prefetch_size();
  const size_t lines = (end - begin + 63) / 64;
  ailego_prefetch_lines(reinterpret_cast<const void *>(begin), lines);
}

// mmap/contiguous variant: resolve vector bodies and optional extra-values
// pointers through the entity's layout-specific accessors.
template <bool HasExtraValues, typename EntityType, typename HeapType,
          typename Visit>
void fast_greedy_search(const EntityType &entity, HeapType &pool,
                        VamanaContext *ctx, VamanaDistCalculator &dc,
                        node_id_t entry_point, uint32_t prefetch_lines,
                        uint32_t prefetch_offset, Visit visit) {
  static constexpr bool kPrefetchGraph =
      std::is_same_v<EntityType, VamanaContiguousStreamerEntity>;
  const uint32_t max_deg = entity.max_degree();

  uint32_t buf_capacity = max_deg;
  auto &neighbor_ids = ctx->search_neighbor_ids_buf();
  auto &dists = ctx->search_dists_buf();
  auto &neighbor_vecs = ctx->search_vecs_buf();
  neighbor_ids.resize(buf_capacity);
  dists.resize(buf_capacity);
  neighbor_vecs.resize(buf_capacity);
  auto &extra_values = ctx->search_extra_values_buf();
  if constexpr (HasExtraValues) {
    extra_values.resize(buf_capacity);
  }

  const auto insert_candidates = [&](uint32_t count) {
    if constexpr (std::is_same_v<HeapType, LinearPool<float>>) {
      for (uint32_t i = 0; i < count; ++i) {
        if constexpr (kPrefetchGraph) {
          bool rewound = false;
          if (pool.insert_with_rewind(static_cast<int>(neighbor_ids[i]),
                                      dists[i], &rewound) &&
              rewound) {
            prefetch_graph_row(entity, neighbor_ids[i]);
          }
        } else {
          pool.insert(static_cast<int>(neighbor_ids[i]), dists[i]);
        }
      }
    } else {
      pool.push_block(dists.data(), neighbor_ids.data(),
                      static_cast<int32_t>(count));
    }
  };

  const auto compute_neighbor_row = [&](const auto &neighbors) {
    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
      if constexpr (HasExtraValues) {
        extra_values.resize(buf_capacity);
      }
    }

    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      const node_id_t node = neighbors[i];
      const void *vec_ptr = entity.get_vector_ptr(node);
      ailego_prefetch_lines(vec_ptr, prefetch_lines);
      neighbor_ids[i] = node;
      neighbor_vecs[i] = vec_ptr;
      if constexpr (HasExtraValues) {
        extra_values[i] = entity.get_extra_values_ptr(node, vec_ptr);
      }
    }
    if (neighbors.size() == 0) return;

    dc.batch_dist(neighbor_vecs.data(), neighbors.size(), dists.data(),
                  HasExtraValues ? extra_values.data() : nullptr);
  };

  // Step 1: Descend one best neighbor at a time to a local optimum, caching
  // the last row's distances. Do not mark descent nodes visited: the pool
  // phase must still be able to explore their other neighbors.
  node_id_t start = entry_point;
  dist_t start_dist = dc.batch_dist(start);
  if (ailego_unlikely(dc.error())) {
    return;
  }

  node_id_t cached_row = kInvalidNodeId;
  for (uint32_t depth = 0; depth < 100; ++depth) {
    const node_id_t before = start;
    cached_row = before;
    if constexpr (kPrefetchGraph) {
      prefetch_graph_row(entity, before);
    }
    const auto neighbors = entity.get_neighbors_typed(before);
    compute_neighbor_row(neighbors);
    if (ailego_unlikely(dc.error())) {
      return;
    }
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      if (dists[i] < start_dist) {
        start = neighbor_ids[i];
        start_dist = dists[i];
      }
    }
    if (start == before) break;
  }

  // Step 2: Seed the pool with the landing row, then the landing node.
  // Preserve this insertion order and visited deduplication: ties and
  // capacity pruning depend on them.
  const auto start_neighbors = entity.get_neighbors_typed(start);
  // The depth limit can leave the cached row at the preceding node.
  if (cached_row != start) {
    compute_neighbor_row(start_neighbors);
    if (ailego_unlikely(dc.error())) {
      return;
    }
  }

  uint32_t initial_count = 0;
  for (uint32_t i = 0; i < start_neighbors.size(); ++i) {
    const node_id_t node = start_neighbors[i];
    if (visit.visited(node)) continue;
    visit.set_visited(node);
    neighbor_ids[initial_count] = node;
    dists[initial_count] = dists[i];
    ++initial_count;
  }
  if (initial_count != 0) {
    insert_candidates(initial_count);
  }

  if (!visit.visited(start)) {
    visit.set_visited(start);
    neighbor_ids[0] = start;
    dists[0] = start_dist;
    insert_candidates(1);
  }

  // Step 3: Expand the bounded greedy pool, collecting unvisited neighbors.
  while (pool.has_next()) {
    node_id_t current_node;
    if constexpr (kPrefetchGraph) {
      node_id_t next = kInvalidNodeId;
      current_node = static_cast<node_id_t>(pool.pop_with_next(&next));
      prefetch_graph_row(entity, current_node);
      if (next != kInvalidNodeId) {
        prefetch_graph_row(entity, next);
      }
    } else {
      current_node = static_cast<node_id_t>(pool.pop());
    }

    const auto neighbors = entity.get_neighbors_typed(current_node);
    if constexpr (kPrefetchGraph) {
      if (ailego_unlikely(!entity.is_contiguous())) {
        ailego_prefetch(neighbors.data);
      }
    } else {
      ailego_prefetch(neighbors.data);
    }

    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
      if constexpr (HasExtraValues) {
        extra_values.resize(buf_capacity);
      }
    }

    const uint32_t po =
        std::min(static_cast<uint32_t>(neighbors.size()), prefetch_offset);
    uint32_t unvisited_count = 0;
    uint32_t i = 0;

    for (; i < po; ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      const void *vec_ptr = entity.get_vector_ptr(node);
      ailego_prefetch_lines(vec_ptr, prefetch_lines);
      neighbor_ids[unvisited_count] = node;
      neighbor_vecs[unvisited_count] = vec_ptr;
      if constexpr (HasExtraValues) {
        extra_values[unvisited_count] =
            entity.get_extra_values_ptr(node, vec_ptr);
      }
      unvisited_count++;
    }
    for (; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      const void *vec_ptr = entity.get_vector_ptr(node);
      neighbor_ids[unvisited_count] = node;
      neighbor_vecs[unvisited_count] = vec_ptr;
      if constexpr (HasExtraValues) {
        extra_values[unvisited_count] =
            entity.get_extra_values_ptr(node, vec_ptr);
      }
      unvisited_count++;
    }

    if (unvisited_count == 0) continue;
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data(),
                  HasExtraValues ? extra_values.data() : nullptr);
    if (ailego_unlikely(dc.error())) {
      return;
    }
    insert_candidates(unvisited_count);
  }
}

// ============================================================================
// dual_heap_greedy_search: shared core for the fallback dual-heap path.
//
// Maintains a candidate min-heap + topk heap + VisitFilter.  Uses plain
// batch_dist.
// ============================================================================
template <typename EntityType, typename MemBlockType, typename Visit,
          typename FilterFn>
void dual_heap_greedy_search(const EntityType &entity, VamanaContext *ctx,
                             VamanaDistCalculator &dc, node_id_t entry_point,
                             TopkHeap &topk_heap, Visit visit,
                             FilterFn &&filter) {
  const uint32_t prefetch_offset = ctx->po();
  const uint32_t vector_body_lines =
      static_cast<uint32_t>((entity.vector_data_size() + 63) / 64);
  const uint32_t prefetch_lines = ctx->pl() > 0
                                      ? std::min(ctx->pl(), vector_body_lines)
                                      : vector_body_lines;

  uint32_t buf_capacity = entity.max_degree();
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<MemBlockType> neighbor_vec_blocks;
  neighbor_vec_blocks.reserve(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);
  const bool has_extra_values = entity.extra_values_size() != 0;
  std::vector<const void *> extra_values(has_extra_values ? buf_capacity : 0);

  CandidateHeap &candidates = ctx->candidates();

  // Initialize with entry point using batch_dist (single-element batch).
  // We must NOT use dc.dist(entry_point) here because dist() calls
  // distance_() which is squared_euclidean_int8_distance (sign/abs trick,
  // expects two raw int8 inputs), but query_ has been preprocessed by
  // reset_query (+128 shift to uint8). batch_dist() correctly calls
  // batch_distance_() which expects the preprocessed uint8 query.
  dist_t entry_dist = dc.batch_dist(entry_point);
  if (ailego_unlikely(dc.error())) {
    return;
  }
  visit.set_visited(entry_point);
  if (!filter(entry_point)) {
    topk_heap.emplace(entry_point, entry_dist);
  }
  candidates.emplace(entry_point, entry_dist);

  while (!candidates.empty() && !ctx->reach_scan_limit()) {
    auto top = candidates.begin();
    node_id_t current_node = top->first;
    dist_t current_dist = top->second;

    // Early termination: if the closest candidate is worse than the worst
    // result in a full topk heap, we won't find anything better.
    if (topk_heap.full() && current_dist > topk_heap[0].second) {
      break;
    }

    candidates.pop();

    // Expand neighbors using typed access (no virtual dispatch)
    const auto neighbors = entity.get_neighbors_typed(current_node);
    ailego_prefetch(neighbors.data);

    // Resize buffers if this node has more neighbors than expected
    if (neighbors.size() > buf_capacity) {
      buf_capacity = neighbors.size();
      neighbor_ids.resize(buf_capacity);
      dists.resize(buf_capacity);
      neighbor_vecs.resize(buf_capacity);
      if (has_extra_values) {
        extra_values.resize(buf_capacity);
      }
    }

    // Collect unvisited neighbors (reuse pre-allocated buffer)
    uint32_t unvisited_count = 0;
    for (uint32_t i = 0; i < neighbors.size(); ++i) {
      node_id_t node = neighbors[i];
      if (visit.visited(node)) continue;
      visit.set_visited(node);
      neighbor_ids[unvisited_count++] = node;
    }
    if (unvisited_count == 0) continue;

    // Batch fetch vectors using typed access (reuse pre-allocated buffer)
    neighbor_vec_blocks.clear();
    int ret = entity.get_vector_typed(neighbor_ids.data(), unvisited_count,
                                      neighbor_vec_blocks);
    if (ailego_unlikely(ret != 0)) break;

    for (uint32_t i = 0; i < std::min(prefetch_offset, unvisited_count); ++i) {
      const char *base =
          static_cast<const char *>(neighbor_vec_blocks[i].data());
      for (uint32_t cl = 0; cl < prefetch_lines; ++cl) {
        ailego_prefetch(base + cl * 64);
      }
    }

    // Batch distance computation (reuse pre-allocated buffers).
    for (uint32_t i = 0; i < unvisited_count; ++i) {
      neighbor_vecs[i] = neighbor_vec_blocks[i].data();
      if (has_extra_values) {
        extra_values[i] =
            entity.get_extra_values_ptr(neighbor_ids[i], neighbor_vecs[i]);
      }
    }
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data(),
                  has_extra_values ? extra_values.data() : nullptr);

    for (uint32_t i = 0; i < unvisited_count; ++i) {
      node_id_t node = neighbor_ids[i];
      dist_t node_dist = dists[i];
      if ((!topk_heap.full()) || node_dist < topk_heap[0].second) {
        candidates.emplace(node, node_dist);
        if (!filter(node)) {
          topk_heap.emplace(node, node_dist);
        }
      }
    }
  }
}

// ============================================================================
// greedy_search: Dispatch the heap prepared by the query or construction path.
//
// Unfiltered mmap/contiguous queries use fast_greedy_search. Construction,
// filtered queries and BufferPool use dual_heap_greedy_search, which enforces
// the scan limit. Retain the query's active heap/pool for result collection.
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::greedy_search(node_id_t entry_point,
                                               VamanaContext *ctx) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  const IndexFilter &index_filter =
      static_cast<const IndexContext *>(ctx)->filter();
  auto no_filter = [](node_id_t) { return false; };

  bool dispatched = false;
  auto run_with_heap = [&](auto &heap) {
    using Heap = std::decay_t<decltype(heap)>;
    dispatched = dispatch_visit_filter(ctx->visit_filter(), [&](auto visit) {
      if constexpr (std::is_same_v<Heap, TopkHeap>) {
        if (index_filter.is_valid()) {
          auto filter = [&](node_id_t id) {
            return index_filter(entity.get_key_typed(id));
          };
          greedy_search_impl(entry_point, ctx, heap, visit, filter);
          return;
        }
      }
      greedy_search_impl(entry_point, ctx, heap, visit, no_filter);
    });
  };
  if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
    ctx->search_heap().dispatch(run_with_heap);
  } else {
    // BufferPool/external entities have only the prepared dual-heap path.
    run_with_heap(ctx->search_heap().topk());
  }
  if (ailego_unlikely(!dispatched)) {
    LOG_ERROR("Failed to dispatch Vamana visit filter, mode %d",
              ctx->visit_filter().get_mode());
    return IndexError_Runtime;
  }
  return 0;
}

template <typename EntityType>
template <typename Heap, typename Visit, typename Filter>
void VamanaAlgorithm<EntityType>::greedy_search_impl(node_id_t entry_point,
                                                     VamanaContext *ctx,
                                                     Heap &heap, Visit visit,
                                                     Filter &&filter) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  VamanaDistCalculator &dc = ctx->dist_calculator();
  if constexpr (std::is_same_v<Heap, TopkHeap>) {
    dual_heap_greedy_search<EntityType, MemBlockType>(
        entity, ctx, dc, entry_point, heap, visit,
        std::forward<Filter>(filter));
  } else {
    static_assert(std::is_same_v<MemBlockType, MmapMemoryBlock>);
    const uint32_t vector_body_lines =
        static_cast<uint32_t>((entity.vector_data_size() + 63) / 64);
    const uint32_t prefetch_lines = ctx->pl() > 0
                                        ? std::min(ctx->pl(), vector_body_lines)
                                        : vector_body_lines;
    if (entity.extra_values_size() != 0) {
      fast_greedy_search<true>(entity, heap, ctx, dc, entry_point,
                               prefetch_lines, ctx->po(), visit);
    } else {
      fast_greedy_search<false>(entity, heap, ctx, dc, entry_point,
                                prefetch_lines, ctx->po(), visit);
    }
  }
}

template <typename EntityType>
int VamanaAlgorithm<EntityType>::refine_node(node_id_t id, float alpha,
                                             VamanaContext *ctx) {
  const void *query_vec = entity_.get_vector(id);
  if (ailego_unlikely(query_vec == nullptr)) {
    return IndexError_ReadData;
  }

  node_id_t entry_point = entity_.entry_point();
  if (entry_point == id) {
    entry_point = kInvalidNodeId;
    const uint32_t n = entity_.doc_cnt();
    for (node_id_t candidate = 0; candidate < n; ++candidate) {
      if (candidate == id) continue;
      if (entity_.get_key(candidate) == kInvalidKey) continue;
      if (entity_.get_vector(candidate) == nullptr) continue;
      entry_point = candidate;
      break;
    }
    if (entry_point == kInvalidNodeId) {
      return 0;
    }
  }

  ctx->clear();
  auto &build_heap =
      ctx->search_heap().reset<TopkHeap>(entity_.search_list_size());
  ctx->visit_filter().clear();
  ctx->candidates().clear();
  ctx->dist_calculator().clear_compare_cnt();
  ctx->reset_query(query_vec);

  int ret = greedy_search(entry_point, ctx);
  if (ailego_unlikely(ret != 0)) {
    return ret;
  }

  const TopkHeap &search_candidates = build_heap;
  const Neighbors current_neighbors = entity_.get_neighbors(id);

  // Unlike add_node(), the node being refined is already visible in the
  // graph, so GreedySearch can return `id` itself with distance zero. Remove
  // it before RobustPrune: although RobustPrune also skips self-loops, leaving
  // it here would consume one of the max_occlusion_size candidate slots.
  TopkHeap &candidates = ctx->update_heap();
  candidates.clear();
  candidates.limit(search_candidates.size() + current_neighbors.size() + 1);

  // GreedySearch's VisitFilter is a superset of the bounded top-k heap. Its
  // search lifetime ends here, so rebuild it from the retained candidates and
  // reuse it as membership while merging the node's existing outgoing edges.
  VisitFilter &candidate_membership = ctx->visit_filter();
  candidate_membership.clear();
  for (const auto &candidate : search_candidates) {
    if (candidate.first == id) continue;
    candidates.emplace(candidate);
    candidate_membership.set_visited(candidate.first);
  }

  const dist_t *cached_dists = entity_.get_neighbor_dists(id);
  for (uint32_t i = 0; i < current_neighbors.size(); ++i) {
    node_id_t neighbor = current_neighbors[i];
    if (neighbor == id || candidate_membership.visited(neighbor)) continue;
    const void *neighbor_vec = entity_.get_vector(neighbor);
    if (neighbor_vec == nullptr) continue;
    dist_t dist = cached_dists
                      ? cached_dists[i]
                      : ctx->dist_calculator().dist(query_vec, neighbor_vec);
    candidates.emplace(neighbor, dist);
    candidate_membership.set_visited(neighbor);
  }

  robust_prune(id, candidates, alpha, entity_.max_degree(), ctx);
  auto pruned_neighbors = ctx->prune_result();

  entity_.update_neighbors(id, pruned_neighbors);
  entity_.update_neighbor_dists(id, pruned_neighbors);
  update_neighbors_and_reverse_links(id, pruned_neighbors, ctx);

  return 0;
}

// ============================================================================
// robust_prune: Select up to max_degree diverse neighbors from candidates.
//
// Faithfully follows DiskANN's occlude_list algorithm:
//   1. Sort candidates by distance (ascending)
//   2. Truncate to max_occlusion_size (DiskANN's maxc parameter)
//   3. Multi-round alpha sweep: cur_alpha starts at 1.0, increments by *1.2
//      each round until reaching alpha. This progressively relaxes the
//      occlusion criterion.
//   4. For each candidate, compute occlude_factor as:
//        max over all selected neighbors p of: dist(query, candidate) / dist(p,
//        candidate)
//      If occlude_factor > cur_alpha, the candidate is occluded in this round.
//   5. After all rounds, if _saturate_graph and alpha > 1, fill remaining
//      slots with any un-selected candidates.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::robust_prune(node_id_t id,
                                               TopkHeap &candidates,
                                               float alpha, uint32_t max_degree,
                                               VamanaContext *ctx) const {
  auto &result = ctx->prune_result();
  result.clear();

  if (candidates.size() == 0) return;

  // Sort candidates by distance (ascending — closest first)
  candidates.sort();

  VamanaDistCalculator &dc = ctx->dist_calculator();
  size_t n = candidates.size();

  // Build-time distance offset: shifts the internal distance into a
  // non-negative range so the ratio-based occlude_factor below is
  // geometrically meaningful. Zero for metrics whose internal distance is
  // already non-negative (SquaredEuclidean, etc.); 1.0 for the quantized int8
  // Cosine / NormalizedCosine path (internal distance is -cos, so
  // offset=1 yields 1-cos which matches DiskANN's normalized-L2 semantics).
  const float dist_offset = ctx->build_distance_offset();

  // Truncate to max_occlusion_size (DiskANN's maxc parameter)
  size_t maxc = entity_.max_occlusion_size();
  if (maxc > 0 && n > maxc) {
    n = maxc;
  }

  // Pre-cache all candidate vectors at once
  auto &vec_cache = ctx->prune_vec_cache();
  vec_cache.resize(n);
  for (size_t i = 0; i < n; ++i) {
    vec_cache[i] = entity_.get_vector(candidates[i].first);
  }

  // occlude_factor: tracks the maximum occlusion ratio for each candidate
  // (DiskANN: occlude_factor[t] = max over selected p of
  // dist_to_query/dist_to_p)
  auto &occlude_factor = ctx->prune_occlude_factor();
  occlude_factor.assign(n, 0.0f);

  // Pre-allocated buffers for batch distance computation
  auto &batch_vecs = ctx->batch_vecs_buf();
  auto &batch_dists = ctx->batch_dists_buf();
  auto &batch_indices = ctx->batch_indices_buf();
  batch_vecs.resize(n);
  batch_dists.resize(n);
  batch_indices.resize(n);

  // Multi-round alpha sweep (DiskANN: cur_alpha starts at 1, increments *1.2)
  float cur_alpha = 1.0f;
  while (cur_alpha <= alpha + 1e-6f && result.size() < max_degree) {
    for (size_t i = 0; i < n && result.size() < max_degree; ++i) {
      if (occlude_factor[i] > cur_alpha) {
        continue;
      }

      // Mark as consumed so it won't be reconsidered
      occlude_factor[i] = std::numeric_limits<float>::max();

      // Skip self-loops
      if (candidates[i].first == id) continue;

      const void *selected_vec = vec_cache[i];
      if (ailego_unlikely(selected_vec == nullptr)) continue;

      // Add this candidate as a neighbor
      node_id_t candidate_id = candidates[i].first;
      dist_t candidate_dist = candidates[i].second;
      result.emplace_back(candidate_id, candidate_dist);

      // Update occlude_factor for remaining candidates
      // Collect candidates that haven't been consumed yet
      uint32_t batch_count = 0;
      for (size_t j = i + 1; j < n; ++j) {
        if (occlude_factor[j] > alpha) continue;  // already fully occluded
        if (ailego_unlikely(vec_cache[j] == nullptr)) continue;
        batch_vecs[batch_count] = vec_cache[j];
        batch_indices[batch_count] = static_cast<uint32_t>(j);
        batch_count++;
      }

      if (batch_count > 0) {
        // Compute distances from selected candidate to remaining candidates.
        // distance_ is the symmetric data-to-data kernel (no pairwise split).
        for (uint32_t k = 0; k < batch_count; ++k) {
          batch_dists[k] = dc.dist(selected_vec, batch_vecs[k]);
        }

        // DiskANN (L2/Cosine):
        //   occlude_factor[t] = max(occlude_factor[t], dist_to_query /
        //   dist_between)
        // where dist_to_query = candidates[j].second (distance from query to j)
        //       dist_between = batch_dists[k] (distance from selected to j)
        //
        // `dist_offset` shifts both distances into a non-negative range for
        // metrics whose internal distance can be negative (e.g. quantized
        // int8 cosine stores -cos(m,q) in [-1, 1]). Without this shift the
        // ratio loses its geometric meaning and RobustPrune produces a
        // poor-quality graph (low recall at low ef_search).
        for (uint32_t k = 0; k < batch_count; ++k) {
          uint32_t j = batch_indices[k];
          float dist_selected_to_candidate = batch_dists[k] + dist_offset;
          if (dist_selected_to_candidate <= 0.0f) {
            occlude_factor[j] = std::numeric_limits<float>::max();
          } else {
            occlude_factor[j] = std::max(occlude_factor[j],
                                         (candidates[j].second + dist_offset) /
                                             dist_selected_to_candidate);
          }
        }
      }
    }
    cur_alpha *= 1.2f;
  }

  // Saturate graph: if enabled and alpha > 1, fill remaining slots with any
  // un-selected candidates. This improves graph connectivity (better recall)
  // at the cost of slightly more distance computations during search.
  // Configurable via proxima.vamana.streamer.saturate_graph (default: false,
  // matching DiskANN's default behavior).
  if (entity_.saturate_graph() && alpha > 1.0f) {
    for (size_t i = 0; i < n && result.size() < max_degree; ++i) {
      if (candidates[i].first == id) continue;
      bool already_selected = false;
      for (const auto &r : result) {
        if (r.first == candidates[i].first) {
          already_selected = true;
          break;
        }
      }
      if (!already_selected) {
        result.emplace_back(candidates[i].first, candidates[i].second);
      }
    }
  }
}

// ============================================================================
// update_neighbors_and_reverse_links: For each new neighbor of `id`,
// add a reverse link from neighbor back to `id`. If the neighbor's degree
// exceeds max_degree, prune it using RobustPrune.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::update_neighbors_and_reverse_links(
    node_id_t id,
    const std::vector<std::pair<node_id_t, dist_t>> &new_neighbors,
    VamanaContext *ctx) {
  for (const auto &[neighbor_id, dist] : new_neighbors) {
    reverse_update_neighbor(id, neighbor_id, dist, ctx);
  }
}

// ============================================================================
// reverse_update_neighbor: Add `id` as a neighbor of `neighbor_id`.
// If neighbor_id already has max_degree neighbors, collect all neighbors
// + the new one into a candidate set and RobustPrune.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::reverse_update_neighbor(node_id_t id,
                                                          node_id_t neighbor_id,
                                                          dist_t dist,
                                                          VamanaContext *ctx) {
  std::lock_guard<std::mutex> lock(lock_pool_[neighbor_id & kLockMask]);

  const Neighbors current_neighbors = entity_.get_neighbors(neighbor_id);
  uint32_t current_size = current_neighbors.size();
  uint32_t max_deg = entity_.max_degree();

  // Check if `id` is already a neighbor
  for (uint32_t i = 0; i < current_size; ++i) {
    if (current_neighbors[i] == id) return;
  }

  if (current_size < max_deg) {
    // Simply append and record distance
    entity_.add_neighbor(neighbor_id, current_size, id);
    entity_.set_neighbor_dist(neighbor_id, current_size, dist);
    return;
  }

  // Need to prune: collect current neighbors + new node into candidates
  VamanaDistCalculator &dc = ctx->dist_calculator();

  // Reuse update_heap from context instead of creating a new TopkHeap each time
  TopkHeap &prune_candidates = ctx->update_heap();
  prune_candidates.clear();
  prune_candidates.limit(max_deg + 1);

  // Add existing neighbors — use cached distances when available
  const dist_t *cached_dists = entity_.get_neighbor_dists(neighbor_id);
  if (cached_dists != nullptr) {
    // Fast path: read distances from storage, no recomputation needed
    for (uint32_t i = 0; i < current_size; ++i) {
      prune_candidates.emplace(current_neighbors[i], cached_dists[i]);
    }
  } else {
    // Fallback: compute distances (first time or dist storage not loaded)
    const void *neighbor_vec = entity_.get_vector(neighbor_id);
    if (ailego_unlikely(neighbor_vec == nullptr)) return;
    for (uint32_t i = 0; i < current_size; ++i) {
      node_id_t nbr = current_neighbors[i];
      const void *nbr_vec = entity_.get_vector(nbr);
      if (ailego_unlikely(nbr_vec == nullptr)) continue;
      dist_t nbr_dist = dc.dist(neighbor_vec, nbr_vec);
      prune_candidates.emplace(nbr, nbr_dist);
    }
  }

  // Add the new reverse link
  prune_candidates.emplace(id, dist);

  // RobustPrune from neighbor_id's perspective
  robust_prune(neighbor_id, prune_candidates, entity_.alpha(), max_deg, ctx);

  // Update neighbor_id's neighbor list and distances
  const auto &prune_result = ctx->prune_result();
  entity_.update_neighbors(neighbor_id, prune_result);
  entity_.update_neighbor_dists(neighbor_id, prune_result);
}

// Explicit template instantiation for all entity types
template class VamanaAlgorithm<VamanaMmapStreamerEntity>;
template class VamanaAlgorithm<VamanaBufferPoolStreamerEntity>;
template class VamanaAlgorithm<VamanaContiguousStreamerEntity>;

}  // namespace core
}  // namespace zvec
