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
  ctx->topk_heap().clear();
  ctx->topk_heap().limit(search_list_size);
  ctx->dist_calculator().clear_compare_cnt();

  // Set query to the new node's vector. Use reset_query (same as search path)
  // so that greedy_search works with the search-optimized distance kernel.
  const void *query_vec = entity_.get_vector(id);
  if (ailego_unlikely(query_vec == nullptr)) {
    LOG_ERROR("Failed to get vector for node %u", id);
    return IndexError_ReadData;
  }
  ctx->reset_query(query_vec);

  greedy_search(entry_point, ctx, /*use_pool=*/false);

  auto &topk_heap = ctx->topk_heap();

  // Step 2: RobustPrune to select diverse neighbors
  robust_prune(id, topk_heap, entity_.alpha(), entity_.max_degree(), ctx);
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
// search: Greedy search for approximate nearest neighbors.
// ============================================================================
template <typename EntityType>
int VamanaAlgorithm<EntityType>::search(VamanaContext *ctx) const {
  spin_lock_.lock();
  auto entry_point = entity_.entry_point();
  spin_lock_.unlock();

  if (ailego_unlikely(entry_point == kInvalidNodeId)) {
    return 0;
  }

  auto &topk_heap = ctx->topk_heap();
  topk_heap.clear();

  // Use ef (query-time parameter) instead of entity.search_list_size()
  // (build-time L parameter). search_list_size controls construction;
  // ef controls search quality and is user-configurable at query time.
  uint32_t ef_search = std::max(static_cast<uint32_t>(ctx->topk()), ctx->ef());
  topk_heap.limit(ef_search);

  greedy_search(entry_point, ctx, /*use_pool=*/true);

  return 0;
}

// ============================================================================
// greedy_search helper templates
//
// Two specialized inner loops, dispatched from greedy_search():
//
//   fast_greedy_search:       mmap/contiguous with direct vector pointers.
//                             Uses batch_dist on a pointer array.
//   slow_greedy_search:       BufferPool-backed storage: must fetch MemBlock
//                             wrappers via get_vector_typed to pin pages.
//
// Both accept either BlockHeap or LinearPool as `HeapType` because the
// two expose the same reset(n, ef, block_size) / push_block(dists, ids, n)
// surface (LinearPool adapts via push_block and ignores the block_size hint).
// ============================================================================

// mmap/contiguous variant: resolve vectors via get_vector_ptr
// and dispatch to the classic pointer-array batch_dist.
template <typename EntityType, typename HeapType>
void fast_greedy_search(const EntityType &entity, HeapType &pool,
                        VisitFilter &visit, VamanaDistCalculator &dc,
                        uint32_t topk, uint32_t ef, node_id_t entry_point,
                        uint32_t prefetch_lines, uint32_t prefetch_offset) {
  const uint32_t max_deg = entity.max_degree();
  const uint32_t cap = std::max(topk, ef);
  pool.reset(static_cast<int32_t>(cap), static_cast<int32_t>(max_deg));
  visit.clear();

  dist_t ep_dist = dc.batch_dist(entry_point);
  visit.set_visited(entry_point);
  pool.push_block(&ep_dist, &entry_point, 1);

  uint32_t buf_capacity = max_deg;
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);

  while (pool.has_next()) {
    auto current_node = pool.pop();

    const auto neighbors = entity.get_neighbors_typed(current_node);
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
// dual_heap_greedy_search: shared core for the fallback dual-heap path.
//
// Maintains a candidate min-heap + topk heap + VisitFilter.  Uses plain
// batch_dist.
// ============================================================================
template <typename EntityType, typename MemBlockType, typename FilterFn>
void dual_heap_greedy_search(const EntityType &entity, VamanaContext *ctx,
                             VamanaDistCalculator &dc, node_id_t entry_point,
                             FilterFn &&filter) {
  const uint32_t prefetch_offset = ctx->po();
  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;

  uint32_t buf_capacity = entity.max_degree();
  std::vector<node_id_t> neighbor_ids(buf_capacity);
  std::vector<MemBlockType> neighbor_vec_blocks;
  neighbor_vec_blocks.reserve(buf_capacity);
  std::vector<float> dists(buf_capacity);
  std::vector<const void *> neighbor_vecs(buf_capacity);

  VisitFilter &visit = ctx->visit_filter();
  CandidateHeap &candidates = ctx->candidates();
  auto &topk_heap = ctx->topk_heap();
  candidates.clear();
  visit.clear();

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
    }
    dc.batch_dist(neighbor_vecs.data(), unvisited_count, dists.data());

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
// greedy_search: Beam search from entry_point.
//
// Maintains a candidate min-heap (ordered by distance) and a visited set.
// At each step, pops the closest unvisited candidate, expands its neighbors,
// and adds unvisited neighbors to both the candidate heap and the topk heap.
// Stops when the closest candidate is farther than the worst in topk, or
// when the scan limit is reached.
// ============================================================================
template <typename EntityType>
void VamanaAlgorithm<EntityType>::greedy_search(node_id_t entry_point,
                                                VamanaContext *ctx,
                                                bool use_pool) const {
  const auto &entity = static_cast<const EntityType &>(ctx->get_entity());
  VamanaDistCalculator &dc = ctx->dist_calculator();

  const IndexFilter &index_filter =
      static_cast<const IndexContext *>(ctx)->filter();

  const uint32_t prefetch_lines =
      ctx->pl() > 0 ? ctx->pl() : (entity.vector_size() + 63) / 64;

  if (!use_pool || index_filter.is_valid()) {
    // Fallback path used by add_node (use_pool=false) and filtered search.
    // Dispatched to dual_heap_greedy_search (plain batch_dist).
    auto run_with_filter = [&](auto &&filter) {
      dual_heap_greedy_search<EntityType, MemBlockType>(
          entity, ctx, dc, entry_point, std::forward<decltype(filter)>(filter));
    };

    if (index_filter.is_valid()) {
      auto filter = [&](node_id_t id) {
        return index_filter(entity.get_key_typed(id));
      };
      run_with_filter(filter);
    } else {
      auto filter = [](node_id_t) { return false; };
      run_with_filter(filter);
    }
  } else {
    // Fast pool-based path for mmap/contiguous entities that support
    // direct pointer access. BlockHeap (AVX2) or LinearPool (scalar)
    // are used for top-k tracking. BufferPool entities fall back to
    // dual_heap_greedy_search since they lack direct pointer access.
    if constexpr (std::is_same_v<MemBlockType, MmapMemoryBlock>) {
      const uint32_t topk_v = static_cast<uint32_t>(ctx->topk());
      const uint32_t ef_v = ctx->ef();
      const bool avx2_ok =
          zvec::ailego::internal::CpuFeatures::static_flags_.AVX2;
      auto &topk_heap = ctx->topk_heap();

      auto &visit = ctx->visit_filter();

      if (avx2_ok) {
        auto &bpool = ctx->block_pool();
        fast_greedy_search(entity, bpool, visit, dc, topk_v, ef_v, entry_point,
                           prefetch_lines, ctx->po());
        copy_pool_to_topk(bpool, topk_heap);
      } else {
        auto &lpool = ctx->pool();
        fast_greedy_search(entity, lpool, visit, dc, topk_v, ef_v, entry_point,
                           prefetch_lines, ctx->po());
        copy_pool_to_topk(lpool, topk_heap);
      }
    } else {
      // BufferPool entities: fallback to dual-heap path.
      auto filter = [](node_id_t) { return false; };
      dual_heap_greedy_search<EntityType, MemBlockType>(entity, ctx, dc,
                                                        entry_point, filter);
    }
  }
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
