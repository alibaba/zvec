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

#include <array>
#include <vector>

namespace zvec::core_interface {

/**
 * @brief Training record structure for OMEGA adaptive search.
 *
 * This structure captures features collected during a single comparison
 * in the HNSW/OMEGA search process. Multiple records are collected per query
 * to train models for early stopping prediction.
 *
 * Features (11 dimensions total):
 * - query_id: Unique identifier for the query
 * - hops_visited: Number of hops in the search graph
 * - cmps_visited: Number of node comparisons performed
 * - dist_1st: Distance to the first (best) result found so far
 * - dist_start: Distance to the starting node
 * - traversal_window_stats: 7 statistical features of the traversal window
 *   (avg, var, min, max, median, percentile25, percentile75)
 * - label: Binary label (1 if collected enough GT results, 0 otherwise)
 *          Computed in real-time during search (memory optimized)
 */
struct TrainingRecord {
  int query_id;
  int hops_visited;
  int cmps_visited;
  float dist_1st;
  float dist_start;
  std::array<float, 7> traversal_window_stats;
  int label;  // Computed in real-time during search

  TrainingRecord()
      : query_id(0),
        hops_visited(0),
        cmps_visited(0),
        dist_1st(0.0f),
        dist_start(0.0f),
        traversal_window_stats{},
        label(0) {}
};

/**
 * @brief Ground truth cmps data for OMEGA table generation.
 *
 * For each query, stores the cmps value when each ground truth result was
 * found. This data is used to generate gt_collected_table and
 * gt_cmps_all_table.
 *
 * gt_cmps[query_id][rank] = cmps value when GT[rank] was collected
 *                         = total_cmps if GT[rank] was never found
 */
struct GtCmpsData {
  // gt_cmps[query_id][rank] = cmps when GT of rank was found
  std::vector<std::vector<int>> gt_cmps;

  // total_cmps[query_id] = total comparisons for this query
  std::vector<int> total_cmps;

  // topk value used during training
  size_t topk = 0;

  // Number of queries
  size_t num_queries = 0;
};

}  // namespace zvec::core_interface
