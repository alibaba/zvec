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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <zvec/ailego/pattern/expected.hpp>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>
#include "db/index/segment/segment.h"

namespace zvec {

/**
 * @brief Configuration options for training data collection
 */
struct TrainingDataCollectorOptions {
  // Number of training queries to generate
  size_t num_training_queries = 1000;

  // Gaussian noise scale for query generation
  float noise_scale = 0.01f;

  // ef parameter for training searches (large value for recall ≈ 1)
  int ef_training = 1000;

  // ef parameter for ground truth computation (0 = brute force, >0 = HNSW with this ef)
  // Using HNSW with large ef is much faster than brute force while maintaining high accuracy
  int ef_groundtruth = 0;

  // Top-K results to retrieve per query
  size_t topk = 100;

  // K_train: number of ground truth results that must be collected for label=1
  // Label=1 iff the top K_train ground truth nodes are all in collected_node_ids
  // Typically set to 1 (i.e., label=1 when the 1st ground truth is found)
  size_t k_train = 1;

  // Random seed for reproducibility
  uint64_t seed = 42;

  // Number of threads for parallel operations (0 = hardware_concurrency)
  size_t num_threads = 0;
};

/**
 * @brief Result of training data collection, includes both records and gt_cmps
 */
struct TrainingDataCollectorResult {
  std::vector<core_interface::TrainingRecord> records;
  core_interface::GtCmpsData gt_cmps_data;
};

/**
 * @brief Collector for OMEGA training data
 *
 * This class collects training data by:
 * 1. Generating training queries from base vectors
 * 2. Computing ground truth with brute force search
 * 3. Performing searches in training mode with large ef
 * 4. Labeling collected records based on ground truth
 */
class TrainingDataCollector {
 public:
  /**
   * @brief Collect training data from a persisted segment
   *
   * @param segment The segment to collect data from (must be persisted)
   * @param field_name Vector field name to train on
   * @param options Collection options
   * @param indexers Optional specific indexers to use (if empty, will use segment->get_vector_indexer)
   * @return Training records with labels filled
   */
  static Result<std::vector<core_interface::TrainingRecord>> CollectTrainingData(
      const Segment::Ptr& segment,
      const std::string& field_name,
      const TrainingDataCollectorOptions& options,
      const std::vector<VectorColumnIndexer::Ptr>& indexers = {});

  /**
   * @brief Collect training data with gt_cmps information for table generation
   *
   * This is the extended version that also computes gt_cmps data needed for
   * generating gt_collected_table and gt_cmps_all_table.
   *
   * @param segment The segment to collect data from (must be persisted)
   * @param field_name Vector field name to train on
   * @param options Collection options
   * @param indexers Optional specific indexers to use
   * @return TrainingDataCollectorResult with records and gt_cmps_data
   */
  static Result<TrainingDataCollectorResult> CollectTrainingDataWithGtCmps(
      const Segment::Ptr& segment,
      const std::string& field_name,
      const TrainingDataCollectorOptions& options,
      const std::vector<VectorColumnIndexer::Ptr>& indexers = {});

 private:
  /**
   * @brief Compute ground truth using brute force or HNSW search
   *
   * @param segment The segment to search
   * @param field_name Vector field name
   * @param queries Training query vectors
   * @param topk Number of top results to retrieve
   * @param num_threads Number of threads (0 = hardware_concurrency)
   * @param query_doc_ids Optional doc_ids of query vectors (for self-exclusion in held-out mode)
   * @param ef_groundtruth ef value for HNSW search (0 = brute force, >0 = HNSW)
   * @param metric_type Distance metric type (L2, IP, COSINE)
   * @param indexers Optional pre-opened indexers (for HNSW GT, avoids using stale indexers from segment)
   * @return Ground truth doc IDs for each query
   */
  static std::vector<std::vector<uint64_t>> ComputeGroundTruth(
      const Segment::Ptr& segment,
      const std::string& field_name,
      const std::vector<std::vector<float>>& queries,
      size_t topk,
      size_t num_threads,
      const std::vector<uint64_t>& query_doc_ids = {},
      int ef_groundtruth = 0,
      MetricType metric_type = MetricType::IP,
      const std::vector<VectorColumnIndexer::Ptr>& indexers = {});

  /**
   * @brief Fill labels in training records based on ground truth
   *
   * Label=1 iff the top K_train ground truth nodes are ALL in collected_node_ids.
   * Label=0 otherwise.
   *
   * This follows big-ann-benchmarks labeling strategy:
   * - Training records represent search states
   * - label=1 means "we've found enough results, can stop now"
   * - label=0 means "need to continue searching"
   *
   * @param records Training records to fill (modified in-place)
   * @param ground_truth Ground truth doc IDs per query (sorted by distance)
   * @param search_results Search result doc IDs per query (unused but kept for compatibility)
   * @param k_train Number of top ground truth results that must be collected
   */
  static void FillLabels(
      std::vector<core_interface::TrainingRecord>* records,
      const std::vector<std::vector<uint64_t>>& ground_truth,
      const std::vector<std::vector<uint64_t>>& search_results,
      size_t k_train);

  /**
   * @brief Compute gt_cmps data from training records and ground truth
   *
   * For each query and each GT rank, find the cmps value when that GT was first
   * collected. This data is used to generate gt_collected_table and gt_cmps_all_table.
   *
   * @param records Training records (must be sorted by query_id, then by cmps)
   * @param ground_truth Ground truth doc IDs per query
   * @param topk Number of top results per query
   * @return GtCmpsData structure with computed gt_cmps
   */
  static core_interface::GtCmpsData ComputeGtCmps(
      const std::vector<core_interface::TrainingRecord>& records,
      const std::vector<std::vector<uint64_t>>& ground_truth,
      size_t topk);
};

}  // namespace zvec
