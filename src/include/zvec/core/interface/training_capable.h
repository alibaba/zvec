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

#include <vector>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>

namespace zvec {
namespace core_interface {

/**
 * @brief Training capability interface for indexes that support OMEGA training mode.
 *
 * This interface follows the Capability Pattern, allowing indexes to optionally
 * provide training functionality without polluting the base Index class.
 *
 * Example usage:
 * @code
 *   if (auto* training = index->GetTrainingCapability()) {
 *       training->EnableTrainingMode(true);
 *       // ... perform searches ...
 *       auto records = training->GetTrainingRecords();
 *   }
 * @endcode
 */
class ITrainingCapable {
 public:
  virtual ~ITrainingCapable() = default;

  /**
   * @brief Enable or disable training mode for collecting training features.
   *
   * When training mode is enabled:
   * - Early stopping is disabled (complete HNSW search)
   * - Training features are collected for each visited node
   * - query_id must be set via SetCurrentQueryId() before each search
   *
   * @param enable True to enable training mode, false to disable
   * @return Status indicating success or failure
   */
  virtual zvec::Status EnableTrainingMode(bool enable) = 0;

  /**
   * @brief Set the query ID for the next search operation.
   *
   * Must be called before search when training mode is enabled.
   * The query_id will be included in all training records collected
   * during that search.
   *
   * @param query_id Unique identifier for the query
   */
  virtual void SetCurrentQueryId(int query_id) = 0;

  /**
   * @brief Get all collected training records.
   *
   * Returns a copy of all training records collected since training mode
   * was enabled or since the last ClearTrainingRecords() call.
   *
   * @return Vector of TrainingRecord structures
   */
  virtual std::vector<TrainingRecord> GetTrainingRecords() const = 0;

  /**
   * @brief Clear all collected training records.
   *
   * Removes all training records from internal storage. Useful for
   * starting a fresh training data collection session.
   */
  virtual void ClearTrainingRecords() = 0;

  /**
   * @brief Set ground truth for training queries.
   *
   * Ground truth is used for real-time label computation during training.
   * Labels are computed as: label=1 iff top k_train GT nodes are in current topk.
   *
   * @param ground_truth 2D vector: ground_truth[query_id][rank] = node_id
   * @param k_train Number of GT nodes to check for label (typically 1)
   */
  virtual void SetTrainingGroundTruth(const std::vector<std::vector<uint64_t>>& ground_truth,
                                       int k_train = 1) = 0;
};

}  // namespace core_interface
}  // namespace zvec
