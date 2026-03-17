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

#include <zvec/core/framework/index_framework.h>
#include <zvec/core/interface/training.h>
#include <zvec/db/status.h>
#include "../hnsw/hnsw_searcher.h"
#include <omega/omega_api.h>
#include <mutex>
#include <vector>

namespace zvec {
namespace core {

//! OMEGA Index Searcher - extends HNSW with adaptive search
class OmegaSearcher : public HnswSearcher {
 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  OmegaSearcher(void);
  ~OmegaSearcher(void);

  OmegaSearcher(const OmegaSearcher &) = delete;
  OmegaSearcher &operator=(const OmegaSearcher &) = delete;

 public:
  // OMEGA Training Mode Support
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
  zvec::Status EnableTrainingMode(bool enable);

  /**
   * @brief Set the query ID for the next search operation.
   *
   * Must be called before search_impl() when training mode is enabled.
   * The query_id will be included in all training records collected
   * during that search.
   *
   * @param query_id Unique identifier for the query
   */
  void SetCurrentQueryId(int query_id);

  /**
   * @brief Get all collected training records.
   *
   * Returns a copy of all training records collected since training mode
   * was enabled or since the last ClearTrainingRecords() call.
   *
   * @return Vector of TrainingRecord structures
   */
  std::vector<core_interface::TrainingRecord> GetTrainingRecords() const;

  /**
   * @brief Clear all collected training records.
   *
   * Removes all training records from internal storage. Useful for
   * starting a fresh training data collection session.
   */
  void ClearTrainingRecords();

  /**
   * @brief Set ground truth for training queries.
   *
   * Ground truth is used for real-time label computation during training.
   * Labels are computed as: label=1 iff top k_train GT nodes are in current topk.
   *
   * @param ground_truth 2D vector: ground_truth[query_id][rank] = node_id
   * @param k_train Number of GT nodes to check for label (typically 1)
   */
  void SetTrainingGroundTruth(const std::vector<std::vector<uint64_t>>& ground_truth,
                               int k_train = 1);

  /**
   * @brief Public search method for OmegaStreamer to call
   *
   * This allows OmegaStreamer to delegate search to OmegaSearcher
   * without needing to access protected methods.
   */
  int search(const void *query, const IndexQueryMeta &qmeta,
             uint32_t count, ContextPointer &context) const {
    return search_impl(query, qmeta, count, context);
  }

 protected:
  //! Initialize Searcher
  virtual int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  virtual int cleanup(void) override;

  //! Load Index from storage
  virtual int load(IndexStorage::Pointer container,
                   IndexMetric::Pointer metric) override;

  //! Unload index from storage
  virtual int unload(void) override;

  //! KNN Search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          ContextPointer &context) const override {
    return search_impl(query, qmeta, 1, context);
  }

  //! KNN Search with OMEGA adaptive search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          ContextPointer &context) const override;

  //! Create a searcher context (creates OmegaContext instead of HnswContext)
  virtual ContextPointer create_context() const override;

  // NOTE: The commented-out delegation methods below are intentionally not used.
  // OmegaSearcher inherits from HnswSearcher and overrides only the necessary methods.
  // The base class implementations are sufficient for the remaining functionality.
  /*
  //! Fetch vector by key (delegate to HNSW)
  virtual const void *get_vector(uint64_t key) const override {
    return hnsw_searcher_->get_vector(key);
  }

  //! Create a searcher context (delegate to HNSW)
  virtual ContextPointer create_context() const override {
    return hnsw_searcher_->create_context();
  }

  //! Create a new iterator (delegate to HNSW)
  virtual IndexProvider::Pointer create_provider(void) const override {
    return hnsw_searcher_->create_provider();
  }

  //! Retrieve statistics (delegate to HNSW)
  virtual const Stats &stats(void) const override {
    return hnsw_searcher_->stats();
  }

  //! Retrieve meta of index (delegate to HNSW)
  virtual const IndexMeta &meta(void) const override {
    return hnsw_searcher_->meta();
  }

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const override {
    return params_;
  }

  virtual void print_debug_info() override {
    hnsw_searcher_->print_debug_info();
  }
  */

 private:
  //! Check if OMEGA mode should be used
  bool should_use_omega() const {
    // Use OMEGA adaptive search if:
    // 1. Training mode is enabled (to collect features even without model), OR
    // 2. OMEGA is enabled and model is loaded
    if (training_mode_enabled_) {
      return true;  // Always use adaptive_search in training mode
    }
    return omega_enabled_ && use_omega_mode_ &&
           omega_model_ != nullptr &&
           omega_model_is_loaded(omega_model_);
  }

  //! Adaptive search with OMEGA predictions
  int adaptive_search(const void *query, const IndexQueryMeta &qmeta,
                      uint32_t count, ContextPointer &context) const;

 private:
  // OMEGA components
  OmegaModelHandle omega_model_;
  bool omega_enabled_;
  bool use_omega_mode_;
  float target_recall_;
  uint32_t min_vector_threshold_;
  size_t current_vector_count_;
  int window_size_;

  // Training mode support
  bool training_mode_enabled_;
  int current_query_id_;
  mutable std::mutex training_mutex_;
  mutable std::vector<core_interface::TrainingRecord> collected_records_;
  std::vector<std::vector<uint64_t>> training_ground_truth_;  // [query_id][rank] = node_id
  int training_k_train_;  // Number of GT nodes to check for label
};

}  // namespace core
}  // namespace zvec
