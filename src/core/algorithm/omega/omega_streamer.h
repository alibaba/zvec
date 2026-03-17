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

#include "../hnsw/hnsw_streamer.h"
#include "omega_context.h"
#include <zvec/core/interface/training.h>
#include <omega/omega_api.h>
#include <atomic>
#include <mutex>
#include <memory>

namespace zvec {
namespace core {

/**
 * @brief OMEGA Index Streamer
 *
 * Inherits from HnswStreamer and overrides dump() to set "OmegaSearcher"
 * as the searcher type, ensuring that disk-persisted indices will use
 * OmegaSearcher (with training support) when loaded.
 *
 * Supports both training mode (feature collection) and inference mode
 * (adaptive search with learned early stopping).
 */
class OmegaStreamer : public HnswStreamer {
 public:
  OmegaStreamer(void) : HnswStreamer(), omega_model_(nullptr) {}
  virtual ~OmegaStreamer(void) {
    if (omega_model_ != nullptr) {
      omega_model_destroy(omega_model_);
      omega_model_ = nullptr;
    }
  }

  OmegaStreamer(const OmegaStreamer &streamer) = delete;
  OmegaStreamer &operator=(const OmegaStreamer &streamer) = delete;

  // Training mode support
  void EnableTrainingMode(bool enable) { training_mode_enabled_ = enable; }
  void SetCurrentQueryId(int query_id) { current_query_id_ = query_id; }
  void SetTrainingGroundTruth(const std::vector<std::vector<uint64_t>>& ground_truth,
                               int k_train = 1) {
    training_ground_truth_ = ground_truth;
    training_k_train_ = k_train;
  }

  // Inference mode support
  bool LoadModel(const std::string& model_dir);
  bool IsModelLoaded() const;
  void SetTargetRecall(float target_recall) { target_recall_ = target_recall; }
  void SetWindowSize(int window_size) { window_size_ = window_size; }

 protected:
  /**
   * @brief Override search to use OMEGA adaptive search
   *
   * In training mode: collects features without early stopping
   * In inference mode: uses OMEGA model for adaptive early stopping
   */
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override;

  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override;

  /**
   * @brief Override open to auto-load omega_model from the index directory.
   */
  virtual int open(IndexStorage::Pointer stg) override;

  /**
   * @brief Override create_context to return OmegaContext
   */
  virtual Context::Pointer create_context() const override;

  /**
   * @brief Override dump to set "OmegaSearcher" instead of "HnswSearcher"
   */
  virtual int dump(const IndexDumper::Pointer &dumper) override;

 private:
  // Perform OMEGA adaptive search (shared between training and inference mode)
  int omega_search_impl(const void *query, const IndexQueryMeta &qmeta,
                        uint32_t count, Context::Pointer &context,
                        bool enable_early_stopping) const;

  // Training mode state
  mutable bool training_mode_enabled_{false};
  mutable int current_query_id_{0};
  std::vector<std::vector<uint64_t>> training_ground_truth_;
  int training_k_train_{1};

  // Inference mode state
  mutable OmegaModelHandle omega_model_{nullptr};
  mutable std::mutex model_mutex_;
  mutable std::atomic<bool> debug_stats_logged_{false};
  float target_recall_{0.95f};
  int window_size_{100};
};

}  // namespace core
}  // namespace zvec
