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
#include <zvec/core/interface/training.h>
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
 * For in-memory indices, currently delegates to parent HNSW search.
 * Future: Implement adaptive search with OMEGA C API directly.
 */
class OmegaStreamer : public HnswStreamer {
 public:
  OmegaStreamer(void) : HnswStreamer() {}
  virtual ~OmegaStreamer(void) {}

  OmegaStreamer(const OmegaStreamer &streamer) = delete;
  OmegaStreamer &operator=(const OmegaStreamer &streamer) = delete;

  // Training mode support (for future implementation)
  void EnableTrainingMode(bool enable) { training_mode_enabled_ = enable; }
  void SetCurrentQueryId(int query_id) { current_query_id_ = query_id; }
  std::vector<core_interface::TrainingRecord> GetTrainingRecords() const {
    std::lock_guard<std::mutex> lock(training_mutex_);
    return collected_records_;
  }
  void ClearTrainingRecords() {
    std::lock_guard<std::mutex> lock(training_mutex_);
    collected_records_.clear();
  }

 protected:
  /**
   * @brief Override search to potentially use OMEGA adaptive search
   *
   * Currently delegates to parent HNSW search.
   * Future: Implement OMEGA adaptive search with learned early stopping.
   */
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override;

  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override;

  /**
   * @brief Override dump to set "OmegaSearcher" instead of "HnswSearcher"
   */
  virtual int dump(const IndexDumper::Pointer &dumper) override;

 private:
  // Training mode state (for future implementation)
  bool training_mode_enabled_{false};
  int current_query_id_{0};
  mutable std::mutex training_mutex_{};
  mutable std::vector<core_interface::TrainingRecord> collected_records_{};
};

}  // namespace core
}  // namespace zvec
