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

#include "../hnsw/hnsw_context.h"
#include "omega_params.h"
#include <zvec/core/interface/training.h>

namespace zvec {
namespace core {

/**
 * OmegaContext extends HnswContext to support OMEGA-specific parameters
 * like target_recall that can be set per-query.
 *
 * Training records are stored per-context (no shared state, no locks needed).
 */
class OmegaContext : public HnswContext {
 public:
  //! Constructor
  OmegaContext(size_t dimension, const IndexMetric::Pointer &metric,
               const HnswEntity::Pointer &entity)
      : HnswContext(dimension, metric, entity), target_recall_(0.95f), training_query_id_(-1) {}

  //! Constructor
  OmegaContext(const IndexMetric::Pointer &metric,
               const HnswEntity::Pointer &entity)
      : HnswContext(metric, entity), target_recall_(0.95f), training_query_id_(-1) {}

  //! Destructor
  virtual ~OmegaContext() = default;

  //! Get target recall for this query
  float target_recall() const {
    return target_recall_;
  }

  //! Get training query ID for this query (-1 means not set, use global)
  int training_query_id() const {
    return training_query_id_;
  }

  //! Get training records collected during this search (no locks needed)
  const std::vector<core_interface::TrainingRecord>& training_records() const {
    return training_records_;
  }

  //! Move training records out (override base class virtual method)
  std::vector<core_interface::TrainingRecord> take_training_records() override {
    return std::move(training_records_);
  }

  //! Add a training record
  void add_training_record(core_interface::TrainingRecord record) {
    training_records_.push_back(std::move(record));
  }

  //! Clear training records (override base class virtual method)
  //! Called before each search when context is reused from pool
  void clear_training_records() override {
    training_records_.clear();
  }

  //! Update context parameters (overrides HnswContext::update)
  int update(const ailego::Params &params) override {
    // First call parent to update HNSW parameters
    int ret = HnswContext::update(params);
    if (ret != 0) {
      return ret;
    }

    // Extract OMEGA-specific parameters
    if (params.has(PARAM_OMEGA_SEARCHER_TARGET_RECALL)) {
      params.get(PARAM_OMEGA_SEARCHER_TARGET_RECALL, &target_recall_);
    }
    if (params.has(PARAM_OMEGA_SEARCHER_TRAINING_QUERY_ID)) {
      params.get(PARAM_OMEGA_SEARCHER_TRAINING_QUERY_ID, &training_query_id_);
    }

    return 0;
  }

 private:
  float target_recall_;  // Per-query target recall
  int training_query_id_;  // Per-query training query ID for parallel training
  std::vector<core_interface::TrainingRecord> training_records_;  // Per-query training records
};

}  // namespace core
}  // namespace zvec
