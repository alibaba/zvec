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

namespace zvec {
namespace core {

/**
 * OmegaContext extends HnswContext to support OMEGA-specific parameters
 * like target_recall that can be set per-query.
 */
class OmegaContext : public HnswContext {
 public:
  //! Constructor
  OmegaContext(size_t dimension, const IndexMetric::Pointer &metric,
               const HnswEntity::Pointer &entity)
      : HnswContext(dimension, metric, entity), target_recall_(0.95f) {}

  //! Constructor
  OmegaContext(const IndexMetric::Pointer &metric,
               const HnswEntity::Pointer &entity)
      : HnswContext(metric, entity), target_recall_(0.95f) {}

  //! Destructor
  virtual ~OmegaContext() = default;

  //! Get target recall for this query
  float target_recall() const {
    return target_recall_;
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

    return 0;
  }

 private:
  float target_recall_;  // Per-query target recall
};

}  // namespace core
}  // namespace zvec
