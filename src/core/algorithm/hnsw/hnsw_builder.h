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

#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_builder.h>
#include "hnsw_algorithm.h"
#include "hnsw_builder_entity.h"

namespace zvec {
namespace core {

class HnswBuilder : public IndexBuilder {
 public:
  //! Constructor
  HnswBuilder();

  //! Initialize the builder
  virtual int init(const IndexMeta &meta,
                   const ailego::Params &params) override;

  //! Cleanup the builder
  virtual int cleanup(void) override;

  //! Train the data
  virtual int train(IndexThreads::Pointer,
                    IndexHolder::Pointer holder) override;

  //! Train the data
  virtual int train(const IndexTrainer::Pointer &trainer) override;


  //! Build the index
  virtual int build(IndexThreads::Pointer threads,
                    IndexHolder::Pointer holder) override;

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override {
    return stats_;
  }

 private:
  void do_build(node_id_t idx, size_t step_size,
                std::atomic<node_id_t> *finished);

  constexpr static uint32_t kDefaultLogIntervalSecs = 15U;
  constexpr static uint32_t kMaxNeighborCnt = 65535;

 private:
  enum BUILD_STATE {
    BUILD_STATE_INIT = 0,
    BUILD_STATE_INITED = 1,
    BUILD_STATE_TRAINED = 2,
    BUILD_STATE_BUILT = 3
  };

  HnswBuilderEntity entity_{};
  HnswAlgorithm::UPointer alg_;  // impl graph algorithm
  uint32_t thread_cnt_{0};
  uint32_t min_neighbor_cnt_{0};
  uint32_t upper_max_neighbor_cnt_{HnswEntity::kDefaultUpperMaxNeighborCnt};
  uint32_t l0_max_neighbor_cnt_{HnswEntity::kDefaultL0MaxNeighborCnt};
  uint32_t ef_construction_{HnswEntity::kDefaultEfConstruction};
  uint32_t scaling_factor_{HnswEntity::kDefaultScalingFactor};
  uint32_t check_interval_secs_{kDefaultLogIntervalSecs};

  int errcode_{0};
  std::atomic_bool error_{false};
  IndexMeta meta_{};
  IndexMetric::Pointer metric_{};
  std::mutex mutex_{};
  std::condition_variable cond_{};
  Stats stats_{};

  BUILD_STATE state_{BUILD_STATE_INIT};
};

}  // namespace core
}  // namespace zvec
