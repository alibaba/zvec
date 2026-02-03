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
#include "hnsw_sparse_algorithm.h"
#include "hnsw_sparse_builder_entity.h"

namespace zvec {
namespace core {

class HnswSparseBuilder : public IndexBuilder {
 public:
  //! Constructor
  HnswSparseBuilder();

  //! Initialize the builder
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup the builder
  int cleanup(void) override;

  //! Train the data
  int train(IndexThreads::Pointer, IndexSparseHolder::Pointer holder) override;

  //! Train the data
  int train(const IndexTrainer::Pointer &trainer) override;

  //! Build the index
  int build(IndexThreads::Pointer threads,
            IndexSparseHolder::Pointer holder) override;

  //! Build the index with indptr format
  int build(IndexThreads::Pointer threads, const IndexQueryMeta &qmeta,
            size_t count, const uint64_t *keys, const uint64_t *sparse_indptr,
            const uint32_t *sparse_indices, const void *sparse_data) override;

  //! Build the index with indptr format
  int build(IndexThreads::Pointer threads, size_t count, const uint64_t *keys,
            const uint64_t *sparse_indptr, const uint32_t *sparse_indices,
            const void *sparse_data) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

 private:
  int build_graph(IndexThreads::Pointer threads,
                  std::atomic<node_id_t> &finished);
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

  HnswSparseBuilderEntity entity_{};
  HnswSparseAlgorithm::UPointer alg_;  // impl graph algorithm
  uint32_t thread_cnt_{0};
  uint32_t l0_max_neighbor_cnt_{HnswSparseEntity::kDefaultL0MaxNeighborCnt};
  uint32_t min_neighbor_cnt_{0};
  uint32_t upper_max_neighbor_cnt_{
      HnswSparseEntity::kDefaultUpperMaxNeighborCnt};
  uint32_t ef_construction_{HnswSparseEntity::kDefaultEfConstruction};
  uint32_t scaling_factor_{HnswSparseEntity::kDefaultScalingFactor};
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
