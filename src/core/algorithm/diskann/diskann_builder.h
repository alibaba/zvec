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
#include "diskann_algorithm.h"
#include "diskann_builder_entity.h"
#include "diskann_pq_trainer.h"

namespace zvec {
namespace core {

class DiskAnnBuilder : public IndexBuilder {
 public:
  //! Constructor
  DiskAnnBuilder() = default;

  //! Initialize the builder
  virtual int init(const IndexMeta &meta,
                   const ailego::Params &params) override;

  //! Cleanup the builder
  virtual int cleanup(void) override;

  //! Train the data
  virtual int train(IndexThreads::Pointer threads,
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

  int do_norm(const void *data_ptr, std::string *norm_data);

 private:
  int train_quantized_data(IndexThreads::Pointer threads);
  int generate_quantized_data(IndexThreads::Pointer threads);
  int build_internal(IndexThreads::Pointer threads);
  int prune_internal(IndexThreads::Pointer threads);

  void do_build(uint64_t idx, size_t step_size,
                std::atomic<uint64_t> *finished);

  void do_prune(uint64_t idx, size_t step_size,
                std::atomic<uint64_t> *finished);

  int calculate_entry_point();

  int calculate_pq_chunk_num();

  double get_memory_in_bytes(double search_ram_budget) {
    return search_ram_budget * 1024 * 1024 * 1024;
  }

 private:
  enum BUILD_STATE {
    BUILD_STATE_INIT = 0,
    BUILD_STATE_INITED = 1,
    BUILD_STATE_TRAINED = 2,
    BUILD_STATE_BUILT = 3
  };

  constexpr static uint32_t kDefaultLogIntervalSecs = 15U;
  constexpr static uint32_t kDefaultListSize = 50U;
  constexpr static uint32_t kDefaultMaxDegree = 100U;
  constexpr static uint32_t kDefaultPqChunkNum = -1U;

  std::string data_file_;

  uint32_t max_degree_{kDefaultMaxDegree};
  uint32_t list_size_{kDefaultListSize};
  double memory_limit_{0.0};
  bool memory_limit_set_{false};
  uint32_t max_pq_chunk_num_{kDefaultPqChunkNum};
  uint32_t pq_chunk_num_{kDefaultPqChunkNum};
  uint32_t build_thread_count_{0};
  uint32_t max_train_sample_count_{PQTable::kMaxTrainSampleCount};
  double train_sample_ratio_{PQTable::kTrainSampleRatio};
  std::string universal_label_{""};
  std::string codebook_prefix_{""};
  std::string index_path_prefix_{"./diskann"};

  BUILD_STATE state_;
  Stats stats_;

  int errcode_{0};
  std::atomic_bool error_{false};

  IndexMetric::Pointer metric_{};

  std::mutex mutex_{};
  std::condition_variable cond_{};

  IndexMeta raw_meta_;
  IndexMeta build_meta_;

  DiskAnnBuilderEntity entity_{};

  IndexHolder::Pointer holder_;

  DiskAnnAlgorithm::UPointer algo_;
  DiskAnnPqTrainer::UPointer trainer_;

  uint32_t check_interval_secs_{kDefaultLogIntervalSecs};
};

}  // namespace core
}  // namespace zvec
