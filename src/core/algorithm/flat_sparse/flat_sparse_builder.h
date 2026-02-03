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

#include <cstdint>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_dumper.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_holder.h>

namespace zvec {
namespace core {

/*! Brute Force Sparse Builder
 */
class FlatSparseBuilder : public IndexBuilder {
 public:
  //! Constructor
  FlatSparseBuilder();

  //! Initialize the builder
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup the builder
  int cleanup(void) override;

  //! Train the data
  int train(IndexThreads::Pointer, IndexSparseHolder::Pointer holder) override;

  //! Train the data
  int train(const IndexTrainer::Pointer &trainer) override;

  int train(IndexThreads::Pointer /*threads*/,
            IndexHolder::Pointer /*holder*/) override {
    return IndexError_NotImplemented;
  }

  int build(IndexThreads::Pointer /*threads*/,
            IndexHolder::Pointer /*holder*/) override {
    return IndexError_NotImplemented;
  }

  //! Build the index
  int build(IndexThreads::Pointer threads,
            IndexSparseHolder::Pointer holder) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

 private:
  int do_dump(const IndexDumper::Pointer &dumper, uint32_t *dump_count);

  int dump_meta(IndexDumper *dumper);

  int dump_keys(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  int dump_mapping(const std::vector<uint64_t> &keys, IndexDumper *dumper);

  int dump_vector_and_offset(IndexDumper *dumper, std::vector<uint64_t> *keys);

  int write_vector_data(const uint32_t sparse_count,
                        const uint32_t *sparse_indices, const void *sparse_vec,
                        IndexDumper *dumper, uint32_t *length);

 private:
  enum BUILD_STATE {
    BUILD_STATE_INIT = 0,
    BUILD_STATE_INITED = 1,
    BUILD_STATE_TRAINED = 2,
    BUILD_STATE_BUILT = 3
  };

  IndexSparseHolder::Pointer holder_{};

  std::atomic_bool error_{false};
  IndexMeta meta_{};
  IndexMetric::Pointer measure_{};
  std::mutex mutex_{};
  std::condition_variable cond_{};
  Stats stats_{};

  BUILD_STATE state_{BUILD_STATE_INIT};
};


}  // namespace core
}  // namespace zvec
