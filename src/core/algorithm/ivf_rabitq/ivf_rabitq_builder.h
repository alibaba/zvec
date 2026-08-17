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

#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_builder.h>
#include <zvec/core/framework/index_meta.h>
#include "ivf_rabitq_index_format.h"

namespace zvec {
namespace core {

class IvfRabitqReformer;
class RabitqConverter;

/*! IVF RaBitQ Builder
 * Implements the standard IndexBuilder interface (init → train → build → dump)
 * for IVF RaBitQ indexes.
 */
class IvfRabitqBuilder : public IndexBuilder {
 public:
  IvfRabitqBuilder();
  ~IvfRabitqBuilder() override;

  IvfRabitqBuilder(const IvfRabitqBuilder &) = delete;
  IvfRabitqBuilder &operator=(const IvfRabitqBuilder &) = delete;

  //! Initialize the builder
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Cleanup the builder
  int cleanup() override;

  //! Train the data (KMeans clustering + rotator)
  int train(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override;

  //! Build the index (assign vectors to centroids, quantize)
  int build(IndexThreads::Pointer threads,
            IndexHolder::Pointer holder) override;

  //! Dump index into file system
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats() const override {
    return stats_;
  }

 private:
  enum State { INIT = 0, INITED = 1, TRAINED = 2, BUILT = 3 };
  State state_{INIT};
  Stats stats_;
  IndexMeta meta_;
  IndexMeta rabitq_meta_;
  ailego::Params params_;

  // IVF RaBitQ parameters
  uint32_t nlist_{1024};
  uint32_t total_bits_{7};
  uint32_t sample_count_{0};
  uint32_t thread_count_{0};
  std::string metric_name_;

  // Training results
  std::shared_ptr<RabitqConverter> converter_;
  std::shared_ptr<IvfRabitqReformer> reformer_;

  // Build results (stored in memory until dump)
  IvfRabitqHeader header_{};
  std::vector<char> batch_data_buf_;
  std::vector<char> ex_data_buf_;
  std::vector<IvfRabitqClusterMeta> cluster_metas_;
  std::vector<uint64_t> keys_buf_;
  std::vector<uint32_t> mapping_buf_;
};

}  // namespace core
}  // namespace zvec
