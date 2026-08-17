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
#include "zvec/core/framework/index_streamer.h"
#include "ivf_rabitq_params.h"

namespace zvec {
namespace core {

class IvfRabitqReformer;
class IvfRabitqEntity;

/*! IVF RaBitQ Streamer
 * Combines IVF partitioning with RaBitQ quantization for fast approximate
 * nearest neighbor search over a built index.
 */
class IvfRabitqStreamer : public IndexStreamer {
 public:
  IvfRabitqStreamer() = default;
  ~IvfRabitqStreamer() override = default;

  //! Initialize Streamer
  int init(const IndexMeta &meta, const ailego::Params &params) override;

  //! Open index from storage
  int open(IndexStorage::Pointer storage) override;

  //! Flush index
  int flush(uint64_t /*check_point*/) override {
    return 0;
  }

  //! Close index
  int close() override {
    return this->unload();
  }

  //! Cleanup Streamer
  int cleanup() override;

  //! Unload index
  int unload() override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Create a search context
  Context::Pointer create_context(void) const override;

  //! Create a new iterator
  IndexProvider::Pointer create_provider(void) const override;

  //! Similarity search
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Similarity search (single query)
  int search_impl(const void *query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override;

  //! Brute force search (for ground truth generation)
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override;
  int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                     uint32_t count, Context::Pointer &context) const override;

  int search_bf_by_p_keys_impl(const void *query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               Context::Pointer &context) const override;

  //! Fetch vector by key
  const void *get_vector(uint64_t key) const override;

  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override;

  //! Fetch vector by id
  const void *get_vector_by_id(uint32_t id) const override {
    return get_vector(id);
  }

  int get_vector_by_id(const uint32_t id,
                       IndexStorage::MemoryBlock &block) const override {
    return get_vector(id, block);
  }

 private:
  //! Load index from storage
  int load_index(IndexStorage::Pointer storage);

  int search_impl_internal(const void *query, const IndexQueryMeta &qmeta,
                           uint32_t count, Context::Pointer &context,
                           bool force_brute_force) const;

  int search_group_by_impl_internal(const void *query,
                                    const IndexQueryMeta &qmeta, uint32_t count,
                                    Context::Pointer &context,
                                    bool force_brute_force) const;

  //! Internal state
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  IndexMeta meta_;
  IndexMeta rabitq_meta_;
  ailego::Params params_;
  IndexStorage::Pointer storage_;
  Stats stats_;
  State state_{STATE_INIT};

  // Core components
  std::shared_ptr<IvfRabitqReformer> reformer_;
  std::shared_ptr<IvfRabitqEntity> entity_;

  // Parameters
  uint32_t nprobe_{kDefaultIvfRabitqNprobe};
  float scan_ratio_{kDefaultIvfRabitqScanRatio};
  uint32_t brute_force_threshold_{kDefaultIvfRabitqBruteForceThreshold};
  std::string metric_name_;
};

}  // namespace core
}  // namespace zvec
