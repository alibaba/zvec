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

#include <zvec/core/framework/index_framework.h>
#include "hnsw_searcher_entity.h"
#include "hnsw_streamer.h"

namespace zvec {
namespace core {

class HnswSearcher : public IndexSearcher {
 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  HnswSearcher(void);
  ~HnswSearcher(void);

  HnswSearcher(const HnswSearcher &) = delete;
  HnswSearcher &operator=(const HnswSearcher &) = delete;

 protected:
  //! Initialize Searcher
  virtual int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  virtual int cleanup(void) override;

  //! Load Index from storage
  virtual int load(IndexStorage::Pointer container,
                   IndexMetric::Pointer metric) override;

  //! Unload index from storage
  virtual int unload(void) override;

  //! KNN Search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          ContextPointer &context) const override {
    return search_impl(query, qmeta, 1, context);
  }

  //! KNN Search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          ContextPointer &context) const override;

  //! Linear Search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             ContextPointer &context) const override {
    return search_bf_impl(query, qmeta, 1, context);
  }

  //! Linear Search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count,
                             ContextPointer &context) const override;

  //! Linear search by primary keys
  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, uint32_t count,
      ContextPointer &context) const override;

  //! Fetch vector by key
  virtual const void *get_vector(uint64_t key) const override;

  //! Create a searcher context
  virtual ContextPointer create_context() const override;

  //! Create a new iterator
  virtual IndexProvider::Pointer create_provider(void) const override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve params of index
  virtual const ailego::Params &params(void) const override {
    return params_;
  }

  virtual void print_debug_info() override;

 private:
  //! To share ctx across streamer/searcher, we need to update the context for
  //! current streamer/searcher
  int update_context(HnswContext *ctx) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  HnswSearcherEntity entity_{};
  HnswAlgorithm::UPointer alg_;  // impl graph algorithm

  IndexMetric::Pointer metric_{};
  IndexMeta meta_{};
  ailego::Params params_{};
  Stats stats_;
  uint32_t ef_{HnswEntity::kDefaultEf};
  uint32_t max_scan_num_{0U};
  uint32_t bruteforce_threshold_{HnswEntity::kDefaultBruteForceThreshold};
  float max_scan_ratio_{HnswEntity::kDefaultScanRatio};
  bool bf_enabled_{false};
  bool check_crc_enabled_{false};
  bool neighbors_in_memory_enabled_{false};
  bool force_padding_topk_enabled_{false};
  float bf_negative_probability_{HnswEntity::kDefaultBFNegativeProbability};
  uint32_t magic_{0U};

  State state_{STATE_INIT};
};

}  // namespace core
}  // namespace zvec