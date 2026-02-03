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
#include "hnsw_sparse_searcher_entity.h"
#include "hnsw_sparse_streamer.h"

namespace zvec {
namespace core {

class HnswSparseSearcher : public IndexSearcher {
 public:
  using ContextPointer = IndexSearcher::Context::Pointer;

 public:
  HnswSparseSearcher(void);
  virtual ~HnswSparseSearcher(void);

  HnswSparseSearcher(const HnswSparseSearcher &) = delete;
  HnswSparseSearcher &operator=(const HnswSparseSearcher &) = delete;

 protected:
  //! Initialize Searcher
  int init(const ailego::Params &params) override;

  //! Cleanup Searcher
  int cleanup(void) override;

  //! Load Index from storage
  int load(IndexStorage::Pointer container,
           IndexMetric::Pointer measure) override;

  //! Unload index from storage
  int unload(void) override;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override {
    return search_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                       context);
  }

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override {
    return search_bf_impl(&sparse_count, sparse_indices, sparse_query, qmeta, 1,
                          context);
  }

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t *sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta, uint32_t count,
                     Context::Pointer &context) const override;

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const uint32_t sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta,
                               ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(&sparse_count, sparse_indices, sparse_query,
                                    p_keys, qmeta, 1, context);
  }

  //! Linear search by primary keys
  int search_bf_by_p_keys_impl(const uint32_t *sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               ContextPointer &context) const override;

  //! Fetch sparser vector by key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override;

  //! Create a searcher context
  ContextPointer create_context() const override;

  //! Create a new iterator
  IndexSearcher::SparseProvider::Pointer create_sparse_provider(
      void) const override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  //! Retrieve params of index
  const ailego::Params &params(void) const override {
    return params_;
  }

  void print_debug_info() override;

 private:
  //! To share ctx across streamer/searcher, we need to update the context for
  //! current streamer/searcher
  int update_context(HnswSparseContext *ctx) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_LOADED = 2 };

  HnswSparseSearcherEntity entity_{};
  HnswSparseAlgorithm::UPointer alg_;  // impl graph algorithm

  IndexMetric::Pointer metric_{};
  IndexMeta meta_{};
  ailego::Params params_{};
  Stats stats_;
  uint32_t ef_{HnswSparseEntity::kDefaultEf};
  uint32_t max_scan_num_{0U};
  uint32_t bruteforce_threshold_{HnswSparseEntity::kDefaultBruteForceThreshold};
  float max_scan_ratio_{HnswSparseEntity::kDefaultScanRatio};
  bool bf_enabled_{false};
  bool check_crc_enabled_{false};
  bool neighbors_in_memory_enabled_{false};
  bool force_padding_topk_enabled_{false};
  float bf_negative_probability_{
      HnswSparseEntity::kDefaultBFNegativeProbability};

  bool query_filtering_enabled_{false};
  float query_filtering_ratio_{HnswSparseEntity::kDefaultQueryFilteringRatio};

  uint32_t magic_{0U};

  State state_{STATE_INIT};
};

}  // namespace core
}  // namespace zvec
