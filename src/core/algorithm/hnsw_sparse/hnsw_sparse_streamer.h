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

#include <ailego/parallel/lock.h>
#include <zvec/core/framework/index_framework.h>
#include "hnsw_sparse_algorithm.h"
#include "hnsw_sparse_streamer_entity.h"

namespace zvec {
namespace core {

class HnswSparseStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

  HnswSparseStreamer(void);
  virtual ~HnswSparseStreamer(void);

  HnswSparseStreamer(const HnswSparseStreamer &streamer) = delete;
  HnswSparseStreamer &operator=(const HnswSparseStreamer &streamer) = delete;

 protected:
  //! Initialize Streamer
  int init(const IndexMeta &imeta, const ailego::Params &params) override;

  //! Cleanup Streamer
  int cleanup(void) override;

  //! Create a context
  Context::Pointer create_context(void) const override;

  //! Create a new sparse iterator
  IndexStreamer::SparseProvider::Pointer create_sparse_provider(
      void) const override;

  int add_impl(uint64_t pkey, const uint32_t sparse_count,
               const uint32_t *sparse_indices, const void *sparse_query,
               const IndexQueryMeta &qmeta, Context::Pointer &context) override;

  int add_with_id_impl(uint32_t id, const uint32_t sparse_count,
                       const uint32_t *sparse_indices, const void *sparse_query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  Context::Pointer &context) const override;

  //! Similarity search with sparse inputs
  int search_impl(const uint32_t *sparse_count, const uint32_t *sparse_indices,
                  const void *sparse_query, const IndexQueryMeta &qmeta,
                  uint32_t count, Context::Pointer &context) const override;

  //! Similarity brute force search with sparse inputs
  int search_bf_impl(const uint32_t sparse_count,
                     const uint32_t *sparse_indices, const void *sparse_query,
                     const IndexQueryMeta &qmeta,
                     Context::Pointer &context) const override;

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
                               ContextPointer &context) const override;

  //! Linear search by primary keys with sparse inputs
  int search_bf_by_p_keys_impl(const uint32_t *sparse_count,
                               const uint32_t *sparse_indices,
                               const void *sparse_query,
                               const std::vector<std::vector<uint64_t>> &p_keys,
                               const IndexQueryMeta &qmeta, uint32_t count,
                               ContextPointer &context) const override;

  //! Fetch sparse vector by key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override {
    return entity_.get_sparse_vector_by_key(
        key, sparse_count, sparse_indices_buffer, sparse_values_buffer);
  }

  //! Fetch vector by id
  int get_sparse_vector_by_id(
      uint32_t id, uint32_t *sparse_count, std::string *sparse_indices_buffer,
      std::string *sparse_values_buffer) const override {
    return entity_.get_sparse_vector_by_id(
        id, sparse_count, sparse_indices_buffer, sparse_values_buffer);
  }

  //! Open index from file path
  int open(IndexStorage::Pointer stg) override;

  //! Close file
  int close(void) override;

  //! flush file
  int flush(uint64_t checkpoint) override;

  //! Dump index into storage
  int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve sparse meta of index
  const IndexMeta &meta(void) const override {
    return meta_;
  }

  void print_debug_info() override;

 private:
  inline int check_params(const IndexQueryMeta &qmeta) const {
    if (ailego_unlikely(qmeta.data_type() != meta_.data_type())) {
      LOG_ERROR("Unsupported query meta");
      return IndexError_Mismatch;
    }
    return 0;
  }

  inline int check_sparse_count_is_zero(const uint32_t *sparse_count,
                                        uint32_t count) const {
    for (uint32_t i = 0; i < count; ++i) {
      if (sparse_count[i] != 0)
        LOG_ERROR("Sparse cout is not empty. Index: %u, Sparse Count: %u", i,
                  sparse_count[i]);
      return IndexError_InvalidArgument;
    }

    return 0;
  }

 private:
  //! To share ctx across streamer/searcher, we need to update the context for
  //! current streamer/searcher
  int update_context(HnswSparseContext *ctx) const;

 private:
  enum State { STATE_INIT = 0, STATE_INITED = 1, STATE_OPENED = 2 };
  class Stats : public IndexStreamer::Stats {
   public:
    void clear(void) {
      set_revision_id(0u);
      set_loaded_count(0u);
      set_added_count(0u);
      set_discarded_count(0u);
      set_index_size(0u);
      set_dumped_size(0u);
      set_check_point(0u);
      set_create_time(0u);
      set_update_time(0u);
      clear_attributes();
    }
  };

  HnswSparseStreamerEntity entity_;
  HnswSparseAlgorithm::UPointer alg_;
  IndexMeta meta_{};
  IndexMetric::Pointer metric_{};

  IndexMetric::MatrixSparseDistance add_distance_{};
  IndexMetric::MatrixSparseDistance search_distance_{};
  Stats stats_{};
  std::mutex mutex_{};

  size_t max_index_size_{0UL};
  size_t chunk_size_{HnswSparseEntity::kDefaultChunkSize};
  size_t docs_hard_limit_{HnswSparseEntity::kDefaultDocsHardLimit};
  size_t docs_soft_limit_{0UL};
  uint32_t min_neighbor_cnt_{0u};
  uint32_t upper_max_neighbor_cnt_{
      HnswSparseEntity::kDefaultUpperMaxNeighborCnt};
  uint32_t l0_max_neighbor_cnt_{HnswSparseEntity::kDefaultL0MaxNeighborCnt};
  uint32_t ef_{HnswSparseEntity::kDefaultEf};
  uint32_t ef_construction_{HnswSparseEntity::kDefaultEfConstruction};
  uint32_t scaling_factor_{HnswSparseEntity::kDefaultScalingFactor};
  size_t bruteforce_threshold_{HnswSparseEntity::kDefaultBruteForceThreshold};
  size_t max_scan_limit_{HnswSparseEntity::kDefaultMaxScanLimit};
  size_t min_scan_limit_{HnswSparseEntity::kDefaultMinScanLimit};
  float bf_negative_prob_{HnswSparseEntity::kDefaultBFNegativeProbability};
  float max_scan_ratio_{HnswSparseEntity::kDefaultScanRatio};
  float sparse_neighbor_ratio_{HnswSparseEntity::kDefaultSparseNeighborRatio};
  uint32_t sparse_neighbor_cnt_{0UL};
  uint32_t sparse_min_neighbor_cnt_{0UL};
  uint32_t upper_sparse_neighbor_cnt_{0UL};

  bool query_filtering_enabled_{false};
  float query_filtering_ratio_{HnswSparseEntity::kDefaultQueryFilteringRatio};

  uint32_t magic_{0U};
  State state_{STATE_INIT};
  bool bf_enabled_{false};
  bool check_crc_enabled_{false};
  bool filter_same_key_{false};
  bool get_vector_enabled_{false};
  bool force_padding_topk_enabled_{false};

  //! avoid add vector while dumping index
  ailego::SharedMutex shared_mutex_{};
};

}  // namespace core
}  // namespace zvec
