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
#include "hnsw_algorithm.h"
#include "hnsw_streamer_entity.h"

namespace zvec {
namespace core {

class HnswStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

  HnswStreamer(void);
  virtual ~HnswStreamer(void);

  HnswStreamer(const HnswStreamer &streamer) = delete;
  HnswStreamer &operator=(const HnswStreamer &streamer) = delete;

 protected:
  //! Initialize Streamer
  virtual int init(const IndexMeta &imeta,
                   const ailego::Params &params) override;

  //! Cleanup Streamer
  virtual int cleanup(void) override;

  //! Create a context
  virtual Context::Pointer create_context(void) const override;

  //! Create a new iterator
  virtual IndexProvider::Pointer create_provider(void) const override;

  //! Add a vector into index
  virtual int add_impl(uint64_t pkey, const void *query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override;

  //! Add a vector with id into index
  virtual int add_with_id_impl(uint32_t id, const void *query,
                               const IndexQueryMeta &qmeta,
                               Context::Pointer &context) override;

  //! Similarity search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override;

  //! Similarity search
  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const override;

  //! Similarity brute force search
  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count,
                             Context::Pointer &context) const override;

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
  virtual const void *get_vector(uint64_t key) const override {
    return entity_.get_vector_by_key(key);
  }

  virtual int get_vector(const uint64_t key,
                         IndexStorage::MemoryBlock &block) const override {
    return entity_.get_vector_by_key(key, block);
  }

  //! Fetch vector by id
  virtual const void *get_vector_by_id(uint32_t id) const override {
    return entity_.get_vector(id);
  }

  virtual int get_vector_by_id(
      const uint32_t id, IndexStorage::MemoryBlock &block) const override {
    return entity_.get_vector(id, block);
  }

  //! Open index from file path
  virtual int open(IndexStorage::Pointer stg) override;

  //! Close file
  virtual int close(void) override;

  //! flush file
  virtual int flush(uint64_t checkpoint) override;

  //! Dump index into storage
  virtual int dump(const IndexDumper::Pointer &dumper) override;

  //! Retrieve statistics
  virtual const Stats &stats(void) const override {
    return stats_;
  }

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const override {
    return meta_;
  }

  virtual void print_debug_info() override;

 private:
  inline int check_params(const void *query,
                          const IndexQueryMeta &qmeta) const {
    if (ailego_unlikely(!query)) {
      LOG_ERROR("null query");
      return IndexError_InvalidArgument;
    }
    if (ailego_unlikely(qmeta.dimension() != meta_.dimension() ||
                        qmeta.data_type() != meta_.data_type() ||
                        qmeta.element_size() != meta_.element_size())) {
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
  int update_context(HnswContext *ctx) const;

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

  HnswStreamerEntity entity_;
  HnswAlgorithm::UPointer alg_;
  IndexMeta meta_{};
  IndexMetric::Pointer metric_{};

  IndexMetric::MatrixDistance add_distance_{};
  IndexMetric::MatrixDistance search_distance_{};

  IndexMetric::MatrixBatchDistance add_batch_distance_{};
  IndexMetric::MatrixBatchDistance search_batch_distance_{};

  Stats stats_{};
  std::mutex mutex_{};

  size_t max_index_size_{0UL};
  size_t chunk_size_{HnswEntity::kDefaultChunkSize};
  size_t docs_hard_limit_{HnswEntity::kDefaultDocsHardLimit};
  size_t docs_soft_limit_{0UL};
  uint32_t min_neighbor_cnt_{0u};
  uint32_t upper_max_neighbor_cnt_{HnswEntity::kDefaultUpperMaxNeighborCnt};
  uint32_t l0_max_neighbor_cnt_{HnswEntity::kDefaultL0MaxNeighborCnt};
  uint32_t ef_{HnswEntity::kDefaultEf};
  uint32_t ef_construction_{HnswEntity::kDefaultEfConstruction};
  uint32_t scaling_factor_{HnswEntity::kDefaultScalingFactor};
  size_t bruteforce_threshold_{HnswEntity::kDefaultBruteForceThreshold};
  size_t max_scan_limit_{HnswEntity::kDefaultMaxScanLimit};
  size_t min_scan_limit_{HnswEntity::kDefaultMinScanLimit};
  float bf_negative_prob_{HnswEntity::kDefaultBFNegativeProbability};
  float max_scan_ratio_{HnswEntity::kDefaultScanRatio};

  uint32_t magic_{0U};
  State state_{STATE_INIT};
  bool bf_enabled_{false};
  bool check_crc_enabled_{false};
  bool filter_same_key_{false};
  bool get_vector_enabled_{false};
  bool force_padding_topk_enabled_{false};
  bool use_id_map_{true};

  //! avoid add vector while dumping index
  ailego::SharedMutex shared_mutex_{};
};

}  // namespace core
}  // namespace zvec
