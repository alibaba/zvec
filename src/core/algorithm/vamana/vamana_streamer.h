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
#include "vamana_algorithm.h"
#include "vamana_streamer_entity.h"

namespace zvec {
namespace core {

class VamanaStreamer : public IndexStreamer {
 public:
  using ContextPointer = IndexStreamer::Context::Pointer;

  VamanaStreamer(void);
  virtual ~VamanaStreamer(void);

  VamanaStreamer(const VamanaStreamer &) = delete;
  VamanaStreamer &operator=(const VamanaStreamer &) = delete;

 protected:
  virtual int init(const IndexMeta &imeta,
                   const ailego::Params &params) override;

  virtual int cleanup(void) override;

  virtual Context::Pointer create_context(void) const override;

  virtual IndexProvider::Pointer create_provider(void) const override;

  virtual int add_impl(uint64_t pkey, const void *query,
                       const IndexQueryMeta &qmeta,
                       Context::Pointer &context) override;

  virtual int add_with_id_impl(uint32_t id, const void *query,
                               const IndexQueryMeta &qmeta,
                               Context::Pointer &context) override;

  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          Context::Pointer &context) const override;

  virtual int search_impl(const void *query, const IndexQueryMeta &qmeta,
                          uint32_t count,
                          Context::Pointer &context) const override;

  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             Context::Pointer &context) const override;

  virtual int search_bf_impl(const void *query, const IndexQueryMeta &qmeta,
                             uint32_t count,
                             Context::Pointer &context) const override;

  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, ContextPointer &context) const override {
    return search_bf_by_p_keys_impl(query, p_keys, qmeta, 1, context);
  }

  virtual int search_bf_by_p_keys_impl(
      const void *query, const std::vector<std::vector<uint64_t>> &p_keys,
      const IndexQueryMeta &qmeta, uint32_t count,
      ContextPointer &context) const override;

  virtual const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  virtual int get_vector(const uint64_t key,
                         IndexStorage::MemoryBlock &block) const override {
    return entity_->get_vector_by_key(key, block);
  }

  virtual const void *get_vector_by_id(uint32_t id) const override {
    return entity_->get_vector(id);
  }

  virtual int get_vector_by_id(
      const uint32_t id, IndexStorage::MemoryBlock &block) const override {
    return entity_->get_vector(id, block);
  }

  virtual int open(IndexStorage::Pointer stg) override;

  virtual int close(void) override;

  virtual int flush(uint64_t checkpoint) override;

  virtual int dump(const IndexDumper::Pointer &dumper) override;

  virtual const Stats &stats(void) const override {
    return stats_;
  }

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

  int setup_entity();
  int update_context(VamanaContext *ctx) const;

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

  std::unique_ptr<VamanaStreamerEntity> entity_;
  VamanaAlgorithmBase::UPointer alg_;
  IndexMeta meta_{};
  IndexMetric::Pointer metric_{};

  IndexMetric::MatrixDistance add_distance_{};
  IndexMetric::MatrixDistance search_distance_{};
  IndexMetric::MatrixBatchDistance add_batch_distance_{};
  IndexMetric::MatrixBatchDistance search_batch_distance_{};

  Stats stats_{};
  std::mutex mutex_{};

  size_t max_index_size_{0UL};
  size_t chunk_size_{VamanaEntity::kDefaultChunkSize};
  size_t docs_hard_limit_{VamanaEntity::kDefaultDocsHardLimit};
  size_t docs_soft_limit_{0UL};
  uint32_t max_degree_{VamanaEntity::kDefaultMaxDegree};
  uint32_t search_list_size_{VamanaEntity::kDefaultSearchListSize};
  uint32_t max_occlusion_size_{VamanaEntity::kDefaultMaxOcclusionSize};
  float alpha_{VamanaEntity::kDefaultAlpha};
  uint32_t ef_{VamanaEntity::kDefaultEf};
  size_t bruteforce_threshold_{VamanaEntity::kDefaultBruteForceThreshold};
  size_t max_scan_limit_{VamanaEntity::kDefaultMaxScanLimit};
  size_t min_scan_limit_{VamanaEntity::kDefaultMinScanLimit};
  float bf_negative_prob_{VamanaEntity::kDefaultBFNegativeProbability};
  float max_scan_ratio_{VamanaEntity::kDefaultScanRatio};

  uint32_t magic_{0U};
  State state_{STATE_INIT};
  bool check_crc_enabled_{false};
  bool get_vector_enabled_{false};
  bool force_padding_topk_enabled_{false};
  bool use_id_map_{true};
  bool saturate_graph_{VamanaEntity::kDefaultSaturateGraph};
  bool use_contiguous_memory_{false};

  ailego::SharedMutex shared_mutex_{};
};

}  // namespace core
}  // namespace zvec
