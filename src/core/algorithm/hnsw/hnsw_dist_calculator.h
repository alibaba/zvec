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

#include <zvec/core/framework/index_meta.h>
#include <zvec/core/framework/index_metric.h>
#include <zvec/core/framework/index_provider.h>
#include "hnsw_entity.h"

namespace zvec {
namespace core {

class HnswDistCalculator {
 public:
  typedef std::shared_ptr<HnswDistCalculator> Pointer;

 public:
  enum DistType {
    DIST_NONE = 0,
    DIST_DENSE = 1,
    DIST_HYBRID = 2,
    DIST_SPARSE = 3
  };

 public:
  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric, uint32_t dim)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {}

  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric, uint32_t dim,
                     const void *query)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(query),
        dim_(dim),
        compare_cnt_(0) {}

  //! Constructor
  HnswDistCalculator(const HnswEntity *entity,
                     const IndexMetric::Pointer &metric)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {}

  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric) {
    entity_ = entity;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
  }

  void update(const HnswEntity *entity, const IndexMetric::Pointer &metric,
              uint32_t dim) {
    entity_ = entity;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
    dim_ = dim;
  }

  inline void update_distance(
      const IndexMetric::MatrixDistance &distance,
      const IndexMetric::MatrixBatchDistance &batch_distance) {
    distance_ = distance;
    batch_distance_ = batch_distance;
  }

  //! Update the dimension used by distance computation
  inline void set_dim(uint32_t dim) {
    dim_ = dim;
  }

  //! Reset query vector data
  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
  }

  //! Returns distance
  inline dist_t dist(const void *vec_lhs, const void *vec_rhs) {
    if (ailego_unlikely(vec_lhs == nullptr || vec_rhs == nullptr)) {
      LOG_ERROR("Nullptr of dense vector");
      error_ = true;
      return 0.0f;
    }

    float score{0.0f};

    distance_(vec_lhs, vec_rhs, dim_, &score);

    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;

    return dist(vec, query_);
  }

  //! Return distance between query and node id.
  inline dist_t dist(node_id_t id) {
    compare_cnt_++;
    IndexStorage::MemoryBlock vec_block;
    int ret = get_vector(id, vec_block);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block.data();
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }

    return dist(feat, query_);
  }

  //! Return dist node lhs between node rhs
  inline dist_t dist(node_id_t lhs, node_id_t rhs) {
    compare_cnt_++;


    IndexStorage::MemoryBlock vec_block_feat;
    int ret = get_vector(lhs, vec_block_feat);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", lhs);
      error_ = true;
      return 0.0f;
    }
    const void *feat = vec_block_feat.data();

    IndexStorage::MemoryBlock vec_block_query;
    ret = get_vector(rhs, vec_block_query);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Get nullptr vector, id=%u", rhs);
      error_ = true;
      return 0.0f;
    }
    const void *query = vec_block_query.data();
    if (ailego_unlikely(feat == nullptr || query == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }

    return dist(feat, query);
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
  }

  dist_t operator()(node_id_t i) {
    return dist(i);
  }

  dist_t operator()(node_id_t lhs, node_id_t rhs) {
    return dist(lhs, rhs);
  }

  void batch_dist(const void **vecs, size_t num, dist_t *distances,
                  const void **extra_values) {
    compare_cnt_++;

    batch_distance_(vecs, query_, num, dim_, distances, extra_values);
  }

  inline dist_t batch_dist(const void *feat, const void *extra_values) {
    compare_cnt_++;

    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }
    dist_t score = 0;
    if (extra_values != nullptr) {
      batch_distance_(&feat, query_, 1, dim_, &score, &extra_values);
    } else {
      batch_distance_(&feat, query_, 1, dim_, &score, nullptr);
    }

    return score;
  }

  inline void clear() {
    compare_cnt_ = 0;
    error_ = false;
  }

  inline void clear_compare_cnt() {
    compare_cnt_ = 0;
  }

  inline bool error() const {
    return error_;
  }

  //! Get distances compute times
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }

  inline uint32_t dimension() const {
    return dim_;
  }

  //! Bind a provider which supplies the original vectors, so vector
  //! fetches by node id go through it instead of the entity
  void set_provider(IndexProvider::Pointer provider) {
    provider_ = std::move(provider);
  }

  inline bool has_provider() const {
    return provider_ != nullptr;
  }

  //! Get a vector by node id, from the provider when set
  int get_vector(node_id_t id, IndexStorage::MemoryBlock &block) const {
    if (provider_) {
      key_t key = entity_->get_key(id);
      if (ailego_unlikely(key == kInvalidKey)) {
        return IndexError_NoExist;
      }
      return provider_->get_vector(key, block);
    }
    return entity_->get_vector(id, block);
  }

  //! Batch get vectors by node ids
  int get_vector(const node_id_t *ids, uint32_t count,
                 std::vector<IndexStorage::MemoryBlock> &vec_blocks) const {
    vec_blocks.reserve(vec_blocks.size() + count);
    for (uint32_t i = 0; i < count; ++i) {
      IndexStorage::MemoryBlock block;
      int ret = get_vector(ids[i], block);
      if (ailego_unlikely(ret != 0)) {
        return ret;
      }
      vec_blocks.push_back(std::move(block));
    }
    return 0;
  }

 private:
  HnswDistCalculator(const HnswDistCalculator &) = delete;
  HnswDistCalculator &operator=(const HnswDistCalculator &) = delete;

 private:
  const HnswEntity *entity_;

  IndexMetric::MatrixDistance distance_;
  IndexMetric::MatrixBatchDistance batch_distance_;

  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;  // record distance compute times
  // uint32_t compare_cnt_batch_;  // record batch distance compute time
  bool error_{false};

  // get original vector, used to build graph
  IndexProvider::Pointer provider_{};
};

}  // namespace core
}  // namespace zvec
