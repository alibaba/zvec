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
#include "vamana_entity.h"

namespace zvec {
namespace core {

class VamanaDistCalculator {
 public:
  typedef std::shared_ptr<VamanaDistCalculator> Pointer;

  VamanaDistCalculator(const VamanaEntity *entity,
                       const IndexMetric::Pointer &metric, uint32_t dim)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {}

  VamanaDistCalculator(const VamanaEntity *entity,
                       const IndexMetric::Pointer &metric, uint32_t dim,
                       const void *query)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(query),
        dim_(dim),
        compare_cnt_(0) {}

  VamanaDistCalculator(const VamanaEntity *entity,
                       const IndexMetric::Pointer &metric)
      : entity_(entity),
        distance_(metric->distance()),
        batch_distance_(metric->batch_distance()),
        query_(nullptr),
        dim_(0),
        compare_cnt_(0) {}

  void update(const VamanaEntity *entity, const IndexMetric::Pointer &metric) {
    entity_ = entity;
    distance_ = metric->distance();
    batch_distance_ = metric->batch_distance();
  }

  void update(const VamanaEntity *entity, const IndexMetric::Pointer &metric,
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

  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
  }

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

  inline dist_t dist(const void *vec) {
    compare_cnt_++;
    return dist(vec, query_);
  }

  inline dist_t dist(node_id_t id) {
    compare_cnt_++;
    const void *feat = entity_->get_vector(id);
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    return dist(feat, query_);
  }

  inline dist_t dist(node_id_t lhs, node_id_t rhs) {
    compare_cnt_++;
    const void *feat = entity_->get_vector(lhs);
    const void *query = entity_->get_vector(rhs);
    if (ailego_unlikely(feat == nullptr || query == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }
    return dist(feat, query);
  }

  inline void batch_dist(const void **vecs, uint32_t count, float *dists) {
    compare_cnt_ += count;
    batch_distance_(vecs, query_, count, dim_, dists);
  }

  // Single-node batch distance: compute distance between query and a stored
  // node using batch_distance_. Consistent with HnswDistCalculator::batch_dist.
  inline dist_t batch_dist(node_id_t id) {
    compare_cnt_++;
    const void *feat = entity_->get_vector(id);
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }
    dist_t score = 0;
    batch_distance_(&feat, query_, 1, dim_, &score);
    return score;
  }

  // Batch distance computation between a base vector and multiple target
  // vectors. Does NOT use query_ and does NOT increment compare_cnt. Used for
  // inter-candidate distance computation in robust_prune.
  //
  // Uses the single distance function (distance_) in a loop rather than
  // batch_distance_, because batch_distance_ (turbo AVX512-VNNI) expects
  // the second argument to be a preprocessed uint8 query (+128 shift),
  // while base_vec here is a raw int8 stored vector. The single distance
  // function (AVX2 sign/abs trick) correctly handles two raw int8 inputs.
  inline void batch_dist_pair(const void *base_vec, const void **vecs,
                              uint32_t count, float *dists) {
    for (uint32_t i = 0; i < count; ++i) {
      distance_(base_vec, vecs[i], dim_, &dists[i]);
    }
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
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }
  inline uint32_t dimension() const {
    return dim_;
  }

 private:
  VamanaDistCalculator(const VamanaDistCalculator &) = delete;
  VamanaDistCalculator &operator=(const VamanaDistCalculator &) = delete;

  const VamanaEntity *entity_;
  IndexMetric::MatrixDistance distance_;
  IndexMetric::MatrixBatchDistance batch_distance_;
  const void *query_;
  uint32_t dim_;
  uint32_t compare_cnt_;
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
