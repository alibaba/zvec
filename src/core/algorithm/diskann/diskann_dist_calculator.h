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
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_meta.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

class DistCalculator {
 public:
  typedef std::shared_ptr<DistCalculator> Pointer;

 public:
  //! Constructor
  DistCalculator(const DiskAnnEntity *entity,
                 const IndexMetric::Pointer &measure, uint32_t dim)
      : entity_(entity),
        distance_(measure->distance()),
        query_(nullptr),
        dim_(dim),
        compare_cnt_(0) {}

  void update(const IndexMetric::Pointer &measure, uint32_t dim) {
    distance_ = measure->distance();
    dim_ = dim;
  }

  inline void update_distance(const IndexMetric::MatrixDistance &distance) {
    distance_ = distance;
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

  inline dist_t dist(diskann_id_t id) {
    compare_cnt_++;

    const void *vec = entity_->get_vector(id);
    if (ailego_unlikely(vec == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }

    return dist(vec, query_);
  }

  inline dist_t dist(diskann_id_t lhs, diskann_id_t rhs) {
    compare_cnt_++;

    const void *vec_lhs = entity_->get_vector(lhs);
    if (ailego_unlikely(vec_lhs == nullptr)) {
      LOG_ERROR("Get nullptr vector, lhs id=%u", lhs);
      error_ = true;
      return 0.0f;
    }

    const void *vec_rhs = entity_->get_vector(rhs);
    if (ailego_unlikely(vec_rhs == nullptr)) {
      LOG_ERROR("Get nullptr vector, rhs id=%u", rhs);
      error_ = true;
      return 0.0f;
    }

    return dist(vec_lhs, vec_rhs);
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
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

 private:
  DistCalculator(const DistCalculator &) = delete;
  DistCalculator &operator=(const DistCalculator &) = delete;

 private:
  const DiskAnnEntity *entity_;

  IndexMetric::MatrixDistance distance_;
  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;  // record distance compute times
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
