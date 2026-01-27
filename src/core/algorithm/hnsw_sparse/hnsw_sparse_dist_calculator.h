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
#include "hnsw_sparse_entity.h"

namespace zvec {
namespace core {

class HnswSparseDistCalculator {
 public:
  typedef std::shared_ptr<HnswSparseDistCalculator> Pointer;

 public:
  //! Constructor
  HnswSparseDistCalculator(const HnswSparseEntity *entity,
                           const IndexMetric::Pointer &metric)
      : entity_(entity),
        distance_(metric->sparse_distance()),
        query_{nullptr},
        compare_cnt_(0) {}

  //! Constructor
  HnswSparseDistCalculator(const HnswSparseEntity *entity,
                           const IndexMetric::Pointer &metric,
                           const void *query)
      : entity_(entity),
        distance_(metric->sparse_distance()),
        query_(query),
        compare_cnt_(0) {}

  void update(const HnswSparseEntity *entity,
              const IndexMetric::Pointer &metric) {
    entity_ = entity;
    distance_ = metric->sparse_distance();
  }

  inline void update_distance(
      const IndexMetric::MatrixSparseDistance &distance) {
    distance_ = distance;
  }

  //! Reset query vector data
  inline void reset_query(const void *query) {
    error_ = false;
    query_ = query;
  }

  //! Returns distance
  inline dist_t dist(const void *sparse_data_lhs, const void *sparse_data_rhs) {
    float score{0.0f};

    if (ailego_unlikely(sparse_data_lhs == nullptr ||
                        sparse_data_rhs == nullptr)) {
      // LOG_WARN("Nullptr of sparse vector. Return dense score only");
      // error_ = true;
      return score;
    }

    distance_(sparse_data_lhs, sparse_data_rhs, &score);

    return score;
  }

  //! Returns distance between query and vec.
  inline dist_t dist(const void *vec) {
    compare_cnt_++;

    auto sparse_data = entity_->get_sparse_data_from_vector(vec);
    if (sparse_data.first == nullptr) {
      error_ = true;
      return 0.0f;
    }

    return dist(sparse_data.first, query_);
  }

  //! Return distance between query and node id.
  inline dist_t dist(node_id_t id) {
    compare_cnt_++;

    const void *feat = entity_->get_vector_meta(id);
    if (ailego_unlikely(feat == nullptr)) {
      LOG_ERROR("Get nullptr vector, id=%u", id);
      error_ = true;
      return 0.0f;
    }

    auto sparse_data = entity_->get_sparse_data_from_vector(feat);
    if (sparse_data.first == nullptr) {
      error_ = true;
      return 0.0f;
    }

    return dist(sparse_data.first, query_);
  }

  //! Return dist node lhs between node rhs
  inline dist_t dist(node_id_t lhs, node_id_t rhs) {
    compare_cnt_++;

    const void *feat = entity_->get_vector_meta(lhs);
    const void *query = entity_->get_vector_meta(rhs);
    if (ailego_unlikely(feat == nullptr || query == nullptr)) {
      LOG_ERROR("Get nullptr vector");
      error_ = true;
      return 0.0f;
    }

    auto feat_sparse_data = entity_->get_sparse_data_from_vector(feat);
    if (feat_sparse_data.first == nullptr) {
      error_ = true;
      return 0.0f;
    }

    auto query_sparse_data = entity_->get_sparse_data_from_vector(query);
    if (query_sparse_data.first == nullptr) {
      error_ = true;
      return 0.0f;
    }

    return dist(feat_sparse_data.first, query_sparse_data.first);
  }

  dist_t operator()(const void *vec) {
    return dist(vec);
  }

  dist_t operator()(id_t i) {
    return dist(i);
  }

  dist_t operator()(id_t lhs, id_t rhs) {
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

  //! Get distances compute times
  inline uint32_t compare_cnt() const {
    return compare_cnt_;
  }

 private:
  HnswSparseDistCalculator(const HnswSparseDistCalculator &) = delete;
  HnswSparseDistCalculator &operator=(const HnswSparseDistCalculator &) =
      delete;

 private:
  const HnswSparseEntity *entity_;

  IndexMetric::MatrixSparseDistance distance_;

  const void *query_;

  uint32_t compare_cnt_;  // record distance compute times
  bool error_{false};
};

}  // namespace core
}  // namespace zvec
