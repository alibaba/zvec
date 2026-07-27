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
#include <vector>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_meta.h>
#include "quantizer/quantizer.h"
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

  //! Bind the PQ state used by quantize_pq_query() / pq_dist().  Called by
  //! the searcher before each search.
  inline void bind_pq(const turbo::Quantizer *quantizer, const uint8_t *codes,
                      uint32_t chunk_num) {
    pq_quantizer_ = quantizer;
    pq_codes_ = codes;
    pq_chunk_num_ = chunk_num;
  }

  //! Build the ADC LUT for the rotated query and bind it into a DistanceImpl
  //! functor (owns the quantized query bytes and the dispatched ADC kernels).
  //! PqInt8Quantizer::quantize_query expects an FP32 query; DiskAnn may store
  //! FP16 vectors, so convert the rotated query into FP32 when needed.
  void quantize_pq_query(const void *query_rotated, IndexMeta::DataType dtype) {
    const void *pq_query = query_rotated;
    if (dtype == IndexMeta::DataType::DT_FP16) {
      uint32_t pq_dim = static_cast<uint32_t>(pq_quantizer_->dim());
      pq_query_scratch_.resize(pq_dim);
      const auto *s = static_cast<const ailego::Float16 *>(query_rotated);
      for (uint32_t d = 0; d < pq_dim; ++d) {
        pq_query_scratch_[d] = static_cast<float>(s[d]);
      }
      pq_query = pq_query_scratch_.data();
    }
    pq_lut_scratch_.resize(pq_quantizer_->quantized_query_vector_length() /
                           sizeof(float));
    pq_quantizer_->quantize_query(pq_query, pq_lut_scratch_.data());
    pq_dist_impl_ =
        pq_quantizer_->distance(pq_lut_scratch_.data(), IndexQueryMeta());
  }

  //! Batched PQ ADC distances between the quantized query and the codes of
  //! the given ids, via the DistanceImpl functor bound in quantize_pq_query().
  void pq_dist(const diskann_id_t *ids, uint32_t n, float *dists) {
    pq_dp_list_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      pq_dp_list_[i] = pq_codes_ + static_cast<size_t>(ids[i]) * pq_chunk_num_;
    }
    pq_dist_impl_.batch(pq_dp_list_.data(), n, dists);
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

  //! PQ state bound by the searcher (see bind_pq).
  const turbo::Quantizer *pq_quantizer_{nullptr};
  const uint8_t *pq_codes_{nullptr};
  uint32_t pq_chunk_num_{0};

  //! Per-query ADC distance functor (see quantize_pq_query).
  turbo::DistanceImpl pq_dist_impl_{};

  //! Reused scratch buffers for pq_dist() / quantize_pq_query().
  std::vector<const void *> pq_dp_list_;
  std::vector<float> pq_query_scratch_;
  std::vector<float> pq_lut_scratch_;
};

}  // namespace core
}  // namespace zvec
