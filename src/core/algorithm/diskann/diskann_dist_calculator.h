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
#include <turbo/quantizer/quantizer.h>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_context.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_meta.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

class DistCalculator {
 public:
  typedef std::shared_ptr<DistCalculator> Pointer;

 public:
  //! Constructor
  DistCalculator(const DiskAnnEntity *entity, const IndexMeta &meta,
                 const IndexMetric::Pointer &measure)
      : entity_(entity),
        query_(nullptr),
        dim_(meta.dimension()),
        compare_cnt_(0) {
    bind_distance(meta, measure);
  }

  void update(const IndexMeta &meta, const IndexMetric::Pointer &measure) {
    bind_distance(meta, measure);
    dim_ = meta.dimension();
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

  inline void bind_quantizer(const turbo::Quantizer *quantizer,
                             const uint8_t *codes, uint32_t code_size) {
    quantizer_ = quantizer;
    quant_codes_ = codes;
    quant_code_size_ = code_size;
  }

  void quantize_query(const void *query_rotated) {
    quant_query_scratch_.resize(quantizer_->quantized_query_vector_length());
    quantizer_->quantize_query(query_rotated, quant_query_scratch_.data());
    quant_dist_impl_ =
        quantizer_->distance(quant_query_scratch_.data(), IndexQueryMeta());
  }

  void quantized_dist(const diskann_id_t *ids, uint32_t n, float *dists) {
    quant_dp_list_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
      quant_dp_list_[i] =
          quant_codes_ + static_cast<size_t>(ids[i]) * quant_code_size_;
    }
    quant_dist_impl_.batch(quant_dp_list_.data(), n, dists);
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

  void bind_distance(const IndexMeta &meta,
                     const IndexMetric::Pointer &measure) {
    data_quantizer_.reset();

    const char *name = nullptr;
    // FP32 stays on the SIMD-optimized ailego metric: the turbo FP32
    // kernels are scalar-only and much slower for graph build / rerank.
    if (meta.data_type() == IndexMeta::DataType::DT_FP16) {
      name = "Fp16Quantizer";
    }
    if (name != nullptr) {
      turbo::Quantizer::Pointer quantizer = IndexFactory::CreateQuantizer(name);
      if (quantizer) {
        IndexMeta quant_meta = meta;
        if (meta.metric_name() == "Cosine") {
          quant_meta.set_dimension(meta.dimension() -
                                   sizeof(float) / meta.unit_size());
        }
        if (quantizer->init(quant_meta, quant_meta.metric_params()) == 0) {
          turbo::DistanceImpl impl = quantizer->distance("", IndexQueryMeta());
          if (impl.valid()) {
            data_quantizer_ = std::move(quantizer);
            distance_ = [func = impl.func(), quant_dim = static_cast<size_t>(
                                                 data_quantizer_->dim())](
                            const void *m, const void *q, size_t /*dim*/,
                            float *out) { func(m, q, quant_dim, out); };
            return;
          }
        }
      }
    }
    distance_ = measure->distance();
  }

 private:
  const DiskAnnEntity *entity_;

  turbo::Quantizer::Pointer data_quantizer_{};

  IndexMetric::MatrixDistance distance_;
  const void *query_;
  uint32_t dim_;

  uint32_t compare_cnt_;
  bool error_{false};

  const turbo::Quantizer *quantizer_{nullptr};
  const uint8_t *quant_codes_{nullptr};
  uint32_t quant_code_size_{0};

  turbo::DistanceImpl quant_dist_impl_{};

  std::vector<const void *> quant_dp_list_;
  std::vector<uint8_t> quant_query_scratch_;
};

}  // namespace core
}  // namespace zvec
