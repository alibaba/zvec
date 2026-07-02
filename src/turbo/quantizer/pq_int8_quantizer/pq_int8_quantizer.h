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

#include <cstdint>
#include <vector>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
#include "quantizer/quantizer.h"

namespace zvec {
namespace turbo {

using namespace zvec::core;

//! Product Quantizer with 8-bit sub-codes (num_bits=8, 256 centroids).
//!
//! Datapoints are encoded as uint8_t[num_subquantizers] codes.
//! Queries are encoded as a float LUT of size [num_subquantizers * 256]
//! via compute_distance_table().  Distance between a PQ code and a
//! query uses ADC (LUT look-up); distance between two PQ codes uses
//! SDC (centroid-to-centroid distance table).
class PqInt8Quantizer : public Quantizer {
 public:
  PqInt8Quantizer() {
    type_ = QuantizeType::kPQ;
  }

  ~PqInt8Quantizer() override = default;

  int init(const IndexMeta &meta, const ailego::Params &params) override;

  const IndexMeta &meta() const override {
    return meta_;
  }

  DataType input_data_type() const override {
    return DataType::kFp32;
  }

  QuantizeType type() const override {
    return type_;
  }

  int dim() const override {
    return static_cast<int>(original_dim_);
  }

  bool require_train() const override {
    return true;
  }

  int train(IndexHolder::Pointer holder) override;

  size_t quantized_datapoint_vector_length() const override {
    return num_subquantizers_;
  }

  size_t quantized_query_vector_length() const override {
    return static_cast<size_t>(num_subquantizers_) * kNumCentroids *
           sizeof(float);
  }

  void quantize_data(const void *input, void *output) const override;

  void quantize_query(const void *input, void *output) const override;

  float calc_distance_dp_query(const void *dp,
                               const void *query) const override;

  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override;

  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *query) const override;

  void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const override;

  float calc_distance_dp_dp(const void *dp1, const void *dp2) const override;

  int quantize(const void *query, const IndexQueryMeta &qmeta,
               std::string *out, IndexQueryMeta *ometa) const override;

  DistanceImpl distance(const void *query,
                        const IndexQueryMeta &qmeta) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

 private:
  //! Train a single sub-quantizer (KMeans, k=256) on the sub-vectors
  //! extracted from holder.  sub_idx selects which sub-quantizer to train.
  void train_subquantizer(const float *data, size_t num, size_t stride,
                          size_t sub_idx);

  //! Compute the centroid-to-centroid distance table for SDC.
  void compute_dist_table();

  //! Find the nearest centroid for a single sub-vector.
  size_t find_nearest_centroid(const float *sub_vec, size_t sub_idx) const;

  static constexpr uint32_t kNumCentroids = 256;
  static constexpr uint32_t kMaxKmeansIters = 25;

  IndexMeta meta_{};
  uint32_t original_dim_{0};
  uint32_t num_subquantizers_{0};
  uint32_t sub_dim_{0};

  //! Centroids: [num_subquantizers * kNumCentroids * sub_dim]
  std::vector<float> centroids_;

  //! Centroid-to-centroid distance table for SDC:
  //! [num_subquantizers * kNumCentroids * kNumCentroids]
  std::vector<float> dist_table_;

  //! ISA-dispatched kernel function pointers (ADC / SDC).
  PqAdcDistanceFunc adc_fn_{nullptr};
  PqSdcKernelFunc sdc_fn_{nullptr};

  //! Reused fp32 batch distance function for LUT computation and SDC
  //! dist_table precomputation.  Obtained from get_batch_distance_func()
  //! which is metric-aware and already SIMD-optimized.
  BatchDistanceFunc fp32_batch_fn_{};
};

}  // namespace turbo
}  // namespace zvec
