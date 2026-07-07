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

//! Product Quantizer with 4-bit sub-codes (num_bits=4, 16 centroids).
//!
//! Datapoints are encoded as packed nibbles: two 4-bit indices per byte.
//!   byte[i] = (code[2*i+1] << 4) | code[2*i]
//! num_subquantizers MUST be even for clean byte packing.
//!
//! Queries are encoded as a float LUT of size [num_subquantizers * 16]
//! via quantize_query().  Distance between a PQ code and a query uses
//! ADC (LUT look-up); distance between two PQ codes uses SDC
//! (centroid-to-centroid distance table of size [nsq * 16 * 16]).
class PqInt4Quantizer : public Quantizer {
 public:
  PqInt4Quantizer() {
    type_ = QuantizeType::kPQ;  // shares kPQ with int8; DataType distinguishes
  }

  ~PqInt4Quantizer() override = default;

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

  int train(IndexHolder::Pointer holder, int thread_count) override;

  // Packed nibbles: num_subquantizers / 2 bytes + optional Cosine norm.
  size_t quantized_datapoint_vector_length() const override {
    return static_cast<size_t>(num_subquantizers_) / 2 + extra_meta_size_;
  }

  // LUT: [num_subquantizers * 16] floats.
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

  int dequantize(const void *in, const IndexQueryMeta &qmeta,
                 std::string *out) const override;

  DistanceImpl distance(const void *query,
                        const IndexQueryMeta &qmeta) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

 private:
  //! Train a single sub-quantizer (KMeans, k=16) on the sub-vectors.
  void train_subquantizer(const float *data, size_t num, size_t stride,
                          size_t sub_idx);

  //! Compute the centroid-to-centroid distance table for SDC.
  void compute_dist_table();

  //! Build centroid_ptrs_cache_ from current centroids_.
  void build_centroid_ptrs_cache();

  static constexpr uint32_t kNumCentroids = 16;
  static constexpr uint32_t kMaxKmeansIters = 25;
  static constexpr size_t kMaxTrainVectors = 65535;
  static constexpr uint32_t kExtraMetaSizeCosine = sizeof(float);

  IndexMeta meta_{};
  uint32_t original_dim_{0};
  uint32_t num_subquantizers_{0};
  uint32_t sub_dim_{0};

  //! Centroids: [num_subquantizers * kNumCentroids * sub_dim]
  std::vector<float> centroids_;

  //! Centroid-to-centroid distance table for SDC:
  //! [num_subquantizers * kNumCentroids * kNumCentroids]
  std::vector<float> dist_table_;

  //! Pre-built centroid pointer arrays for each sub-quantizer.
  std::vector<std::vector<const void *>> centroid_ptrs_cache_;

  //! ISA-dispatched kernel function pointers (ADC / SDC / Batch ADC).
  //! These point to int4-specific kernels (packed nibble codes).
  PqAdcDistanceFunc adc_fn_{nullptr};
  PqSdcKernelFunc sdc_fn_{nullptr};
  PqBatchAdcFunc batch_adc_fn_{nullptr};

  //! Metric-aware fp32 batch distance for search-side LUT and SDC table.
  BatchDistanceFunc fp32_batch_fn_{};

  //! L2-only fp32 batch distance for encoding and KMeans training.
  BatchDistanceFunc fp32_l2_batch_fn_{};
};

}  // namespace turbo
}  // namespace zvec
