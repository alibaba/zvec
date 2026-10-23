// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <vector>
#include "preprocessor/fht_rotator/fht_rotator.h"
#if RABITQ_SUPPORTED
#include <cstdint>
#endif
#include "quantizer/quantizer.h"
#include "rabitq_params.h"

namespace zvec::turbo {

// Centroid-based split RaBitQ: independent 1-bit codes/factors screen graph
// neighbors; extra bits refine surviving candidates. Graph building uses raw
// vectors. The portable kernels share the dedicated index's scalar encoder.
class RabitqQuantizer final : public Quantizer {
 public:
  RabitqQuantizer() : Quantizer(QuantizeType::kRabit) {}
  int init(const IndexMeta &, const ailego::Params &) override;
  const IndexMeta &meta() const override {
    return meta_;
  }
  DataType input_data_type() const override {
    return DataType::kFp32;
  }
  int dim() const override {
    return dim_;
  }
  bool require_train() const override {
    return centroids_.empty();
  }
  using Quantizer::train;
  // Training consumes an IndexHolder; never report success for an unused
  // contiguous batch through the base class's default implementation.
  int train(const void *, size_t, size_t) override {
    return kErrUnsupported;
  }
  int train(IndexHolder::Pointer holder) override;
  bool requires_original_vectors() const override {
    return true;
  }
  size_t quantized_datapoint_vector_length() const override {
    return 9 * sizeof(float) + static_cast<size_t>(padded_dim_) * bits_ / 8;
  }
  size_t quantized_query_vector_length() const override {
#if RABITQ_SUPPORTED
    // [rotated query][bit-plane transposed 4-bit query bin (p*4/8 bytes)]
    // [delta, vl, k1xsumq, kbxsumq][g_add per cluster][g_error per cluster]
    return static_cast<size_t>(padded_dim_) * sizeof(float) +
           static_cast<size_t>(padded_dim_) * 4 / 8 +
           (4 + 2 * num_clusters_) * sizeof(float);
#else
    return (2 * static_cast<size_t>(padded_dim_) + 1 + 2 * num_clusters_) *
           sizeof(float);
#endif
  }
  IndexQueryMeta quantized_query_meta() const override {
    IndexQueryMeta result;
    result.set_meta(IndexMeta::DT_FP32, dim_, static_cast<uint32_t>(type_),
                    quantized_query_vector_length() - dim_ * sizeof(float));
    return result;
  }
  void quantize_data(const void *, void *) const override;
  void quantize_query(const void *, void *) const override;
  float calc_distance_dp_query(const void *, const void *) const override;
  bool supports_distance_refinement() const override {
    return true;
  }
  DistanceEstimate estimate_distance_dp_query(const void *,
                                              const void *) const override;
  void calc_distance_dp_query_batch(const void *const *, int, const void *,
                                    float *) const override;
  float calc_distance_dp_query_unquantized(const void *,
                                           const void *) const override;
  void calc_distance_dp_query_batch_unquantized(const void *const *, int,
                                                const void *,
                                                float *) const override;
  float calc_distance_dp_dp(const void *, const void *) const override;
  int quantize(const void *, const IndexQueryMeta &, std::string *,
               IndexQueryMeta *) const override;
  int quantize_datapoint(const void *, const IndexQueryMeta &, std::string *,
                         IndexQueryMeta *) const override;
  int dequantize(const void *, const IndexQueryMeta &,
                 std::string *) const override;
  DistanceImpl distance(const void *, const IndexQueryMeta &) const override;
  size_t scan_block_size() const override;
  int pack_scan_block(const void *, size_t, size_t, void *) const override;
  std::unique_ptr<BlockScanner> block_scanner(
      const DistanceImpl &) const override;
  bool support_score_normalization() const override {
    return metric_ == MetricType::kInnerProduct;
  }
  void normalize_score(float *score) const override {
    *score = -*score;
  }
  void denormalize_score(float *score) const override {
    *score = -*score;
  }
  int serialize(std::string *) const override;
  int deserialize(std::string &in) override {
    return deserialize(in.data(), in.size());
  }
  int deserialize(const void *, size_t) override;

 private:
  bool valid_input(const IndexQueryMeta &) const;
  float rotate(const void *, std::vector<float> *) const;
  unsigned code(const void *, int) const;
  unsigned sign(const void *, int) const;
  uint32_t nearest_centroid(const std::vector<float> &) const;
  float estimate(const void *, const void *, bool full, float *lower) const;
  IndexMeta meta_;
  int dim_{0};
  int bits_{7};
  int padded_dim_{0};
  uint32_t num_clusters_{16};
  size_t sample_count_{0};
  int niters_{20};
  std::vector<float> centroids_;  // rotated centroids, immutable after training
  MetricType metric_{MetricType::kUnknown};
  double rescale_{-1};
  // Precomputed 4-bit query rescale (ex_bits=3); deterministic, so it is
  // recomputed on every init() instead of being serialized.
  double query_rescale_{-1};
  FhtRotator::Pointer rotator_;
#if RABITQ_SUPPORTED
  // Dispatched ex-code inner product kernel, resolved once at init.
  float (*ex_ipfunc_)(const float *, const uint8_t *, size_t) = nullptr;
#endif
};

}  // namespace zvec::turbo
