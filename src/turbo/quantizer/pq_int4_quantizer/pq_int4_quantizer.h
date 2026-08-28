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
#include <limits>
#include <vector>
#include <zvec/ailego/utility/float_helper.h>
#include <zvec/core/framework/index_holder.h>
#include <zvec/core/framework/index_meta.h>
// Rooted at src/ so this header stays includable from core (ivf_entity).
#include <turbo/quantizer/common/pq_quantizer/precompute_table_quantizer.h>
#include <turbo/quantizer/quantizer.h>

namespace zvec {
namespace turbo {

using namespace zvec::core;

//! Product Quantizer with 4-bit sub-codes (num_bits=4, 16 centroids).
//!
//! Datapoints are encoded as nibble-packed codes: two 4-bit sub-codes share
//! one byte (sub-quantizer m occupies the low nibble of byte m/2 when m is
//! even, the high nibble when m is odd).  An odd num_chunk pads the last
//! byte's high nibble with 0.  Total code length is ceil(num_chunk / 2) bytes.
//!
//! Queries are encoded as a float LUT of size [num_chunk * 16] via
//! quantize_query().  Distance between a PQ code and a query uses ADC (LUT
//! look-up); distance between two PQ codes uses SDC (centroid-to-centroid
//! distance table).
class PqInt4Quantizer : public Quantizer, public PrecomputeTableQuantizer {
 public:
  PqInt4Quantizer() : Quantizer(QuantizeType::kPQ) {}

  ~PqInt4Quantizer() override = default;

  int init(const IndexMeta &meta, const ailego::Params &params) override;

  const IndexMeta &meta() const override {
    return meta_;
  }

  DataType input_data_type() const override {
    return input_data_type_;
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
    return packed_code_length() + extra_meta_size_;
  }

  size_t quantized_query_vector_length() const override {
    return static_cast<size_t>(num_chunk_) * kNumCentroids * sizeof(float);
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

  int quantize(const void *query, const IndexQueryMeta &qmeta, std::string *out,
               IndexQueryMeta *ometa) const override;

  int dequantize(const void *in, const IndexQueryMeta &qmeta,
                 std::string *out) const override;

  DistanceImpl distance(const void *query,
                        const IndexQueryMeta &qmeta) const override;

  DistanceImpl sym_distance(const void *query,
                            const IndexQueryMeta &qmeta) const;

  //! Precomputed residual distance table support (see
  //! PrecomputeTableQuantizer).  The fused per-list scan table keeps the
  //! float LUT shape [num_chunk * kNumCentroids] of quantize_query().
  int build_centroid_distance_table(const void *centroids, size_t centroid_num,
                                    std::string *table) const override;

  int quantize_precomputed_query(const void *query, const IndexQueryMeta &qmeta,
                                 std::string *out,
                                 IndexQueryMeta *ometa) const override;

  int merge_query_distance_table(const void *query_table,
                                 const std::string &centroid_table,
                                 size_t centroid_id,
                                 std::string *out) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

 private:
  //! Train a single sub-quantizer (KMeans, k=16) on the sub-vectors.
  //! Templated on the data type T (float or ailego::Float16) so that
  //! NumericalKmeans<T> operates natively in the input precision.
  //! sub_idx selects which sub-quantizer to train.
  template <typename T>
  void train_subquantizer(const T *data, size_t num, size_t stride,
                          size_t sub_idx);

  //! L2-normalize a batch of vectors in-place (train-time use).
  template <typename T>
  void normalize(T *data, size_t num) const;

  //! Compute the per-dimension mean (accumulated in float to avoid FP16
  //! overflow) and subtract it from all training vectors in-place.
  template <typename T>
  void compute_and_subtract_center(T *data, size_t num);

  //! L2-normalize a single vector in-place; optionally writes the norm out.
  template <typename T>
  void normalize(T *vec, float *norm_out = nullptr) const;

  //! Subtract the pre-computed centroid_ from a single vector.
  template <typename T>
  void subtract_center(T *vec) const;

  //! Compute the centroid-to-centroid distance table for SDC.
  void compute_dist_table();

  //! Compute ||c_m[j]||^2 for every sub-centroid, consumed by
  //! build_centroid_distance_table().  Built in train() and deserialize().
  void compute_sub_centroid_norms();

  //! Build centroid_ptrs_cache_ from current centroids_.
  //! Called after train() and deserialize() when centroids are available.
  void build_centroid_ptrs_cache();

  //! Re-dispatch kernels and batch distance functions (init/deserialize).
  //! Yields kErrUnsupported if any of them is unavailable for the configured
  //! metric / data type.  Sets extra_meta_size_ but leaves pushing it into
  //! meta_ to the caller.
  int setup_functions();

  //! Byte size of one element in the original data type.
  uint32_t element_size() const {
    return (input_data_type_ == DataType::kFp16)
               ? static_cast<uint32_t>(sizeof(ailego::Float16))
               : static_cast<uint32_t>(sizeof(float));
  }

  //! Packed code length in bytes: two 4-bit codes per byte, last byte padded.
  size_t packed_code_length() const {
    return (static_cast<size_t>(num_chunk_) + 1) / 2;
  }

  static constexpr uint32_t kNumCentroids = 16;
  static constexpr uint32_t kMaxKmeansIters = 25;
  static constexpr size_t kMaxTrainVectors = 65536;
  static constexpr uint32_t kExtraMetaSizeCosine = sizeof(float);

  //! Actual input data type (kFp32 or kFp16).
  DataType input_data_type_{DataType::kFp32};

  //! Thread count for KMeans training (0 = hardware_concurrency).
  //! Read from params in init(), aligned with multi_chunk_cluster.
  uint32_t thread_count_{0};

  //! KMC2 Markov chain length (aligned with multi_chunk_cluster default 32).
  uint32_t markov_chain_length_{32};

  //! Cost-based convergence threshold (aligned with multi_chunk_cluster).
  double epsilon_{std::numeric_limits<float>::epsilon()};

  //! Whether to apply zero-mean centering before training/encoding.
  //! When enabled, the per-dimension mean of training data is subtracted
  //! from all vectors (train, encode, query) and added back on dequantize.
  bool use_zero_mean_{false};

  //! Set by a successful init().  deserialize() requires it: the metric policy
  //! comes from meta_, and a default-constructed IndexMeta silently reports
  //! "SquaredEuclidean", so its value cannot tell initialized from fresh.
  bool initialized_{false};

  IndexMeta meta_{};
  uint32_t original_dim_{0};
  uint32_t num_chunk_{0};
  uint32_t sub_dim_{0};

  //! Centroids stored as raw bytes in the original data type:
  //! [num_chunk * kNumCentroids * sub_dim * sizeof(T)]
  //! T = float for kFp32, ailego::Float16 for kFp16.
  std::vector<uint8_t> centroids_;

  //! Global centroid (per-dimension mean) for zero-mean centering.
  //! Size: original_dim_ floats.  Only populated when use_zero_mean_ = true.
  std::vector<float> centroid_;

  //! Centroid-to-centroid distance table for SDC:
  //! [num_chunk * kNumCentroids * kNumCentroids]
  std::vector<float> dist_table_;

  //! Squared norms of the sub-centroids: [num_chunk * kNumCentroids].
  //! Used by build_centroid_distance_table().
  std::vector<float> sub_centroid_norms_;

  //! Pre-built centroid pointer arrays for each sub-quantizer.
  //! Layout: centroid_ptrs_cache_[sub_idx][centroid_idx] = pointer to centroid.
  //! Built once during init/deserialize, reused by compute_dist_table
  //! and quantize_query to avoid repeated allocations.
  std::vector<std::vector<const void *>> centroid_ptrs_cache_;

  //! ISA-dispatched kernel function pointers (ADC / SDC / Batch ADC).
  CodebookAsymmetricDistanceFunc adc_fn_{nullptr};
  CodebookSymmetricDistanceFunc sdc_fn_{nullptr};
  CodebookBatchAsymmetricDistanceFunc batch_adc_fn_{nullptr};

  //! Metric-aware batch distance function for search-side LUT
  //! computation and SDC dist_table.  Data type matches input_data_type_.
  //! Obtained from get_batch_distance_func() with the configured metric.
  BatchDistanceFunc batch_fn_{};

  //! L2-only batch distance function for encoding (quantize_data).
  //! Data type matches input_data_type_.  PQ encoding always minimizes L2
  //! quantization error regardless of the search metric.
  BatchDistanceFunc l2_batch_fn_{};

  //! Inner-product batch distance function for the precomputed residual
  //! tables (build_centroid_distance_table / quantize_precomputed_query).
  //! The table terms are pure inner products against residual centroids
  //! regardless of the search metric, so this is always assembled for
  //! MetricType::kInnerProduct.  Independent of the configured metric;
  //! returns -<a, b> per element.
  BatchDistanceFunc ip_batch_fn_{};
};

}  // namespace turbo
}  // namespace zvec
