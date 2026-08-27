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
#include <turbo/quantizer/common/pq_quantizer/packed_code_quantizer.h>
#include <turbo/quantizer/common/pq_quantizer/precompute_table_quantizer.h>
#include <turbo/quantizer/quantizer.h>

namespace zvec {
namespace turbo {

using namespace zvec::core;

//! FastScan Product Quantizer (num_bits=4, 16 centroids per sub-quantizer).
//!
//! Codes are nibble-packed exactly like PqInt4Quantizer, but MUST be stored in
//! packed 32-vector blocks (PackedCodeQuantizer capability) so the FastScan
//! kernel can look up 32 codes per sub-space with one SIMD byte shuffle.
//!
//! Queries are a uint8 affine-quantized LUT (single min/max over the whole
//! [num_chunk * 16] float table), packed and followed by a dequantization tail:
//!   [packed u8 LUT | float delta = (hi - lo) / 255 | float bias = M * lo]
//! with dist = accu * delta + bias.
//!
//! calc_distance_packed_block() is the only read path: single-code ADC, SDC and
//! reconstruction are unavailable (distance() / sym_distance() return empty
//! handles, dequantize() fails), so this quantizer must not be used for HNSW
//! graph construction, and callers needing the vector back must read it from
//! the unquantized side.  Metrics: SquaredEuclidean, InnerProduct and Cosine
//! (= normalize + L2, with the original vector norm stored after each code).
class PqFastQuantizer : public Quantizer,
                        public PackedCodeQuantizer,
                        public PrecomputeTableQuantizer {
 public:
  PqFastQuantizer() : Quantizer(QuantizeType::kPQFast) {}

  ~PqFastQuantizer() override = default;

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

  int train(IndexHolder::Pointer holder, int thread_count) override;

  size_t quantized_datapoint_vector_length() const override {
    return packed_code_length() + extra_meta_size_;
  }

  size_t quantized_query_vector_length() const override;

  void quantize_data(const void *input, void *output) const override;

  void quantize_query(const void *input, void *output) const override;

  //! Pack up to 32 plain nibble codes (as returned by quantize_data()) into
  //! one interleaved block consumable by calc_distance_packed_block().
  int pack_codes(const void *codes, size_t num, size_t stride,
                 void *out) const override;

  //! Unsupported, see the class comment: asserts and yields NaN.
  float calc_distance_dp_query(const void *dp,
                               const void *query) const override;

  //! Unsupported, see the class comment: asserts and yields NaN.
  void calc_distance_dp_query_batch(const void *const *dp_list, int dp_num,
                                    const void *query,
                                    float *dist_list) const override;

  //! PackedCodeQuantizer capability: native block scan over packed 32-vector
  //! blocks (see pack_codes).  This is the read-side counterpart of packing
  //! and the only path that runs the SIMD FastScan kernel.
  void calc_distance_packed_block(const void *block, size_t num,
                                  const void *query,
                                  float *dist_list) const override;

  //! Unsupported, see the class comment: asserts and yields NaN.
  float calc_distance_dp_query_unquantized(const void *dp,
                                           const void *query) const override;

  //! Unsupported, see the class comment: asserts and yields NaN.
  void calc_distance_dp_query_batch_unquantized(
      const void *const *dp_list, int dp_num, const void *query,
      float *dist_list) const override;

  //! Unsupported, see the class comment: asserts and yields NaN.
  float calc_distance_dp_dp(const void *dp1, const void *dp2) const override;

  DistanceImpl distance(const void *query,
                        const IndexQueryMeta &qmeta) const override;

  DistanceImpl sym_distance(const void *query,
                            const IndexQueryMeta &qmeta) const;

  //! Precomputed residual distance table support (see
  //! PrecomputeTableQuantizer).  The merged per-list LUT is affine-quantized
  //! internally, so the output keeps the packed-u8 FastScan query format
  //! consumed by the block scan.  fp32 + plain L2 only: anything else
  //! returns kErrUnsupported and IVF keeps the per-list path.
  int build_centroid_distance_table(const void *centroids, size_t centroid_num,
                                    std::string *table) const override;

  int quantize_precomputed_query(const void *query, const IndexQueryMeta &qmeta,
                                 std::string *out,
                                 IndexQueryMeta *ometa) const override;

  int merge_query_distance_table(const void *query_table,
                                 const std::string &centroid_table,
                                 size_t centroid_id,
                                 std::string *out) const override;

  int quantize(const void *query, const IndexQueryMeta &qmeta, std::string *out,
               IndexQueryMeta *ometa) const override;

  //! Unsupported, see the class comment: asserts and yields kErrUnsupported.
  int dequantize(const void *in, const IndexQueryMeta &qmeta,
                 std::string *out) const override;

  int serialize(std::string *out) const override;

  int deserialize(std::string &in) override;

  int deserialize(const void *data, size_t len) override;

 private:
  //! Train a single sub-quantizer (KMeans, k=16) on the sub-vectors.
  template <typename T>
  void train_subquantizer(const T *data, size_t num, size_t stride,
                          size_t sub_idx);

  //! Compute the per-dimension mean (accumulated in float to avoid FP16
  //! overflow) and subtract it from all training vectors in-place.
  template <typename T>
  void compute_and_subtract_center(T *data, size_t num);

  //! Subtract the pre-computed centroid_ from a single vector.
  template <typename T>
  void subtract_center(T *vec) const;

  //! Normalize a batch of vectors in-place (L2).
  template <typename T>
  void normalize_batch(T *data, size_t num) const;

  //! Normalize one vector in-place; optionally report the original norm.
  template <typename T>
  void normalize_single(T *vec, float *norm_out = nullptr) const;

  //! Compute the float ADC LUT [num_chunk * 16] for a raw query
  //! (zero-mean centering applied when enabled).
  void compute_float_lut(const void *input, float *lut) const;

  //! Codebook helpers backing the precomputed residual table; kept private
  //! so IVF never handles float ADC tables directly.

  //! Preprocess a raw query into the codebook space (Cosine normalization
  //! then zero-mean centering), writing dim() floats into out.
  int preprocess_query(const void *input, float *out) const;

  //! Total number of float LUT entries (num_chunk * kNumCentroids).
  size_t lut_entry_count() const;

  //! Affine-quantize a float LUT into the packed-u8 FastScan query layout
  //! ([packed u8 LUT | float delta | float bias]).
  int quantize_lut(const float *lut, void *out) const;

  //! Compute ||c_m[j]||^2 for every sub-centroid, consumed by
  //! build_centroid_distance_table().  Built in train() and deserialize().
  void compute_sub_centroid_norms();

  //! Build centroid_ptrs_cache_ from current centroids_.
  void build_centroid_ptrs_cache();

  //! Re-dispatch kernels and batch distance functions (init/deserialize).
  //! Yields kErrUnsupported if any of them is unavailable for the configured
  //! metric / data type.
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
  uint32_t thread_count_{0};

  //! KMC2 Markov chain length (aligned with multi_chunk_cluster default 32).
  uint32_t markov_chain_length_{32};

  //! Cost-based convergence threshold (aligned with multi_chunk_cluster).
  double epsilon_{std::numeric_limits<float>::epsilon()};

  //! Whether to apply zero-mean centering before training/encoding.
  bool use_zero_mean_{false};

  //! Set by a successful init().  deserialize() requires it: the metric policy
  //! comes from meta_, and a default-constructed IndexMeta silently reports
  //! "SquaredEuclidean", so its value cannot tell initialized from fresh.
  bool initialized_{false};

  //! Extra bytes appended to each code (Cosine: original vector norm for
  //! dequantize).
  uint32_t extra_meta_size_{0};

  IndexMeta meta_{};
  uint32_t original_dim_{0};
  uint32_t num_chunk_{0};
  uint32_t sub_dim_{0};

  //! Centroids stored as raw bytes in the original data type:
  //! [num_chunk * kNumCentroids * sub_dim * sizeof(T)]
  std::vector<uint8_t> centroids_;

  //! Global centroid (per-dimension mean) for zero-mean centering.
  std::vector<float> centroid_;

  //! Pre-built centroid pointer arrays for each sub-quantizer.
  std::vector<std::vector<const void *>> centroid_ptrs_cache_;

  //! ISA-dispatched FastScan scan32 kernel.
  CodebookFastScanFunc scan_fn_{nullptr};

  //! Metric-aware batch distance function for the search-side LUT
  //! (L2: squared euclidean, IP: -dot).  Data type matches input.
  BatchDistanceFunc batch_fn_{};

  //! L2-only batch distance function for encoding (quantize_data).
  BatchDistanceFunc l2_batch_fn_{};

  //! Inner-product batch distance function for the precomputed residual
  //! tables.  The table terms are pure inner products regardless of the
  //! search metric; the kernel returns -<a, b> per element.
  BatchDistanceFunc ip_batch_fn_{};

  //! Squared norms of the sub-centroids: [num_chunk * kNumCentroids].
  //! Used by build_centroid_distance_table().
  std::vector<float> sub_centroid_norms_;
};

}  // namespace turbo
}  // namespace zvec
