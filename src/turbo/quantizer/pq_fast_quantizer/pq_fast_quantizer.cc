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

#include "quantizer/pq_fast_quantizer/pq_fast_quantizer.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <ailego/algorithm/kmeans.h>
#include <ailego/math/normalizer.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_threads.h>
#include "common/fast_scan_common.h"

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// FastScan PQ serialization payload (follows the QuantizerSerHeader).
// ---------------------------------------------------------------------------
struct PqFastSerPayload {
  uint32_t original_dim;
  uint32_t num_chunk;
  uint32_t sub_dim;
  uint32_t num_centroids;  // always 16
  uint8_t use_zero_mean;
  uint8_t input_data_type;  // turbo DataType: kFp32=3, kFp16=2
  uint8_t reserved[2];
};

int PqFastQuantizer::setup_functions() {
  const QuantizeType input_quantize_type = input_data_type_ == DataType::kFp16
                                               ? QuantizeType::kFp16
                                               : QuantizeType::kFp32;

  // ISA-dispatched FastScan kernel: the packed block scan is the only distance
  // kernel of the kPQFast family.
  auto pq_k = get_pq_kernels(DataType::kInt4, QuantizeType::kPQFast);
  scan_fn_ = pq_k.fast_scan;

  // L2-only batch distance for encoding and KMeans training: the PQ
  // codebook is trained/encoded in L2 space regardless of the search metric.
  l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, input_data_type_,
                              input_quantize_type, CpuArchType::kAuto);

  // Inner-product batch distance for the precomputed residual tables.  The
  // table terms are pure inner products regardless of the search metric.
  ip_batch_fn_ =
      get_batch_distance_func(MetricType::kInnerProduct, input_data_type_,
                              input_quantize_type, CpuArchType::kAuto);

  // The search LUT always reuses one of the two kernels above, so no third
  // dispatch is needed: Cosine = normalize + L2 is monotonic with
  // squared-Euclidean after normalization (aligned with PqInt4Quantizer), and
  // only InnerProduct needs the IP kernel.  init() rejects every other metric,
  // so the remaining cases are unreachable and leave batch_fn_ empty.
  switch (metric_from_name(meta_.metric_name())) {
    case MetricType::kInnerProduct:
      batch_fn_ = ip_batch_fn_;
      break;
    case MetricType::kSquaredEuclidean:
      batch_fn_ = l2_batch_fn_;
      break;
    case MetricType::kCosine:
      batch_fn_ = l2_batch_fn_;
      extra_meta_size_ = kExtraMetaSizeCosine;
      meta_.set_extra_meta_size(extra_meta_size_);
      break;
    case MetricType::kMipsSquaredEuclidean:
    case MetricType::kUnknown:
      break;
  }

  // Only these four matter: kPQFast deliberately leaves asymmetric_distance /
  // symmetric_distance / batch_asymmetric_distance null, since the packed
  // block scan is its sole read path.
  if (!scan_fn_ || !l2_batch_fn_ || !ip_batch_fn_ || !batch_fn_) {
    return kErrUnsupported;
  }
  return 0;
}

int PqFastQuantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  initialized_ = false;
  meta_ = meta;

  // Map core IndexMeta::DataType to turbo DataType.
  if (meta.data_type() == IndexMeta::DataType::DT_FP16) {
    input_data_type_ = DataType::kFp16;
  } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
    input_data_type_ = DataType::kFp32;
  } else {
    return kErrUnsupported;
  }

  // FastScan is a batch-scan quantizer for L2 / IP / Cosine (Cosine =
  // normalize + L2, aligned with PqInt4Quantizer).
  const auto &metric_name = meta_.metric_name();
  if (metric_name != "SquaredEuclidean" && metric_name != "InnerProduct" &&
      metric_name != "Cosine") {
    return kErrUnsupported;
  }

  uint32_t d = meta.dimension();
  original_dim_ = d;

  // Read num_chunk from params (required).
  uint32_t nsq = 0;
  if (!params.get("num_chunk", &nsq) || nsq == 0) {
    return kErrUnsupported;
  }
  if (d % nsq != 0) {
    return kErrUnsupported;
  }

  num_chunk_ = nsq;
  sub_dim_ = d / nsq;

  // Pre-allocate centroids as byte buffer (filled by train()).
  centroids_.resize(static_cast<size_t>(num_chunk_) * kNumCentroids * sub_dim_ *
                    element_size());

  if (setup_functions() != 0) {
    return kErrUnsupported;
  }

  // Read optional training params (aligned with multi_chunk_cluster)
  params.get("thread_count", &thread_count_);
  params.get("markov_chain_length", &markov_chain_length_);
  params.get("epsilon", &epsilon_);
  params.get("use_zero_mean", &use_zero_mean_);

  // Zero-mean centering shifts the space, which breaks the IP LUT
  // decomposition -- keep it for L2 / Cosine (Cosine runs in normalized
  // space where L2 semantics apply).
  if (use_zero_mean_ && metric_name != "SquaredEuclidean" &&
      metric_name != "Cosine") {
    use_zero_mean_ = false;
  }

  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);
  meta_.set_extra_meta_size(extra_meta_size_);
  initialized_ = true;
  return 0;
}

// ---------------------------------------------------------------------------
// Training (identical paradigm to PqInt4Quantizer: L2 KMeans, k=16).
// ---------------------------------------------------------------------------

void PqFastQuantizer::build_centroid_ptrs_cache() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  const uint8_t *base = centroids_.data();
  const size_t type_size = element_size();
  centroid_ptrs_cache_.resize(num_chunk_);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    auto &ptrs = centroid_ptrs_cache_[m];
    ptrs.resize(k);
    for (size_t c = 0; c < k; ++c) {
      ptrs[c] = base + (static_cast<size_t>(m) * k * d + c * d) * type_size;
    }
  }
}

template <typename T>
void PqFastQuantizer::train_subquantizer(const T *data, size_t num,
                                         size_t stride, size_t sub_idx) {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  uint8_t *centroids_m =
      centroids_.data() + static_cast<size_t>(sub_idx) * k * d * sizeof(T);

  // Non-spherical L2 KMeans: the PQ codebook must minimize L2 reconstruction
  // error, so centroids are the true (magnitude-preserving) means.
  ailego::NumericalKmeans<T, SingleQueueIndexThreads> algorithm(k, d);

  // Append sub-vectors (NumericalKmeans handles transpose internally)
  for (size_t i = 0; i < num; ++i) {
    const T *sub_vec =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(data) +
                                    i * stride) +
        sub_idx * d;
    algorithm.append(sub_vec, d);
  }

  // Single-threaded pool — parallelism is at the sub-quantizer level.
  auto local_threads = std::make_shared<SingleQueueIndexThreads>(1, false);

  // KMC2 centroid initialization.
  ailego::Kmc2CentroidsGenerator<
      ailego::NumericalKmeans<T, SingleQueueIndexThreads>,
      SingleQueueIndexThreads>
      gen;
  gen.set_chain_length(markov_chain_length_);
  gen.set_assumption_free(false);
  algorithm.init_centroids(*local_threads, gen);

  // Lloyd iterations
  double cost = 0.0;
  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    double old_cost = cost;
    bool result = algorithm.cluster_once(*local_threads, &cost);
    if (!result) {
      break;
    }
    double new_epsilon = std::abs(cost - old_cost);
    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // Extract centroids into the flat centroids_ byte buffer
  const auto &cents = algorithm.centroids();
  for (size_t c = 0; c < cents.count(); ++c) {
    std::memcpy(centroids_m + c * d * sizeof(T), cents[c], d * sizeof(T));
  }
}

int PqFastQuantizer::train(IndexHolder::Pointer holder) {
  return train(holder, static_cast<int>(thread_count_));
}

int PqFastQuantizer::train(IndexHolder::Pointer holder, int thread_count) {
  if (!holder) {
    return kErrUnsupported;
  }

  size_t num = holder->count();
  const uint32_t elem_size = element_size();

  // Collect all data into a contiguous byte buffer (original data type).
  auto iter = holder->create_iterator();
  std::vector<uint8_t> all_data(num * original_dim_ * elem_size);
  size_t row = 0;
  for (; iter->is_valid(); iter->next(), ++row) {
    std::memcpy(all_data.data() + row * original_dim_ * elem_size, iter->data(),
                original_dim_ * elem_size);
  }

  // Subsample if the dataset exceeds the training limit.
  if (num > kMaxTrainVectors) {
    std::mt19937 rng(42);
    // Fisher-Yates partial shuffle: randomly place kMaxTrainVectors vectors
    // at the front of the buffer.
    for (size_t i = 0; i < kMaxTrainVectors; ++i) {
      std::uniform_int_distribution<size_t> dist(i, num - 1);
      size_t j = dist(rng);
      if (i != j) {
        // Swap full vectors (dim-sized chunks in bytes).
        size_t vec_bytes = original_dim_ * elem_size;
        for (size_t b = 0; b < vec_bytes; ++b) {
          std::swap(all_data[i * vec_bytes + b], all_data[j * vec_bytes + b]);
        }
      }
    }
    num = kMaxTrainVectors;
    all_data.resize(num * original_dim_ * elem_size);
    all_data.shrink_to_fit();
  }

  size_t data_stride = original_dim_ * elem_size;

  // For Cosine: normalize training data so centroids are learned in
  // normalized space (L2 minimization == maximizing cosine similarity).
  if (meta_.metric_name() == "Cosine") {
    switch (input_data_type_) {
      case DataType::kFp16:
        normalize_batch(reinterpret_cast<ailego::Float16 *>(all_data.data()),
                        num);
        break;
      case DataType::kFp32:
        normalize_batch(reinterpret_cast<float *>(all_data.data()), num);
        break;
      default:
        break;
    }
  }

  // Zero-mean centering: subtract the per-dimension mean; the centroid is
  // saved for quantize_data / quantize_query / dequantize.  For Cosine this
  // runs AFTER normalization, so all paths keep the same order
  // (normalize -> center; dequantize: un-center -> rescale).
  if (use_zero_mean_) {
    switch (input_data_type_) {
      case DataType::kFp16:
        compute_and_subtract_center(
            reinterpret_cast<ailego::Float16 *>(all_data.data()), num);
        break;
      case DataType::kFp32:
        compute_and_subtract_center(reinterpret_cast<float *>(all_data.data()),
                                    num);
        break;
      default:
        break;
    }
  }

  // Create thread pool.
  auto threads = std::make_shared<SingleQueueIndexThreads>(
      static_cast<uint32_t>(thread_count), false);
  auto task_group = threads->make_group();

  // Distribute sub-quantizers across threads.
  std::atomic<size_t> finished{0};
  size_t pool_count = threads->count();

  auto submit_training = [&](const auto *typed_data) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(typed_data)>>;
    for (size_t i = 0; i < pool_count; ++i) {
      task_group->submit(ailego::Closure::New(
          [this, typed_data, num, data_stride, i, pool_count, &finished]() {
            for (uint32_t m = static_cast<uint32_t>(i); m < num_chunk_;
                 m += static_cast<uint32_t>(pool_count)) {
              train_subquantizer<T>(typed_data, num, data_stride, m);
              finished++;
            }
          }));
    }
  };

  switch (input_data_type_) {
    case DataType::kFp16:
      submit_training(
          reinterpret_cast<const ailego::Float16 *>(all_data.data()));
      break;
    case DataType::kFp32:
      submit_training(reinterpret_cast<const float *>(all_data.data()));
      break;
    default:
      break;
  }
  task_group->wait_finish();

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  // Pre-compute sub-centroid norms for the precomputed residual table.
  compute_sub_centroid_norms();
  return 0;
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

void PqFastQuantizer::quantize_data(const void *input, void *output) const {
  uint8_t *code = reinterpret_cast<uint8_t *>(output);
  const uint32_t elem_size = element_size();

  // For Cosine: normalize FIRST (codebook is trained in normalized space);
  // the original norm is stored after the PQ code for dequantize().
  std::vector<uint8_t> norm_vec_storage;
  float vec_norm = 0.0f;
  const void *vec = input;

  if (meta_.metric_name() == "Cosine") {
    norm_vec_storage.resize(original_dim_ * elem_size);
    std::memcpy(norm_vec_storage.data(), input, original_dim_ * elem_size);
    switch (input_data_type_) {
      case DataType::kFp16:
        normalize_single(
            reinterpret_cast<ailego::Float16 *>(norm_vec_storage.data()),
            &vec_norm);
        break;
      case DataType::kFp32:
        normalize_single(reinterpret_cast<float *>(norm_vec_storage.data()),
                         &vec_norm);
        break;
      default:
        break;
    }
    vec = norm_vec_storage.data();
  }

  // Zero-mean centering: subtract centroid before encoding.
  std::vector<uint8_t> centered_vec_storage;
  if (use_zero_mean_) {
    centered_vec_storage.resize(original_dim_ * elem_size);
    std::memcpy(centered_vec_storage.data(), vec, original_dim_ * elem_size);
    switch (input_data_type_) {
      case DataType::kFp16:
        subtract_center(
            reinterpret_cast<ailego::Float16 *>(centered_vec_storage.data()));
        break;
      case DataType::kFp32:
        subtract_center(reinterpret_cast<float *>(centered_vec_storage.data()));
        break;
      default:
        break;
    }
    vec = centered_vec_storage.data();
  }

  // Zero the packed code buffer first: nibble packing ORs codes in, and an
  // odd num_chunk leaves the last byte's high nibble as the (zero) pad.
  const size_t packed_len = packed_code_length();
  std::memset(code, 0, packed_len);

  // Encode with L2-only batch distance (search-metric independent),
  // fusing argmin into the distance loop.
  float dists[kNumCentroids];
  const uint8_t *vec_bytes = reinterpret_cast<const uint8_t *>(vec);

  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const void *sub_vec =
        vec_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size;
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];

    // Compute L2 distances from this sub-vector to all 16 centroids.
    // const_cast: .data() returns const void* const* but the kernel expects
    // const void** and never modifies the pointer array.
    l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub_vec,
                 kNumCentroids, sub_dim_, dists);

    // Argmin: find nearest centroid.  Seeding with +infinity skips NaN
    // distances from dead centroids; a dists[0] seed would pin them to 0
    // because (x < NaN) is always false.
    float best_dist = std::numeric_limits<float>::infinity();
    uint32_t best_idx = 0;
    for (uint32_t j = 0; j < kNumCentroids; ++j) {
      if (dists[j] < best_dist) {
        best_dist = dists[j];
        best_idx = j;
      }
    }

    // Pack the 4-bit code: low nibble for even m, high nibble for odd m.
    code[m >> 1] |= static_cast<uint8_t>((m & 1) ? (best_idx << 4) : best_idx);
  }

  // Store norm after the packed PQ code for Cosine dequantize support.
  // memcpy: packed_len is arbitrary, so code + packed_len carries no float
  // alignment guarantee (dequantize() reads it back the same way).
  if (meta_.metric_name() == "Cosine") {
    std::memcpy(code + packed_len, &vec_norm, sizeof(vec_norm));
  }
}

void PqFastQuantizer::compute_float_lut(const void *input, float *lut) const {
  const uint32_t elem_size = element_size();

  // For Cosine: normalize FIRST (Cosine uses an L2 LUT on normalized data),
  // consistent with train() / quantize_data().
  std::vector<uint8_t> norm_query_storage;
  const void *query = input;

  if (meta_.metric_name() == "Cosine") {
    norm_query_storage.resize(original_dim_ * elem_size);
    std::memcpy(norm_query_storage.data(), input, original_dim_ * elem_size);
    switch (input_data_type_) {
      case DataType::kFp16:
        normalize_single(
            reinterpret_cast<ailego::Float16 *>(norm_query_storage.data()));
        break;
      case DataType::kFp32:
        normalize_single(reinterpret_cast<float *>(norm_query_storage.data()));
        break;
      default:
        break;
    }
    query = norm_query_storage.data();
  }

  // Zero-mean centering: subtract centroid before LUT computation.
  std::vector<uint8_t> centered_query_storage;
  if (use_zero_mean_) {
    centered_query_storage.resize(original_dim_ * elem_size);
    std::memcpy(centered_query_storage.data(), query,
                original_dim_ * elem_size);
    switch (input_data_type_) {
      case DataType::kFp16:
        subtract_center(
            reinterpret_cast<ailego::Float16 *>(centered_query_storage.data()));
        break;
      case DataType::kFp32:
        subtract_center(
            reinterpret_cast<float *>(centered_query_storage.data()));
        break;
      default:
        break;
    }
    query = centered_query_storage.data();
  }

  // LUT[m][j] = distance(q_m, c_m[j]) via the metric-aware batch_fn_:
  // L2/Cosine: ||q_m - c_m[j]||^2   IP: -dot(q_m, c_m[j]).
  const uint8_t *query_bytes = reinterpret_cast<const uint8_t *>(query);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    const void *sub_query =
        query_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size;
    batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub_query,
              kNumCentroids, sub_dim_, lut + m * kNumCentroids);
  }

  // For Cosine, pre-scale the L2 entries by 0.5 so that every ADC path sums
  // to 0.5 * ||qn - cn||^2 = 1 - cos(q, c), i.e. the cosine distance
  // directly (aligned with PqInt4Quantizer; the u8 affine quantization
  // absorbs the scale into delta/bias).
  if (meta_.metric_name() == "Cosine") {
    const size_t lut_size = static_cast<size_t>(num_chunk_) * kNumCentroids;
    for (size_t i = 0; i < lut_size; ++i) {
      lut[i] *= 0.5f;
    }
  }
}

size_t PqFastQuantizer::quantized_query_vector_length() const {
  return fast_scan_packed_lut_size(num_chunk_) + 2 * sizeof(float);
}

size_t PqFastQuantizer::lut_entry_count() const {
  // The query encoding appends a delta/bias tail after the packed u8 LUT;
  // the pre-quantization float table holds only the real subquantizers.
  return static_cast<size_t>(num_chunk_) * kNumCentroids;
}

int PqFastQuantizer::quantize_lut(const float *lut, void *output) const {
  const size_t lut_size = static_cast<size_t>(num_chunk_) * kNumCentroids;

  // Sentinel-seeded and non-finite entries excluded: std::min/std::max keep
  // their first argument for a NaN operand, so a NaN reaching lo/hi is
  // unrecoverable; one +Inf makes delta infinite and all distances 0 * Inf.
  float lo = std::numeric_limits<float>::max();
  float hi = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < lut_size; ++i) {
    if (std::isfinite(lut[i])) {
      lo = std::min(lo, lut[i]);
      hi = std::max(hi, lut[i]);
    }
  }
  if (lo > hi) {  // no finite entry at all
    lo = 0.0f;
    hi = 0.0f;
  }
  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  const float delta = std::max(hi - lo, epsilon) / 255.0f;
  const float inv_delta = 1.0f / delta;

  std::vector<uint8_t> u8_lut(lut_size);
  for (size_t i = 0; i < lut_size; ++i) {
    // Non-finite entries go to the far end: NaN slips through the clamp
    // untouched and would cast to code 0, i.e. the *smallest* distance.
    const float q =
        std::isfinite(lut[i])
            ? std::min(std::max(std::round((lut[i] - lo) * inv_delta), 0.0f),
                       255.0f)
            : 255.0f;
    u8_lut[i] = static_cast_from_float_to_uint8(q);
  }

  // [packed u8 LUT | float delta | float bias]
  uint8_t *out = reinterpret_cast<uint8_t *>(output);
  fast_scan_pack_lut(u8_lut.data(), num_chunk_, out);
  const float bias = static_cast<float>(num_chunk_) * lo;
  std::memcpy(out + fast_scan_packed_lut_size(num_chunk_), &delta,
              sizeof(float));
  std::memcpy(out + fast_scan_packed_lut_size(num_chunk_) + sizeof(float),
              &bias, sizeof(float));
  return 0;
}

void PqFastQuantizer::quantize_query(const void *input, void *output) const {
  std::vector<float> lut(static_cast<size_t>(num_chunk_) * kNumCentroids);
  compute_float_lut(input, lut.data());
  quantize_lut(lut.data(), output);
}

// ---------------------------------------------------------------------------
// Precomputed residual table support (faiss use_precomputed_table=1).
// fp32 + plain L2 only: Cosine re-normalizes the residual query (non-linear,
// breaks the expansion), IP already has a query-only LUT, fp16 callers fall
// back to the per-list path.  Same term2/term3 contract as PqInt8Quantizer,
// but the merged per-list LUT is affine-quantized in merge_query_distance
// _table() so the scan keeps consuming the packed-u8 FastScan query format.
// ---------------------------------------------------------------------------

void PqFastQuantizer::compute_sub_centroid_norms() {
  const size_t k = kNumCentroids;
  sub_centroid_norms_.resize(static_cast<size_t>(num_chunk_) * k);

  // ||c_m[j]||^2 = dist(zero, c_m[j]): reuse the L2 batch kernel with a
  // zero query vector instead of hand-rolling a norm loop.
  std::vector<uint8_t> zero(static_cast<size_t>(sub_dim_) * element_size(), 0);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()), zero.data(),
                 k, sub_dim_, sub_centroid_norms_.data() + m * k);
  }
}

int PqFastQuantizer::build_centroid_distance_table(const void *centroids,
                                                   size_t centroid_num,
                                                   std::string *table) const {
  if (input_data_type_ != DataType::kFp32 ||
      meta_.metric_name() != "SquaredEuclidean") {
    return kErrUnsupported;
  }
  //! Refuse zero-mean: the shift applied here (term2) and in
  //! quantize_precomputed_query() (term3) cancels on merge, so the scan would
  //! rank against ||q - c_i - c_m[j]||^2 and drop the mean the codes carry.
  if (use_zero_mean_) {
    return kErrUnsupported;
  }
  if (centroids == nullptr || centroid_num == 0 || table == nullptr ||
      num_chunk_ == 0 || centroids_.empty() || sub_centroid_norms_.empty()) {
    return kErrInvalidArgument;
  }

  const size_t row_floats = lut_entry_count();
  //! Mimic faiss precomputed_table_max_bytes: refuse oversized tables so
  //! the caller can fall back to the per-list path.
  const size_t kMaxTableBytes = 1ULL << 30;
  if (centroid_num * row_floats * sizeof(float) > kMaxTableBytes) {
    return kErrUnsupported;
  }

  table->resize(centroid_num * row_floats * sizeof(float));
  float *tab = reinterpret_cast<float *>(&(*table)[0]);

  std::vector<float> buf(original_dim_);
  const float *src = reinterpret_cast<const float *>(centroids);
  float dists[kNumCentroids];
  for (size_t i = 0; i < centroid_num; ++i) {
    std::memcpy(buf.data(), src + i * original_dim_,
                original_dim_ * sizeof(float));
    const uint8_t *buf_bytes = reinterpret_cast<const uint8_t *>(buf.data());
    const uint32_t elem_size = element_size();
    float *row = tab + i * row_floats;
    for (uint32_t m = 0; m < num_chunk_; ++m) {
      //! term2 = ||c_m[j]||^2 + 2<c_i^m, c_m[j]>.  The IP kernel returns
      //! the negated inner product, i.e. dists[j] = -<c_i^m, c_m[j]>.
      const auto &centroid_ptrs = centroid_ptrs_cache_[m];
      ip_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                   buf_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size,
                   kNumCentroids, sub_dim_, dists);
      const float *rn = sub_centroid_norms_.data() + m * kNumCentroids;
      float *out_m = row + m * kNumCentroids;
      for (uint32_t j = 0; j < kNumCentroids; ++j) {
        out_m[j] = rn[j] - 2.0f * dists[j];
      }
    }
  }
  return 0;
}

int PqFastQuantizer::quantize_precomputed_query(const void *query,
                                                const IndexQueryMeta &qmeta,
                                                std::string *out,
                                                IndexQueryMeta *ometa) const {
  if (input_data_type_ != DataType::kFp32 ||
      meta_.metric_name() != "SquaredEuclidean") {
    return kErrUnsupported;
  }
  //! Same zero-mean restriction as build_centroid_distance_table().
  if (use_zero_mean_) {
    return kErrUnsupported;
  }
  if (query == nullptr || out == nullptr || centroids_.empty() ||
      qmeta.unit_size() != sizeof(float)) {
    return kErrInvalidArgument;
  }

  //! Preprocess into the codebook space (Cosine and zero-mean both gated
  //! out above), then build the term3 table.  It stays UNQUANTIZED here:
  //! merge_query_distance_table() adds the per-list term2 row first and
  //! affine-quantizes the merged table once.
  std::vector<float> prep(original_dim_);
  int ret = preprocess_query(query, prep.data());
  if (ret != 0) {
    return ret;
  }

  out->resize(lut_entry_count() * sizeof(float));
  float *lut = reinterpret_cast<float *>(&(*out)[0]);
  const uint8_t *prep_bytes = reinterpret_cast<const uint8_t *>(prep.data());
  const uint32_t elem_size = element_size();
  float dists[kNumCentroids];
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    //! term3 = -2<q^m, c_m[j]>; dists[j] = -<q^m, c_m[j]>, so LUT = 2*dists.
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    ip_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                 prep_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size,
                 kNumCentroids, sub_dim_, dists);
    float *lut_m = lut + m * kNumCentroids;
    for (uint32_t j = 0; j < kNumCentroids; ++j) {
      lut_m[j] = 2.0f * dists[j];
    }
  }

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP32, original_dim_,
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

int PqFastQuantizer::merge_query_distance_table(
    const void *query_table, const std::string &centroid_table,
    size_t centroid_id, std::string *out) const {
  const size_t row_floats = lut_entry_count();
  if (query_table == nullptr || out == nullptr || num_chunk_ == 0 ||
      row_floats == 0 ||
      centroid_table.size() < (centroid_id + 1) * row_floats * sizeof(float)) {
    return kErrInvalidArgument;
  }

  //! term2 + term3 in the float domain, then affine-quantize once, so the
  //! output keeps the packed-u8 FastScan query format consumed by the scan.
  std::vector<float> merged(row_floats);
  const float *qtab = reinterpret_cast<const float *>(query_table);
  const float *ctab = reinterpret_cast<const float *>(centroid_table.data()) +
                      centroid_id * row_floats;
  for (size_t i = 0; i < row_floats; ++i) {
    merged[i] = qtab[i] + ctab[i];
  }
  out->resize(quantized_query_vector_length());
  return quantize_lut(merged.data(), &(*out)[0]);
}

int PqFastQuantizer::preprocess_query(const void *input, float *out) const {
  if (input_data_type_ != DataType::kFp32) {
    return kErrNotImplemented;
  }
  // Mirror the query-side handling of compute_float_lut(): Cosine
  // normalizes first, then zero-mean centering applies.  Any divergence
  // here silently breaks the precomputed-table decomposition, so this must
  // stay in lockstep with compute_float_lut().
  const float *q = reinterpret_cast<const float *>(input);
  std::memcpy(out, q, static_cast<size_t>(original_dim_) * sizeof(float));
  if (meta_.metric_name() == "Cosine") {
    normalize_single(out);
  }
  if (use_zero_mean_) {
    subtract_center(out);
  }
  return 0;
}

int PqFastQuantizer::pack_codes(const void *codes, size_t num, size_t stride,
                                void *out) const {
  if (!codes || !out) {
    return kErrInvalidArgument;
  }
  fast_scan_pack_codes(reinterpret_cast<const uint8_t *>(codes), num, stride,
                       num_chunk_, reinterpret_cast<uint8_t *>(out));
  return 0;
}

// ---------------------------------------------------------------------------
// Distance computation
// ---------------------------------------------------------------------------

float PqFastQuantizer::calc_distance_dp_query(const void * /*dp*/,
                                              const void * /*query*/) const {
  assert(false && "PqFastQuantizer: packed-block only, no single-code ADC");
  return std::numeric_limits<float>::quiet_NaN();
}

void PqFastQuantizer::calc_distance_dp_query_batch(
    const void *const * /*dp_list*/, int dp_num, const void * /*query*/,
    float *dist_list) const {
  assert(false && "PqFastQuantizer: packed-block only, no gather-style batch");
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = std::numeric_limits<float>::quiet_NaN();
  }
}

void PqFastQuantizer::calc_distance_packed_block(const void *block, size_t num,
                                                 const void *query,
                                                 float *dist_list) const {
  // `block` is one (or several back-to-back) packed 32-vector block(s)
  // produced by pack_codes().
  const uint8_t *q = reinterpret_cast<const uint8_t *>(query);
  float delta = 0.0f;
  float bias = 0.0f;
  const uint8_t *tail = q + fast_scan_packed_lut_size(num_chunk_);
  std::memcpy(&delta, tail, sizeof(float));
  std::memcpy(&bias, tail + sizeof(float), sizeof(float));

  const uint8_t *packed = reinterpret_cast<const uint8_t *>(block);
  const size_t block_bytes = fast_scan_packed_block_size(num_chunk_);
  int32_t accu32[kFastScanBlockSize];
  size_t done = 0;
  while (done < num) {
    const size_t n = std::min<size_t>(kFastScanBlockSize, num - done);
    scan_fn_(packed, q, num_chunk_, accu32);
    for (size_t i = 0; i < n; ++i) {
      dist_list[done + i] = static_cast<float>(accu32[i]) * delta + bias;
    }
    packed += block_bytes;
    done += n;
  }
}

float PqFastQuantizer::calc_distance_dp_query_unquantized(
    const void * /*dp*/, const void * /*query*/) const {
  assert(false && "PqFastQuantizer: packed-block only, no single-code ADC");
  return std::numeric_limits<float>::quiet_NaN();
}

void PqFastQuantizer::calc_distance_dp_query_batch_unquantized(
    const void *const * /*dp_list*/, int dp_num, const void * /*query*/,
    float *dist_list) const {
  assert(false && "PqFastQuantizer: packed-block only, no gather-style batch");
  for (int i = 0; i < dp_num; ++i) {
    dist_list[i] = std::numeric_limits<float>::quiet_NaN();
  }
}

float PqFastQuantizer::calc_distance_dp_dp(const void * /*dp1*/,
                                           const void * /*dp2*/) const {
  // NaN rather than a plausible value, so callers that skip the capability
  // check cannot silently build on all-equal distances.
  assert(false && "PqFastQuantizer: no SDC, code-vs-code distance unavailable");
  return std::numeric_limits<float>::quiet_NaN();
}

int PqFastQuantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                              std::string *out, IndexQueryMeta *ometa) const {
  // Validate unit_size against the input data type.
  size_t expected_unit = 0;
  switch (input_data_type_) {
    case DataType::kFp16:
      expected_unit = sizeof(ailego::Float16);
      break;
    case DataType::kFp32:
      expected_unit = sizeof(float);
      break;
    default:
      break;
  }
  if (qmeta.unit_size() != expected_unit) {
    return kErrUnsupported;
  }

  size_t lut_bytes = quantized_query_vector_length();
  out->resize(lut_bytes);
  quantize_query(query, &(*out)[0]);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_INT4, num_chunk_,
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

int PqFastQuantizer::dequantize(const void * /*in*/,
                                const IndexQueryMeta & /*qmeta*/,
                                std::string * /*out*/) const {
  // Reconstruction needs a plain nibble code, which only exists between
  // quantize_data() and pack_codes(): stored blocks keep neither the
  // per-code byte boundary nor the Cosine norm tail.  Callers that need the
  // vector back must read it from the unquantized side (an IVF features
  // segment or the refiner index), never from a FastScan code.
  assert(false && "PqFastQuantizer: packed-block only, cannot reconstruct");
  return kErrUnsupported;
}

DistanceImpl PqFastQuantizer::distance(const void * /*query*/,
                                       const IndexQueryMeta & /*qmeta*/) const {
  // No gather-style handle: callers must fall back to
  // calc_distance_packed_block() rather than receive a callable that would
  // mis-read a plain code.
  return DistanceImpl{};
}

DistanceImpl PqFastQuantizer::sym_distance(
    const void * /*query*/, const IndexQueryMeta & /*qmeta*/) const {
  // FastScan keeps no SDC dist_table, so code-vs-code distance is not
  // available.  Return an empty handle so callers fall back instead of the
  // base-class forward, which would mis-read a PQ code as a LUT.
  return DistanceImpl{};
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
int PqFastQuantizer::serialize(std::string *out) const {
  if (!out) return kErrUnsupported;

  QuantizerSerHeader hdr{};
  hdr.magic = kQuantizerMagic;
  hdr.version = kQuantizerSerVersion;
  hdr.quant_type = static_cast<uint16_t>(QuantizeType::kPQFast);
  hdr.dim = original_dim_;
  hdr.metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));
  hdr.data_type = static_cast<uint16_t>(DataType::kInt4);

  PqFastSerPayload payload{};
  payload.original_dim = original_dim_;
  payload.num_chunk = num_chunk_;
  payload.sub_dim = sub_dim_;
  payload.num_centroids = kNumCentroids;
  payload.use_zero_mean = use_zero_mean_ ? 1 : 0;
  payload.input_data_type = static_cast<uint8_t>(input_data_type_);

  size_t centroids_bytes = centroids_.size();  // already byte buffer
  size_t centroid_bytes = use_zero_mean_ ? centroid_.size() * sizeof(float) : 0;
  hdr.payload_size =
      static_cast<uint32_t>(sizeof(payload) + centroids_bytes + centroid_bytes);

  out->clear();
  out->append(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out->append(reinterpret_cast<const char *>(&payload), sizeof(payload));
  out->append(reinterpret_cast<const char *>(centroids_.data()),
              centroids_bytes);
  // Append zero-mean centroid for centering support.
  if (use_zero_mean_) {
    out->append(reinterpret_cast<const char *>(centroid_.data()),
                centroid_bytes);
  }
  return 0;
}

int PqFastQuantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int PqFastQuantizer::deserialize(const void *data, size_t len) {
  if (!initialized_) return kErrUnsupported;
  if (len < sizeof(QuantizerSerHeader) + sizeof(PqFastSerPayload)) {
    return kErrUnsupported;
  }

  const char *ptr = reinterpret_cast<const char *>(data);
  QuantizerSerHeader hdr;
  std::memcpy(&hdr, ptr, sizeof(hdr));
  ptr += sizeof(hdr);

  if (hdr.magic != kQuantizerMagic) return kErrUnsupported;
  if (hdr.quant_type != static_cast<uint16_t>(QuantizeType::kPQFast)) {
    return kErrUnsupported;
  }
  // The codebook is only decodable under the metric it was trained in.
  if (hdr.metric !=
      static_cast<uint32_t>(metric_from_name(meta_.metric_name()))) {
    return kErrUnsupported;
  }

  PqFastSerPayload payload;
  std::memcpy(&payload, ptr, sizeof(payload));
  ptr += sizeof(payload);

  original_dim_ = payload.original_dim;
  num_chunk_ = payload.num_chunk;
  sub_dim_ = payload.sub_dim;
  input_data_type_ = static_cast<DataType>(payload.input_data_type);
  if (input_data_type_ != DataType::kFp32 &&
      input_data_type_ != DataType::kFp16) {
    return kErrUnsupported;
  }

  // Restore centroids (raw bytes in original data type).
  size_t centroids_bytes = static_cast<size_t>(num_chunk_) * kNumCentroids *
                           sub_dim_ * element_size();
  size_t centroid_bytes =
      payload.use_zero_mean ? static_cast<size_t>(original_dim_) * sizeof(float)
                            : 0;
  // Both blob sizes are derived from payload fields, so a truncated or
  // corrupted buffer would otherwise be read past its end.
  if (len - sizeof(QuantizerSerHeader) - sizeof(PqFastSerPayload) <
      centroids_bytes + centroid_bytes) {
    return kErrUnsupported;
  }
  centroids_.resize(centroids_bytes);
  std::memcpy(centroids_.data(), ptr, centroids_bytes);
  ptr += centroids_bytes;

  // Restore zero-mean centroid if centering was enabled.
  if (payload.use_zero_mean) {
    use_zero_mean_ = true;
    centroid_.resize(original_dim_);
    std::memcpy(centroid_.data(), ptr, centroid_bytes);
    ptr += centroid_bytes;
  }

  // Re-dispatch kernels and batch distance functions.
  if (setup_functions() != 0) {
    return kErrUnsupported;
  }

  // Set output meta: the quantized representation is INT4 codes
  // (+ extra_meta_size_ for Cosine norm storage).
  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);
  meta_.set_extra_meta_size(extra_meta_size_);

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  // Pre-compute sub-centroid norms for the precomputed residual table.
  compute_sub_centroid_norms();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqFastQuantizer);

// ---------------------------------------------------------------------------
// Template helper implementations (type-dispatched at call sites)
// ---------------------------------------------------------------------------

template <typename T>
void PqFastQuantizer::compute_and_subtract_center(T *data, size_t num) {
  centroid_.assign(original_dim_, 0.0f);
  for (size_t i = 0; i < num; ++i) {
    const T *v = data + i * original_dim_;
    for (uint32_t d = 0; d < original_dim_; ++d) {
      centroid_[d] += static_cast<float>(v[d]);
    }
  }
  for (uint32_t d = 0; d < original_dim_; ++d) {
    centroid_[d] /= static_cast<float>(num);
  }
  for (size_t i = 0; i < num; ++i) {
    T *v = data + i * original_dim_;
    for (uint32_t d = 0; d < original_dim_; ++d) {
      v[d] -= centroid_[d];
    }
  }
}

template <typename T>
void PqFastQuantizer::subtract_center(T *vec) const {
  for (uint32_t d = 0; d < original_dim_; ++d) {
    vec[d] -= centroid_[d];
  }
}

template <typename T>
void PqFastQuantizer::normalize_batch(T *data, size_t num) const {
  for (size_t i = 0; i < num; ++i) {
    float norm = 0.0f;
    ailego::Normalizer<T>::L2(data + i * original_dim_, original_dim_, &norm);
  }
}

template <typename T>
void PqFastQuantizer::normalize_single(T *vec, float *norm_out) const {
  float norm = 0.0f;
  ailego::Normalizer<T>::L2(vec, original_dim_, &norm);
  if (norm_out) {
    *norm_out = norm;
  }
}

}  // namespace turbo
}  // namespace zvec
