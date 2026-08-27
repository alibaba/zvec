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

#include "quantizer/pq_int4_quantizer/pq_int4_quantizer.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <ailego/algorithm/kmeans.h>
#include <ailego/math/normalizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// PQ serialization payload (follows the QuantizerSerHeader).
// ---------------------------------------------------------------------------
struct PqInt4SerPayload {
  uint32_t original_dim;
  uint32_t num_chunk;
  uint32_t chunk_dim;
  uint32_t num_centroids;  // always 16 for int4
  uint8_t use_zero_mean;
  uint8_t input_data_type;  // turbo DataType: kFp32=3, kFp16=2
  uint8_t reserved[2];
};

int PqInt4Quantizer::setup_functions() {
  const QuantizeType input_quantize_type = input_data_type_ == DataType::kFp16
                                               ? QuantizeType::kFp16
                                               : QuantizeType::kFp32;

  // Dispatch ISA kernels for the int4 (nibble-packed, 16-centroid) layout.
  auto pq_k = get_pq_kernels(DataType::kInt4);
  adc_fn_ = pq_k.asymmetric_distance;
  sdc_fn_ = pq_k.symmetric_distance;
  batch_adc_fn_ = pq_k.batch_asymmetric_distance;

  // L2-only batch distance for encoding and KMeans training: the PQ codebook
  // is trained in L2 space regardless of the search metric.
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
  // squared-Euclidean after normalization, and only InnerProduct needs the IP
  // kernel.  Metrics with no registered kernel leave batch_fn_ empty and are
  // rejected by the check below.
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
      break;
    case MetricType::kMipsSquaredEuclidean:
    case MetricType::kUnknown:
      break;
  }

  if (!adc_fn_ || !sdc_fn_ || !batch_adc_fn_ || !l2_batch_fn_ ||
      !ip_batch_fn_ || !batch_fn_) {
    return kErrUnsupported;
  }
  return 0;
}

int PqInt4Quantizer::init(const IndexMeta &meta, const ailego::Params &params) {
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

  if (use_zero_mean_ && meta_.metric_name() != "SquaredEuclidean" &&
      meta_.metric_name() != "Cosine") {
    use_zero_mean_ = false;
  }

  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);
  meta_.set_extra_meta_size(extra_meta_size_);
  initialized_ = true;
  return 0;
}

// ---------------------------------------------------------------------------
// Simple Lloyd's KMeans for one chunk.
// ---------------------------------------------------------------------------

void PqInt4Quantizer::build_centroid_ptrs_cache() {
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
void PqInt4Quantizer::train_subquantizer(const T *data, size_t num,
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

  // Single-threaded pool — parallelism is at the chunk level.
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

int PqInt4Quantizer::train(IndexHolder::Pointer holder) {
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

  // Subsample if the dataset exceeds the training limit (aligned with
  // faiss/vsag: 256 centroids * 256 max_points_per_centroid ~= 65535).
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
        normalize(reinterpret_cast<ailego::Float16 *>(all_data.data()), num);
        break;
      case DataType::kFp32:
        normalize(reinterpret_cast<float *>(all_data.data()), num);
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
  auto threads =
      std::make_shared<SingleQueueIndexThreads>(thread_count_, false);
  auto task_group = threads->make_group();

  // Distribute chunks across threads.
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

  // Pre-build centroid pointer cache (needed by compute_dist_table).
  build_centroid_ptrs_cache();

  // Pre-compute SDC dist_table.
  compute_dist_table();

  // Pre-compute sub-centroid norms for the precomputed residual table.
  compute_sub_centroid_norms();
  return 0;
}

void PqInt4Quantizer::compute_dist_table() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  dist_table_.resize(static_cast<size_t>(num_chunk_) * k * k, 0.0f);

  // Centroid-to-centroid distances via the metric-aware batch_fn_:
  // L2:  dist_table[m][i][j] = ||c_m[i] - c_m[j]||^2
  // IP:  dist_table[m][i][j] = -dot(c_m[i], c_m[j])
  // Cosine: centroids trained on normalized data, uses L2.
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    float *table_m = dist_table_.data() + m * k * k;

    // Use pre-built centroid pointer cache.
    // const_cast: .data() returns const void* const* but batch_fn_
    // expects const void**.  The kernel never modifies the pointer array.
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    const void *centroid_i = centroid_ptrs[0];
    for (uint32_t i = 0; i < k; ++i) {
      batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                reinterpret_cast<const uint8_t *>(centroid_i) +
                    static_cast<size_t>(i) * d * element_size(),
                k, d, table_m + i * k);
    }
  }
}

void PqInt4Quantizer::compute_sub_centroid_norms() {
  const size_t k = kNumCentroids;
  sub_centroid_norms_.resize(static_cast<size_t>(num_chunk_) * k);

  // ||c_m[j]||^2 = dist(zero, c_m[j]): reuse the L2 batch kernel with a
  // zero query vector instead of hand-rolling a norm loop.
  std::vector<uint8_t> zero(static_cast<size_t>(sub_dim_) * element_size(), 0);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    // const_cast: see compute_dist_table for rationale.
    l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()), zero.data(),
                 k, sub_dim_, sub_centroid_norms_.data() + m * k);
  }
}

void PqInt4Quantizer::quantize_data(const void *input, void *output) const {
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
        normalize(reinterpret_cast<ailego::Float16 *>(norm_vec_storage.data()),
                  &vec_norm);
        break;
      case DataType::kFp32:
        normalize(reinterpret_cast<float *>(norm_vec_storage.data()),
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

void PqInt4Quantizer::quantize_query(const void *input, void *output) const {
  float *lut = reinterpret_cast<float *>(output);
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
        normalize(
            reinterpret_cast<ailego::Float16 *>(norm_query_storage.data()));
        break;
      case DataType::kFp32:
        normalize(reinterpret_cast<float *>(norm_query_storage.data()));
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
  // const_cast: see compute_dist_table for rationale.
  const uint8_t *query_bytes = reinterpret_cast<const uint8_t *>(query);
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    const void *sub_query =
        query_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size;
    batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub_query,
              kNumCentroids, sub_dim_, lut + m * kNumCentroids);
  }

  // Cosine: the LUT holds ||q_m - c_m[j]||^2 on L2-normalized vectors, and
  // ||q - c||^2 = 2 - 2*cos_sim = 2*(1 - cos_sim).  Scale by 0.5 so the ADC
  // sum yields cosine distance (1 - cos_sim) directly on every ADC path.
  if (meta_.metric_name() == "Cosine") {
    const size_t lut_size = static_cast<size_t>(num_chunk_) * kNumCentroids;
    for (size_t i = 0; i < lut_size; ++i) {
      lut[i] *= 0.5f;
    }
  }
}

float PqInt4Quantizer::calc_distance_dp_query(const void *dp,
                                              const void *query) const {
  // dp = packed pq_code (nibble codes)
  // query = LUT (float[num_chunk * 16])
  float d = 0.0f;
  adc_fn_(reinterpret_cast<const uint8_t *>(dp),
          reinterpret_cast<const float *>(query), num_chunk_, &d);
  // Cosine LUT is pre-scaled by 0.5 in quantize_query, so the ADC sum is
  // already the cosine distance — no conversion needed here.
  return d;
}

void PqInt4Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                   int dp_num,
                                                   const void *query,
                                                   float *dist_list) const {
  // ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: batch_adc_fn_ expects const void**; kernel is read-only.
  batch_adc_fn_(const_cast<const void **>(dp_list), query,
                static_cast<size_t>(dp_num), num_chunk_, dist_list);
  // Cosine LUT is pre-scaled by 0.5 in quantize_query, so the batch ADC sums
  // are already cosine distances — no conversion applied here.
}

float PqInt4Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  // Build LUT on the fly, then use ADC.
  std::vector<float> lut(static_cast<size_t>(num_chunk_) * kNumCentroids);
  quantize_query(query, lut.data());
  float d = 0.0f;
  adc_fn_(reinterpret_cast<const uint8_t *>(dp), lut.data(), num_chunk_, &d);
  return d;
}

void PqInt4Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::vector<float> lut(static_cast<size_t>(num_chunk_) * kNumCentroids);
  quantize_query(query, lut.data());
  // Use ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: see calc_distance_dp_query_batch for rationale.
  batch_adc_fn_(const_cast<const void **>(dp_list), lut.data(),
                static_cast<size_t>(dp_num), num_chunk_, dist_list);
}

float PqInt4Quantizer::calc_distance_dp_dp(const void *dp1,
                                           const void *dp2) const {
  float d = 0.0f;
  sdc_fn_(reinterpret_cast<const uint8_t *>(dp1),
          reinterpret_cast<const uint8_t *>(dp2), dist_table_.data(),
          num_chunk_, &d);
  return d;
}

int PqInt4Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
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

int PqInt4Quantizer::build_centroid_distance_table(const void *centroids,
                                                   size_t centroid_num,
                                                   std::string *table) const {
  //! The three-term decomposition below is pure L2 algebra.  IVF forces the
  //! residual meta's metric to L2 on both the build and the restore side, so
  //! any other metric here means that contract was broken.
  if (meta_.metric_name() != "SquaredEuclidean") {
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

  const size_t row_floats = static_cast<size_t>(num_chunk_) * kNumCentroids;
  //! Mimic faiss precomputed_table_max_bytes: refuse oversized tables so
  //! the caller can fall back to the per-list path.
  const size_t kMaxTableBytes = 1ULL << 30;
  if (centroid_num * row_floats * sizeof(float) > kMaxTableBytes) {
    return kErrUnsupported;
  }

  table->resize(centroid_num * row_floats * sizeof(float));
  float *tab = reinterpret_cast<float *>(&(*table)[0]);
  const uint32_t elem_size = element_size();

  auto build_for_type = [&](auto *typed_dummy) {
    using T = std::remove_pointer_t<decltype(typed_dummy)>;
    std::vector<T> buf(original_dim_);
    float dists[kNumCentroids];
    const T *src = reinterpret_cast<const T *>(centroids);
    for (size_t i = 0; i < centroid_num; ++i) {
      std::memcpy(buf.data(), src + i * original_dim_,
                  original_dim_ * sizeof(T));
      const uint8_t *buf_bytes = reinterpret_cast<const uint8_t *>(buf.data());
      float *row = tab + i * row_floats;
      for (uint32_t m = 0; m < num_chunk_; ++m) {
        //! term2 = ||c_m[j]||^2 + 2<c_i^m, c_m[j]>.  The IP kernel returns
        //! the negated inner product, i.e. dists[j] = -<c_i^m, c_m[j]>.
        const auto &centroid_ptrs = centroid_ptrs_cache_[m];
        // const_cast: see compute_dist_table for rationale.
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
  };

  switch (input_data_type_) {
    case DataType::kFp16:
      build_for_type(static_cast<ailego::Float16 *>(nullptr));
      break;
    case DataType::kFp32:
      build_for_type(static_cast<float *>(nullptr));
      break;
    default:
      return kErrUnsupported;
  }
  return 0;
}

int PqInt4Quantizer::quantize_precomputed_query(const void *query,
                                                const IndexQueryMeta &qmeta,
                                                std::string *out,
                                                IndexQueryMeta *ometa) const {
  //! Same L2-only contract as build_centroid_distance_table().
  if (meta_.metric_name() != "SquaredEuclidean") {
    return kErrUnsupported;
  }
  //! Same zero-mean restriction as build_centroid_distance_table().
  if (use_zero_mean_) {
    return kErrUnsupported;
  }

  // Validate unit_size against the input data type (same as quantize()).
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
  if (query == nullptr || out == nullptr ||
      qmeta.unit_size() != expected_unit || centroids_.empty()) {
    return kErrInvalidArgument;
  }

  const uint32_t elem_size = element_size();

  //! Preprocessing mirrors quantize_query() so the query lands in the same
  //! space as the codebook: Cosine normalization only (inert on the residual
  //! path, whose metric is intrinsically L2); zero-mean is gated out above.
  std::vector<uint8_t> norm_query_storage;
  const void *prep = query;
  if (meta_.metric_name() == "Cosine") {
    norm_query_storage.resize(original_dim_ * elem_size);
    std::memcpy(norm_query_storage.data(), query, original_dim_ * elem_size);
    switch (input_data_type_) {
      case DataType::kFp16:
        normalize(
            reinterpret_cast<ailego::Float16 *>(norm_query_storage.data()));
        break;
      case DataType::kFp32:
        normalize(reinterpret_cast<float *>(norm_query_storage.data()));
        break;
      default:
        break;
    }
    prep = norm_query_storage.data();
  }

  std::vector<uint8_t> centered_query_storage;
  if (use_zero_mean_) {
    centered_query_storage.resize(original_dim_ * elem_size);
    std::memcpy(centered_query_storage.data(), prep, original_dim_ * elem_size);
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
    prep = centered_query_storage.data();
  }

  //! term3 LUT: -2<q^m, c_m[j]>.  The IP kernel returns the negated inner
  //! product, i.e. dists[j] = -<q^m, c_m[j]>, so LUT = 2 * dists.  The
  //! merged LUT keeps the plain float[num_chunk * kNumCentroids] layout
  //! consumed by the ADC kernels.
  out->resize(quantized_query_vector_length());
  float *lut = reinterpret_cast<float *>(&(*out)[0]);
  const uint8_t *prep_bytes = reinterpret_cast<const uint8_t *>(prep);
  float dists[kNumCentroids];
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const uint8_t *sub =
        prep_bytes + static_cast<size_t>(m) * sub_dim_ * elem_size;
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    // const_cast: see compute_dist_table for rationale.
    ip_batch_fn_(const_cast<const void **>(centroid_ptrs.data()), sub,
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

int PqInt4Quantizer::merge_query_distance_table(
    const void *query_table, const std::string &centroid_table,
    size_t centroid_id, std::string *out) const {
  const size_t row_bytes = quantized_query_vector_length();
  if (query_table == nullptr || out == nullptr || num_chunk_ == 0 ||
      centroid_table.size() < (centroid_id + 1) * row_bytes) {
    return kErrInvalidArgument;
  }

  out->resize(row_bytes);
  const float *qtab = reinterpret_cast<const float *>(query_table);
  const float *ctab = reinterpret_cast<const float *>(centroid_table.data()) +
                      centroid_id * (row_bytes / sizeof(float));
  float *merged = reinterpret_cast<float *>(&(*out)[0]);
  const size_t floats = row_bytes / sizeof(float);
  //! term2 + term3: element-wise sum; term1 is added back by the caller.
  for (size_t i = 0; i < floats; ++i) {
    merged[i] = qtab[i] + ctab[i];
  }
  return 0;
}

int PqInt4Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                                std::string *out) const {
  (void)qmeta;
  const uint8_t *code = reinterpret_cast<const uint8_t *>(in);
  size_t byte_size = static_cast<size_t>(original_dim_) * sizeof(float);
  out->resize(byte_size);
  float *result = reinterpret_cast<float *>(&(*out)[0]);

  // Reconstruct by concatenating the selected centroids per chunk,
  // in the space the codebook was trained in (normalized for Cosine).
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  const uint32_t elem_size = element_size();
  for (uint32_t m = 0; m < num_chunk_; ++m) {
    const uint8_t *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * k * d * elem_size;
    // Unpack the 4-bit code: low nibble for even m, high nibble for odd m.
    uint8_t c = static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
    const uint8_t *centroid =
        centroids_m + static_cast<size_t>(c) * d * elem_size;
    switch (input_data_type_) {
      case DataType::kFp16: {
        const ailego::Float16 *src =
            reinterpret_cast<const ailego::Float16 *>(centroid);
        for (size_t j = 0; j < d; ++j) {
          result[m * d + j] = static_cast<float>(src[j]);
        }
        break;
      }
      case DataType::kFp32:
        std::memcpy(result + m * d, centroid, d * sizeof(float));
        break;
      default:
        break;
    }
  }

  // Undo zero-mean centering: add the centroid back FIRST (centering was
  // applied in normalized space during encode).
  if (use_zero_mean_) {
    for (uint32_t j = 0; j < original_dim_; ++j) {
      result[j] += centroid_[j];
    }
  }

  // For Cosine: rescale the un-centered unit-space vector back to the
  // original magnitude using the stored norm.
  if (meta_.metric_name() == "Cosine") {
    float norm = 0.0f;
    std::memcpy(&norm, code + packed_code_length(), sizeof(float));
    for (uint32_t j = 0; j < original_dim_; ++j) {
      result[j] *= norm;
    }
  }
  return 0;
}

DistanceImpl PqInt4Quantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  // ADC: CodebookAsymmetricDistanceFunc matches DistanceFunc directly (no
  // lambda needed).
  DistanceFunc adc_func = adc_fn_;

  // Batch ADC: ISA-dispatched batch4 kernel, no lambda needed.
  BatchDistanceFunc batch_func = batch_adc_fn_;

  // The query is already quantized (LUT) by the caller; copy it directly.
  size_t lut_bytes = quantized_query_vector_length();
  std::string lut_storage(static_cast<const char *>(query), lut_bytes);

  return DistanceImpl(std::move(adc_func), std::move(batch_func),
                      std::move(lut_storage), static_cast<size_t>(num_chunk_));
}

DistanceImpl PqInt4Quantizer::sym_distance(const void *query,
                                           const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  // SDC kernel needs a lambda: 5 parameters (extra dist_table pointer)
  // vs DistanceFunc's 4.
  auto sdc = sdc_fn_;
  const void *dt = dist_table_.data();
  DistanceFunc sdc_func = [sdc, dt](const void *a, const void *b, size_t dim,
                                    float *out) { sdc(a, b, dt, dim, out); };

  // The query is a packed PQ code, NOT a LUT: SDC compares two PQ codes via
  // the centroid-to-centroid dist_table_.
  size_t code_bytes = quantized_datapoint_vector_length();
  std::string code_storage(static_cast<const char *>(query), code_bytes);

  // SDC has no batch kernel — use the 3-arg constructor.
  return DistanceImpl(std::move(sdc_func), std::move(code_storage),
                      static_cast<size_t>(num_chunk_));
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
int PqInt4Quantizer::serialize(std::string *out) const {
  if (!out) return kErrUnsupported;

  QuantizerSerHeader hdr{};
  hdr.magic = kQuantizerMagic;
  hdr.version = kQuantizerSerVersion;
  hdr.quant_type = static_cast<uint16_t>(QuantizeType::kPQ);
  hdr.dim = original_dim_;
  hdr.metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));
  hdr.data_type = static_cast<uint16_t>(DataType::kInt4);

  PqInt4SerPayload payload{};
  payload.original_dim = original_dim_;
  payload.num_chunk = num_chunk_;
  payload.chunk_dim = sub_dim_;
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
  // dist_table_ is NOT serialized: it is a build-phase-only derivative of the
  // codebook (used by SDC), recomputable on demand and unneeded after
  // deserialization (search uses ADC).
  return 0;
}

int PqInt4Quantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int PqInt4Quantizer::deserialize(const void *data, size_t len) {
  if (!initialized_) return kErrUnsupported;
  if (len < sizeof(QuantizerSerHeader) + sizeof(PqInt4SerPayload)) {
    return kErrUnsupported;
  }

  const char *ptr = reinterpret_cast<const char *>(data);
  QuantizerSerHeader hdr;
  std::memcpy(&hdr, ptr, sizeof(hdr));
  ptr += sizeof(hdr);

  if (hdr.magic != kQuantizerMagic) return kErrUnsupported;
  // Reject foreign code types (e.g. int8 PQ blobs sharing quant_type == kPQ).
  // kInt4 == 0, so legacy blobs written before data_type existed stay loadable.
  if (hdr.data_type != static_cast<uint16_t>(DataType::kInt4)) {
    return kErrUnsupported;
  }
  // The codebook is only decodable under the metric it was trained in.
  if (hdr.metric !=
      static_cast<uint32_t>(metric_from_name(meta_.metric_name()))) {
    return kErrUnsupported;
  }

  PqInt4SerPayload payload;
  std::memcpy(&payload, ptr, sizeof(payload));
  ptr += sizeof(payload);

  original_dim_ = payload.original_dim;
  num_chunk_ = payload.num_chunk;
  sub_dim_ = payload.chunk_dim;

  // Restore input data type.  Old payloads have input_data_type == 0
  // (was reserved), which maps to kInt4 -- treat as kFp32 for compat.
  if (payload.input_data_type == 0 ||
      payload.input_data_type == static_cast<uint8_t>(DataType::kInt4) ||
      payload.input_data_type == static_cast<uint8_t>(DataType::kInt8)) {
    input_data_type_ = DataType::kFp32;
  } else {
    input_data_type_ = static_cast<DataType>(payload.input_data_type);
  }
  if (input_data_type_ != DataType::kFp16 &&
      input_data_type_ != DataType::kFp32) {
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
  if (len - sizeof(QuantizerSerHeader) - sizeof(PqInt4SerPayload) <
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
  // dist_table_ is intentionally not restored: SDC is only needed during
  // offline build, not after deserialization (search uses ADC).

  // Re-dispatch kernels and batch distance functions.
  if (setup_functions() != 0) {
    return kErrUnsupported;
  }

  // Set output meta: the quantized representation is INT4 codes with
  // num_chunk_ bytes (+ extra_meta_size_ for Cosine norm storage).
  meta_.set_meta(IndexMeta::DataType::DT_INT4, num_chunk_);
  meta_.set_extra_meta_size(extra_meta_size_);

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  // Pre-compute sub-centroid norms for the precomputed residual table.
  compute_sub_centroid_norms();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqInt4Quantizer);

// ---------------------------------------------------------------------------
// Template helper implementations (type-dispatched at call sites)
// ---------------------------------------------------------------------------

template <typename T>
void PqInt4Quantizer::normalize(T *data, size_t num) const {
  for (size_t i = 0; i < num; ++i) {
    float norm = 0.0f;
    ailego::Normalizer<T>::L2(data + i * original_dim_, original_dim_, &norm);
  }
}

template <typename T>
void PqInt4Quantizer::compute_and_subtract_center(T *data, size_t num) {
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
void PqInt4Quantizer::normalize(T *vec, float *norm_out) const {
  float norm = 0.0f;
  ailego::Normalizer<T>::L2(vec, original_dim_, &norm);
  if (norm_out) {
    *norm_out = norm;
  }
}

template <typename T>
void PqInt4Quantizer::subtract_center(T *vec) const {
  for (uint32_t d = 0; d < original_dim_; ++d) {
    vec[d] -= centroid_[d];
  }
}

}  // namespace turbo
}  // namespace zvec
