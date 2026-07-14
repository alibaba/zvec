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

#include "quantizer/pq_int8_quantizer/pq_int8_quantizer.h"
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
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/framework/index_threads.h>

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// PQ serialization payload (follows the QuantizerSerHeader).
// ---------------------------------------------------------------------------
struct PqInt8SerPayload {
  uint32_t original_dim;
  uint32_t num_subquantizers;
  uint32_t sub_dim;
  uint32_t num_centroids;  // always 256 for int8
};

int PqInt8Quantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  meta_ = meta;

  uint32_t d = meta.dimension();
  original_dim_ = d;

  // Read num_subquantizers from params (required).
  uint32_t nsq = 0;
  if (!params.get("num_subquantizers", &nsq) || nsq == 0) {
    LOG_ERROR("PqInt8Quantizer: num_subquantizers not set or zero");
    return kErrUnsupported;
  }
  if (d % nsq != 0) {
    LOG_ERROR(
        "PqInt8Quantizer: dim (%u) is not divisible by num_subquantizers (%u)",
        d, nsq);
    return kErrUnsupported;
  }

  num_subquantizers_ = nsq;
  sub_dim_ = d / nsq;

  // Pre-allocate centroids (filled by train()).
  centroids_.resize(
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * sub_dim_, 0.0f);

  // Dispatch ISA kernels (scalar only for now).
  auto pq_k = get_pq_kernels(DataType::kInt8);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;
  batch_adc_fn_ = pq_k.batch_adc_distance;

  // Resolve the configured metric type (local only — not stored).
  auto mt = metric_from_name(meta_.metric_name());

  // L2-only batch distance: always used for encoding (quantize_data) and
  // KMeans training (train_subquantizer).  PQ codebook is trained in L2
  // space, so encoding must minimize L2 quantization error regardless
  // of the search metric.
  fp32_l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // Metric-aware batch distance: used for search-side LUT computation
  // (quantize_query) and SDC dist_table.  For IP/Cosine this computes
  // inner-product-based distances instead of L2.
  fp32_batch_fn_ = get_batch_distance_func(
      mt, DataType::kFp32, QuantizeType::kDefault, CpuArchType::kAuto);

  // For Cosine: fall back to IP for the metric-aware batch function,
  // because cosine = normalize + IP.  The normalization is applied
  // explicitly in quantize_query.
  if (meta_.metric_name() == "Cosine") {
    fp32_batch_fn_ =
        get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  }

  // For Cosine: reserve extra space to store the L2 norm (one float)
  // alongside each PQ code, enabling dequantize() to rescale.
  if (meta_.metric_name() == "Cosine") {
    extra_meta_size_ = kExtraMetaSizeCosine;
    meta_.set_extra_meta_size(extra_meta_size_);
  }

  // Read optional training params (aligned with multi_chunk_cluster)
  params.get("thread_count", &thread_count_);
  params.get("markov_chain_length", &markov_chain_length_);
  params.get("epsilon", &epsilon_);

  meta_.set_meta(IndexMeta::DataType::DT_FP32, d);
  return 0;
}

// ---------------------------------------------------------------------------
// Simple Lloyd's KMeans for one sub-quantizer.
// ---------------------------------------------------------------------------

void PqInt8Quantizer::build_centroid_ptrs_cache() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  centroid_ptrs_cache_.resize(num_subquantizers_);
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * k * d;
    auto &ptrs = centroid_ptrs_cache_[m];
    ptrs.resize(k);
    for (size_t c = 0; c < k; ++c) {
      ptrs[c] = centroids_m + c * d;
    }
  }
}

void PqInt8Quantizer::train_subquantizer(const float *data, size_t num,
                                         size_t stride, size_t sub_idx) {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  float *centroids_m = centroids_.data() + static_cast<size_t>(sub_idx) * k * d;

  // Create KMeans algorithm (L2 distance via NumericalKmeansContext).
  // Enable spherical mode for IP / Cosine so that centroids are updated
  // on the unit sphere, improving convergence for angular objectives.
  const bool spherical =
      (meta_.metric_name() == "InnerProduct" ||
       meta_.metric_name() == "Cosine");
  ailego::NumericalKmeans<float, SingleQueueIndexThreads> algorithm(
      k, d, spherical);

  // Append sub-vectors (NumericalKmeans handles transpose internally)
  for (size_t i = 0; i < num; ++i) {
    const float *sub_vec =
        reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(data) + i * stride) +
        sub_idx * d;
    algorithm.append(sub_vec, d);
  }

  // Single-threaded pool — parallelism is at sub-quantizer level
  // (aligned with multi_chunk_cluster.h L184-185)
  auto local_threads = std::make_shared<SingleQueueIndexThreads>(1, false);

  // KMC2 centroid initialization (aligned with multi_chunk_cluster.h L191-196)
  ailego::Kmc2CentroidsGenerator<
      ailego::NumericalKmeans<float, SingleQueueIndexThreads>,
      SingleQueueIndexThreads>
      gen;
  gen.set_chain_length(markov_chain_length_);
  gen.set_assumption_free(false);
  algorithm.init_centroids(*local_threads, gen);

  // Lloyd iterations (aligned with multi_chunk_cluster.h L200-216)
  double cost = 0.0;
  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    double old_cost = cost;
    bool result = algorithm.cluster_once(*local_threads, &cost);
    if (!result) {
      LOG_ERROR("sub[%zu] cluster_once failed at iter %u", sub_idx, iter);
      break;
    }
    double new_epsilon = std::abs(cost - old_cost);
    if (new_epsilon < epsilon_) {
      break;
    }
  }

  // Extract centroids into the flat centroids_ buffer
  const auto &cents = algorithm.centroids();
  for (size_t c = 0; c < cents.count(); ++c) {
    std::memcpy(centroids_m + c * d, cents[c], d * sizeof(float));
  }
}

int PqInt8Quantizer::train(IndexHolder::Pointer holder) {
  return train(holder, static_cast<int>(thread_count_));
}

int PqInt8Quantizer::train(IndexHolder::Pointer holder, int thread_count) {
  if (!holder) {
    return kErrUnsupported;
  }

  size_t num = holder->count();

  // Collect all data into a contiguous buffer.
  auto iter = holder->create_iterator();
  std::vector<float> all_data(num * original_dim_);
  size_t row = 0;
  for (; iter->is_valid(); iter->next(), ++row) {
    const float *src = reinterpret_cast<const float *>(iter->data());
    std::memcpy(all_data.data() + row * original_dim_, src,
                original_dim_ * sizeof(float));
  }

  // Subsample if the dataset exceeds the training limit (aligned with
  // faiss/vsag: 256 centroids * 256 max_points_per_centroid ≈ 65535).
  if (num > kMaxTrainVectors) {
    LOG_INFO("PQ training: subsampling %zu -> %zu vectors", num,
             kMaxTrainVectors);
    std::mt19937 rng(42);
    // Fisher-Yates partial shuffle: randomly place kMaxTrainVectors vectors
    // at the front of the buffer.
    for (size_t i = 0; i < kMaxTrainVectors; ++i) {
      std::uniform_int_distribution<size_t> dist(i, num - 1);
      size_t j = dist(rng);
      if (i != j) {
        // Swap full vectors (dim-sized chunks).
        for (uint32_t d = 0; d < original_dim_; ++d) {
          std::swap(all_data[i * original_dim_ + d],
                    all_data[j * original_dim_ + d]);
        }
      }
    }
    num = kMaxTrainVectors;
    all_data.resize(num * original_dim_);
    all_data.shrink_to_fit();
  }

  size_t data_stride = original_dim_ * sizeof(float);

  // For Cosine: normalize training data to unit length so that KMeans
  // centroids are learned in normalized space.  After normalization,
  // L2 minimization is equivalent to maximizing cosine similarity.
  if (meta_.metric_name() == "Cosine") {
    for (size_t i = 0; i < num; ++i) {
      float *v = all_data.data() + i * original_dim_;
      float norm = 0.0f;
      ailego::Normalizer<float>::L2(v, original_dim_, &norm);
    }
  }

  LOG_INFO(
      "PQ training: %zu vectors, dim=%u, nsq=%u, sub_dim=%u, "
      "max_iters=%u, threads=%d",
      num, original_dim_, num_subquantizers_, sub_dim_, kMaxKmeansIters,
      thread_count);

  // Create thread pool (aligned with multi_chunk_cluster L183-185)
  auto threads = std::make_shared<SingleQueueIndexThreads>(
      static_cast<uint32_t>(thread_count), false);
  auto task_group = threads->make_group();

  // Distribute sub-quantizers across threads (aligned with L198-201)
  std::atomic<size_t> finished{0};
  size_t pool_count = threads->count();
  for (size_t i = 0; i < pool_count; ++i) {
    task_group->submit(ailego::Closure::New(
        [this, &all_data, num, data_stride, i, pool_count, &finished]() {
          for (uint32_t m = static_cast<uint32_t>(i);
               m < num_subquantizers_;
               m += static_cast<uint32_t>(pool_count)) {
            train_subquantizer(all_data.data(), num, data_stride, m);
            finished++;
          }
        }));
  }
  task_group->wait_finish();

  LOG_INFO("  all %u sub-quantizers trained (%zu threads)",
           num_subquantizers_, pool_count);

  // Pre-build centroid pointer cache (needed by compute_dist_table).
  build_centroid_ptrs_cache();

  // Pre-compute SDC dist_table.
  LOG_INFO("Computing SDC dist_table ...");
  compute_dist_table();

  LOG_INFO("PQ training complete.");
  return 0;
}

void PqInt8Quantizer::compute_dist_table() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  dist_table_.resize(static_cast<size_t>(num_subquantizers_) * k * k, 0.0f);

  // For each sub-quantizer, compute centroid-to-centroid distances via
  // the metric-aware fp32 batch distance function (fp32_batch_fn_).
  // L2:  dist_table[m][i][j] = ||c_m[i] - c_m[j]||^2
  // IP:  dist_table[m][i][j] = -dot(c_m[i], c_m[j])
  // Cosine: centroids trained on normalized data, same as IP.
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m = centroids_.data() + m * k * d;
    float *table_m = dist_table_.data() + m * k * k;

    // Use pre-built centroid pointer cache.
    // const_cast: .data() returns const void* const* but fp32_batch_fn_
    // expects const void**.  The kernel never modifies the pointer array.
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    for (uint32_t i = 0; i < k; ++i) {
      fp32_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                     reinterpret_cast<const void *>(centroids_m + i * d), k, d,
                     table_m + i * k);
    }
  }
}

void PqInt8Quantizer::quantize_data(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  uint8_t *code = reinterpret_cast<uint8_t *>(output);

  // For Cosine: normalize the vector to unit length before encoding,
  // because the codebook was trained on normalized data.  The original
  // norm is stored after the PQ code for later dequantize().
  std::vector<float> norm_vec_storage;
  float vec_norm = 0.0f;
  if (meta_.metric_name() == "Cosine") {
    norm_vec_storage.assign(vec, vec + original_dim_);
    ailego::Normalizer<float>::L2(norm_vec_storage.data(), original_dim_,
                                  &vec_norm);
    vec = norm_vec_storage.data();
  }

  // Encode with L2-only batch distance (search-metric independent).
  // Process one sub-quantizer at a time, reusing a single 256-float
  // scratch buffer and fusing argmin into the distance loop to avoid
  // a second pass over the data.
  float dists[kNumCentroids];

  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *sub_vec = vec + m * sub_dim_;
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];

    // Compute L2 distances from this sub-vector to all 256 centroids.
    fp32_l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                      reinterpret_cast<const void *>(sub_vec), kNumCentroids,
                      sub_dim_, dists);

    // Argmin: find nearest centroid.
    float best_dist = dists[0];
    uint32_t best_idx = 0;
    for (uint32_t j = 1; j < kNumCentroids; ++j) {
      if (dists[j] < best_dist) {
        best_dist = dists[j];
        best_idx = j;
      }
    }
    code[m] = static_cast<uint8_t>(best_idx);
  }

  // Store norm after PQ code for Cosine dequantize support.
  if (meta_.metric_name() == "Cosine") {
    float *norm_out = reinterpret_cast<float *>(code + num_subquantizers_);
    *norm_out = vec_norm;
  }
}

void PqInt8Quantizer::quantize_query(const void *input, void *output) const {
  const float *query = reinterpret_cast<const float *>(input);
  float *lut = reinterpret_cast<float *>(output);

  // For Cosine: normalize the query to unit length first, then use IP
  // as the LUT metric.  After normalization, dot(q_norm, c) equals the
  // cosine similarity when centroids were trained on normalized data.
  // IP is additive across subspaces, so sum of sub-IPs = full IP.
  std::vector<float> norm_query_storage;
  if (meta_.metric_name() == "Cosine") {
    norm_query_storage.assign(query, query + original_dim_);
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(norm_query_storage.data(), original_dim_,
                                  &norm);
    query = norm_query_storage.data();
  }

  // For each sub-quantizer, compute distance from query sub-vector to all
  // 256 centroids via the metric-aware fp32 batch distance function.
  // For L2: LUT[m][j] = ||q_m - c_m[j]||^2
  // For IP: LUT[m][j] = -dot(q_m, c_m[j])
  // For Cosine: (query is normalized) LUT[m][j] = -dot(q_norm_m, c_m[j])
  // const_cast: see compute_dist_table for rationale.
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    fp32_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                   reinterpret_cast<const void *>(query + m * sub_dim_),
                   kNumCentroids, sub_dim_, lut + m * kNumCentroids);
  }
}

float PqInt8Quantizer::calc_distance_dp_query(const void *dp,
                                              const void *query) const {
  // dp = pq_code (uint8_t[num_subquantizers])
  // query = LUT (float[num_subquantizers * 256])
  float d = 0.0f;
  adc_fn_(reinterpret_cast<const uint8_t *>(dp),
          reinterpret_cast<const float *>(query), num_subquantizers_, &d);
  // For Cosine: ADC sum = -cos_sim; convert to cosine distance = 1 - cos_sim.
  if (meta_.metric_name() == "Cosine") {
    d = 1.0f + d;
  }
  return d;
}

void PqInt8Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                   int dp_num,
                                                   const void *query,
                                                   float *dist_list) const {
  // Use ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: dp_list is const void* const* (outer const from vtable
  // signature), but batch_adc_fn_ expects const void**.  Kernel is read-only
  // on the pointer array.
  batch_adc_fn_(const_cast<const void **>(dp_list), query,
                static_cast<size_t>(dp_num), num_subquantizers_, dist_list);
  // For Cosine: convert -cos_sim to cosine distance.
  if (meta_.metric_name() == "Cosine") {
    for (int i = 0; i < dp_num; ++i) {
      dist_list[i] = 1.0f + dist_list[i];
    }
  }
}

float PqInt8Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  // Build LUT on the fly, then use ADC.
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  float d = 0.0f;
  adc_fn_(reinterpret_cast<const uint8_t *>(dp), lut.data(), num_subquantizers_,
          &d);
  if (meta_.metric_name() == "Cosine") {
    d = 1.0f + d;
  }
  return d;
}

void PqInt8Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  // Use ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: see calc_distance_dp_query_batch for rationale.
  batch_adc_fn_(const_cast<const void **>(dp_list), lut.data(),
                static_cast<size_t>(dp_num), num_subquantizers_, dist_list);
  if (meta_.metric_name() == "Cosine") {
    for (int i = 0; i < dp_num; ++i) {
      dist_list[i] = 1.0f + dist_list[i];
    }
  }
}

float PqInt8Quantizer::calc_distance_dp_dp(const void *dp1,
                                           const void *dp2) const {
  float d = 0.0f;
  sdc_fn_(reinterpret_cast<const uint8_t *>(dp1),
          reinterpret_cast<const uint8_t *>(dp2), dist_table_.data(),
          num_subquantizers_, &d);
  return d;
}

int PqInt8Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
                              std::string *out, IndexQueryMeta *ometa) const {
  if (qmeta.unit_size() != sizeof(float)) {
    return kErrUnsupported;
  }

  size_t lut_bytes = quantized_query_vector_length();
  out->resize(lut_bytes);
  quantize_query(query, &(*out)[0]);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP32, original_dim_,
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

int PqInt8Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                                std::string *out) const {
  (void)qmeta;
  const uint8_t *code = reinterpret_cast<const uint8_t *>(in);
  size_t byte_size = static_cast<size_t>(original_dim_) * sizeof(float);
  out->resize(byte_size);
  float *result = reinterpret_cast<float *>(&(*out)[0]);

  // Reconstruct by concatenating the selected centroids from each
  // sub-quantizer.  This yields the PQ approximation of the vector
  // in the space the codebook was trained in (normalized for Cosine).
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * k * d;
    const float *centroid = centroids_m + static_cast<size_t>(code[m]) * d;
    std::memcpy(result + m * d, centroid, d * sizeof(float));
  }

  // For Cosine: the codebook was trained on normalized data and the
  // stored norm lets us rescale back to the original vector's magnitude.
  if (meta_.metric_name() == "Cosine") {
    float norm = 0.0f;
    std::memcpy(&norm, code + num_subquantizers_, sizeof(float));
    for (uint32_t j = 0; j < original_dim_; ++j) {
      result[j] *= norm;
    }
  }
  return 0;
}

DistanceImpl PqInt8Quantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  // ADC: PqAdcDistanceFunc signature now matches DistanceFunc directly
  // (both use void*), so we can assign without a lambda wrapper.
  DistanceFunc adc_func = adc_fn_;

  // Batch ADC: ISA-dispatched batch4 kernel, no lambda needed.
  BatchDistanceFunc batch_func = batch_adc_fn_;

  // The query is already quantized (LUT) by the caller (reset_query)
  // — copy it directly into DistanceImpl storage.
  size_t lut_bytes = quantized_query_vector_length();
  std::string lut_storage(static_cast<const char *>(query), lut_bytes);

  return DistanceImpl(std::move(adc_func), std::move(batch_func),
                      std::move(lut_storage),
                      static_cast<size_t>(num_subquantizers_));
}

DistanceImpl PqInt8Quantizer::sym_distance(const void *query,
                                           const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  // SDC kernel needs a lambda: 5 parameters (extra dist_table pointer)
  // vs DistanceFunc's 4.
  auto sdc = sdc_fn_;
  const void *dt = dist_table_.data();
  DistanceFunc sdc_func = [sdc, dt](const void *a, const void *b, size_t dim,
                                    float *out) { sdc(a, b, dt, dim, out); };

  // The query is a PQ code (uint8_t[num_subquantizers]), NOT a LUT.
  // SDC computes distance between two PQ codes via the centroid-to-centroid
  // dist_table_, so both sides must be PQ codes.
  size_t code_bytes = quantized_datapoint_vector_length();
  std::string code_storage(static_cast<const char *>(query), code_bytes);

  // SDC has no batch kernel — use the 3-arg constructor.
  return DistanceImpl(std::move(sdc_func), std::move(code_storage),
                      static_cast<size_t>(num_subquantizers_));
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
int PqInt8Quantizer::serialize(std::string *out) const {
  if (!out) return kErrUnsupported;

  QuantizerSerHeader hdr{};
  hdr.magic = kQuantizerMagic;
  hdr.version = kQuantizerSerVersion;
  hdr.quant_type = static_cast<uint32_t>(QuantizeType::kPQ);
  hdr.dim = original_dim_;
  hdr.metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));

  PqInt8SerPayload payload{};
  payload.original_dim = original_dim_;
  payload.num_subquantizers = num_subquantizers_;
  payload.sub_dim = sub_dim_;
  payload.num_centroids = kNumCentroids;

  size_t centroids_bytes = centroids_.size() * sizeof(float);
  hdr.payload_size = static_cast<uint32_t>(sizeof(payload) + centroids_bytes);

  out->clear();
  out->append(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out->append(reinterpret_cast<const char *>(&payload), sizeof(payload));
  out->append(reinterpret_cast<const char *>(centroids_.data()),
              centroids_bytes);
  // dist_table_ is NOT serialized: it is a build-phase-only derivative
  // of the codebook (used by SDC during graph construction).  Search
  // uses ADC exclusively, so the table can be recomputed on demand if
  // ever needed, but is unnecessary after deserialization.
  return 0;
}

int PqInt8Quantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int PqInt8Quantizer::deserialize(const void *data, size_t len) {
  if (len < sizeof(QuantizerSerHeader) + sizeof(PqInt8SerPayload)) {
    return kErrUnsupported;
  }

  const char *ptr = reinterpret_cast<const char *>(data);
  QuantizerSerHeader hdr;
  std::memcpy(&hdr, ptr, sizeof(hdr));
  ptr += sizeof(hdr);

  if (hdr.magic != kQuantizerMagic) return kErrUnsupported;

  PqInt8SerPayload payload;
  std::memcpy(&payload, ptr, sizeof(payload));
  ptr += sizeof(payload);

  original_dim_ = payload.original_dim;
  num_subquantizers_ = payload.num_subquantizers;
  sub_dim_ = payload.sub_dim;

  meta_.set_meta(IndexMeta::DataType::DT_FP32, original_dim_);

  size_t centroids_bytes = static_cast<size_t>(num_subquantizers_) *
                           kNumCentroids * sub_dim_ * sizeof(float);

  centroids_.resize(centroids_bytes / sizeof(float));
  std::memcpy(centroids_.data(), ptr, centroids_bytes);
  // dist_table_ is intentionally not restored: SDC is only needed during
  // offline build, not after deserialization (search uses ADC).

  // Re-dispatch kernels.
  auto pq_k = get_pq_kernels(DataType::kInt8);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;
  batch_adc_fn_ = pq_k.batch_adc_distance;

  // L2-only batch distance for encoding (always L2 regardless of metric).
  fp32_l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // Metric-aware batch distance for search LUT.  For Cosine, use IP
  // (normalization is applied explicitly in quantize_query).
  fp32_batch_fn_ = get_batch_distance_func(
      metric_from_name(meta_.metric_name()), DataType::kFp32,
      QuantizeType::kDefault, CpuArchType::kAuto);
  if (meta_.metric_name() == "Cosine") {
    fp32_batch_fn_ =
        get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  }

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqInt8Quantizer);

}  // namespace turbo
}  // namespace zvec
