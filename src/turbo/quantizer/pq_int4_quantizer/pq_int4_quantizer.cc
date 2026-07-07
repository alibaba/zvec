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
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include <ailego/math/normalizer.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace turbo {

// ---------------------------------------------------------------------------
// PQ int4 serialization payload (follows the QuantizerSerHeader).
// ---------------------------------------------------------------------------
struct PqInt4SerPayload {
  uint32_t original_dim;
  uint32_t num_subquantizers;
  uint32_t sub_dim;
  uint32_t num_centroids;  // always 16 for int4
};

int PqInt4Quantizer::init(const IndexMeta &meta, const ailego::Params &params) {
  meta_ = meta;

  uint32_t d = meta.dimension();
  original_dim_ = d;

  // Read num_subquantizers from params (required).
  uint32_t nsq = 0;
  if (!params.get("num_subquantizers", &nsq) || nsq == 0) {
    LOG_ERROR("PqInt4Quantizer: num_subquantizers not set or zero");
    return kErrUnsupported;
  }
  if (d % nsq != 0) {
    LOG_ERROR(
        "PqInt4Quantizer: dim (%u) is not divisible by num_subquantizers (%u)",
        d, nsq);
    return kErrUnsupported;
  }
  // int4 codes are packed nibbles: 2 sub-quantizers per byte.
  // num_subquantizers MUST be even for clean byte packing.
  if (nsq % 2 != 0) {
    LOG_ERROR(
        "PqInt4Quantizer: num_subquantizers (%u) must be even for int4 "
        "packed nibble encoding",
        nsq);
    return kErrUnsupported;
  }

  num_subquantizers_ = nsq;
  sub_dim_ = d / nsq;

  // Pre-allocate centroids (filled by train()).
  centroids_.resize(
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * sub_dim_, 0.0f);

  // Dispatch ISA kernels: int4 packed nibble path.
  auto pq_k = get_pq_kernels(DataType::kInt4, QuantizeType::kPQ);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;
  batch_adc_fn_ = pq_k.batch_adc_distance;

  // Resolve the configured metric type.
  auto mt = metric_from_name(meta_.metric_name());

  // L2-only batch distance: always used for encoding (quantize_data) and
  // KMeans training (train_subquantizer).
  fp32_l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  // Metric-aware batch distance for search-side LUT and SDC table.
  fp32_batch_fn_ = get_batch_distance_func(
      mt, DataType::kFp32, QuantizeType::kDefault, CpuArchType::kAuto);

  // For Cosine: fall back to IP (normalization applied explicitly).
  if (meta_.metric_name() == "Cosine") {
    fp32_batch_fn_ =
        get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  }

  // For Cosine: reserve extra space to store the L2 norm (one float).
  if (meta_.metric_name() == "Cosine") {
    extra_meta_size_ = kExtraMetaSizeCosine;
    meta_.set_extra_meta_size(extra_meta_size_);
  }

  meta_.set_meta(IndexMeta::DataType::DT_FP32, d);
  return 0;
}

// ---------------------------------------------------------------------------
// Simple Lloyd's KMeans for one sub-quantizer (k=16).
// ---------------------------------------------------------------------------

void PqInt4Quantizer::build_centroid_ptrs_cache() {
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

void PqInt4Quantizer::train_subquantizer(const float *data, size_t num,
                                         size_t stride, size_t sub_idx) {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  float *centroids_m = centroids_.data() + static_cast<size_t>(sub_idx) * k * d;

  // -- Initialization: pick k distinct random data points -----------------
  std::mt19937 rng(42 + static_cast<uint32_t>(sub_idx));
  std::vector<size_t> perm(num);
  for (size_t i = 0; i < num; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);

  size_t n_init = std::min(static_cast<size_t>(k), num);
  for (size_t i = 0; i < n_init; ++i) {
    const float *src =
        reinterpret_cast<const float *>(
            reinterpret_cast<const uint8_t *>(data) + perm[i] * stride) +
        sub_idx * d;
    std::memcpy(centroids_m + i * d, src, d * sizeof(float));
  }
  for (size_t i = n_init; i < k; ++i) {
    std::memcpy(centroids_m + i * d, centroids_m + (i % n_init) * d,
                d * sizeof(float));
  }

  // -- Lloyd iterations ---------------------------------------------------
  std::vector<uint32_t> assignments(num);
  std::vector<float> new_centroids(k * d);
  std::vector<uint32_t> counts(k);

  std::vector<const void *> centroid_ptrs(k);
  for (size_t c = 0; c < k; ++c) {
    centroid_ptrs[c] = centroids_m + c * d;
  }
  std::vector<float> dists(k);

  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    bool changed = false;

    for (size_t i = 0; i < num; ++i) {
      const float *sub_vec =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(data) + i * stride) +
          sub_idx * d;

      fp32_l2_batch_fn_(centroid_ptrs.data(),
                        reinterpret_cast<const void *>(sub_vec), k, d,
                        dists.data());
      uint32_t best_idx = static_cast<uint32_t>(
          std::min_element(dists.begin(), dists.end()) - dists.begin());

      if (assignments[i] != best_idx) {
        changed = true;
        assignments[i] = best_idx;
      }
    }

    if (!changed) {
      LOG_INFO("  sub[%zu] converged at iter %u", sub_idx, iter + 1);
      break;
    }

    std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0);

    for (size_t i = 0; i < num; ++i) {
      const float *sub_vec =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(data) + i * stride) +
          sub_idx * d;
      uint32_t c = assignments[i];
      counts[c]++;
      float *cent = new_centroids.data() + c * d;
      for (size_t j = 0; j < d; ++j) {
        cent[j] += sub_vec[j];
      }
    }

    for (size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) continue;
      float inv = 1.0f / static_cast<float>(counts[c]);
      float *cent = new_centroids.data() + c * d;
      for (size_t j = 0; j < d; ++j) {
        cent[j] *= inv;
      }
    }

    for (size_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        std::memcpy(centroids_m + c * d, new_centroids.data() + c * d,
                    d * sizeof(float));
      }
    }
  }
}

int PqInt4Quantizer::train(IndexHolder::Pointer holder) {
  return train(holder, 1);
}

int PqInt4Quantizer::train(IndexHolder::Pointer holder, int thread_count) {
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

  // Subsample if the dataset exceeds the training limit.
  if (num > kMaxTrainVectors) {
    LOG_INFO("PQ int4 training: subsampling %zu -> %zu vectors", num,
             kMaxTrainVectors);
    std::mt19937 rng(42);
    for (size_t i = 0; i < kMaxTrainVectors; ++i) {
      std::uniform_int_distribution<size_t> dist(i, num - 1);
      size_t j = dist(rng);
      if (i != j) {
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

  // For Cosine: normalize training data to unit length.
  if (meta_.metric_name() == "Cosine") {
    for (size_t i = 0; i < num; ++i) {
      float *v = all_data.data() + i * original_dim_;
      float norm = 0.0f;
      ailego::Normalizer<float>::L2(v, original_dim_, &norm);
    }
  }

  thread_count =
      std::max(1, std::min(thread_count, static_cast<int>(num_subquantizers_)));

  LOG_INFO(
      "PQ int4 training: %zu vectors, dim=%u, nsq=%u, sub_dim=%u, "
      "max_iters=%u, threads=%d",
      num, original_dim_, num_subquantizers_, sub_dim_, kMaxKmeansIters,
      thread_count);

  if (thread_count == 1) {
    for (uint32_t m = 0; m < num_subquantizers_; ++m) {
      train_subquantizer(all_data.data(), num, data_stride, m);
      LOG_INFO("  sub-quantizer [%u/%u] done", m + 1, num_subquantizers_);
    }
  } else {
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t) {
      uint32_t m_begin = static_cast<uint32_t>(
          (static_cast<size_t>(t) * num_subquantizers_) / thread_count);
      uint32_t m_end = static_cast<uint32_t>(
          (static_cast<size_t>(t + 1) * num_subquantizers_) / thread_count);

      threads.emplace_back(
          [this, &all_data, num, data_stride, m_begin, m_end]() {
            for (uint32_t m = m_begin; m < m_end; ++m) {
              train_subquantizer(all_data.data(), num, data_stride, m);
            }
          });
    }

    for (auto &th : threads) {
      th.join();
    }

    LOG_INFO("  all %u sub-quantizers trained (%d threads)", num_subquantizers_,
             thread_count);
  }

  build_centroid_ptrs_cache();

  LOG_INFO("Computing SDC dist_table ...");
  compute_dist_table();

  LOG_INFO("PQ int4 training complete.");
  return 0;
}

void PqInt4Quantizer::compute_dist_table() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  dist_table_.resize(static_cast<size_t>(num_subquantizers_) * k * k, 0.0f);

  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m = centroids_.data() + m * k * d;
    float *table_m = dist_table_.data() + m * k * k;

    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    for (uint32_t i = 0; i < k; ++i) {
      fp32_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                     reinterpret_cast<const void *>(centroids_m + i * d), k, d,
                     table_m + i * k);
    }
  }
}

void PqInt4Quantizer::quantize_data(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  uint8_t *code = reinterpret_cast<uint8_t *>(output);

  // For Cosine: normalize the vector before encoding.
  std::vector<float> norm_vec_storage;
  float vec_norm = 0.0f;
  if (meta_.metric_name() == "Cosine") {
    norm_vec_storage.assign(vec, vec + original_dim_);
    ailego::Normalizer<float>::L2(norm_vec_storage.data(), original_dim_,
                                  &vec_norm);
    vec = norm_vec_storage.data();
  }

  // Clear packed nibble code buffer (num_subquantizers / 2 bytes).
  size_t code_bytes = static_cast<size_t>(num_subquantizers_) / 2;
  std::memset(code, 0, code_bytes);

  // Encode each sub-quantizer and pack into nibbles.
  float dists[kNumCentroids];

  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *sub_vec = vec + m * sub_dim_;
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];

    fp32_l2_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                      reinterpret_cast<const void *>(sub_vec), kNumCentroids,
                      sub_dim_, dists);

    // Argmin: find nearest centroid (0..15).
    float best_dist = dists[0];
    uint32_t best_idx = 0;
    for (uint32_t j = 1; j < kNumCentroids; ++j) {
      if (dists[j] < best_dist) {
        best_dist = dists[j];
        best_idx = j;
      }
    }

    // Pack nibble: even m -> low 4 bits; odd m -> high 4 bits.
    if (m & 1u) {
      code[m >> 1] |= static_cast<uint8_t>(best_idx << 4);
    } else {
      code[m >> 1] |= static_cast<uint8_t>(best_idx & 0x0Fu);
    }
  }

  // Store norm after PQ code for Cosine dequantize support.
  if (meta_.metric_name() == "Cosine") {
    float *norm_out = reinterpret_cast<float *>(code + code_bytes);
    *norm_out = vec_norm;
  }
}

void PqInt4Quantizer::quantize_query(const void *input, void *output) const {
  const float *query = reinterpret_cast<const float *>(input);
  float *lut = reinterpret_cast<float *>(output);

  // For Cosine: normalize query, then use IP as LUT metric.
  std::vector<float> norm_query_storage;
  if (meta_.metric_name() == "Cosine") {
    norm_query_storage.assign(query, query + original_dim_);
    float norm = 0.0f;
    ailego::Normalizer<float>::L2(norm_query_storage.data(), original_dim_,
                                  &norm);
    query = norm_query_storage.data();
  }

  // For each sub-quantizer, compute distance from query sub-vector to all
  // 16 centroids.  LUT stride = kNumCentroids = 16.
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    fp32_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                   reinterpret_cast<const void *>(query + m * sub_dim_),
                   kNumCentroids, sub_dim_, lut + m * kNumCentroids);
  }
}

float PqInt4Quantizer::calc_distance_dp_query(const void *dp,
                                              const void *query) const {
  float d = 0.0f;
  adc_fn_(dp, query, num_subquantizers_, &d);
  if (meta_.metric_name() == "Cosine") {
    d = 1.0f + d;
  }
  return d;
}

void PqInt4Quantizer::calc_distance_dp_query_batch(const void *const *dp_list,
                                                   int dp_num,
                                                   const void *query,
                                                   float *dist_list) const {
  batch_adc_fn_(const_cast<const void **>(dp_list), query,
                static_cast<size_t>(dp_num), num_subquantizers_, dist_list);
  if (meta_.metric_name() == "Cosine") {
    for (int i = 0; i < dp_num; ++i) {
      dist_list[i] = 1.0f + dist_list[i];
    }
  }
}

float PqInt4Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  float d = 0.0f;
  adc_fn_(dp, lut.data(), num_subquantizers_, &d);
  if (meta_.metric_name() == "Cosine") {
    d = 1.0f + d;
  }
  return d;
}

void PqInt4Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  batch_adc_fn_(const_cast<const void **>(dp_list), lut.data(),
                static_cast<size_t>(dp_num), num_subquantizers_, dist_list);
  if (meta_.metric_name() == "Cosine") {
    for (int i = 0; i < dp_num; ++i) {
      dist_list[i] = 1.0f + dist_list[i];
    }
  }
}

float PqInt4Quantizer::calc_distance_dp_dp(const void *dp1,
                                           const void *dp2) const {
  float d = 0.0f;
  sdc_fn_(dp1, dp2, dist_table_.data(), num_subquantizers_, &d);
  return d;
}

int PqInt4Quantizer::quantize(const void *query, const IndexQueryMeta &qmeta,
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

int PqInt4Quantizer::dequantize(const void *in, const IndexQueryMeta &qmeta,
                                std::string *out) const {
  (void)qmeta;
  const uint8_t *code = reinterpret_cast<const uint8_t *>(in);
  size_t byte_size = static_cast<size_t>(original_dim_) * sizeof(float);
  out->resize(byte_size);
  float *result = reinterpret_cast<float *>(&(*out)[0]);

  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * k * d;

    // Decode nibble index from packed code.
    uint8_t byte = code[m >> 1];
    uint8_t idx = (m & 1u) ? static_cast<uint8_t>(byte >> 4)
                           : static_cast<uint8_t>(byte & 0x0Fu);

    const float *centroid = centroids_m + static_cast<size_t>(idx) * d;
    std::memcpy(result + m * d, centroid, d * sizeof(float));
  }

  // For Cosine: rescale by stored norm.
  if (meta_.metric_name() == "Cosine") {
    size_t code_bytes = static_cast<size_t>(num_subquantizers_) / 2;
    float norm = 0.0f;
    std::memcpy(&norm, code + code_bytes, sizeof(float));
    for (uint32_t j = 0; j < original_dim_; ++j) {
      result[j] *= norm;
    }
  }
  return 0;
}

DistanceImpl PqInt4Quantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  DistanceFunc adc_func = adc_fn_;

  auto sdc = sdc_fn_;
  const void *dt = dist_table_.data();
  DistanceFunc sdc_func = [sdc, dt](const void *a, const void *b, size_t dim,
                                    float *out) { sdc(a, b, dt, dim, out); };

  BatchDistanceFunc batch_func = batch_adc_fn_;

  size_t lut_bytes = quantized_query_vector_length();
  std::string lut_storage(static_cast<const char *>(query), lut_bytes);

  return DistanceImpl(std::move(adc_func), std::move(sdc_func),
                      std::move(batch_func), std::move(lut_storage),
                      static_cast<size_t>(num_subquantizers_));
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------
int PqInt4Quantizer::serialize(std::string *out) const {
  if (!out) return kErrUnsupported;

  QuantizerSerHeader hdr{};
  hdr.magic = kQuantizerMagic;
  hdr.version = kQuantizerSerVersion;
  hdr.quant_type = static_cast<uint32_t>(QuantizeType::kPQ);
  hdr.dim = original_dim_;
  hdr.metric = static_cast<uint32_t>(metric_from_name(meta_.metric_name()));

  PqInt4SerPayload payload{};
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
  return 0;
}

int PqInt4Quantizer::deserialize(std::string &in) {
  return deserialize(in.data(), in.size());
}

int PqInt4Quantizer::deserialize(const void *data, size_t len) {
  if (len < sizeof(QuantizerSerHeader) + sizeof(PqInt4SerPayload)) {
    return kErrUnsupported;
  }

  const char *ptr = reinterpret_cast<const char *>(data);
  QuantizerSerHeader hdr;
  std::memcpy(&hdr, ptr, sizeof(hdr));
  ptr += sizeof(hdr);

  if (hdr.magic != kQuantizerMagic) return kErrUnsupported;

  PqInt4SerPayload payload;
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

  // Re-dispatch int4 kernels.
  auto pq_k = get_pq_kernels(DataType::kInt4, QuantizeType::kPQ);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;
  batch_adc_fn_ = pq_k.batch_adc_distance;

  fp32_l2_batch_fn_ =
      get_batch_distance_func(MetricType::kSquaredEuclidean, DataType::kFp32,
                              QuantizeType::kDefault, CpuArchType::kAuto);

  fp32_batch_fn_ = get_batch_distance_func(
      metric_from_name(meta_.metric_name()), DataType::kFp32,
      QuantizeType::kDefault, CpuArchType::kAuto);
  if (meta_.metric_name() == "Cosine") {
    fp32_batch_fn_ =
        get_batch_distance_func(MetricType::kInnerProduct, DataType::kFp32,
                                QuantizeType::kDefault, CpuArchType::kAuto);
  }

  build_centroid_ptrs_cache();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqInt4Quantizer);

}  // namespace turbo
}  // namespace zvec
