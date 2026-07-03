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
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_logger.h>

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

int PqInt8Quantizer::init(const IndexMeta &meta,
                          const ailego::Params &params) {
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
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * sub_dim_,
      0.0f);

  // Dispatch ISA kernels (scalar only for now).
  auto pq_k = get_pq_kernels(QuantizeType::kPQ);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;

  // LUT / dist_table computation reuses the metric-aware fp32 batch distance
  // function (already SIMD-optimized), no need for a hand-written kernel.
  fp32_batch_fn_ = get_batch_distance_func(
      metric_from_name(meta_.metric_name()), DataType::kFp32,
      QuantizeType::kDefault, CpuArchType::kAuto);

  meta_.set_meta(IndexMeta::DataType::DT_FP32, d);
  return 0;
}

// ---------------------------------------------------------------------------
// Simple Lloyd's KMeans for one sub-quantizer.
// ---------------------------------------------------------------------------
void PqInt8Quantizer::train_subquantizer(const float *data, size_t num,
                                         size_t stride, size_t sub_idx) {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  float *centroids_m =
      centroids_.data() + static_cast<size_t>(sub_idx) * k * d;

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
  // Duplicate if num < k (rare edge case).
  for (size_t i = n_init; i < k; ++i) {
    std::memcpy(centroids_m + i * d, centroids_m + (i % n_init) * d,
                d * sizeof(float));
  }

  // -- Lloyd iterations ---------------------------------------------------
  std::vector<uint32_t> assignments(num);
  std::vector<float> new_centroids(k * d);
  std::vector<uint32_t> counts(k);

  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    bool changed = false;

    // Assignment step.
    for (size_t i = 0; i < num; ++i) {
      const float *sub_vec =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(data) + i * stride) +
          sub_idx * d;

      float best_dist = std::numeric_limits<float>::max();
      uint32_t best_idx = 0;
      for (size_t c = 0; c < k; ++c) {
        float dist = 0.0f;
        const float *cent = centroids_m + c * d;
        for (size_t j = 0; j < d; ++j) {
          float diff = sub_vec[j] - cent[j];
          dist += diff * diff;
        }
        if (dist < best_dist) {
          best_dist = dist;
          best_idx = static_cast<uint32_t>(c);
        }
      }
      if (assignments[i] != best_idx) {
        changed = true;
        assignments[i] = best_idx;
      }
    }

    if (!changed) break;

    // Update step.
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

    // Copy back; keep old centroid for empty clusters.
    for (size_t c = 0; c < k; ++c) {
      if (counts[c] > 0) {
        std::memcpy(centroids_m + c * d, new_centroids.data() + c * d,
                    d * sizeof(float));
      }
    }
  }
}

int PqInt8Quantizer::train(IndexHolder::Pointer holder) {
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

  size_t data_stride = original_dim_ * sizeof(float);

  // Train each sub-quantizer independently.
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    train_subquantizer(all_data.data(), num, data_stride, m);
  }

  // Pre-compute SDC dist_table.
  compute_dist_table();

  return 0;
}

void PqInt8Quantizer::compute_dist_table() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  dist_table_.resize(
      static_cast<size_t>(num_subquantizers_) * k * k, 0.0f);

  // For each sub-quantizer, compute centroid-to-centroid distances via the
  // metric-aware fp32 batch distance function.
  std::vector<const void *> centroid_ptrs(k);
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m = centroids_.data() + m * k * d;
    float *table_m = dist_table_.data() + m * k * k;

    for (uint32_t c = 0; c < k; ++c) {
      centroid_ptrs[c] = centroids_m + c * d;
    }
    for (uint32_t i = 0; i < k; ++i) {
      fp32_batch_fn_(centroid_ptrs.data(),
                     reinterpret_cast<const void *>(centroids_m + i * d),
                     k, d, table_m + i * k);
    }
  }
}

size_t PqInt8Quantizer::find_nearest_centroid(const float *sub_vec,
                                              size_t sub_idx) const {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  const float *centroids_m =
      centroids_.data() + sub_idx * k * d;

  float best_dist = std::numeric_limits<float>::max();
  size_t best_idx = 0;
  for (size_t c = 0; c < k; ++c) {
    float dist = 0.0f;
    const float *cent = centroids_m + c * d;
    for (size_t j = 0; j < d; ++j) {
      float diff = sub_vec[j] - cent[j];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_idx = c;
    }
  }
  return best_idx;
}

void PqInt8Quantizer::quantize_data(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  uint8_t *code = reinterpret_cast<uint8_t *>(output);

  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    code[m] = static_cast<uint8_t>(
        find_nearest_centroid(vec + m * sub_dim_, m));
  }
}

void PqInt8Quantizer::quantize_query(const void *input, void *output) const {
  const float *query = reinterpret_cast<const float *>(input);
  float *lut = reinterpret_cast<float *>(output);

  // For each sub-quantizer, compute distance from query sub-vector to all
  // 256 centroids via the metric-aware fp32 batch distance function.
  std::vector<const void *> centroid_ptrs(kNumCentroids);
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m =
        centroids_.data() + static_cast<size_t>(m) * kNumCentroids * sub_dim_;
    for (uint32_t k = 0; k < kNumCentroids; ++k) {
      centroid_ptrs[k] = centroids_m + k * sub_dim_;
    }
    fp32_batch_fn_(centroid_ptrs.data(),
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
  return d;
}

void PqInt8Quantizer::calc_distance_dp_query_batch(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  for (int i = 0; i < dp_num; ++i) {
    float d = 0.0f;
    adc_fn_(reinterpret_cast<const uint8_t *>(dp_list[i]),
            reinterpret_cast<const float *>(query), num_subquantizers_, &d);
    dist_list[i] = d;
  }
}

float PqInt8Quantizer::calc_distance_dp_query_unquantized(
    const void *dp, const void *query) const {
  // Build LUT on the fly, then use ADC.
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  float d = 0.0f;
  adc_fn_(reinterpret_cast<const uint8_t *>(dp), lut.data(),
          num_subquantizers_, &d);
  return d;
}

void PqInt8Quantizer::calc_distance_dp_query_batch_unquantized(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  std::vector<float> lut(static_cast<size_t>(num_subquantizers_) *
                         kNumCentroids);
  quantize_query(query, lut.data());
  for (int i = 0; i < dp_num; ++i) {
    float d = 0.0f;
    adc_fn_(reinterpret_cast<const uint8_t *>(dp_list[i]), lut.data(),
            num_subquantizers_, &d);
    dist_list[i] = d;
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
                               std::string *out,
                               IndexQueryMeta *ometa) const {
  size_t lut_bytes = quantized_query_vector_length();
  out->resize(lut_bytes);
  quantize_query(query, &(*out)[0]);

  *ometa = qmeta;
  ometa->set_meta(IndexMeta::DataType::DT_FP32, original_dim_,
                  static_cast<uint32_t>(type_), 0);
  return 0;
}

DistanceImpl PqInt8Quantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  (void)qmeta;  // reserved for future use (e.g. data-type dispatch)
  // Build the LUT from the (already quantized) query.
  size_t lut_bytes = quantized_query_vector_length();
  std::string lut_storage(static_cast<const char *>(query), lut_bytes);

  // ADC function: DistanceFunc signature
  //   (const void *candidate_pq_code, const void *lut, size_t dim, float *out)
  // dim here is num_subquantizers (passed through DistanceImpl::dim()).
  auto nsq = num_subquantizers_;
  auto adc = adc_fn_;
  DistanceFunc adc_func = [nsq, adc](const void *cand, const void *lut,
                                      size_t /*dim*/, float *out) {
    adc(reinterpret_cast<const uint8_t *>(cand),
        reinterpret_cast<const float *>(lut), nsq, out);
  };

  // SDC (symmetric) function: captures dist_table + sdc kernel.
  auto sdc = sdc_fn_;
  const float *dt = dist_table_.data();
  DistanceFunc sdc_func = [nsq, sdc, dt](const void *a, const void *b,
                                          size_t /*dim*/, float *out) {
    sdc(reinterpret_cast<const uint8_t *>(a),
        reinterpret_cast<const uint8_t *>(b), dt, nsq, out);
  };

  // Batch ADC: iterate over candidates.
  BatchDistanceFunc batch_func =
      [nsq, adc](const void **candidates, const void *lut, size_t num,
                  size_t /*dim*/, float *out) {
        for (size_t i = 0; i < num; ++i) {
          adc(reinterpret_cast<const uint8_t *>(candidates[i]),
              reinterpret_cast<const float *>(lut), nsq, out + i);
        }
      };

  return DistanceImpl(std::move(adc_func), std::move(sdc_func),
                      std::move(batch_func), std::move(lut_storage),
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
  size_t dist_table_bytes = dist_table_.size() * sizeof(float);
  hdr.payload_size = static_cast<uint32_t>(
      sizeof(payload) + centroids_bytes + dist_table_bytes);

  out->clear();
  out->append(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
  out->append(reinterpret_cast<const char *>(&payload), sizeof(payload));
  out->append(reinterpret_cast<const char *>(centroids_.data()),
              centroids_bytes);
  out->append(reinterpret_cast<const char *>(dist_table_.data()),
              dist_table_bytes);
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

  size_t centroids_bytes =
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * sub_dim_ *
      sizeof(float);
  size_t dist_table_bytes =
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * kNumCentroids *
      sizeof(float);

  centroids_.resize(centroids_bytes / sizeof(float));
  std::memcpy(centroids_.data(), ptr, centroids_bytes);
  ptr += centroids_bytes;

  dist_table_.resize(dist_table_bytes / sizeof(float));
  std::memcpy(dist_table_.data(), ptr, dist_table_bytes);

  // Re-dispatch kernels.
  auto pq_k = get_pq_kernels(QuantizeType::kPQ);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;

  fp32_batch_fn_ = get_batch_distance_func(
      metric_from_name(meta_.metric_name()), DataType::kFp32,
      QuantizeType::kDefault, CpuArchType::kAuto);

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqInt8Quantizer);

}  // namespace turbo
}  // namespace zvec
