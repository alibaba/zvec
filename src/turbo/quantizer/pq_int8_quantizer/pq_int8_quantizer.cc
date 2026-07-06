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
#include <thread>
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
  batch_adc_fn_ = pq_k.batch_adc_distance;

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

void PqInt8Quantizer::build_centroid_ptrs_cache() {
  const size_t k = kNumCentroids;
  const size_t d = sub_dim_;
  centroid_ptrs_cache_.resize(num_subquantizers_);
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m = centroids_.data() + static_cast<size_t>(m) * k * d;
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

  // Pre-build centroid pointer array for SIMD batch distance.
  std::vector<const void *> centroid_ptrs(k);
  for (size_t c = 0; c < k; ++c) {
    centroid_ptrs[c] = centroids_m + c * d;
  }
  std::vector<float> dists(k);

  for (uint32_t iter = 0; iter < kMaxKmeansIters; ++iter) {
    bool changed = false;

    // Assignment step — use SIMD-accelerated fp32_batch_fn_ instead of
    // scalar L2 loop.  Each call computes distances from one sub-vector
    // to all k centroids in one batch.
    for (size_t i = 0; i < num; ++i) {
      const float *sub_vec =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(data) + i * stride) +
          sub_idx * d;

      fp32_batch_fn_(centroid_ptrs.data(),
                     reinterpret_cast<const void *>(sub_vec),
                     k, d, dists.data());
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
  return train(holder, 1);
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

  size_t data_stride = original_dim_ * sizeof(float);

  // Clamp thread count to [1, num_subquantizers_].
  thread_count = std::max(1, std::min(thread_count,
                                       static_cast<int>(num_subquantizers_)));

  LOG_INFO("PQ training: %zu vectors, dim=%u, nsq=%u, sub_dim=%u, "
           "max_iters=%u, threads=%d",
           num, original_dim_, num_subquantizers_, sub_dim_,
           kMaxKmeansIters, thread_count);

  if (thread_count == 1) {
    // Single-threaded path.
    for (uint32_t m = 0; m < num_subquantizers_; ++m) {
      train_subquantizer(all_data.data(), num, data_stride, m);
      LOG_INFO("  sub-quantizer [%u/%u] done",
               m + 1, num_subquantizers_);
    }
  } else {
    // Multi-threaded path: each thread handles a contiguous range of
    // sub-quantizers.  Sub-quantizers are fully independent — no
    // synchronization needed.
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int t = 0; t < thread_count; ++t) {
      uint32_t m_begin = static_cast<uint32_t>(
          (static_cast<size_t>(t) * num_subquantizers_) / thread_count);
      uint32_t m_end = static_cast<uint32_t>(
          (static_cast<size_t>(t + 1) * num_subquantizers_) / thread_count);

      threads.emplace_back([this, &all_data, num, data_stride,
                            m_begin, m_end]() {
        for (uint32_t m = m_begin; m < m_end; ++m) {
          train_subquantizer(all_data.data(), num, data_stride, m);
        }
      });
    }

    for (auto &th : threads) {
      th.join();
    }

    LOG_INFO("  all %u sub-quantizers trained (%d threads)",
             num_subquantizers_, thread_count);
  }

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
  dist_table_.resize(
      static_cast<size_t>(num_subquantizers_) * k * k, 0.0f);

  // For each sub-quantizer, compute centroid-to-centroid distances via the
  // metric-aware fp32 batch distance function.
  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *centroids_m = centroids_.data() + m * k * d;
    float *table_m = dist_table_.data() + m * k * k;

    // Use pre-built centroid pointer cache.
    // const_cast: .data() returns const void* const* but fp32_batch_fn_
    // expects const void**.  The kernel never modifies the pointer array.
    const auto &centroid_ptrs = centroid_ptrs_cache_[m];
    for (uint32_t i = 0; i < k; ++i) {
      fp32_batch_fn_(const_cast<const void **>(centroid_ptrs.data()),
                     reinterpret_cast<const void *>(centroids_m + i * d),
                     k, d, table_m + i * k);
    }
  }
}

void PqInt8Quantizer::quantize_data(const void *input, void *output) const {
  const float *vec = reinterpret_cast<const float *>(input);
  uint8_t *code = reinterpret_cast<uint8_t *>(output);

  // Reuse quantize_query to build LUT (distances from each sub-vector to
  // all 256 centroids), then argmin each sub to get the PQ code.
  // This shares the same fp32_batch_fn_ + centroid_ptrs_cache_ path as
  // LUT computation, eliminating the separate find_nearest_centroid.
  const size_t lut_size = static_cast<size_t>(num_subquantizers_) * kNumCentroids;
  std::vector<float> lut(lut_size);
  quantize_query(vec, lut.data());

  for (uint32_t m = 0; m < num_subquantizers_; ++m) {
    const float *row = lut.data() + m * kNumCentroids;
    code[m] = static_cast<uint8_t>(
        std::min_element(row, row + kNumCentroids) - row);
  }
}

void PqInt8Quantizer::quantize_query(const void *input, void *output) const {
  const float *query = reinterpret_cast<const float *>(input);
  float *lut = reinterpret_cast<float *>(output);

  // For each sub-quantizer, compute distance from query sub-vector to all
  // 256 centroids via the metric-aware fp32 batch distance function.
  // Uses pre-built centroid pointer cache to avoid per-sub allocations.
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
  return d;
}

void PqInt8Quantizer::calc_distance_dp_query_batch(
    const void *const *dp_list, int dp_num, const void *query,
    float *dist_list) const {
  // Use ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: dp_list is const void* const* (outer const from vtable
  // signature), but batch_adc_fn_ expects const void**.  Kernel is read-only
  // on the pointer array.
  batch_adc_fn_(const_cast<const void **>(dp_list), query,
                static_cast<size_t>(dp_num),
                num_subquantizers_, dist_list);
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
  // Use ISA-dispatched batch4 ADC kernel (4-way ILP + SIMD gather).
  // const_cast: see calc_distance_dp_query_batch for rationale.
  batch_adc_fn_(const_cast<const void **>(dp_list), lut.data(),
                static_cast<size_t>(dp_num),
                num_subquantizers_, dist_list);
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

DistanceImpl PqInt8Quantizer::distance(const void *query,
                                       const IndexQueryMeta &qmeta) const {
  (void)qmeta;

  // ADC: PqAdcDistanceFunc signature now matches DistanceFunc directly
  // (both use void*), so we can assign without a lambda wrapper.
  DistanceFunc adc_func = adc_fn_;

  // SDC: still needs a lambda because the kernel has 5 parameters
  // (extra dist_table pointer), vs DistanceFunc's 4.
  auto sdc = sdc_fn_;
  const void *dt = dist_table_.data();
  DistanceFunc sdc_func = [sdc, dt](const void *a, const void *b,
                                     size_t dim, float *out) {
    sdc(a, b, dt, dim, out);
  };

  // Batch ADC: ISA-dispatched batch4 kernel, no lambda needed.
  BatchDistanceFunc batch_func = batch_adc_fn_;

  // The query is already quantized (LUT) by the caller (reset_query)
  // — copy it directly into DistanceImpl storage.
  size_t lut_bytes = quantized_query_vector_length();
  std::string lut_storage(static_cast<const char *>(query), lut_bytes);

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

  size_t centroids_bytes =
      static_cast<size_t>(num_subquantizers_) * kNumCentroids * sub_dim_ *
      sizeof(float);

  centroids_.resize(centroids_bytes / sizeof(float));
  std::memcpy(centroids_.data(), ptr, centroids_bytes);
  // dist_table_ is intentionally not restored: SDC is only needed during
  // offline build, not after deserialization (search uses ADC).

  // Re-dispatch kernels.
  auto pq_k = get_pq_kernels(QuantizeType::kPQ);
  adc_fn_ = pq_k.adc_distance;
  sdc_fn_ = pq_k.sdc_distance;
  batch_adc_fn_ = pq_k.batch_adc_distance;

  fp32_batch_fn_ = get_batch_distance_func(
      metric_from_name(meta_.metric_name()), DataType::kFp32,
      QuantizeType::kDefault, CpuArchType::kAuto);

  // Pre-build centroid pointer cache for fast encode/search.
  build_centroid_ptrs_cache();

  return 0;
}

INDEX_FACTORY_REGISTER_QUANTIZER(PqInt8Quantizer);

}  // namespace turbo
}  // namespace zvec
