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
#include "avx2/interleaved_many_to_many.h"
#include <cstring>
#include <type_traits>
#include <zvec/ailego/utility/float_helper.h>
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_AVX2 && (defined(__FMA__) || defined(_MSC_VER))
#include <immintrin.h>
#define ZVEC_INTERLEAVED_AVX2 1
#else
#define ZVEC_INTERLEAVED_AVX2 0
#endif

namespace zvec::turbo::avx2 {
namespace {
#if ZVEC_INTERLEAVED_AVX2
// Each SIMD lane is one candidate, so no transpose or horizontal reduction
// is needed. Narrow blocks use exact-width loads and masked output stores.
template <size_t Rows>
inline __m256i lane_mask() {
  return _mm256_setr_epi32(-1, -1, Rows >= 4 ? -1 : 0, Rows >= 4 ? -1 : 0, 0, 0,
                           0, 0);
}

template <typename T, size_t Rows>
inline __m256 load_candidates(const T *p) {
  if constexpr (std::is_same_v<T, float>) {
    if constexpr (Rows >= 8)
      return _mm256_loadu_ps(p);
    else
      return _mm256_maskload_ps(p, lane_mask<Rows>());
  } else {
    __m128i half;
    if constexpr (Rows >= 8) {
      half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
    } else if constexpr (Rows == 4) {
      half = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
    } else {
      int bits;
      std::memcpy(&bits, p, sizeof(bits));
      half = _mm_cvtsi32_si128(bits);
    }
    return _mm256_cvtph_ps(half);
  }
}

template <typename T>
inline __m256 broadcast_query(const T *p) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm256_broadcast_ss(p);
  } else {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(bits)));
  }
}

template <MetricType Metric>
inline __m256 accumulate(__m256 sum, __m256 x, __m256 q) {
  if constexpr (Metric == MetricType::kInnerProduct) {
    return _mm256_fnmadd_ps(x, q, sum);
  } else {
    const auto delta = _mm256_sub_ps(x, q);
    return _mm256_fmadd_ps(delta, delta, sum);
  }
}

template <size_t Rows>
inline void store_distances(float *out, __m256 sum) {
  if constexpr (Rows >= 8)
    _mm256_storeu_ps(out, sum);
  else
    _mm256_maskstore_ps(out, lane_mask<Rows>(), sum);
}

template <typename T, size_t Rows, MetricType Metric, bool Pair>
void tile(const T *matrix, const T *q0, const T *q1, size_t dim, float *out0,
          float *out1) {
  constexpr size_t Chunks = (Rows + 7) / 8;
  __m256 a[Chunks]{}, b[Chunks]{};
  for (size_t d = 0; d < dim; ++d) {
    const auto qa = broadcast_query(q0 + d);
    __m256 qb{};
    if constexpr (Pair) qb = broadcast_query(q1 + d);
    for (size_t c = 0; c < Chunks; ++c) {
      const auto x = load_candidates<T, Rows>(matrix + d * Rows + c * 8);
      a[c] = accumulate<Metric>(a[c], x, qa);
      if constexpr (Pair) b[c] = accumulate<Metric>(b[c], x, qb);
    }
  }
  for (size_t c = 0; c < Chunks; ++c) {
    store_distances<Rows>(out0 + c * 8, a[c]);
    if constexpr (Pair) store_distances<Rows>(out1 + c * 8, b[c]);
  }
}

template <typename T, size_t Rows, MetricType Metric>
InterleavedManyToManyDistanceFunc kernel() {
  return [](const void *matrix, const void *const *queries, size_t n,
            size_t dim, float *out) {
    const auto *m = static_cast<const T *>(matrix);
    size_t q = 0;
    for (; q + 1 < n; q += 2) {
      tile<T, Rows, Metric, true>(m, static_cast<const T *>(queries[q]),
                                  static_cast<const T *>(queries[q + 1]), dim,
                                  out + q * Rows, out + (q + 1) * Rows);
    }
    if (q < n) {
      tile<T, Rows, Metric, false>(m, static_cast<const T *>(queries[q]),
                                   nullptr, dim, out + q * Rows, nullptr);
    }
  };
}

template <typename T, MetricType Metric>
InterleavedManyToManyDistanceFunc select_rows(size_t rows) {
  switch (rows) {
    case 2:
      return kernel<T, 2, Metric>();
    case 4:
      return kernel<T, 4, Metric>();
    case 8:
      return kernel<T, 8, Metric>();
    case 16:
      return kernel<T, 16, Metric>();
    case 32:
      return kernel<T, 32, Metric>();
    default:
      return {};
  }
}

template <typename T>
InterleavedManyToManyDistanceFunc select_metric(MetricType metric,
                                                size_t rows) {
  if (metric == MetricType::kInnerProduct)
    return select_rows<T, MetricType::kInnerProduct>(rows);
  if (metric == MetricType::kSquaredEuclidean)
    return select_rows<T, MetricType::kSquaredEuclidean>(rows);
  return {};
}
#endif
}  // namespace

InterleavedManyToManyDistanceFunc make_interleaved_many_to_many_distance(
    MetricType metric, DataType type, size_t rows) {
#if ZVEC_INTERLEAVED_AVX2
  if (type == DataType::kFp32) return select_metric<float>(metric, rows);
  if (type == DataType::kFp16)
    return select_metric<ailego::Float16>(metric, rows);
#else
  (void)metric;
  (void)type;
  (void)rows;
#endif
  return {};
}
}  // namespace zvec::turbo::avx2
