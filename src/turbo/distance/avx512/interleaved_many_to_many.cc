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
#include "avx512/interleaved_many_to_many.h"
#include <cstring>
#include <type_traits>
#include <zvec/ailego/utility/float_helper.h>
#include "common/interleaved_matrix.h"
#if defined(__AVX512F__) && (defined(__F16C__) || defined(_MSC_VER))
#include <immintrin.h>
#define ZVEC_INTERLEAVED_AVX512 1
#else
#define ZVEC_INTERLEAVED_AVX512 0
#endif

namespace zvec::turbo::avx512 {
namespace {
#if ZVEC_INTERLEAVED_AVX512
// Each lane accumulates one candidate; two queries share every matrix load.
// Small blocks use exact-width input loads and masked output stores.
template <size_t Rows>
constexpr __mmask16 lane_mask() {
  if constexpr (Rows >= 16) {
    return 0xffff;
  } else {
    return (1U << Rows) - 1;
  }
}

template <typename T, size_t Rows>
inline __m512 load_candidates(const T *p) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm512_maskz_loadu_ps(lane_mask<Rows>(), p);
  } else {
    __m256i half;
    if constexpr (Rows >= 16) {
      half = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    } else {
      __m128i low;
      if constexpr (Rows == 8) {
        low = _mm_loadu_si128(reinterpret_cast<const __m128i *>(p));
      } else if constexpr (Rows == 4) {
        low = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p));
      } else {
        int bits;
        std::memcpy(&bits, p, sizeof(bits));
        low = _mm_cvtsi32_si128(bits);
      }
      half = _mm256_zextsi128_si256(low);
    }
    return _mm512_cvtph_ps(half);
  }
}

template <typename T>
inline __m512 broadcast_query(const T *p) {
  if constexpr (std::is_same_v<T, float>) {
    return _mm512_set1_ps(*p);
  } else {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits))));
  }
}

template <typename T, MetricType Metric>
inline __m512 accumulate(__m512 sum, __m512 x, __m512 q) {
  if constexpr (std::is_same_v<T, ailego::Float16> &&
                Metric == MetricType::kInnerProduct) {
    // Every finite FP16 product is exact in FP32, including before addition.
    return _mm512_fnmadd_ps(x, q, sum);
  } else {
    return accumulate_matrix_lanes<Metric>(sum, x, q);
  }
}

template <size_t Rows>
inline void store_distances(float *out, __m512 sum) {
  _mm512_mask_storeu_ps(out, lane_mask<Rows>(), sum);
}

template <typename T, size_t Rows, MetricType Metric, bool Pair>
void accumulate_tile(const T *matrix, const T *q0, const T *q1, size_t dim,
                     float *out0, float *out1) {
  constexpr size_t Chunks = (Rows + 15) / 16;
  __m512 a[Chunks]{}, b[Chunks]{};
  for (size_t d = 0; d < dim; ++d) {
    const auto qa = broadcast_query(q0 + d);
    __m512 qb{};
    if constexpr (Pair) qb = broadcast_query(q1 + d);
    for (size_t c = 0; c < Chunks; ++c) {
      const auto x = load_candidates<T, Rows>(matrix + d * Rows + c * 16);
      a[c] = accumulate<T, Metric>(a[c], x, qa);
      if constexpr (Pair) b[c] = accumulate<T, Metric>(b[c], x, qb);
    }
  }
  for (size_t c = 0; c < Chunks; ++c) {
    store_distances<Rows>(out0 + c * 16, a[c]);
    if constexpr (Pair) store_distances<Rows>(out1 + c * 16, b[c]);
  }
}

template <typename T, size_t Rows, MetricType Metric, bool Pair>
void tile(const T *matrix, const T *q0, const T *q1, size_t dim, float *out0,
          float *out1) {
  constexpr auto compute = accumulate_tile<T, Rows, Metric, Pair>;
  if constexpr (std::is_same_v<T, ailego::Float16>) {
    fp16_matrix_tiles<Rows, Pair>(compute, matrix, q0, q1, dim, out0, out1);
  } else {
    compute(matrix, q0, q1, dim, out0, out1);
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
#if ZVEC_INTERLEAVED_AVX512
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
}  // namespace zvec::turbo::avx512
