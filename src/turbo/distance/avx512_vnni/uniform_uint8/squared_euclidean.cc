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

// AVX512-VNNI optimized squared L2 for uniform uint8 quantization.
//
// Stored record layout: [dim int8(code - 128) | uint32 sum_sq(raw code)].
// Canonical query layout matches records. Query preprocessing converts a
// private copy to:      [dim raw uint8 code    | int32 query correction].
//
// Build-time pairwise distance uses true L2 between two stored shifted
// vectors. Search-time batch distance includes the query-only correction:
//   distance = sum_sq(record_raw)
//            - 2 * dot(record_shifted, query_raw)
//            + sum_sq(query_raw) - 256 * sum(query_raw)
//            = ||record_raw - query_raw||^2
// VNNI uses vpdpbusd's unsigned x signed contract as:
//   dot(record_shifted, query_raw) = dpbusd(query_raw, record_shifted)
//
// Batch kernel design (hot path for graph search):
//   - four records per block with independent accumulators
//   - software prefetch of future records, including the metadata tail
//   - SIMD horizontal reduction and final score calculation for four records

#include "avx512_vnni/uniform_uint8/squared_euclidean.h"
#include <cstdint>
#include <cstring>
#include <limits>
#include "zvec/ailego/internal/platform.h"

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512_vnni {

namespace {

constexpr size_t kTailBytes = sizeof(uint32_t);
// The uint32 norm and int32 VNNI dot product are both lossless through the
// public 65,536-dimension limit. Larger direct calls use squared differences.
constexpr size_t kMaxIdentityDimension = 65536;
static_assert(uint64_t{kMaxIdentityDimension} * 255 * 255 <=
              (std::numeric_limits<uint32_t>::max)());
static_assert(uint64_t{kMaxIdentityDimension} * 255 * 128 <=
              (std::numeric_limits<int32_t>::max)());
static_assert(uint64_t{kMaxIdentityDimension} * 128 * 128 <=
              (std::numeric_limits<int32_t>::max)());

static inline size_t original_dim(size_t encoded_dim) {
  return encoded_dim > kTailBytes ? encoded_dim - kTailBytes : 0;
}

static inline void uniform_sq_l2_uint8_scalar_single(const void *vector,
                                                     const uint8_t *raw_query,
                                                     size_t orig_dim,
                                                     float *distance) {
  const auto *record = reinterpret_cast<const int8_t *>(vector);
  int64_t result = 0;
  for (size_t d = 0; d < orig_dim; ++d) {
    const int difference =
        static_cast<int>(record[d]) - (static_cast<int>(raw_query[d]) - 128);
    result += static_cast<int64_t>(difference) * difference;
  }
  *distance = static_cast<float>(result);
}

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

static inline uint32_t load_extra_value(const void *extra_values) {
  ailego_assert_with(extra_values != nullptr,
                     "UniformUint8 batch distance requires extra values");
  uint32_t value = 0;
  std::memcpy(&value, extra_values, sizeof(value));
  return value;
}

static inline int32_t query_correction(const void *query, size_t orig_dim) {
  int32_t value = 0;
  std::memcpy(&value, reinterpret_cast<const uint8_t *>(query) + orig_dim,
              sizeof(value));
  return value;
}

// Convert the low four uint32 lanes with AVX512F's native unsigned
// conversion. The upper lanes are unused.
static ailego_force_inline __m128 uint32_to_float(__m128i values) {
  const __m512 converted = _mm512_cvtepu32_ps(_mm512_castsi128_si512(values));
  return _mm512_castps512_ps128(converted);
}

// Sign-extend 32 stored int8 values, subtract without int8 overflow, and use
// VNNI's signed-word dot product to accumulate pairs of squared differences
// into 16 int32 lanes. The full stored range [-128, 127] produces differences
// in [-255, 255], so each int16 product remains exact.
static ailego_force_inline __m512i squared_diff_32(__m512i accumulator,
                                                   const int8_t *lhs,
                                                   const int8_t *rhs) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(lhs)));
  const __m512i rhs16 = _mm512_cvtepi8_epi16(
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(rhs)));
  const __m512i difference16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(accumulator, difference16, difference16);
}

static ailego_force_inline __m512i squared_diff_masked_32(__m512i accumulator,
                                                          const int8_t *lhs,
                                                          const int8_t *rhs,
                                                          __mmask32 mask) {
  const __m512i lhs16 = _mm512_cvtepi8_epi16(
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(lhs)));
  const __m512i rhs16 = _mm512_cvtepi8_epi16(
      _mm256_maskz_loadu_epi8(mask, static_cast<const void *>(rhs)));
  const __m512i difference16 = _mm512_sub_epi16(lhs16, rhs16);
  return _mm512_dpwssd_epi32(accumulator, difference16, difference16);
}

// Widen before the horizontal sum. A uint8 squared distance can exceed
// INT32_MAX (for example 65,536 * 255^2), even though every accumulator lane
// remains in range between periodic flushes.
static ailego_force_inline int64_t
reduce_add_epi32_to_int64(__m512i accumulator) {
  const __m256i low32 = _mm512_castsi512_si256(accumulator);
  const __m256i high32 = _mm512_extracti64x4_epi64(accumulator, 1);
  const __m512i low64 = _mm512_cvtepi32_epi64(low32);
  const __m512i high64 = _mm512_cvtepi32_epi64(high32);
  return _mm512_reduce_add_epi64(_mm512_add_epi64(low64, high64));
}

#endif

}  // namespace

void uniform_squared_euclidean_uint8_distance(const void *a, const void *b,
                                              size_t dim, float *distance) {
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    *distance = 0.0f;
    return;
  }

  const auto *lhs = reinterpret_cast<const int8_t *>(a);
  const auto *rhs = reinterpret_cast<const int8_t *>(b);

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  // Four dependency chains cover 128 bytes per iteration. Each VPDPWSSD lane
  // receives two squares, whose maximum contribution is 2 * 255^2. Flush
  // every 8,192 iterations so each int32 lane stays below 1.1 billion.
  constexpr size_t kBlockBytes = 32;
  constexpr size_t kUnrolledBytes = 4 * kBlockBytes;
  constexpr size_t kFlushIterations = 8192;

  __m512i accumulator0 = _mm512_setzero_si512();
  __m512i accumulator1 = _mm512_setzero_si512();
  __m512i accumulator2 = _mm512_setzero_si512();
  __m512i accumulator3 = _mm512_setzero_si512();
  int64_t result = 0;

  size_t d = 0;
  size_t iterations_since_flush = 0;
  for (; d + kUnrolledBytes <= orig_dim; d += kUnrolledBytes) {
    accumulator0 = squared_diff_32(accumulator0, lhs + d, rhs + d);
    accumulator1 = squared_diff_32(accumulator1, lhs + d + 32, rhs + d + 32);
    accumulator2 = squared_diff_32(accumulator2, lhs + d + 64, rhs + d + 64);
    accumulator3 = squared_diff_32(accumulator3, lhs + d + 96, rhs + d + 96);

    if (++iterations_since_flush == kFlushIterations) {
      result += reduce_add_epi32_to_int64(accumulator0);
      result += reduce_add_epi32_to_int64(accumulator1);
      result += reduce_add_epi32_to_int64(accumulator2);
      result += reduce_add_epi32_to_int64(accumulator3);
      accumulator0 = _mm512_setzero_si512();
      accumulator1 = _mm512_setzero_si512();
      accumulator2 = _mm512_setzero_si512();
      accumulator3 = _mm512_setzero_si512();
      iterations_since_flush = 0;
    }
  }

  for (; d + kBlockBytes <= orig_dim; d += kBlockBytes) {
    accumulator0 = squared_diff_32(accumulator0, lhs + d, rhs + d);
  }

  if (d < orig_dim) {
    const size_t remaining = orig_dim - d;
    const __mmask32 mask =
        static_cast<__mmask32>((uint32_t{1} << remaining) - 1);
    accumulator0 = squared_diff_masked_32(accumulator0, lhs + d, rhs + d, mask);
  }

  result += reduce_add_epi32_to_int64(accumulator0);
  result += reduce_add_epi32_to_int64(accumulator1);
  result += reduce_add_epi32_to_int64(accumulator2);
  result += reduce_add_epi32_to_int64(accumulator3);
  *distance = static_cast<float>(result);
#else
  int64_t result = 0;
  for (size_t i = 0; i < orig_dim; ++i) {
    const int difference = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    result += static_cast<int64_t>(difference) * difference;
  }
  *distance = static_cast<float>(result);
#endif
}

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))

namespace {

// Reduce four zmm int32 accumulators to one xmm containing four scalar sums.
static ailego_force_inline __m128i reduce_add_4x16_epi32(__m512i accumulator0,
                                                         __m512i accumulator1,
                                                         __m512i accumulator2,
                                                         __m512i accumulator3) {
  const __m256i half0 =
      _mm256_add_epi32(_mm512_castsi512_si256(accumulator0),
                       _mm512_extracti64x4_epi64(accumulator0, 1));
  const __m256i half1 =
      _mm256_add_epi32(_mm512_castsi512_si256(accumulator1),
                       _mm512_extracti64x4_epi64(accumulator1, 1));
  const __m256i half2 =
      _mm256_add_epi32(_mm512_castsi512_si256(accumulator2),
                       _mm512_extracti64x4_epi64(accumulator2, 1));
  const __m256i half3 =
      _mm256_add_epi32(_mm512_castsi512_si256(accumulator3),
                       _mm512_extracti64x4_epi64(accumulator3, 1));
  const __m256i pair01 = _mm256_hadd_epi32(half0, half1);
  const __m256i pair23 = _mm256_hadd_epi32(half2, half3);
  const __m256i totals = _mm256_hadd_epi32(pair01, pair23);
  return _mm_add_epi32(_mm256_castsi256_si128(totals),
                       _mm256_extracti128_si256(totals, 1));
}

static ailego_force_inline void uniform_sq_l2_uint8_batch4(
    const void *const *vectors, const uint8_t *raw_query, size_t orig_dim,
    int32_t correction, const void *const *extra_values,
    const void *const *prefetch_vectors, float *distances) {
  __m512i accumulator0 = _mm512_setzero_si512();
  __m512i accumulator1 = _mm512_setzero_si512();
  __m512i accumulator2 = _mm512_setzero_si512();
  __m512i accumulator3 = _mm512_setzero_si512();

  const auto *vector0 = reinterpret_cast<const int8_t *>(vectors[0]);
  const auto *vector1 = reinterpret_cast<const int8_t *>(vectors[1]);
  const auto *vector2 = reinterpret_cast<const int8_t *>(vectors[2]);
  const auto *vector3 = reinterpret_cast<const int8_t *>(vectors[3]);

  size_t d = 0;
  for (; d + 64 <= orig_dim; d += 64) {
    const __m512i query =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(raw_query + d));
    const __m512i record0 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector0 + d));
    const __m512i record1 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector1 + d));
    const __m512i record2 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector2 + d));
    const __m512i record3 =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(vector3 + d));

    for (size_t i = 0; i < 4; ++i) {
      if (prefetch_vectors[i]) {
        _mm_prefetch(reinterpret_cast<const char *>(prefetch_vectors[i]) + d,
                     _MM_HINT_T0);
      }
    }

    accumulator0 = _mm512_dpbusd_epi32(accumulator0, query, record0);
    accumulator1 = _mm512_dpbusd_epi32(accumulator1, query, record1);
    accumulator2 = _mm512_dpbusd_epi32(accumulator2, query, record2);
    accumulator3 = _mm512_dpbusd_epi32(accumulator3, query, record3);
  }

  // The main loop only covers full cache lines, so prefetch the metadata tail
  // of each future record explicitly.
  for (size_t i = 0; i < 4; ++i) {
    if (prefetch_vectors[i]) {
      _mm_prefetch(
          reinterpret_cast<const char *>(prefetch_vectors[i]) + orig_dim,
          _MM_HINT_T0);
    }
  }

  __m128i dot_products = reduce_add_4x16_epi32(accumulator0, accumulator1,
                                               accumulator2, accumulator3);

  if (d < orig_dim) {
    alignas(16) int32_t totals[4];
    _mm_store_si128(reinterpret_cast<__m128i *>(totals), dot_products);
    const int8_t *records[4] = {vector0, vector1, vector2, vector3};
    for (size_t i = 0; i < 4; ++i) {
      int32_t remainder = 0;
      for (size_t j = d; j < orig_dim; ++j) {
        remainder +=
            static_cast<int>(records[i][j]) * static_cast<int>(raw_query[j]);
      }
      totals[i] += remainder;
    }
    dot_products = _mm_load_si128(reinterpret_cast<const __m128i *>(totals));
  }

  alignas(16) const uint32_t tails[4] = {
      load_extra_value(extra_values[0]), load_extra_value(extra_values[1]),
      load_extra_value(extra_values[2]), load_extra_value(extra_values[3])};
  const __m128i sum_squared =
      _mm_load_si128(reinterpret_cast<const __m128i *>(tails));

  // Exact squared L2 can exceed INT32_MAX while still fitting uint32_t for
  // every supported dimension. Preserve its exact bit pattern in packed
  // arithmetic, then convert it as unsigned.
  const __m128i squared_distances =
      _mm_add_epi32(_mm_sub_epi32(sum_squared, _mm_slli_epi32(dot_products, 1)),
                    _mm_set1_epi32(correction));
  _mm_storeu_ps(distances, uint32_to_float(squared_distances));
}

static ailego_force_inline void uniform_sq_l2_uint8_single(
    const void *vector, const uint8_t *raw_query, size_t orig_dim,
    int32_t correction, const void *extra_values, float *distance) {
  const auto *record = reinterpret_cast<const int8_t *>(vector);
  __m512i accumulator = _mm512_setzero_si512();
  size_t d = 0;
  for (; d + 64 <= orig_dim; d += 64) {
    const __m512i query =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(raw_query + d));
    const __m512i stored =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(record + d));
    accumulator = _mm512_dpbusd_epi32(accumulator, query, stored);
  }
  int64_t dot_product = _mm512_reduce_add_epi32(accumulator);
  for (; d < orig_dim; ++d) {
    dot_product += static_cast<int>(record[d]) * static_cast<int>(raw_query[d]);
  }
  *distance =
      static_cast<float>(static_cast<int64_t>(load_extra_value(extra_values)) -
                         2 * dot_product + correction);
}

}  // namespace

#endif

static void uniform_squared_euclidean_uint8_batch_distance_impl(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
#if !defined(__AVX512VNNI__) && !(defined(_MSC_VER) && defined(__AVX512F__))
  (void)extra_values;
#endif
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    for (size_t i = 0; i < n; ++i) {
      distances[i] = 0.0f;
    }
    return;
  }
  if (orig_dim > kMaxIdentityDimension) {
    const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
    for (size_t i = 0; i < n; ++i) {
      uniform_sq_l2_uint8_scalar_single(vectors[i], raw_query, orig_dim,
                                        distances + i);
    }
    return;
  }

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
  const int32_t correction = query_correction(query, orig_dim);

  constexpr size_t kBatchSize = 4;
  const size_t prefetch_step = orig_dim > 256 ? 1 : 2;
  size_t i = 0;
  const void *prefetch_vectors[kBatchSize];
  for (; i + kBatchSize <= n; i += kBatchSize) {
    for (size_t j = 0; j < kBatchSize; ++j) {
      const size_t prefetch_index = i + j + kBatchSize * prefetch_step;
      prefetch_vectors[j] =
          prefetch_index < n ? vectors[prefetch_index] : nullptr;
    }
    uniform_sq_l2_uint8_batch4(vectors + i, raw_query, orig_dim, correction,
                               extra_values + i, prefetch_vectors,
                               distances + i);
  }
  for (; i < n; ++i) {
    uniform_sq_l2_uint8_single(vectors[i], raw_query, orig_dim, correction,
                               extra_values[i], distances + i);
  }
#else
  const auto *raw_query = reinterpret_cast<const uint8_t *>(query);
  for (size_t i = 0; i < n; ++i) {
    uniform_sq_l2_uint8_scalar_single(vectors[i], raw_query, orig_dim,
                                      distances + i);
  }
#endif
}

void uniform_squared_euclidean_uint8_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
  ailego_assert_with(extra_values != nullptr,
                     "UniformUint8 batch distance requires extra values");
  uniform_squared_euclidean_uint8_batch_distance_impl(vectors, query, n, dim,
                                                      distances, extra_values);
}

void uniform_squared_euclidean_uint8_query_preprocess(void *query, size_t dim) {
  const size_t orig_dim = original_dim(dim);
  if (orig_dim == 0) {
    return;
  }

  auto *raw_query = reinterpret_cast<uint8_t *>(query);
  // Match the existing record-quantizer contract: this converts one private
  // canonical query copy exactly once before the query batch kernel uses it.
  uint64_t sum = 0;
  uint64_t sum_squared = 0;
  size_t d = 0;

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
  const __m512i sign_bit = _mm512_set1_epi8(static_cast<char>(0x80));
  const __m512i zero = _mm512_setzero_si512();
  __m512i sums = _mm512_setzero_si512();
  __m512i squared_sums = _mm512_setzero_si512();
  size_t iterations_since_flush = 0;
  constexpr size_t kSquaredSumFlushIterations = 4096;
  for (; d + 64 <= orig_dim; d += 64) {
    const __m512i stored = _mm512_loadu_si512(raw_query + d);
    const __m512i values = _mm512_xor_si512(stored, sign_bit);
    _mm512_storeu_si512(raw_query + d, values);
    sums = _mm512_add_epi64(sums, _mm512_sad_epu8(values, zero));
    const __m512i low_values =
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(values));
    const __m512i high_values =
        _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(values, 1));
    squared_sums = _mm512_dpwssd_epi32(squared_sums, low_values, low_values);
    squared_sums = _mm512_dpwssd_epi32(squared_sums, high_values, high_values);
    if (++iterations_since_flush == kSquaredSumFlushIterations) {
      sum_squared +=
          static_cast<uint64_t>(reduce_add_epi32_to_int64(squared_sums));
      squared_sums = _mm512_setzero_si512();
      iterations_since_flush = 0;
    }
  }
  alignas(64) uint64_t lanes[8];
  _mm512_store_si512(reinterpret_cast<__m512i *>(lanes), sums);
  for (uint64_t lane : lanes) {
    sum += lane;
  }
  sum_squared += static_cast<uint64_t>(reduce_add_epi32_to_int64(squared_sums));
#endif

  for (; d < orig_dim; ++d) {
    raw_query[d] ^= uint8_t{0x80};
    const uint64_t value = raw_query[d];
    sum += value;
    sum_squared += value * value;
  }

  const int64_t correction =
      static_cast<int64_t>(sum_squared) - 256 * static_cast<int64_t>(sum);
  if (correction < (std::numeric_limits<int32_t>::min)() ||
      correction > (std::numeric_limits<int32_t>::max)()) {
    // The public quantizer dimension bound keeps the correction in int32.
    // Oversized direct calls use the scalar squared-difference fallback.
    return;
  }
  const int32_t encoded_correction = static_cast<int32_t>(correction);
  std::memcpy(static_cast<uint8_t *>(query) + orig_dim, &encoded_correction,
              sizeof(encoded_correction));
}

}  // namespace zvec::turbo::avx512_vnni
