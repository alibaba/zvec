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

// Shared AVX512-VNNI inner product kernels for record_quantized_int8 distance
// implementations (cosine, l2, mips_l2, etc.).
//
// All functions are marked always_inline so that when this header is included
// from a per-file-march .cc translation unit, the compiler can fully inline
// and optimize them under the correct -march flag without any cross-TU call
// overhead.

#pragma once

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <array>
#include <cstdint>

#ifdef _MSC_VER
#define TURBO_ALWAYS_INLINE __forceinline
#else
#define TURBO_ALWAYS_INLINE __attribute__((always_inline))
#endif

namespace zvec::turbo::avx512_vnni::internal {

static inline int32_t HorizontalAdd_INT32_V256(__m256i v) {
  __m256i x1 = _mm256_hadd_epi32(v, v);
  __m256i x2 = _mm256_hadd_epi32(x1, x1);
  __m128i x3 = _mm256_extractf128_si256(x2, 1);
  __m128i x4 = _mm_add_epi32(_mm256_castsi256_si128(x2), x3);
  return _mm_cvtsi128_si32(x4);
}

#define FMA_INT8_GENERAL(m, q, sum) sum += static_cast<float>(m * q);

// Compute the raw integer inner product of two int8 vectors of length `size`.
// The result is written to `*distance` as a float.
// Both `a` and `b` must point to int8_t arrays.
static TURBO_ALWAYS_INLINE void ip_int8_avx512_vnni(const void *a,
                                                    const void *b, size_t size,
                                                    float *distance) {
  const __m256i ONES_INT16_AVX = _mm256_set1_epi32(0x00010001);
  const __m128i ONES_INT16_SSE = _mm_set1_epi32(0x00010001);

  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  const int8_t *last = lhs + size;
  const int8_t *last_aligned = lhs + ((size >> 6) << 6);

  float result = 0.0f;

  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_load_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_load_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_load_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_load_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_load_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_load_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  } else {
    for (; lhs != last_aligned; lhs += 64, rhs += 64) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((const __m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((const __m256i *)(lhs + 32));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((const __m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((const __m256i *)(rhs + 32));

      ymm_lhs_0 = _mm256_sign_epi8(ymm_lhs_0, ymm_rhs_0);
      ymm_lhs_1 = _mm256_sign_epi8(ymm_lhs_1, ymm_rhs_1);
      ymm_rhs_0 = _mm256_abs_epi8(ymm_rhs_0);
      ymm_rhs_1 = _mm256_abs_epi8(ymm_rhs_1);

      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_0, ymm_lhs_0),
                            ONES_INT16_AVX),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs_1, ymm_lhs_1),
                            ONES_INT16_AVX),
          ymm_sum_1);
    }

    if (last >= last_aligned + 32) {
      __m256i ymm_lhs = _mm256_loadu_si256((const __m256i *)lhs);
      __m256i ymm_rhs = _mm256_loadu_si256((const __m256i *)rhs);
      ymm_lhs = _mm256_sign_epi8(ymm_lhs, ymm_rhs);
      ymm_rhs = _mm256_abs_epi8(ymm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_madd_epi16(_mm256_maddubs_epi16(ymm_rhs, ymm_lhs),
                            ONES_INT16_AVX),
          ymm_sum_0);
      lhs += 32;
      rhs += 32;
    }

    if (last >= lhs + 16) {
      __m128i xmm_lhs = _mm_loadu_si128((const __m128i *)lhs);
      __m128i xmm_rhs = _mm_loadu_si128((const __m128i *)rhs);
      xmm_lhs = _mm_sign_epi8(xmm_lhs, xmm_rhs);
      xmm_rhs = _mm_abs_epi8(xmm_rhs);
      ymm_sum_0 = _mm256_add_epi32(
          _mm256_set_m128i(_mm_setzero_si128(),
                           _mm_madd_epi16(_mm_maddubs_epi16(xmm_rhs, xmm_lhs),
                                          ONES_INT16_SSE)),
          ymm_sum_0);
      lhs += 16;
      rhs += 16;
    }
  }
  result = static_cast<float>(
      HorizontalAdd_INT32_V256(_mm256_add_epi32(ymm_sum_0, ymm_sum_1)));

  switch (last - lhs) {
    case 15:
      FMA_INT8_GENERAL(lhs[14], rhs[14], result)
      /* FALLTHRU */
    case 14:
      FMA_INT8_GENERAL(lhs[13], rhs[13], result)
      /* FALLTHRU */
    case 13:
      FMA_INT8_GENERAL(lhs[12], rhs[12], result)
      /* FALLTHRU */
    case 12:
      FMA_INT8_GENERAL(lhs[11], rhs[11], result)
      /* FALLTHRU */
    case 11:
      FMA_INT8_GENERAL(lhs[10], rhs[10], result)
      /* FALLTHRU */
    case 10:
      FMA_INT8_GENERAL(lhs[9], rhs[9], result)
      /* FALLTHRU */
    case 9:
      FMA_INT8_GENERAL(lhs[8], rhs[8], result)
      /* FALLTHRU */
    case 8:
      FMA_INT8_GENERAL(lhs[7], rhs[7], result)
      /* FALLTHRU */
    case 7:
      FMA_INT8_GENERAL(lhs[6], rhs[6], result)
      /* FALLTHRU */
    case 6:
      FMA_INT8_GENERAL(lhs[5], rhs[5], result)
      /* FALLTHRU */
    case 5:
      FMA_INT8_GENERAL(lhs[4], rhs[4], result)
      /* FALLTHRU */
    case 4:
      FMA_INT8_GENERAL(lhs[3], rhs[3], result)
      /* FALLTHRU */
    case 3:
      FMA_INT8_GENERAL(lhs[2], rhs[2], result)
      /* FALLTHRU */
    case 2:
      FMA_INT8_GENERAL(lhs[1], rhs[1], result)
      /* FALLTHRU */
    case 1:
      FMA_INT8_GENERAL(lhs[0], rhs[0], result)
  }
  *distance = result;
}

#undef FMA_INT8_GENERAL

// Shift the first `original_dim` bytes of `query` in-place from int8 to uint8
// by adding 128 to each element. The metadata tail beyond `original_dim` is
// left untouched. This prepares the query for use with dpbusd (uint8 * int8).
static TURBO_ALWAYS_INLINE void shift_int8_to_uint8_avx512(
    void *query, size_t original_dim) {
  const int8_t *input = reinterpret_cast<const int8_t *>(query);
  uint8_t *output = reinterpret_cast<uint8_t *>(query);

  // 128 represented as int8_t wraps to -128, but two's complement addition
  // produces the correct uint8 result.
  const __m512i offset = _mm512_set1_epi8(static_cast<int8_t>(128));

  size_t i = 0;
  for (; i + 64 <= original_dim; i += 64) {
    __m512i data =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    __m512i shifted = _mm512_add_epi8(data, offset);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), shifted);
  }
  for (; i < original_dim; ++i) {
    output[i] = static_cast<uint8_t>(static_cast<int>(input[i]) + 128);
  }
}

// Compute raw integer inner products for a batch of int8 vectors against a
// single query. Uses AVX512-VNNI dpbusd instruction.
// `query` is treated as uint8 (preprocessed), `vectors[i]` as int8.
template <size_t batch_size>
TURBO_ALWAYS_INLINE void ip_int8_batch_avx512_vnni_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs,
    size_t dimensionality, float *distances) {
  __m512i accs[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }
  size_t dim = 0;
  for (; dim + 64 <= dimensionality; dim += 64) {
    __m512i q = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
        reinterpret_cast<const int8_t *>(query) + dim));
    __m512i data_regs[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
      data_regs[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
          reinterpret_cast<const int8_t *>(vectors[i]) + dim));
    }
    for (size_t i = 0; i < batch_size; ++i) {
      if (prefetch_ptrs[i]) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + dim),
            _MM_HINT_T0);
      }
      accs[i] = _mm512_dpbusd_epi32(accs[i], q, data_regs[i]);
    }
  }
  std::array<int, batch_size> temp_results{};
  for (size_t i = 0; i < batch_size; ++i) {
    temp_results[i] = _mm512_reduce_add_epi32(accs[i]);
  }
  for (; dim < dimensionality; ++dim) {
    int q = static_cast<int>(reinterpret_cast<const uint8_t *>(query)[dim]);
    for (size_t i = 0; i < batch_size; ++i) {
      temp_results[i] +=
          q *
          static_cast<int>(reinterpret_cast<const int8_t *>(vectors[i])[dim]);
    }
  }
  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] = static_cast<float>(temp_results[i]);
  }
}

// Dispatch batched inner product over all `n` vectors with prefetching.
static TURBO_ALWAYS_INLINE void ip_int8_batch_avx512_vnni(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  static constexpr size_t batch_size = 2;
  static constexpr size_t prefetch_step = 2;
  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      if (i + j + batch_size * prefetch_step < n) {
        prefetch_ptrs[j] = vectors[i + j + batch_size * prefetch_step];
      } else {
        prefetch_ptrs[j] = nullptr;
      }
    }
    ip_int8_batch_avx512_vnni_impl<batch_size>(
        query, &vectors[i], prefetch_ptrs, dim, distances + i);
  }
  for (; i < n; i++) {
    std::array<const void *, 1> prefetch_ptrs{nullptr};
    ip_int8_batch_avx512_vnni_impl<1>(query, &vectors[i], prefetch_ptrs, dim,
                                      distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni::internal

#endif  // defined(__AVX512VNNI__) || (defined(_MSC_VER) &&
        // defined(__AVX512F__))
