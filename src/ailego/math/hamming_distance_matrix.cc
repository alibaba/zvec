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

#include "hamming_distance_matrix.h"
#include <arrow/util/future.h>
#include <zvec/ailego/internal/platform.h>
#include "distance_matrix_popcnt.i"

namespace zvec {
namespace ailego {

#define POPCNT_UINT32_STEP1_SSE HAMMING_UINT32_STEP1_SSE
#define POPCNT_UINT32_STEP2_SSE HAMMING_UINT32_STEP2_SSE
#define POPCNT_UINT32_STEP3_SSE HAMMING_UINT32_STEP3_SSE
#define POPCNT_UINT32_STEP1_AVX HAMMING_UINT32_STEP1_AVX
#define POPCNT_UINT32_STEP2_AVX HAMMING_UINT32_STEP2_AVX
#define POPCNT_UINT32_STEP3_AVX HAMMING_UINT32_STEP3_AVX
#define POPCNT_UINT64_STEP1_AVX HAMMING_UINT64_STEP1_AVX
#define POPCNT_UINT64_STEP2_AVX HAMMING_UINT64_STEP2_AVX

//! Calculate population count (Step 1 SSE)
#define HAMMING_UINT32_STEP1_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_add_epi8(                               \
      VerticalPopCount_INT8_V128(_mm_xor_si128(xmm_m, xmm_q)), xmm_sum);

//! Calculate population count (Step 2 SSE)
#define HAMMING_UINT32_STEP2_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_add_epi16(                              \
      VerticalPopCount_INT16_V128(_mm_xor_si128(xmm_m, xmm_q)), xmm_sum);

//! Calculate population count (Step 3 SSE)
#define HAMMING_UINT32_STEP3_SSE(xmm_m, xmm_q, xmm_sum) \
  xmm_sum = _mm_add_epi32(                              \
      VerticalPopCount_INT32_V128(_mm_xor_si128(xmm_m, xmm_q)), xmm_sum);

//! Calculate population count (Step 1 AVX)
#define HAMMING_UINT32_STEP1_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_add_epi8(                            \
      VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_m, ymm_q)), ymm_sum);

//! Calculate population count (Step 2 AVX)
#define HAMMING_UINT32_STEP2_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_add_epi16(                           \
      VerticalPopCount_INT16_V256(_mm256_xor_si256(ymm_m, ymm_q)), ymm_sum);

//! Calculate population count (Step 3 AVX)
#define HAMMING_UINT32_STEP3_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_add_epi32(                           \
      VerticalPopCount_INT32_V256(_mm256_xor_si256(ymm_m, ymm_q)), ymm_sum);

//! Calculate population count (Step 1 AVX)
#define HAMMING_UINT64_STEP1_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_add_epi8(                            \
      VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_m, ymm_q)), ymm_sum);

//! Calculate population count (Step 2 AVX)
#define HAMMING_UINT64_STEP2_AVX(ymm_m, ymm_q, ymm_sum) \
  ymm_sum = _mm256_add_epi64(                           \
      VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_m, ymm_q)), ymm_sum);

#if defined(__AVX512VL__) && defined(__AVX512DQ__)
#define CONVERT_UINT64_TO_FP32(v, ...) _mm256_cvtepu64_ps(v)
#elif defined(__AVX2__)
static const __m256i CONVERT_UINT32_MASK_AVX =
    _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

#define CONVERT_UINT64_TO_FP32(v, ...)    \
  _mm_cvtepi32_ps(_mm256_castsi256_si128( \
      _mm256_permutevar8x32_epi32(v, CONVERT_UINT32_MASK_AVX)))
#endif  // __AVX512VL__ && __AVX512DQ__

#define SQRT_UINT64_TO_FP32(v, ...) _mm_sqrt_ps(CONVERT_UINT64_TO_FP32(v))
#define SQRT_UINT32_TO_FP32_SSE(v, ...) _mm_sqrt_ps(_mm_cvtepi32_ps(v))
#define SQRT_UINT32_TO_FP32_AVX(v, ...) _mm256_sqrt_ps(_mm256_cvtepi32_ps(v))

#if defined(__AVX2__)
static inline size_t HammingDistanceAVX(const uint32_t *lhs,
                                        const uint32_t *rhs, size_t size) {
  __m256i ymm_sum_0 = _mm256_setzero_si256();
  __m256i ymm_sum_1 = _mm256_setzero_si256();

  const uint32_t *lhs_0 = lhs + ((size >> 4) << 4);
  const uint32_t *lhs_1 = (size > 496 ? lhs + 496 : lhs_0);
  const uint32_t *lhs_2 = lhs + size;

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != lhs_1; lhs += 16, rhs += 16) {
      __m256i ymm_lhs_0 = _mm256_load_si256((__m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((__m256i *)(lhs + 8));
      __m256i ymm_rhs_0 = _mm256_load_si256((__m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((__m256i *)(rhs + 8));

      ymm_sum_0 = _mm256_add_epi8(
          VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_lhs_0, ymm_rhs_0)),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi8(
          VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_lhs_1, ymm_rhs_1)),
          ymm_sum_1);
    }
    ymm_sum_0 = _mm256_sad_epu8(ymm_sum_0, POPCNT_ZERO_AVX);
    ymm_sum_1 = _mm256_sad_epu8(ymm_sum_1, POPCNT_ZERO_AVX);

    for (; lhs != lhs_0; lhs += 16, rhs += 16) {
      __m256i ymm_lhs_0 = _mm256_load_si256((__m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_load_si256((__m256i *)(lhs + 8));
      __m256i ymm_rhs_0 = _mm256_load_si256((__m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_load_si256((__m256i *)(rhs + 8));

      ymm_sum_0 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs_0, ymm_rhs_0)),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs_1, ymm_rhs_1)),
          ymm_sum_1);
    }

    if (lhs_2 >= lhs + 8) {
      __m256i ymm_lhs = _mm256_load_si256((__m256i *)(lhs));
      __m256i ymm_rhs = _mm256_load_si256((__m256i *)(rhs));
      ymm_sum_0 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs, ymm_rhs)),
          ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  } else {
    for (; lhs != lhs_1; lhs += 16, rhs += 16) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((__m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((__m256i *)(lhs + 8));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((__m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((__m256i *)(rhs + 8));

      ymm_sum_0 = _mm256_add_epi8(
          VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_lhs_0, ymm_rhs_0)),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi8(
          VerticalPopCount_INT8_V256(_mm256_xor_si256(ymm_lhs_1, ymm_rhs_1)),
          ymm_sum_1);
    }
    ymm_sum_0 = _mm256_sad_epu8(ymm_sum_0, POPCNT_ZERO_AVX);
    ymm_sum_1 = _mm256_sad_epu8(ymm_sum_1, POPCNT_ZERO_AVX);

    for (; lhs != lhs_0; lhs += 16, rhs += 16) {
      __m256i ymm_lhs_0 = _mm256_loadu_si256((__m256i *)(lhs + 0));
      __m256i ymm_lhs_1 = _mm256_loadu_si256((__m256i *)(lhs + 8));
      __m256i ymm_rhs_0 = _mm256_loadu_si256((__m256i *)(rhs + 0));
      __m256i ymm_rhs_1 = _mm256_loadu_si256((__m256i *)(rhs + 8));

      ymm_sum_0 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs_0, ymm_rhs_0)),
          ymm_sum_0);
      ymm_sum_1 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs_1, ymm_rhs_1)),
          ymm_sum_1);
    }

    if (lhs_2 >= lhs + 8) {
      __m256i ymm_lhs = _mm256_loadu_si256((__m256i *)(lhs));
      __m256i ymm_rhs = _mm256_loadu_si256((__m256i *)(rhs));
      ymm_sum_0 = _mm256_add_epi64(
          VerticalPopCount_INT64_V256(_mm256_xor_si256(ymm_lhs, ymm_rhs)),
          ymm_sum_0);
      lhs += 8;
      rhs += 8;
    }
  }

  size_t count =
      (size_t)HorizontalAdd_INT64_V256(_mm256_add_epi64(ymm_sum_0, ymm_sum_1));
  switch (lhs_2 - lhs) {
    case 7:
      count += ailego_popcount32(lhs[6] ^ rhs[6]);
      /* FALLTHRU */
    case 6:
      count += ailego_popcount32(lhs[5] ^ rhs[5]);
      /* FALLTHRU */
    case 5:
      count += ailego_popcount32(lhs[4] ^ rhs[4]);
      /* FALLTHRU */
    case 4:
      count += ailego_popcount32(lhs[3] ^ rhs[3]);
      /* FALLTHRU */
    case 3:
      count += ailego_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += ailego_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += ailego_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}

static inline size_t HammingDistanceAVX(const uint64_t *lhs,
                                        const uint64_t *rhs, size_t size) {
  return HammingDistanceAVX(reinterpret_cast<const uint32_t *>(lhs),
                            reinterpret_cast<const uint32_t *>(rhs),
                            (size << 1));
}
#endif  // __AVX2__

#if defined(AILEGO_M64)
static inline size_t HammingDistance(const uint32_t *lhs, const uint32_t *rhs,
                                     size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    count += ailego_popcount64(*(uint64_t *)(&lhs[6]) ^ *(uint64_t *)(&rhs[6]));
    count += ailego_popcount64(*(uint64_t *)(&lhs[4]) ^ *(uint64_t *)(&rhs[4]));
    count += ailego_popcount64(*(uint64_t *)(&lhs[2]) ^ *(uint64_t *)(&rhs[2]));
    count += ailego_popcount64(*(uint64_t *)(&lhs[0]) ^ *(uint64_t *)(&rhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      count += ailego_popcount32(lhs[6] ^ rhs[6]);
      /* FALLTHRU */
    case 6:
      count += ailego_popcount32(lhs[5] ^ rhs[5]);
      /* FALLTHRU */
    case 5:
      count += ailego_popcount32(lhs[4] ^ rhs[4]);
      /* FALLTHRU */
    case 4:
      count += ailego_popcount32(lhs[3] ^ rhs[3]);
      /* FALLTHRU */
    case 3:
      count += ailego_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += ailego_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += ailego_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}

static inline size_t HammingDistance(const uint64_t *lhs, const uint64_t *rhs,
                                     size_t size) {
  const uint64_t *last = lhs + size;
  const uint64_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += ailego_popcount64(lhs[3] ^ rhs[3]);
    count += ailego_popcount64(lhs[2] ^ rhs[2]);
    count += ailego_popcount64(lhs[1] ^ rhs[1]);
    count += ailego_popcount64(lhs[0] ^ rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += ailego_popcount64(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += ailego_popcount64(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += ailego_popcount64(lhs[0] ^ rhs[0]);
  }
  return count;
}
#else
static inline size_t HammingDistance(const uint32_t *lhs, const uint32_t *rhs,
                                     size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += ailego_popcount32(lhs[3] ^ rhs[3]);
    count += ailego_popcount32(lhs[2] ^ rhs[2]);
    count += ailego_popcount32(lhs[1] ^ rhs[1]);
    count += ailego_popcount32(lhs[0] ^ rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += ailego_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += ailego_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += ailego_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}
#endif  // AILEGO_M64

//! Compute the distance between matrix and query (UINT32, M=1, N=1)
void HammingDistanceMatrix<uint32_t, 1, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  if (cnt > 63) {
    *out = static_cast<float>(HammingDistanceAVX(m, q, cnt));
    return;
  }
#endif
  *out = static_cast<float>(HammingDistance(m, q, cnt));
}

#if defined(__SSSE3__)
//! Compute the distance between matrix and query (UINT32, M=2, N=1)
void HammingDistanceMatrix<uint32_t, 2, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_2X1_AVX(m, q, cnt, out, _mm_cvtepi32_ps)
#else
  POPCNT_UINT32_2X1_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=2, N=2)
void HammingDistanceMatrix<uint32_t, 2, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_2X2_AVX(m, q, cnt, out, _mm_cvtepi32_ps)
#else
  POPCNT_UINT32_2X2_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=1)
void HammingDistanceMatrix<uint32_t, 4, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X1_AVX(m, q, cnt, out, _mm_cvtepi32_ps)
#else
  POPCNT_UINT32_4X1_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=2)
void HammingDistanceMatrix<uint32_t, 4, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X2_AVX(m, q, cnt, out, _mm_cvtepi32_ps)
#else
  POPCNT_UINT32_4X2_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=4)
void HammingDistanceMatrix<uint32_t, 4, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X4_AVX(m, q, cnt, out, _mm_cvtepi32_ps)
#else
  POPCNT_UINT32_4X4_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=1)
void HammingDistanceMatrix<uint32_t, 8, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X1_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_8X1_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=2)
void HammingDistanceMatrix<uint32_t, 8, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X2_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_8X2_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=4)
void HammingDistanceMatrix<uint32_t, 8, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X4_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_8X4_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=8)
void HammingDistanceMatrix<uint32_t, 8, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X8_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_8X8_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=1)
void HammingDistanceMatrix<uint32_t, 16, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X1_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_16X1_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=2)
void HammingDistanceMatrix<uint32_t, 16, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X2_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_16X2_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=4)
void HammingDistanceMatrix<uint32_t, 16, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X4_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_16X4_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=8)
void HammingDistanceMatrix<uint32_t, 16, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X8_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_16X8_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=16)
void HammingDistanceMatrix<uint32_t, 16, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X16_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_16X16_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=1)
void HammingDistanceMatrix<uint32_t, 32, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X1_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X1_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=2)
void HammingDistanceMatrix<uint32_t, 32, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X2_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X2_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=4)
void HammingDistanceMatrix<uint32_t, 32, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X4_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X4_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=8)
void HammingDistanceMatrix<uint32_t, 32, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X8_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X8_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=16)
void HammingDistanceMatrix<uint32_t, 32, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X16_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X16_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=32)
void HammingDistanceMatrix<uint32_t, 32, 32>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X32_AVX(m, q, cnt, out, _mm256_cvtepi32_ps)
#else
  POPCNT_UINT32_32X32_SSE(m, q, cnt, out, _mm_cvtepi32_ps)
#endif
}
#endif  // __SSSE3__

#if defined(AILEGO_M64)
//! Compute the distance between matrix and query (UINT64, M=1, N=1)
void HammingDistanceMatrix<uint64_t, 1, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
#if defined(__AVX2__)
  if (cnt > 31) {
    *out = static_cast<float>(HammingDistanceAVX(m, q, cnt));
    return;
  }
#endif
  *out = static_cast<float>(HammingDistance(m, q, cnt));
}

#if defined(__AVX2__)
//! Compute the distance between matrix and query (UINT64, M=2, N=1)
void HammingDistanceMatrix<uint64_t, 2, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_2X1_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=2, N=2)
void HammingDistanceMatrix<uint64_t, 2, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_2X2_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=1)
void HammingDistanceMatrix<uint64_t, 4, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X1_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=2)
void HammingDistanceMatrix<uint64_t, 4, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X2_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=4)
void HammingDistanceMatrix<uint64_t, 4, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X4_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=1)
void HammingDistanceMatrix<uint64_t, 8, 1>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X1_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=2)
void HammingDistanceMatrix<uint64_t, 8, 2>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X2_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=4)
void HammingDistanceMatrix<uint64_t, 8, 4>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X4_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=8)
void HammingDistanceMatrix<uint64_t, 8, 8>::Compute(const ValueType *m,
                                                    const ValueType *q,
                                                    size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X8_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=1)
void HammingDistanceMatrix<uint64_t, 16, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X1_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=2)
void HammingDistanceMatrix<uint64_t, 16, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X2_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=4)
void HammingDistanceMatrix<uint64_t, 16, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X4_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=8)
void HammingDistanceMatrix<uint64_t, 16, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X8_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=16)
void HammingDistanceMatrix<uint64_t, 16, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X16_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=1)
void HammingDistanceMatrix<uint64_t, 32, 1>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X1_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=2)
void HammingDistanceMatrix<uint64_t, 32, 2>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X2_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=4)
void HammingDistanceMatrix<uint64_t, 32, 4>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X4_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=8)
void HammingDistanceMatrix<uint64_t, 32, 8>::Compute(const ValueType *m,
                                                     const ValueType *q,
                                                     size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X8_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=16)
void HammingDistanceMatrix<uint64_t, 32, 16>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X16_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=32)
void HammingDistanceMatrix<uint64_t, 32, 32>::Compute(const ValueType *m,
                                                      const ValueType *q,
                                                      size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X32_AVX(m, q, cnt, out, CONVERT_UINT64_TO_FP32)
}
#endif  // __AVX2__
#endif  // AILEGO_M64

//! Compute the distance between matrix and query (UINT32, M=1, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 1, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  if (cnt > 63) {
    *out = std::sqrt(static_cast<float>(HammingDistanceAVX(m, q, cnt)));
    return;
  }
#endif
  *out = std::sqrt(static_cast<float>(HammingDistance(m, q, cnt)));
}

#if defined(__SSSE3__)
//! Compute the distance between matrix and query (UINT32, M=2, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 2, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_2X1_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#else
  POPCNT_UINT32_2X1_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=2, N=2)
void HammingSquareRootDistanceMatrix<uint32_t, 2, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_2X2_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#else
  POPCNT_UINT32_2X2_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 4, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X1_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#else
  POPCNT_UINT32_4X1_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=2)
void HammingSquareRootDistanceMatrix<uint32_t, 4, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X2_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#else
  POPCNT_UINT32_4X2_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=4, N=4)
void HammingSquareRootDistanceMatrix<uint32_t, 4, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_4X4_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#else
  POPCNT_UINT32_4X4_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 8, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X1_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_8X1_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=2)
void HammingSquareRootDistanceMatrix<uint32_t, 8, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X2_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_8X2_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=4)
void HammingSquareRootDistanceMatrix<uint32_t, 8, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X4_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_8X4_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=8, N=8)
void HammingSquareRootDistanceMatrix<uint32_t, 8, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_8X8_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_8X8_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 16, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X1_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_16X1_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=2)
void HammingSquareRootDistanceMatrix<uint32_t, 16, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X2_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_16X2_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=4)
void HammingSquareRootDistanceMatrix<uint32_t, 16, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X4_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_16X4_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=8)
void HammingSquareRootDistanceMatrix<uint32_t, 16, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X8_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_16X8_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=16, N=16)
void HammingSquareRootDistanceMatrix<uint32_t, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_16X16_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_16X16_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=1)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X1_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X1_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=2)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X2_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X2_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=4)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X4_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X4_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=8)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X8_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X8_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=16)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X16_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X16_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}

//! Compute the distance between matrix and query (UINT32, M=32, N=32)
void HammingSquareRootDistanceMatrix<uint32_t, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 5);
#if defined(__AVX2__)
  POPCNT_UINT32_32X32_AVX(m, q, cnt, out, SQRT_UINT32_TO_FP32_AVX)
#else
  POPCNT_UINT32_32X32_SSE(m, q, cnt, out, SQRT_UINT32_TO_FP32_SSE)
#endif
}
#endif  // __SSSE3__

#if defined(AILEGO_M64)
//! Compute the distance between matrix and query (UINT64, M=1, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 1, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
#if defined(__AVX2__)
  if (cnt > 31) {
    *out = std::sqrt(static_cast<float>(HammingDistanceAVX(m, q, cnt)));
    return;
  }
#endif
  *out = std::sqrt(static_cast<float>(HammingDistance(m, q, cnt)));
}

#if defined(__AVX2__)
//! Compute the distance between matrix and query (UINT64, M=2, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 2, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_2X1_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=2, N=2)
void HammingSquareRootDistanceMatrix<uint64_t, 2, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_2X2_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 4, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X1_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=2)
void HammingSquareRootDistanceMatrix<uint64_t, 4, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X2_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=4, N=4)
void HammingSquareRootDistanceMatrix<uint64_t, 4, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_4X4_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 8, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X1_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=2)
void HammingSquareRootDistanceMatrix<uint64_t, 8, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X2_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=4)
void HammingSquareRootDistanceMatrix<uint64_t, 8, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X4_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=8, N=8)
void HammingSquareRootDistanceMatrix<uint64_t, 8, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_8X8_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 16, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X1_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=2)
void HammingSquareRootDistanceMatrix<uint64_t, 16, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X2_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=4)
void HammingSquareRootDistanceMatrix<uint64_t, 16, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X4_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=8)
void HammingSquareRootDistanceMatrix<uint64_t, 16, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X8_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=16, N=16)
void HammingSquareRootDistanceMatrix<uint64_t, 16, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_16X16_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=1)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 1>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X1_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=2)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 2>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X2_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=4)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 4>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X4_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=8)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 8>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X8_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=16)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 16>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X16_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}

//! Compute the distance between matrix and query (UINT64, M=32, N=32)
void HammingSquareRootDistanceMatrix<uint64_t, 32, 32>::Compute(
    const ValueType *m, const ValueType *q, size_t dim, float *out) {
  size_t cnt = (dim >> 6);
  POPCNT_UINT64_32X32_AVX(m, q, cnt, out, SQRT_UINT64_TO_FP32)
}
#endif  // __AVX2__
#endif  // AILEGO_M64

}  // namespace ailego
}  // namespace zvec