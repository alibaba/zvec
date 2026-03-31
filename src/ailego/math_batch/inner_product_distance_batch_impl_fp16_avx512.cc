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

#include <array>
#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512F__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, ailego::Float16>, void>
compute_one_to_many_inner_product_avx512f_fp16(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m512 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    __m512i q =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + dim));

    __m512 q1 = _mm512_cvtph_ps(_mm512_castsi512_si256(q));
    __m512 q2 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(q, 1));

    __m512 data_regs_1[dp_batch];
    __m512 data_regs_2[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      __m512i m =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptrs[i] + dim));

      data_regs_1[i] = _mm512_cvtph_ps(_mm512_castsi512_si256(m));
      data_regs_2[i] = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(m, 1));
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm512_fmadd_ps(q1, data_regs_1[i], accs[i]);
      accs[i] = _mm512_fmadd_ps(q2, data_regs_2[i], accs[i]);
    }
  }

  if (dim + 16 <= dimensionality) {
    __m512 q = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + dim)));

    __m512 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm512_cvtph_ps(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptrs[i] + dim)));
      accs[i] = _mm512_fmadd_ps(q, data_regs[i], accs[i]);
    }

    dim += 16;
  }

  __m256 acc_new[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    acc_new[i] = _mm256_add_ps(
        _mm512_castps512_ps256(accs[i]),
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(accs[i]), 1)));
  }

  if (dim + 8 <= dimensionality) {
    __m256 q = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + dim)));

    for (size_t i = 0; i < dp_batch; ++i) {
      __m256 m = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptrs[i] + dim)));
      acc_new[i] = _mm256_fmadd_ps(m, q, acc_new[i]);
    }

    dim += 8;
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP32_V256(acc_new[i]);
  }

  for (; dim < dimensionality; ++dim) {
    for (size_t i = 0; i < dp_batch; ++i) {
      results[i] += (*(query + dim)) * (*(ptrs[i] + dim));
    }
  }
}

void compute_one_to_many_inner_product_avx512f_fp16_1(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs, size_t dim,
    float *sums) {
  return compute_one_to_many_inner_product_avx512f_fp16<ailego::Float16, 1>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

void compute_one_to_many_inner_product_avx512f_fp16_12(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs, size_t dim,
    float *sums) {
  return compute_one_to_many_inner_product_avx512f_fp16<ailego::Float16, 12>(
      query, ptrs, prefetch_ptrs, dim, sums);
}
#endif

}  // namespace zvec::ailego::DistanceBatch
