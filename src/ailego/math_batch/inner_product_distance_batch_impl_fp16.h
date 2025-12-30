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

#pragma once

#include <vector>
#include <ailego/internal/platform.h>
#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512FP16__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, ailego::Float16>, void>
compute_one_to_many_avx512fp16_fp16(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m512h accs[dp_batch];

  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_ph();
  }

  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    __m512h q = _mm512_loadu_ph(query + dim);

    __m512h data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm512_loadu_ph(ptrs[i] + dim);
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm512_fmadd_ph(data_regs[i], q, accs[i]);
    }
  }

  if (dim < dimensionality) {
    __mmask32 mask = (__mmask32)((1 << (dimensionality - dim)) - 1);

    for (size_t i = 0; i < dp_batch; ++i) {
      __m512i zmm_undefined = _mm512_undefined_epi32();

      accs[i] =
          _mm512_mask3_fmadd_ph(_mm512_castsi512_ph(_mm512_mask_loadu_epi16(
                                    zmm_undefined, mask, query + dim)),
                                _mm512_castsi512_ph(_mm512_mask_loadu_epi16(
                                    zmm_undefined, mask, ptrs[i] + dim)),
                                accs[i], mask);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP16_V512(accs[i]);
  }
}

#endif

#if defined(__AVX512F__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, ailego::Float16>, void>
compute_one_to_many_avx512f_fp16(
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

  if (dim + 16 < dimensionality) {
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

  if (dim + 8 < dimensionality) {
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
#endif

#if defined(__AVX2__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, ailego::Float16>, void>
compute_one_to_many_avx2_fp16(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m256 accs[dp_batch];

  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_ps();
  }

  size_t dim = 0;
  for (; dim + 16 <= dimensionality; dim += 16) {
    __m256i q =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + dim));

    __m256 q1 = _mm256_cvtph_ps(_mm256_castsi256_si128(q));
    __m256 q2 = _mm256_cvtph_ps(_mm256_extractf128_si256(q, 1));

    __m256 data_regs_1[dp_batch];
    __m256 data_regs_2[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      __m256i m =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptrs[i] + dim));

      data_regs_1[i] = _mm256_cvtph_ps(_mm256_castsi256_si128(m));
      data_regs_2[i] = _mm256_cvtph_ps(_mm256_extractf128_si256(m, 1));
    }

    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }

    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm256_fmadd_ps(q1, data_regs_1[i], accs[i]);
      accs[i] = _mm256_fmadd_ps(q2, data_regs_2[i], accs[i]);
    }
  }

  if (dim + 8 < dimensionality) {
    __m256 q = _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + dim)));

    __m256 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptrs[i] + dim)));
      accs[i] = _mm256_fmadd_ps(q, data_regs[i], accs[i]);
    }

    dim += 8;
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP32_V256(accs[i]);
  }

  for (; dim < dimensionality; ++dim) {
    for (size_t i = 0; i < dp_batch; ++i) {
      results[i] += (*(query + dim)) * (*(ptrs[i] + dim));
    }
  }
}

#endif


}  // namespace zvec::ailego::DistanceBatch