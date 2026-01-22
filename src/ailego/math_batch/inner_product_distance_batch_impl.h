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
#include <ailego/math/inner_product_matrix.h>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

template <typename ValueType, size_t BatchSize>
static void compute_one_to_many_fallback(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, BatchSize> &prefetch_ptrs, size_t dim,
    float *sums) {
  for (size_t j = 0; j < BatchSize; ++j) {
    sums[j] = 0.0;
    InnerProductMatrix<ValueType, 1, 1>::Compute(ptrs[j], query, dim, sums + j);
    ailego_prefetch(&prefetch_ptrs[j]);
  }
}

#if defined(__AVX2__)

inline float sum4(__m128 v) {
  v = _mm_add_ps(v, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 8)));
  return v[0] + v[1];
}

inline __m128 sum_top_bottom_avx(__m256 v) {
  const __m128 high = _mm256_extractf128_ps(v, 1);
  const __m128 low = _mm256_castps256_ps128(v);
  return _mm_add_ps(high, low);
}

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, float>, void>
compute_one_to_many_avx2_fp32(
    const ValueType *query, const ValueType **ptrs,
    std::array<const ValueType *, dp_batch> &prefetch_ptrs,
    size_t dimensionality, float *results) {
  __m256 accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_ps();
  }
  size_t dim = 0;
  for (; dim + 8 <= dimensionality; dim += 8) {
    __m256 q = _mm256_loadu_ps(query + dim);
    __m256 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_ps(ptrs[i] + dim);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm256_fnmadd_ps(q, data_regs[i], accs[i]);
    }
  }
  __m128 sum128_regs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    sum128_regs[i] = sum_top_bottom_avx(accs[i]);
  }
  if (dim + 4 <= dimensionality) {
    __m128 q = _mm_loadu_ps(query + dim);
    __m128 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_loadu_ps(ptrs[i] + dim);
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      sum128_regs[i] = _mm_fnmadd_ps(q, data_regs[i], sum128_regs[i]);
    }
    dim += 4;
  }
  if (dim + 2 <= dimensionality) {
    __m128 q = _mm_setzero_ps();
    __m128 data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_setzero_ps();
    }

    q = _mm_loadh_pi(q, (const __m64 *)(query + dim));
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm_loadh_pi(data_regs[i], (const __m64 *)(ptrs[i] + dim));
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      sum128_regs[i] = _mm_fnmadd_ps(q, data_regs[i], sum128_regs[i]);
    }
    dim += 2;
  }
  std::array<float, dp_batch> res;
  for (size_t i = 0; i < dp_batch; ++i) {
    res[i] = sum4(sum128_regs[i]);
  }
  if (dim < dimensionality) {
    float q = query[dim];
    for (size_t i = 0; i < dp_batch; ++i) {
      res[i] -= q * ptrs[i][dim];
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = -res[i];
  }
}
#endif


}  // namespace zvec::ailego::DistanceBatch