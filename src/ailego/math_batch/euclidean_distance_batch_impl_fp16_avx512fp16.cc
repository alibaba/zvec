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

#include <ailego/math/matrix_utility.i>
#include <ailego/utility/math_helper.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512FP16__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, ailego::Float16>, void>
compute_one_to_many_squared_euclidean_avx512fp16_fp16(
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
      __m512h diff = _mm512_sub_ph(data_regs[i], q);
      accs[i] = _mm512_fmadd_ph(diff, diff, accs[i]);
    }
  }

  if (dim < dimensionality) {
    __mmask32 mask = (__mmask32)(((uint32_t)1 << (dimensionality - dim)) - 1);

    for (size_t i = 0; i < dp_batch; ++i) {
      __m512i zmm_undefined = _mm512_undefined_epi32();
      __m512h zmm_undefined_ph = _mm512_undefined_ph();
      __m512h zmm_d =
          _mm512_mask_sub_ph(zmm_undefined_ph, mask,
                             _mm512_castsi512_ph(_mm512_mask_loadu_epi16(
                                 zmm_undefined, mask, query + dim)),
                             _mm512_castsi512_ph(_mm512_mask_loadu_epi16(
                                 zmm_undefined, mask, ptrs[i] + dim)));

      accs[i] = _mm512_mask3_fmadd_ph(zmm_d, zmm_d, accs[i], mask);
    }
  }

  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = HorizontalAdd_FP16_V512(accs[i]);
  }
}

void compute_one_to_many_squared_euclidean_avx512fp16_fp16_1(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 1> &prefetch_ptrs, size_t dim,
    float *sums) {
  return compute_one_to_many_squared_euclidean_avx512fp16_fp16<ailego::Float16,
                                                               1>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

void compute_one_to_many_squared_euclidean_avx512fp16_fp16_12(
    const ailego::Float16 *query, const ailego::Float16 **ptrs,
    std::array<const ailego::Float16 *, 12> &prefetch_ptrs, size_t dim,
    float *sums) {
  return compute_one_to_many_squared_euclidean_avx512fp16_fp16<ailego::Float16,
                                                               12>(
      query, ptrs, prefetch_ptrs, dim, sums);
}

#endif

}  // namespace zvec::ailego::DistanceBatch
