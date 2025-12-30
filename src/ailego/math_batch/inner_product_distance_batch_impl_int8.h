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
#include <ailego/utility/math_helper.h>
#include <ailego/utility/type_helper.h>

namespace zvec::ailego::DistanceBatch {

#if defined(__AVX512VNNI__)

static void compute_one_to_many_avx512_vnni_int8_query_preprocess(void *query,
                                                                  size_t dim) {
  const int8_t *input = reinterpret_cast<const int8_t *>(query);
  uint8_t *output = reinterpret_cast<uint8_t *>(query);

  // // AVX512 constant: 128 in each byte (cast to int8_t, which becomes -128
  // // in signed representation, but addition works correctly due to two's
  // // complement arithmetic)
  const __m512i offset = _mm512_set1_epi8(static_cast<int8_t>(128));
  //
  size_t i = 0;
  // // Process 64 bytes at a time using AVX512
  for (; i + 64 <= dim; i += 64) {
    __m512i data =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(input + i));
    __m512i result = _mm512_add_epi8(data, offset);
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(output + i), result);
  }

  // Handle remaining elements with scalar loop
  for (; i < dim; ++i) {
    output[i] = static_cast<uint8_t>(static_cast<int>(input[i]) + 128);
  }
}


// query is unsigned
template <size_t dp_batch>
static void compute_one_to_many_avx512_vnni_int8(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m512i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm512_setzero_si512();
  }
  size_t dim = 0;
  for (; dim + 64 <= dimensionality; dim += 64) {
    __m512i q =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + dim));
    __m512i data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptrs[i] + dim));
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] = _mm512_dpbusd_epi32(accs[i], q, data_regs[i]);
    }
  }
  std::array<int, dp_batch> temp_results{};
  for (size_t i = 0; i < dp_batch; ++i) {
    temp_results[i] = _mm512_reduce_add_epi32(accs[i]);
  }
  for (; dim < dimensionality; ++dim) {
    uint q = reinterpret_cast<const u_int8_t *>(query)[dim];
    for (size_t i = 0; i < dp_batch; ++i) {
      temp_results[i] += q * static_cast<int>(ptrs[i][dim]);
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = static_cast<float>(temp_results[i]);
  }
}
//
// #elif defined(__AVX512BW__)
// // TODO: this version is problematic
// template <typename ValueType, size_t dp_batch>
// static std::enable_if_t<std::is_same_v<ValueType, int8_t>, void>
// compute_one_to_many_avx512_int8(
//     const int8_t *query, const int8_t **ptrs,
//     std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t
//     dimensionality, float *results) {
//   __m512i accs[dp_batch];
//   size_t dim = 0;
//   for (; dim + 64 <= dimensionality; dim += 64) {
//     __m512i q =
//         _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + dim));
//     __m512i data_regs[dp_batch];
//     for (size_t i = 0; i < dp_batch; ++i) {
//       data_regs[i] =
//           _mm512_loadu_si512(reinterpret_cast<const __m512i *>(ptrs[i] +
//           dim));
//     }
//     if (prefetch_ptrs[0]) {
//       for (size_t i = 0; i < dp_batch; ++i) {
//         ailego_prefetch(prefetch_ptrs[i] + dim);
//       }
//     }
//     __m512i q_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 0));
//     __m512i q_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(q, 1));
//     __m512i data_lo[dp_batch];
//     __m512i data_hi[dp_batch];
//     for (size_t i = 0; i < dp_batch; ++i) {
//       data_lo[i] =
//           _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_regs[i], 0));
//       data_hi[i] =
//           _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(data_regs[i], 1));
//     }
//     __m512i prod_lo[dp_batch];
//     __m512i prod_hi[dp_batch];
//     for (size_t i = 0; i < dp_batch; ++i) {
//       prod_lo[i] = _mm512_madd_epi16(q_lo, data_lo[i]);
//       prod_hi[i] = _mm512_madd_epi16(q_hi, data_hi[i]);
//     }
//     for (size_t i = 0; i < dp_batch; ++i) {
//       accs[i] = _mm512_add_epi32(
//           accs[i], _mm512_add_epi32(
//                        _mm512_madd_epi16(prod_lo[i], _mm512_set1_epi16(1)),
//                        _mm512_madd_epi16(prod_hi[i], _mm512_set1_epi16(1))));
//     }
//   }
//   std::array<int, dp_batch> temp_results;
//   for (size_t i = 0; i < dp_batch; ++i) {
//     temp_results[i] = _mm512_reduce_add_epi32(accs[i]);
//   }
//   for (; dim < dimensionality; ++dim) {
//     int8_t q = query[dim];
//     for (size_t i = 0; i < dp_batch; ++i) {
//       temp_results[i] += q * static_cast<int>(ptrs[i][dim]);
//     }
//   }
//   for (size_t i = 0; i < dp_batch; ++i) {
//     results[i] = static_cast<float>(temp_results[i]);
//   }
// }
#endif

#if defined(__AVX2__)

template <typename ValueType, size_t dp_batch>
static std::enable_if_t<std::is_same_v<ValueType, int8_t>, void>
compute_one_to_many_avx2_int8(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, dp_batch> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  __m256i accs[dp_batch];
  for (size_t i = 0; i < dp_batch; ++i) {
    accs[i] = _mm256_setzero_si256();
  }
  size_t dim = 0;
  for (; dim + 32 <= dimensionality; dim += 32) {
    __m256i q = _mm256_loadu_si256((const __m256i *)(query + dim));
    __m256i data_regs[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_regs[i] = _mm256_loadu_si256((const __m256i *)(ptrs[i] + dim));
    }
    if (prefetch_ptrs[0]) {
      for (size_t i = 0; i < dp_batch; ++i) {
        ailego_prefetch(prefetch_ptrs[i] + dim);
      }
    }
    __m256i q_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(q));
    __m256i q_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(q, 1));
    __m256i data_lo[dp_batch];
    __m256i data_hi[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      data_lo[i] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(data_regs[i]));
      data_hi[i] =
          _mm256_cvtepi8_epi16(_mm256_extracti128_si256(data_regs[i], 1));
    }
    __m256i prod_lo[dp_batch];
    __m256i prod_hi[dp_batch];
    for (size_t i = 0; i < dp_batch; ++i) {
      prod_lo[i] = _mm256_madd_epi16(q_lo, data_lo[i]);
      prod_hi[i] = _mm256_madd_epi16(q_hi, data_hi[i]);
    }
    for (size_t i = 0; i < dp_batch; ++i) {
      accs[i] =
          _mm256_add_epi32(accs[i], _mm256_add_epi32(prod_lo[i], prod_hi[i]));
    }
  }
  std::array<int, dp_batch> temp_results;
  for (size_t i = 0; i < dp_batch; ++i) {
    __m128i lo = _mm256_castsi256_si128(accs[i]);
    __m128i hi = _mm256_extracti128_si256(accs[i], 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    temp_results[i] = _mm_cvtsi128_si32(sum128);
  }
  for (; dim < dimensionality; ++dim) {
    int8_t q = query[dim];
    for (size_t i = 0; i < dp_batch; ++i) {
      temp_results[i] += q * static_cast<int>(ptrs[i][dim]);
    }
  }
  for (size_t i = 0; i < dp_batch; ++i) {
    results[i] = static_cast<float>(temp_results[i]);
  }
}

#endif


}  // namespace zvec::ailego::DistanceBatch