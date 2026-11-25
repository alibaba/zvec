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
#include "avx512/raw_int8/distance.h"
#include <algorithm>
#include <zvec/ailego/internal/platform.h>
#include "scalar/raw_numeric.h"
#include "raw_int8.h"

namespace zvec::turbo::avx512 {
namespace {
template <bool InnerProduct>
void Distance(const void *lhs, const void *rhs, size_t dim, float *out) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const auto *a = static_cast<const int8_t *>(lhs);
  const auto *b = static_cast<const int8_t *>(rhs);
  const size_t aligned = dim - dim % 32;
  size_t i = 0;
  int64_t sum = 0;
  while (i < aligned) {
    const size_t end = i + std::min(internal::kRawInt8BlockSize, aligned - i);
    auto acc = _mm512_setzero_si512();
    for (; i < end; i += 32) {
      auto x = _mm512_cvtepi8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)));
      auto y = _mm512_cvtepi8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
      if constexpr (!InnerProduct) {
        x = _mm512_sub_epi16(x, y);
        y = x;
      }
      acc = _mm512_add_epi32(acc, _mm512_madd_epi16(x, y));
    }
    int32_t lanes[16];
    _mm512_storeu_si512(static_cast<void *>(lanes), acc);
    for (int32_t lane : lanes) sum += lane;
  }
  sum += internal::RawInt8Tail<InnerProduct>(a, b, i, dim);
  *out = static_cast<float>(InnerProduct ? -sum : sum);
#else
  scalar::raw_integer_distance<false, InnerProduct>(lhs, rhs, dim, out);
#endif
}
}  // namespace

bool raw_int8_kernels_available() {
#if defined(__AVX512F__) && defined(__AVX512BW__)
  return true;
#else
  return false;
#endif
}

void raw_int8_squared_euclidean(const void *a, const void *b, size_t dim,
                                float *out) {
  Distance<false>(a, b, dim, out);
}
void raw_int8_squared_euclidean_batch(const void *const *vectors,
                                      const void *query, size_t count,
                                      size_t dim, float *out,
                                      const void *const * /*extra_values*/) {
  internal::RawInt8Batch<raw_int8_squared_euclidean>(vectors, query, count, dim,
                                                     out);
}

void raw_int8_inner_product(const void *a, const void *b, size_t dim,
                            float *out) {
  Distance<true>(a, b, dim, out);
}
void raw_int8_inner_product_batch(const void *const *vectors, const void *query,
                                  size_t count, size_t dim, float *out,
                                  const void *const * /*extra_values*/) {
  internal::RawInt8Batch<raw_int8_inner_product>(vectors, query, count, dim,
                                                 out);
}

}  // namespace zvec::turbo::avx512
