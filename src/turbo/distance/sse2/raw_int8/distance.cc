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
#include "sse2/raw_int8/distance.h"
#include <algorithm>
#include <zvec/ailego/internal/platform.h>
#include "scalar/raw_numeric.h"
#include "raw_int8.h"

namespace zvec::turbo::sse2 {
namespace {
template <bool InnerProduct>
void Distance(const void *lhs, const void *rhs, size_t dim, float *out) {
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  const auto *a = static_cast<const int8_t *>(lhs);
  const auto *b = static_cast<const int8_t *>(rhs);
  const size_t aligned = dim - dim % 16;
  size_t i = 0;
  int64_t sum = 0;
  while (i < aligned) {
    const size_t end = i + std::min(internal::kRawInt8BlockSize, aligned - i);
    auto acc = _mm_setzero_si128();
    for (; i < end; i += 16) {
      const auto x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i));
      const auto y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
      const auto sx = _mm_cmpgt_epi8(_mm_setzero_si128(), x);
      const auto sy = _mm_cmpgt_epi8(_mm_setzero_si128(), y);
      auto xl = _mm_unpacklo_epi8(x, sx), xh = _mm_unpackhi_epi8(x, sx);
      auto yl = _mm_unpacklo_epi8(y, sy), yh = _mm_unpackhi_epi8(y, sy);
      if constexpr (!InnerProduct) {
        xl = _mm_sub_epi16(xl, yl);
        xh = _mm_sub_epi16(xh, yh);
        yl = xl;
        yh = xh;
      }
      acc = _mm_add_epi32(acc, _mm_madd_epi16(xl, yl));
      acc = _mm_add_epi32(acc, _mm_madd_epi16(xh, yh));
    }
    int32_t lanes[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(lanes), acc);
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
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
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

}  // namespace zvec::turbo::sse2
