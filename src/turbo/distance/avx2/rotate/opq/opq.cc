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

#include "opq.h"

#if defined(__AVX2__)
#include <immintrin.h>
#else
#include "scalar/rotate/opq/opq.h"
#endif

namespace zvec::turbo::avx2 {

void opq_rotate_avx2(const float *in, float *out, size_t in_dim,
                     size_t /*out_dim*/, void *ctx) {
#if defined(__AVX2__)
  // out = R * in; four independent FMA chains hide the add latency.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  constexpr size_t kChunk = 32;
  for (size_t r = 0; r < in_dim; ++r) {
    const float *row = matrix + r * in_dim;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t c = 0;
    for (; c + kChunk <= in_dim; c += kChunk) {
      acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(in + c),
                             acc0);
      acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8),
                             _mm256_loadu_ps(in + c + 8), acc1);
      acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16),
                             _mm256_loadu_ps(in + c + 16), acc2);
      acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24),
                             _mm256_loadu_ps(in + c + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(acc0);
    __m128 hi = _mm256_extractf128_ps(acc0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; c < in_dim; ++c) {
      sum += row[c] * in[c];
    }
    out[r] = sum;
  }
#else
  scalar::opq_rotate(in, out, in_dim, in_dim, ctx);
#endif
}

void opq_unrotate_avx2(const float *in, float *out, size_t in_dim,
                       size_t /*out_dim*/, void *ctx) {
#if defined(__AVX2__)
  // out = R^T * in via broadcast outer-product accumulation.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  constexpr size_t kChunk = 32;
  for (size_t r = 0; r < in_dim; ++r) {
    out[r] = 0.0f;
  }
  for (size_t c = 0; c < in_dim; ++c) {
    const __m256 xc = _mm256_set1_ps(in[c]);
    const float *row = matrix + c * in_dim;
    size_t r = 0;
    for (; r + kChunk <= in_dim; r += kChunk) {
      _mm256_storeu_ps(out + r, _mm256_fmadd_ps(_mm256_loadu_ps(row + r), xc,
                                                _mm256_loadu_ps(out + r)));
      _mm256_storeu_ps(out + r + 8,
                       _mm256_fmadd_ps(_mm256_loadu_ps(row + r + 8), xc,
                                       _mm256_loadu_ps(out + r + 8)));
      _mm256_storeu_ps(out + r + 16,
                       _mm256_fmadd_ps(_mm256_loadu_ps(row + r + 16), xc,
                                       _mm256_loadu_ps(out + r + 16)));
      _mm256_storeu_ps(out + r + 24,
                       _mm256_fmadd_ps(_mm256_loadu_ps(row + r + 24), xc,
                                       _mm256_loadu_ps(out + r + 24)));
    }
    for (; r < in_dim; ++r) {
      out[r] += row[r] * in[c];
    }
  }
#else
  scalar::opq_unrotate(in, out, in_dim, in_dim, ctx);
#endif
}

}  // namespace zvec::turbo::avx2
