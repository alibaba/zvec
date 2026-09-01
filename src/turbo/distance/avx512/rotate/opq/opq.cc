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

#if defined(__AVX512F__)
#include <immintrin.h>
#else
// Fallback when the build toolchain cannot emit AVX512 code: reuse the scalar
// kernel so the function stays correct instead of becoming a no-op stub.
#include "scalar/rotate/opq/opq.h"
#endif

namespace zvec::turbo::avx512 {

void opq_rotate_avx512(const float *in, float *out, size_t in_dim,
                       size_t /*out_dim*/, void *ctx) {
#if defined(__AVX512F__)
  // ctx is the dim x dim row-major rotation matrix itself.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  // One row dot product per output element; four independent FMA chains
  // (64 floats per iteration) hide the add latency of a single accumulator.
  constexpr size_t kChunk = 64;
  for (size_t r = 0; r < in_dim; ++r) {
    const float *row = matrix + r * in_dim;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    size_t c = 0;
    for (; c + kChunk <= in_dim; c += kChunk) {
      acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c), _mm512_loadu_ps(in + c),
                             acc0);
      acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 16),
                             _mm512_loadu_ps(in + c + 16), acc1);
      acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 32),
                             _mm512_loadu_ps(in + c + 32), acc2);
      acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(row + c + 48),
                             _mm512_loadu_ps(in + c + 48), acc3);
    }
    acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    float sum = _mm512_reduce_add_ps(acc0);
    for (; c < in_dim; ++c) {
      sum += row[c] * in[c];
    }
    out[r] = sum;
  }
#else
  scalar::opq_rotate(in, out, in_dim, in_dim, ctx);
#endif
}

void opq_unrotate_avx512(const float *in, float *out, size_t in_dim,
                         size_t /*out_dim*/, void *ctx) {
#if defined(__AVX512F__)
  // out = R^T * in written as outer-product accumulation: for each column
  // coefficient x[c], a broadcast multiply-add over the contiguous row R[c].
  // Both the matrix row and the output are streamed sequentially.
  const float *matrix = reinterpret_cast<const float *>(ctx);
  constexpr size_t kChunk = 64;
  for (size_t r = 0; r < in_dim; ++r) {
    out[r] = 0.0f;
  }
  for (size_t c = 0; c < in_dim; ++c) {
    const __m512 xc = _mm512_set1_ps(in[c]);
    const float *row = matrix + c * in_dim;
    size_t r = 0;
    for (; r + kChunk <= in_dim; r += kChunk) {
      _mm512_storeu_ps(out + r, _mm512_fmadd_ps(_mm512_loadu_ps(row + r), xc,
                                                _mm512_loadu_ps(out + r)));
      _mm512_storeu_ps(out + r + 16,
                       _mm512_fmadd_ps(_mm512_loadu_ps(row + r + 16), xc,
                                       _mm512_loadu_ps(out + r + 16)));
      _mm512_storeu_ps(out + r + 32,
                       _mm512_fmadd_ps(_mm512_loadu_ps(row + r + 32), xc,
                                       _mm512_loadu_ps(out + r + 32)));
      _mm512_storeu_ps(out + r + 48,
                       _mm512_fmadd_ps(_mm512_loadu_ps(row + r + 48), xc,
                                       _mm512_loadu_ps(out + r + 48)));
    }
    for (; r < in_dim; ++r) {
      out[r] += row[r] * in[c];
    }
  }
#else
  scalar::opq_unrotate(in, out, in_dim, in_dim, ctx);
#endif
}

}  // namespace zvec::turbo::avx512
