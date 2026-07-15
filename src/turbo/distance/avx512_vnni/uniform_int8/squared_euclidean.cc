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

// AVX512-VNNI optimized squared Euclidean distance for uniform-quantized INT8.
//
// Since all vectors share a single global scale/bias, the distance is simply:
//   sum((a[i] - b[i])^2)
// computed entirely in the integer domain.  No per-vector reconstruction or
// scalar dequantization is needed.
//
// Algorithm for each 64-element chunk (VNNI abs trick):
//   1. Load 64 int8 values from each vector                (zmm load)
//   2. Subtract int8 vectors: diff = a - b                  (vpsubb)
//   3. Absolute value: |diff|                               (vpabsb)
//   4. Squared accumulate via VNNI: acc += |diff| * |diff|  (vpdpbusd)
//
// Constraint: input values MUST be in [0, 127] so that the int8
// subtraction does not overflow (max |diff| = 127 fits in both
// uint8 and int8 for the VNNI multiply).
//
// This processes 64 bytes per iteration (2x throughput vs int16 widening)
// and uses only 3 core SIMD ops in the inner loop.
//
// This file is compiled with per-file -march=avx512vnni (set in
// CMakeLists.txt).

#include "avx512_vnni/uniform_int8/squared_euclidean.h"
#include "zvec/ailego/internal/platform.h"

#if defined(__AVX512VNNI__) || (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#include <array>
#include <cstdint>

namespace zvec::turbo::avx512_vnni {

// ---------------------------------------------------------------------------
// Batch kernel template: compute squared L2 for `batch_size` database vectors
// against a single query, with software prefetching of future vectors.
//
// Uses VNNI abs trick: sub_epi8 → abs_epi8 → vpdpbusd, processing 64 bytes
// per iteration.  Two-phase load/compute: load ALL vectors first, then compute
// (allows CPU to issue multiple loads in parallel, hiding memory latency).
// ---------------------------------------------------------------------------
template <size_t batch_size>
static ailego_force_inline void uniform_sq_l2_int8_batch_impl(
    const void *query, const void *const *vectors,
    const std::array<const void *, batch_size> &prefetch_ptrs, size_t dim,
    float *distances) {
  const int8_t *q = reinterpret_cast<const int8_t *>(query);

  __m512i accs[batch_size];
  for (size_t i = 0; i < batch_size; ++i) {
    accs[i] = _mm512_setzero_si512();
  }

  // Process 64 bytes (one cache line) per iteration.
  size_t d = 0;
  for (; d + 64 <= dim; d += 64) {
    // Load 64 query bytes
    __m512i q_zmm =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(q + d));

    // Phase 1: load all data vectors into registers first
    __m512i data_regs[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
      data_regs[i] = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
          reinterpret_cast<const int8_t *>(vectors[i]) + d));
    }

    // Phase 2: prefetch + compute (data already in registers)
    for (size_t i = 0; i < batch_size; ++i) {
      if (prefetch_ptrs[i]) {
        _mm_prefetch(
            reinterpret_cast<const char *>(
                reinterpret_cast<const int8_t *>(prefetch_ptrs[i]) + d),
            _MM_HINT_T0);
      }
      __m512i diff = _mm512_sub_epi8(data_regs[i], q_zmm);
      diff = _mm512_abs_epi8(diff);
      accs[i] = _mm512_dpbusd_epi32(accs[i], diff, diff);
    }
  }

  // Horizontal reduce each accumulator
  std::array<int, batch_size> results{};
  for (size_t i = 0; i < batch_size; ++i) {
    results[i] = _mm512_reduce_add_epi32(accs[i]);
  }

  // Handle remaining elements (dim not a multiple of 64)
  for (; d < dim; ++d) {
    int qv = static_cast<int>(q[d]);
    for (size_t i = 0; i < batch_size; ++i) {
      int diff = qv - static_cast<int>(
                          reinterpret_cast<const int8_t *>(vectors[i])[d]);
      results[i] += diff * diff;
    }
  }

  for (size_t i = 0; i < batch_size; ++i) {
    distances[i] = static_cast<float>(results[i]);
  }
}

// ---------------------------------------------------------------------------
// Public: single-vector squared Euclidean distance (int8, VNNI abs trick)
// ---------------------------------------------------------------------------
void uniform_squared_euclidean_int8_distance(const void *a, const void *b,
                                             size_t dim, float *distance) {
  const int8_t *lhs = reinterpret_cast<const int8_t *>(a);
  const int8_t *rhs = reinterpret_cast<const int8_t *>(b);

  // Four independent accumulators to break the data-dependency chain.
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  __m512i acc2 = _mm512_setzero_si512();
  __m512i acc3 = _mm512_setzero_si512();

  size_t d = 0;

  // Main loop: process 256 bytes (4 × 64) per iteration.
  for (; d + 256 <= dim; d += 256) {
    __m512i diff0 = _mm512_abs_epi8(_mm512_sub_epi8(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 0)),
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 0))));
    __m512i diff1 = _mm512_abs_epi8(_mm512_sub_epi8(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 64)),
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 64))));
    __m512i diff2 = _mm512_abs_epi8(_mm512_sub_epi8(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 128)),
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 128))));
    __m512i diff3 = _mm512_abs_epi8(_mm512_sub_epi8(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d + 192)),
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d + 192))));

    acc0 = _mm512_dpbusd_epi32(acc0, diff0, diff0);
    acc1 = _mm512_dpbusd_epi32(acc1, diff1, diff1);
    acc2 = _mm512_dpbusd_epi32(acc2, diff2, diff2);
    acc3 = _mm512_dpbusd_epi32(acc3, diff3, diff3);
  }

  // Bridge loop: 64-byte chunks for the remaining (dim % 256) bytes.
  for (; d + 64 <= dim; d += 64) {
    __m512i diff = _mm512_abs_epi8(_mm512_sub_epi8(
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(lhs + d)),
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(rhs + d))));
    acc0 = _mm512_dpbusd_epi32(acc0, diff, diff);
  }

  // Reduce four accumulators -> one, then horizontally to a scalar.
  __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1),
                                 _mm512_add_epi32(acc2, acc3));
  int result = _mm512_reduce_add_epi32(acc);

  // Scalar tail (dim not a multiple of 64).
  for (; d < dim; ++d) {
    int diff = static_cast<int>(lhs[d]) - static_cast<int>(rhs[d]);
    result += diff * diff;
  }

  *distance = static_cast<float>(result);
}

// ---------------------------------------------------------------------------
// Public: batch squared Euclidean distance (int8, no tail, no preprocessing)
// ---------------------------------------------------------------------------
void uniform_squared_euclidean_int8_batch_distance(const void *const *vectors,
                                                   const void *query, size_t n,
                                                   size_t dim,
                                                   float *distances) {
  static constexpr size_t batch_size = 4;
  static constexpr size_t prefetch_step = 2;

  size_t i = 0;
  for (; i + batch_size <= n; i += batch_size) {
    std::array<const void *, batch_size> prefetch_ptrs;
    for (size_t j = 0; j < batch_size; ++j) {
      size_t pi = i + j + batch_size * prefetch_step;
      prefetch_ptrs[j] = (pi < n) ? vectors[pi] : nullptr;
    }
    uniform_sq_l2_int8_batch_impl<batch_size>(query, &vectors[i], prefetch_ptrs,
                                              dim, distances + i);
  }
  // Tail (n % batch_size vectors): delegate to the single-vector kernel.
  // It already uses 4-way independent accumulators (see P1-2) and avoids
  // both an extra `batch_size=1` template instantiation and the per-call
  // std::array setup that the batch_impl path requires.
  for (; i < n; ++i) {
    uniform_squared_euclidean_int8_distance(vectors[i], query, dim,
                                            distances + i);
  }
}

}  // namespace zvec::turbo::avx512_vnni

#else  // no AVX512 support

namespace zvec::turbo::avx512_vnni {

void uniform_squared_euclidean_int8_distance(const void * /*a*/,
                                             const void * /*b*/, size_t /*dim*/,
                                             float * /*distance*/) {}

void uniform_squared_euclidean_int8_batch_distance(
    const void *const * /*vectors*/, const void * /*query*/, size_t /*n*/,
    size_t /*dim*/, float * /*distances*/) {}

}  // namespace zvec::turbo::avx512_vnni

#endif
