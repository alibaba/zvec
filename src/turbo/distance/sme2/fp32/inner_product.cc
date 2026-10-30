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

// Copyright 2026-present the gemv_bench contributors
// SPDX-License-Identifier: Apache-2.0

#include "sme2/fp32/inner_product.h"
#include <arm_sme.h>
#include "sme2/fp32/cosine.h"

namespace zvec::turbo::sme2 {
namespace {

// The ZA accumulator layout shares query loads across candidate rows: a ZA
// group contains four candidate rows for one dimension-vector position.
// Transposing the four loaded tuples in register names makes each FMLS
// broadcast one query vector to four rows.
__attribute__((always_inline)) inline void accumulate16(
    const void *const *vectors, size_t d, svcount_t pg,
    svfloat32x4_t q) __arm_streaming __arm_inout("za") {
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < 16; r += 4) {
    const auto a = svld1_f32_x4(pg, static_cast<const float *>(vectors[r]) + d);
    const auto b =
        svld1_f32_x4(pg, static_cast<const float *>(vectors[r + 1]) + d);
    const auto c =
        svld1_f32_x4(pg, static_cast<const float *>(vectors[r + 2]) + d);
    const auto e =
        svld1_f32_x4(pg, static_cast<const float *>(vectors[r + 3]) + d);
#define SME2_QUARTER(K)                                                   \
  svmls_single_za32_f32_vg1x4(                                            \
      r + K,                                                              \
      svcreate4_f32(svget4_f32(a, K), svget4_f32(b, K), svget4_f32(c, K), \
                    svget4_f32(e, K)),                                    \
      svget4_f32(q, K))
    SME2_QUARTER(0);
    SME2_QUARTER(1);
    SME2_QUARTER(2);
    SME2_QUARTER(3);
#undef SME2_QUARTER
  }
}

// Finish cosine before leaving streaming mode, avoiding a second ordinary
// NEON pass immediately after the SME2 kernel's streaming transition.
template <bool Cosine>
__attribute__((always_inline)) inline float finish_distance(
    float value) __arm_streaming __arm_in("za") {
  if constexpr (Cosine) {
    return value + 1.0f;
  } else {
    return value;
  }
}

template <bool Cosine>
__attribute__((always_inline)) inline void reduce16(
    float *distances) __arm_streaming __arm_inout("za") {
  const auto pg = svptrue_b32();
#pragma clang loop unroll(full)
  for (unsigned r = 0; r < 16; r += 4) {
    // Balanced reduction of the four dimension-vector positions; each ZA
    // addition operates on all four candidates together.
    svadd_za32_f32_vg1x4(r, svread_za32_f32_vg1x4(r + 1));
    svadd_za32_f32_vg1x4(r + 2, svread_za32_f32_vg1x4(r + 3));
    svadd_za32_f32_vg1x4(r, svread_za32_f32_vg1x4(r + 2));
    const auto sum = svread_za32_f32_vg1x4(r);
    distances[r] = finish_distance<Cosine>(svaddv_f32(pg, svget4_f32(sum, 0)));
    distances[r + 1] =
        finish_distance<Cosine>(svaddv_f32(pg, svget4_f32(sum, 1)));
    distances[r + 2] =
        finish_distance<Cosine>(svaddv_f32(pg, svget4_f32(sum, 2)));
    distances[r + 3] =
        finish_distance<Cosine>(svaddv_f32(pg, svget4_f32(sum, 3)));
  }
}

__attribute__((always_inline)) inline float reduce(
    unsigned slice) __arm_streaming __arm_in("za") {
  const auto sum = svread_za32_f32_vg1x4(slice);
  const auto pg = svptrue_b32();
  const auto lo = svadd_f32_x(pg, svget4_f32(sum, 0), svget4_f32(sum, 1));
  const auto hi = svadd_f32_x(pg, svget4_f32(sum, 2), svget4_f32(sum, 3));
  return svaddv_f32(pg, svadd_f32_x(pg, lo, hi));
}

template <bool Cosine>
__attribute__((always_inline)) inline void tile16(
    const void *const *vectors, const float *query, size_t dim, size_t width,
    float *distances) __arm_streaming __arm_inout("za") {
  svzero_za();
  const auto all = svptrue_c32();
  size_t d = 0;
  for (; dim - d >= width; d += width) {
    const auto q = svld1_f32_x4(all, query + d);
    accumulate16(vectors, d, all, q);
  }
  if (d != dim) {
    // Predication suppresses every access past the logical end of each row.
    const auto tail = svwhilelt_c32_u64(d, dim, 4);
    const auto q = svld1_f32_x4(tail, query + d);
    accumulate16(vectors, d, tail, q);
  }
  reduce16<Cosine>(distances);
}

template <bool Cosine>
__attribute__((always_inline)) inline void batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) __arm_streaming __arm_inout("za") {
  const auto *q = static_cast<const float *>(query);
  // ZA holds svcntw() independent groups of four FP32 vectors. The main
  // 16-row tile is valid at SVL >= 512 bits; shorter vector lengths use the
  // bounded generic path below rather than aliasing ZA accumulator slices.
  const size_t lanes = svcntw();
  const size_t width = lanes * 4;
  size_t r = 0;
  if (lanes >= 16) {
    for (; n - r >= 16; r += 16)
      tile16<Cosine>(vectors + r, q, dim, width, distances + r);
  }
  while (r != n) {
    const size_t count = n - r < lanes ? n - r : lanes;
    svzero_za();
    for (size_t d = 0; d < dim;) {
      const auto pg = svwhilelt_c32_u64(d, dim, 4);
      const auto query_part = svld1_f32_x4(pg, q + d);
      for (size_t i = 0; i < count; ++i) {
        const auto row =
            svld1_f32_x4(pg, static_cast<const float *>(vectors[r + i]) + d);
        svmls_za32_f32_vg1x4(static_cast<unsigned>(i), row, query_part);
      }
      // Avoid size_t overflow for a dimension near SIZE_MAX.
      d += dim - d < width ? dim - d : width;
    }
    for (size_t i = 0; i < count; ++i)
      distances[r + i] =
          finish_distance<Cosine>(reduce(static_cast<unsigned>(i)));
    r += count;
  }
}

}  // namespace

// The compiler implements streaming transitions and the ZA lazy-save ABI.
// Keeping these attributes on the definitions preserves Turbo's existing
// non-streaming function-pointer type in callers compiled for NEON.
__arm_locally_streaming __arm_new("za") void inner_product_fp32_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
  (void)extra_values;
  batch_distance<false>(vectors, query, n, dim, distances);
}

__arm_locally_streaming __arm_new("za") void cosine_fp32_batch_distance(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances, const void *const *extra_values) {
  (void)extra_values;
  batch_distance<true>(vectors, query, n, dim, distances);
}

}  // namespace zvec::turbo::sme2
