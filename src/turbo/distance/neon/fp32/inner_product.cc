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

#include "neon/fp32/inner_product.h"
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#else
#include "scalar/fp32/inner_product.h"
#endif

namespace zvec::turbo::neon {

#if defined(AILEGO_ARM64_NEON)
namespace {

#if defined(__clang__)
#define ZVEC_NEON_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define ZVEC_NEON_UNROLL _Pragma("GCC unroll 8")
#else
#define ZVEC_NEON_UNROLL
#endif

// Reuse each query load across several vectors. Independent FMA chains hide
// instruction latency; limiting the largest block to 16 accumulators keeps it
// within AArch64's SIMD register file without spilling to the stack.
template <size_t Batch, size_t Accumulators>
inline void inner_product_block(const void *const *vectors, const float *query,
                                size_t dim, float *distances) {
  static_assert(Accumulators == 2 || Accumulators == 4);
  const float *data[Batch];
  float32x4_t sums[Batch][Accumulators];
ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    data[b] = static_cast<const float *>(vectors[b]);
ZVEC_NEON_UNROLL
    for (size_t u = 0; u < Accumulators; ++u) {
      sums[b][u] = vdupq_n_f32(0.0f);
    }
  }

  size_t offset = 0;
  for (; dim - offset >= Accumulators * 4; offset += Accumulators * 4) {
    float32x4_t q[Accumulators];
ZVEC_NEON_UNROLL
    for (size_t u = 0; u < Accumulators; ++u) {
      q[u] = vld1q_f32(query + offset + u * 4);
    }
ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
ZVEC_NEON_UNROLL
      for (size_t u = 0; u < Accumulators; ++u) {
        sums[b][u] = vfmaq_f32(sums[b][u], q[u],
                              vld1q_f32(data[b] + offset + u * 4));
      }
    }
  }
  for (; dim - offset >= 4; offset += 4) {
    const float32x4_t q = vld1q_f32(query + offset);
ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
      sums[b][0] = vfmaq_f32(sums[b][0], q, vld1q_f32(data[b] + offset));
    }
  }

  float results[Batch];
ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    float32x4_t sum = vaddq_f32(sums[b][0], sums[b][1]);
    if constexpr (Accumulators == 4) {
      sum = vaddq_f32(sum, vaddq_f32(sums[b][2], sums[b][3]));
    }
    results[b] = vaddvq_f32(sum);
  }
  // Keep the scalar remainder explicitly bounded to three elements. Otherwise
  // compilers may emit a second general vector loop after the SIMD loops.
  const size_t tail = dim & 3;
ZVEC_NEON_UNROLL
  for (size_t t = 0; t < tail; ++t) {
    const float q = query[offset + t];
ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
      results[b] += q * data[b][offset + t];
    }
  }
ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    distances[b] = -results[b];
  }
}

// Keep the large batch register frame out of the single-vector fast path.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void inner_product_batch(const void *const *vectors, const void *query, size_t n,
                         size_t dim, float *distances) {
  const float *typed_query = static_cast<const float *>(query);
  size_t i = 0;
  for (; n - i >= 8; i += 8) {
    inner_product_block<8, 2>(vectors + i, typed_query, dim, distances + i);
  }
  if (n - i >= 4) {
    inner_product_block<4, 2>(vectors + i, typed_query, dim, distances + i);
    i += 4;
  }
  if (n - i >= 2) {
    inner_product_block<2, 4>(vectors + i, typed_query, dim, distances + i);
    i += 2;
  }
  if (i < n) {
    inner_product_fp32_distance(vectors[i], query, dim, distances + i);
  }
}

#undef ZVEC_NEON_UNROLL

}  // namespace
#endif

void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(AILEGO_ARM64_NEON)
  // Small vectors cannot amortize four-accumulator setup and reduction.
  if (dim < 32) {
    inner_product_block<1, 2>(&a, static_cast<const float *>(b), dim, distance);
  } else {
    inner_product_block<1, 4>(&a, static_cast<const float *>(b), dim, distance);
  }
#else
  scalar::inner_product_fp32_distance(a, b, dim, distance);
#endif
}

void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
#if defined(AILEGO_ARM64_NEON)
  if (dim == 0) {
    for (size_t i = 0; i < n; ++i) {
      distances[i] = -0.0f;
    }
    return;
  }
  if (n == 1) {
    inner_product_fp32_distance(vectors[0], query, dim, distances);
    return;
  }

  inner_product_batch(vectors, query, n, dim, distances);
#else
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
#endif
}

}  // namespace zvec::turbo::neon
