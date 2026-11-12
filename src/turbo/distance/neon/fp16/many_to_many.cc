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

#include "neon/fp16/many_to_many.h"
#include <zvec/ailego/utility/float_helper.h>
#include "common/fp16_common.h"
#include "neon/fp16/cosine.h"
#include "neon/fp16/inner_product.h"
#if defined(AILEGO_ARM64_GNU_LIKE)
#include <arm_neon.h>
#endif

namespace zvec::turbo::neon {
namespace {
#if defined(AILEGO_ARM64_GNU_LIKE)
// Four candidates and two queries share FP16 conversion. Two accumulators
// per pair keep the dependency chains short while fitting ARM64 registers.
template <size_t Rows>
void tile(const void *const *vectors, const void *const *queries, size_t dim,
          size_t stride, float *out) {
  const auto *q0 = static_cast<const float16_t *>(queries[0]);
  const auto *q1 = static_cast<const float16_t *>(queries[1]);
  float32x4_t lo[2][Rows], hi[2][Rows];
  for (size_t j = 0; j < 2; ++j) {
    for (size_t i = 0; i < Rows; ++i) {
      lo[j][i] = hi[j][i] = vdupq_n_f32(0.0f);
    }
  }
  size_t d = 0;
  for (; d + 8 <= dim; d += 8) {
    const auto y0 = vld1q_f16(q0 + d), y1 = vld1q_f16(q1 + d);
    const auto q0lo = vcvt_f32_f16(vget_low_f16(y0));
    const auto q0hi = vcvt_high_f32_f16(y0);
    const auto q1lo = vcvt_f32_f16(vget_low_f16(y1));
    const auto q1hi = vcvt_high_f32_f16(y1);
    for (size_t i = 0; i < Rows; ++i) {
      const auto x = vld1q_f16(static_cast<const float16_t *>(vectors[i]) + d);
      const auto xlo = vcvt_f32_f16(vget_low_f16(x));
      const auto xhi = vcvt_high_f32_f16(x);
      lo[0][i] = vfmaq_f32(lo[0][i], xlo, q0lo);
      hi[0][i] = vfmaq_f32(hi[0][i], xhi, q0hi);
      lo[1][i] = vfmaq_f32(lo[1][i], xlo, q1lo);
      hi[1][i] = vfmaq_f32(hi[1][i], xhi, q1hi);
    }
  }
  for (size_t j = 0; j < 2; ++j) {
    const auto *q = static_cast<const ailego::Float16 *>(queries[j]);
    for (size_t i = 0; i < Rows; ++i) {
      float sum = vaddvq_f32(vaddq_f32(lo[j][i], hi[j][i]));
      const auto *v = static_cast<const ailego::Float16 *>(vectors[i]);
      for (size_t tail = d; tail < dim; ++tail) {
        sum += float(v[tail]) * float(q[tail]);
      }
      out[j * stride + i] = -sum;
    }
  }
}
#endif
}  // namespace

void inner_product_fp16_many_to_many_distance(const void *const *vectors,
                                              const void *const *queries,
                                              size_t m, size_t n, size_t dim,
                                              float *out,
                                              const void *const *extra_values) {
  if (m == 0 || n == 0) return;
  if (m == 1) {
    // IP is symmetric; one candidate can reuse batch over all queries.
    inner_product_fp16_batch_distance(queries, vectors[0], n, dim, out,
                                      nullptr);
    return;
  }
  size_t j = 0;
#if defined(AILEGO_ARM64_GNU_LIKE)
  for (; j + 2 <= n; j += 2) {
    size_t i = 0;
    for (; i + 4 <= m; i += 4) {
      tile<4>(vectors + i, queries + j, dim, m, out + j * m + i);
    }
    if (i + 2 <= m) {
      tile<2>(vectors + i, queries + j, dim, m, out + j * m + i);
      i += 2;
    }
    if (i < m) {
      for (size_t k = j; k < j + 2; ++k) {
        inner_product_fp16_batch_distance(
            vectors + i, queries[k], m - i, dim, out + k * m + i,
            extra_values ? extra_values + i : nullptr);
      }
    }
  }
#endif
  for (; j < n; ++j) {
    inner_product_fp16_batch_distance(vectors, queries[j], m, dim, out + j * m,
                                      extra_values);
  }
}

void cosine_fp16_many_to_many_distance(const void *const *vectors,
                                       const void *const *queries, size_t m,
                                       size_t n, size_t dim, float *out,
                                       const void *const *extra_values) {
  if (m == 0 || n == 0) return;
  if (n == 1) {
    cosine_fp16_batch_distance(vectors, queries[0], m, dim, out, extra_values);
    return;
  }
  if (m == 1) {
    cosine_fp16_batch_distance(queries, vectors[0], n, dim, out, nullptr);
    return;
  }
  inner_product_fp16_many_to_many_distance(vectors, queries, m, n, dim, out,
                                           extra_values);
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < m; ++i) out[j * m + i] += 1.0f;
  }
}

}  // namespace zvec::turbo::neon
