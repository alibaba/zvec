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

// Native ARMv8.2 FP16 (FEAT_FP16) variants of the FP16 MIPS-euclidean
// kernels. This TU is compiled with an FP16-capable -march (e.g.
// -march=armv8.2-a+fp16, see src/ailego/CMakeLists.txt), which defines
// __ARM_FEATURE_FP16_VECTOR_ARITHMETIC. Callers must runtime-gate
// invocation on CpuFeatures FP16.

#include "distance_matrix_accum_fp16.i"
#include "distance_matrix_mips_utility.i"
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(AILEGO_ARM64_GNU_LIKE) && \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
//! Compute the Inner Product between p and q, and each Squared L2-Norm value
static float InnerProductAndSquaredNormFp16NEONFP16(const Float16 *lhs,
                                                    const Float16 *rhs,
                                                    size_t size, float *sql,
                                                    float *sqr) {
  const Float16 *last = lhs + size;
  const Float16 *last_aligned = lhs + ((size >> 3) << 3);
  float16x8_t v_sum = vdupq_n_f16(0);
  float16x8_t v_sum_norm1 = vdupq_n_f16(0);
  float16x8_t v_sum_norm2 = vdupq_n_f16(0);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    float16x8_t v_lhs = vld1q_f16((const float16_t *)lhs);
    float16x8_t v_rhs = vld1q_f16((const float16_t *)rhs);
    v_sum = vfmaq_f16(v_sum, v_lhs, v_rhs);
    v_sum_norm1 = vfmaq_f16(v_sum_norm1, v_lhs, v_lhs);
    v_sum_norm2 = vfmaq_f16(v_sum_norm2, v_rhs, v_rhs);
  }
  if (last >= last_aligned + 4) {
    float16x8_t v_lhs = vcombine_f16(vld1_f16((const float16_t *)lhs),
                                     vreinterpret_f16_u64(vdup_n_u64(0ul)));
    float16x8_t v_rhs = vcombine_f16(vld1_f16((const float16_t *)rhs),
                                     vreinterpret_f16_u64(vdup_n_u64(0ul)));
    v_sum = vfmaq_f16(v_sum, v_lhs, v_rhs);
    v_sum_norm1 = vfmaq_f16(v_sum_norm1, v_lhs, v_lhs);
    v_sum_norm2 = vfmaq_f16(v_sum_norm2, v_rhs, v_rhs);
    lhs += 4;
    rhs += 4;
  }

  float result = HorizontalAdd_FP16_NEON(v_sum);
  float norm1 = HorizontalAdd_FP16_NEON(v_sum_norm1);
  float norm2 = HorizontalAdd_FP16_NEON(v_sum_norm2);

  switch (last - lhs) {
    case 3:
      FMA_FP16_GENERAL(lhs[2], rhs[2], result, norm1, norm2);
      /* FALLTHRU */
    case 2:
      FMA_FP16_GENERAL(lhs[1], rhs[1], result, norm1, norm2);
      /* FALLTHRU */
    case 1:
      FMA_FP16_GENERAL(lhs[0], rhs[0], result, norm1, norm2);
  }
  *sql = norm1;
  *sqr = norm2;
  return result;
}

float MipsEuclideanDistanceSphericalInjectionFp16NEONFP16(const Float16 *lhs,
                                                          const Float16 *rhs,
                                                          size_t size,
                                                          float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp16NEONFP16(lhs, rhs, size, &u2, &v2);

  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionFp16NEONFP16(
    const Float16 *lhs, const Float16 *rhs, size_t size, size_t m, float e2) {
  float u2{0.0f};
  float v2{0.0f};
  float sum{0.0f};

  sum = InnerProductAndSquaredNormFp16NEONFP16(lhs, rhs, size, &u2, &v2);

  sum = e2 * (u2 + v2 - 2 * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    sum += (u2 - v2) * (u2 - v2);
    u2 = u2 * u2;
    v2 = v2 * v2;
  }

  return sum;
}
#endif  // AILEGO_ARM64_GNU_LIKE && __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

}  // namespace ailego
}  // namespace zvec
