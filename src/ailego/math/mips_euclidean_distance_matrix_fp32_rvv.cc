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

#include <zvec/ailego/internal/platform.h>
#include "mips_euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__riscv_vector)
namespace {

static inline float HorizontalReduceF32M8(vfloat32m8_t value, size_t vlmax) {
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t red = __riscv_vfredusum_vs_f32m8_f32m1(value, zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(red);
}

static inline float InnerProductRVV(const float *lhs, const float *rhs,
                                    size_t size) {
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (size != 0) {
    size_t vl = __riscv_vsetvl_e32m8(size);
    vfloat32m8_t v_lhs = __riscv_vle32_v_f32m8(lhs, vl);
    vfloat32m8_t v_rhs = __riscv_vle32_v_f32m8(rhs, vl);
    v_sum = __riscv_vfmacc_vv_f32m8_tu(v_sum, v_lhs, v_rhs, vl);
    lhs += vl;
    rhs += vl;
    size -= vl;
  }

  return HorizontalReduceF32M8(v_sum, vlmax);
}

static inline float SquaredNormRVV(const float *src, size_t size) {
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (size != 0) {
    size_t vl = __riscv_vsetvl_e32m8(size);
    vfloat32m8_t v_src = __riscv_vle32_v_f32m8(src, vl);
    v_sum = __riscv_vfmacc_vv_f32m8_tu(v_sum, v_src, v_src, vl);
    src += vl;
    size -= vl;
  }

  return HorizontalReduceF32M8(v_sum, vlmax);
}

}  // namespace

float MipsEucldeanDistanceSphericalInjectionRVV(const float *lhs,
                                                const float *rhs, size_t size,
                                                float e2) {
  float sum = InnerProductRVV(lhs, rhs, size);
  float u2 = SquaredNormRVV(lhs, size);
  float v2 = SquaredNormRVV(rhs, size);
  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEucldeanDistanceRepeatedQuadraticInjectionRVV(const float *lhs,
                                                        const float *rhs,
                                                        size_t size, size_t m,
                                                        float e2) {
  float sum = InnerProductRVV(lhs, rhs, size);
  float u2 = SquaredNormRVV(lhs, size);
  float v2 = SquaredNormRVV(rhs, size);

  sum = e2 * (u2 + v2 - 2.0f * sum);
  u2 *= e2;
  v2 *= e2;
  for (size_t i = 0; i < m; ++i) {
    float d = u2 - v2;
    sum += d * d;
    u2 = u2 * u2;
    v2 = v2 * v2;
  }
  return sum;
}

#endif  // __riscv_vector

}  // namespace ailego
}  // namespace zvec
