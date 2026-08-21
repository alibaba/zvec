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

#if defined(__riscv_zvfh)
namespace {

static inline float HorizontalReduceF32M8(vfloat32m8_t value, size_t vlmax) {
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t red = __riscv_vfredusum_vs_f32m8_f32m1(value, zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(red);
}

static inline float InnerProductRVV(const Float16 *lhs, const Float16 *rhs,
                                    size_t size) {
  const _Float16 *lhs_fp16 = reinterpret_cast<const _Float16 *>(lhs);
  const _Float16 *rhs_fp16 = reinterpret_cast<const _Float16 *>(rhs);
  const size_t vlmax = __riscv_vsetvlmax_e16m4();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (size != 0) {
    const size_t vl = __riscv_vsetvl_e16m4(size);
    vfloat16m4_t v_lhs = __riscv_vle16_v_f16m4(lhs_fp16, vl);
    vfloat16m4_t v_rhs = __riscv_vle16_v_f16m4(rhs_fp16, vl);
    v_sum = __riscv_vfwmacc_vv_f32m8_tu(v_sum, v_lhs, v_rhs, vl);
    lhs_fp16 += vl;
    rhs_fp16 += vl;
    size -= vl;
  }

  return HorizontalReduceF32M8(v_sum, vlmax);
}

static inline float SquaredNormRVV(const Float16 *src, size_t size) {
  const _Float16 *src_fp16 = reinterpret_cast<const _Float16 *>(src);
  const size_t vlmax = __riscv_vsetvlmax_e16m4();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (size != 0) {
    const size_t vl = __riscv_vsetvl_e16m4(size);
    vfloat16m4_t v_src = __riscv_vle16_v_f16m4(src_fp16, vl);
    v_sum = __riscv_vfwmacc_vv_f32m8_tu(v_sum, v_src, v_src, vl);
    src_fp16 += vl;
    size -= vl;
  }

  return HorizontalReduceF32M8(v_sum, vlmax);
}

}  // namespace

float MipsEuclideanDistanceSphericalInjectionRVV(const Float16 *lhs,
                                                 const Float16 *rhs,
                                                 size_t size, float e2) {
  float sum = InnerProductRVV(lhs, rhs, size);
  float u2 = SquaredNormRVV(lhs, size);
  float v2 = SquaredNormRVV(rhs, size);
  return ComputeSphericalInjection(sum, u2, v2, e2);
}

float MipsEuclideanDistanceRepeatedQuadraticInjectionRVV(const Float16 *lhs,
                                                         const Float16 *rhs,
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

#endif  // __riscv_zvfh

}  // namespace ailego
}  // namespace zvec
