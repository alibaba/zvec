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
#include "distance_matrix_euclidean_utility.i"
#include "euclidean_distance_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__riscv_vector)
namespace {

static inline float SquaredEuclideanDistanceRVVImpl(const float *lhs,
                                                    const float *rhs,
                                                    size_t size) {
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (size != 0) {
    size_t vl = __riscv_vsetvl_e32m8(size);
    vfloat32m8_t v_lhs = __riscv_vle32_v_f32m8(lhs, vl);
    vfloat32m8_t v_rhs = __riscv_vle32_v_f32m8(rhs, vl);
    vfloat32m8_t v_d = __riscv_vfsub_vv_f32m8(v_lhs, v_rhs, vl);
    v_sum = __riscv_vfmacc_vv_f32m8_tu(v_sum, v_d, v_d, vl);
    lhs += vl;
    rhs += vl;
    size -= vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(v_red);
}

}  // namespace

//! Squared Euclidean Distance
float SquaredEuclideanDistanceRVV(const float *lhs, const float *rhs,
                                  size_t size) {
  return SquaredEuclideanDistanceRVVImpl(lhs, rhs, size);
}

#endif  // __riscv_vector

}  // namespace ailego
}  // namespace zvec
