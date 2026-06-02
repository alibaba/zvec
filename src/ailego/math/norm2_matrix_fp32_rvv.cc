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

#include <cmath>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

#if defined(__riscv_vector)
namespace {

float SquaredNorm2RVVImpl(const float *m, size_t dim) {
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (dim != 0) {
    const size_t vl = __riscv_vsetvl_e32m8(dim);
    vfloat32m8_t v_m = __riscv_vle32_v_f32m8(m, vl);
    v_sum = __riscv_vfmacc_vv_f32m8_tu(v_sum, v_m, v_m, vl);
    m += vl;
    dim -= vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t v_reduce =
      __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(v_reduce);
}

}  // namespace

float SquaredNorm2RVV(const float *m, size_t dim) {
  return SquaredNorm2RVVImpl(m, dim);
}

float Norm2RVV(const float *m, size_t dim) {
  return std::sqrt(SquaredNorm2RVVImpl(m, dim));
}
#endif  // __riscv_vector

}  // namespace ailego
}  // namespace zvec
