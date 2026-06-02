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
#include "norm2_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__riscv_zvfh)
namespace {

static inline float SquaredNorm2RVVImpl(const Float16 *m, size_t dim) {
  const _Float16 *m_fp16 = reinterpret_cast<const _Float16 *>(m);
  const size_t vlmax = __riscv_vsetvlmax_e16m4();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (dim != 0) {
    const size_t vl = __riscv_vsetvl_e16m4(dim);
    vfloat16m4_t v_m = __riscv_vle16_v_f16m4(m_fp16, vl);
    v_sum = __riscv_vfwmacc_vv_f32m8_tu(v_sum, v_m, v_m, vl);
    m_fp16 += vl;
    dim -= vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t v_reduce =
      __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(v_reduce);
}

}  // namespace

float SquaredNorm2RVV(const Float16 *m, size_t dim) {
  return SquaredNorm2RVVImpl(m, dim);
}

float Norm2RVV(const Float16 *m, size_t dim) {
  return std::sqrt(SquaredNorm2RVVImpl(m, dim));
}
#endif  // __riscv_zvfh

}  // namespace ailego
}  // namespace zvec
