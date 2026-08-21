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
#include "inner_product_matrix.h"

namespace zvec {
namespace ailego {

#if defined(__riscv_vector)
namespace {

static inline float InnerProductRVVImpl(const int8_t *lhs, const int8_t *rhs,
                                        size_t size) {
  const size_t vlmax = __riscv_vsetvlmax_e8m2();
  vint32m8_t v_sum = __riscv_vmv_v_x_i32m8(0, vlmax);

  while (size != 0) {
    const size_t vl = __riscv_vsetvl_e8m2(size);
    const vint8m2_t v_lhs8 = __riscv_vle8_v_i8m2(lhs, vl);
    const vint8m2_t v_rhs8 = __riscv_vle8_v_i8m2(rhs, vl);
    const vint16m4_t v_lhs16 = __riscv_vsext_vf2_i16m4(v_lhs8, vl);
    const vint16m4_t v_rhs16 = __riscv_vsext_vf2_i16m4(v_rhs8, vl);
    v_sum = __riscv_vwmacc_vv_i32m8_tu(v_sum, v_lhs16, v_rhs16, vl);
    lhs += vl;
    rhs += vl;
    size -= vl;
  }

  const vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t v_red = __riscv_vredsum_vs_i32m8_i32m1(v_sum, v_zero, vlmax);
  return static_cast<float>(__riscv_vmv_x_s_i32m1_i32(v_red));
}

}  // namespace

float InnerProductRVV(const int8_t *lhs, const int8_t *rhs, size_t size) {
  return InnerProductRVVImpl(lhs, rhs, size);
}

float MinusInnerProductRVV(const int8_t *lhs, const int8_t *rhs, size_t size) {
  return -InnerProductRVVImpl(lhs, rhs, size);
}

#endif  // __riscv_vector

}  // namespace ailego
}  // namespace zvec
