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

#include <array>
#include <zvec/ailego/internal/platform.h>

namespace zvec::ailego::DistanceBatch {

#if defined(__riscv_vector)

void compute_one_to_many_inner_product_rvv_int8_1(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 1> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  const int8_t *q_ptr = query;
  const int8_t *m_ptr = ptrs[0];

  const size_t vlmax = __riscv_vsetvlmax_e8m2();
  vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlmax);

  size_t dim = 0;
  while (dim < dimensionality) {
    size_t vl = __riscv_vsetvl_e8m2(dimensionality - dim);

    vint8m2_t q8 = __riscv_vle8_v_i8m2(q_ptr + dim, vl);
    vint8m2_t m8 = __riscv_vle8_v_i8m2(m_ptr + dim, vl);
    vint16m4_t q16 = __riscv_vsext_vf2_i16m4(q8, vl);
    vint16m4_t m16 = __riscv_vsext_vf2_i16m4(m8, vl);

    acc = __riscv_vwmacc_vv_i32m8_tu(acc, q16, m16, vl);

    if (prefetch_ptrs[0] != nullptr) {
      ailego_prefetch(prefetch_ptrs[0] + dim);
    }

    dim += vl;
  }

  vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, zero, vlmax);
  results[0] = static_cast<float>(__riscv_vmv_x_s_i32m1_i32(red));
}

void compute_one_to_many_inner_product_rvv_int8_12(
    const int8_t *query, const int8_t **ptrs,
    std::array<const int8_t *, 12> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  const int8_t *q_ptr = query;

  const size_t vlmax = __riscv_vsetvlmax_e8m2();

  for (size_t channel = 0; channel < 12; ++channel) {
    const int8_t *m_ptr = ptrs[channel];
    vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlmax);

    size_t dim = 0;
    while (dim < dimensionality) {
      size_t vl = __riscv_vsetvl_e8m2(dimensionality - dim);

      vint8m2_t q8 = __riscv_vle8_v_i8m2(q_ptr + dim, vl);
      vint8m2_t m8 = __riscv_vle8_v_i8m2(m_ptr + dim, vl);
      vint16m4_t q16 = __riscv_vsext_vf2_i16m4(q8, vl);
      vint16m4_t m16 = __riscv_vsext_vf2_i16m4(m8, vl);

      acc = __riscv_vwmacc_vv_i32m8_tu(acc, q16, m16, vl);

      if (prefetch_ptrs[channel] != nullptr) {
        ailego_prefetch(prefetch_ptrs[channel] + dim);
      }

      dim += vl;
    }

    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t red = __riscv_vredsum_vs_i32m8_i32m1(acc, zero, vlmax);
    results[channel] = static_cast<float>(__riscv_vmv_x_s_i32m1_i32(red));
  }
}

#endif

}  // namespace zvec::ailego::DistanceBatch
