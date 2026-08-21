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

void compute_one_to_many_inner_product_rvv_fp32_1(
    const float *query, const float **ptrs,
    std::array<const float *, 1> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  const float *q_ptr = query;
  const float *m_ptr = ptrs[0];

  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  size_t dim = 0;
  while (dim < dimensionality) {
    size_t vl = __riscv_vsetvl_e32m8(dimensionality - dim);

    vfloat32m8_t q = __riscv_vle32_v_f32m8(q_ptr + dim, vl);
    vfloat32m8_t m = __riscv_vle32_v_f32m8(m_ptr + dim, vl);

    acc = __riscv_vfmacc_vv_f32m8_tu(acc, q, m, vl);

    if (prefetch_ptrs[0] != nullptr) {
      ailego_prefetch(prefetch_ptrs[0] + dim);
    }

    dim += vl;
  }

  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t red = __riscv_vfredusum_vs_f32m8_f32m1(acc, zero, vlmax);
  results[0] = __riscv_vfmv_f_s_f32m1_f32(red);
}

void compute_one_to_many_inner_product_rvv_fp32_12(
    const float *query, const float **ptrs,
    std::array<const float *, 12> &prefetch_ptrs, size_t dimensionality,
    float *results) {
  const float *q_ptr = query;

  const size_t vlmax = __riscv_vsetvlmax_e32m8();

  for (size_t channel = 0; channel < 12; ++channel) {
    const float *m_ptr = ptrs[channel];
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

    size_t dim = 0;
    while (dim < dimensionality) {
      size_t vl = __riscv_vsetvl_e32m8(dimensionality - dim);

      vfloat32m8_t q = __riscv_vle32_v_f32m8(q_ptr + dim, vl);
      vfloat32m8_t m = __riscv_vle32_v_f32m8(m_ptr + dim, vl);

      acc = __riscv_vfmacc_vv_f32m8_tu(acc, q, m, vl);

      if (prefetch_ptrs[channel] != nullptr) {
        ailego_prefetch(prefetch_ptrs[channel] + dim);
      }

      dim += vl;
    }

    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m8_f32m1(acc, zero, vlmax);
    results[channel] = __riscv_vfmv_f_s_f32m1_f32(red);
  }
}

#endif

}  // namespace zvec::ailego::DistanceBatch
