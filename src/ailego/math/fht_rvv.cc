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
#include <cstddef>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace ailego {

void fht_flip_sign_rvv(const uint8_t *flip, float *data, size_t dim) {
  const uint32_t sign_bit = 0x80000000u;
  size_t i = 0;
  while (i < dim) {
    size_t vl = __riscv_vsetvl_e32m8(dim - i);
    vbool4_t mask = __riscv_vlm_v_b4(flip + i / 8, vl);
    vfloat32m8_t v = __riscv_vle32_v_f32m8(data + i, vl);
    vuint32m8_t bits = __riscv_vreinterpret_v_f32m8_u32m8(v);
    bits = __riscv_vxor_vx_u32m8_mu(mask, bits, bits, sign_bit, vl);
    __riscv_vse32_v_f32m8(data + i, __riscv_vreinterpret_v_u32m8_f32m8(bits),
                          vl);
    i += vl;
  }
}

void fht_kacs_walk_rvv(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  size_t i = 0;
  while (i < half) {
    size_t vl = __riscv_vsetvl_e32m8(half - i);
    vfloat32m8_t x = __riscv_vle32_v_f32m8(data + i, vl);
    vfloat32m8_t y = __riscv_vle32_v_f32m8(data + i + offset, vl);
    __riscv_vse32_v_f32m8(data + i, __riscv_vfadd_vv_f32m8(x, y, vl), vl);
    __riscv_vse32_v_f32m8(data + i + offset, __riscv_vfsub_vv_f32m8(x, y, vl),
                          vl);
    i += vl;
  }
  if (base != 0) {
    data[half] *= std::sqrt(2.0f);
  }
}

void fht_inv_kacs_walk_rvv(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  if (base != 0) {
    data[half] *= std::sqrt(0.5f);
  }
  size_t i = 0;
  while (i < half) {
    size_t vl = __riscv_vsetvl_e32m8(half - i);
    vfloat32m8_t a = __riscv_vle32_v_f32m8(data + i, vl);
    vfloat32m8_t b = __riscv_vle32_v_f32m8(data + i + offset, vl);
    vfloat32m8_t t = __riscv_vfadd_vv_f32m8(a, b, vl);
    __riscv_vse32_v_f32m8(data + i, __riscv_vfmul_vf_f32m8(t, 0.5f, vl), vl);
    t = __riscv_vfsub_vv_f32m8(a, b, vl);
    __riscv_vse32_v_f32m8(data + i + offset,
                          __riscv_vfmul_vf_f32m8(t, 0.5f, vl), vl);
    i += vl;
  }
}

void fht_inplace_rvv(float *data, size_t n) {
  for (size_t len = 1; len < n; len <<= 1) {
    size_t step = len << 1;
    for (size_t i = 0; i < n; i += step) {
      size_t j = 0;
      while (j < len) {
        size_t vl = __riscv_vsetvl_e32m8(len - j);
        vfloat32m8_t u = __riscv_vle32_v_f32m8(data + i + j, vl);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(data + i + j + len, vl);
        __riscv_vse32_v_f32m8(data + i + j, __riscv_vfadd_vv_f32m8(u, v, vl),
                              vl);
        __riscv_vse32_v_f32m8(data + i + j + len,
                              __riscv_vfsub_vv_f32m8(u, v, vl), vl);
        j += vl;
      }
    }
  }
}

}  // namespace ailego
}  // namespace zvec
