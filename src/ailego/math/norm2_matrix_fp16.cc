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
#include "ailego/internal/cpu_features.h"
#include "norm2_matrix.h"
#include "norm_matrix_fp16.i"

namespace zvec {
namespace ailego {

#define NORM_FP32_STEP_GENERAL SS_FP32_GENERAL
#define NORM_FP32_STEP_SSE SS_FP32_SSE
#define NORM_FP32_STEP_AVX SS_FP32_AVX
#define NORM_FP32_STEP_AVX512 SS_FP32_AVX512
#define NORM_FP32_STEP_NEON SS_FP32_NEON
#define NORM_FP16_STEP_GENERAL SS_FP16_GENERAL
#define NORM_FP16_STEP_NEON SS_FP16_NEON

//! Calculate sum of squared (GENERAL)
#define SS_FP32_GENERAL(m, sum) sum += (m) * (m);

//! Calculate sum of squared (SSE)
#define SS_FP32_SSE(xmm_m, xmm_sum) \
  xmm_sum = _mm_fmadd_ps(xmm_m, xmm_m, xmm_sum);

//! Calculate sum of squared (AVX)
#define SS_FP32_AVX(ymm_m, ymm_sum) \
  ymm_sum = _mm256_fmadd_ps(ymm_m, ymm_m, ymm_sum);

//! Calculate sum of squared (AVX512)
#define SS_FP32_AVX512(zmm_m, zmm_sum) \
  zmm_sum = _mm512_fmadd_ps(zmm_m, zmm_m, zmm_sum);

//! Calculate sum of squared (NEON)
#define SS_FP32_NEON(v_m, v_sum) v_sum = vfmaq_f32(v_sum, v_m, v_m);

//! Calculate sum of squared (GENERAL)
#define SS_FP16_GENERAL(m, sum) sum += (m) * (m);

//! Calculate sum of squared (NEON)
#define SS_FP16_NEON(v_m, v_sum) v_sum = vfmaq_f16(v_sum, v_m, v_m);

#if defined(__riscv_zvfh)
//! Compute the squared L2-norm of vector (RVV)
static inline float SquaredNorm2RVV(const Float16 *m, size_t dim) {
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

//! Compute the L2-norm of vector (RVV)
static inline float Norm2RVV(const Float16 *m, size_t dim) {
  return std::sqrt(SquaredNorm2RVV(m, dim));
}
#endif  // __riscv_zvfh

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__)) || defined(__riscv_zvfh)
//! Compute the L2-norm of vectors (FP16, M=1)
void Norm2Matrix<Float16, 1>::Compute(const ValueType *m, size_t dim,
                                      float *out) {
#if defined(__ARM_NEON)
  NORM_FP16_1_NEON(m, dim, out, std::sqrt)
#elif defined(__riscv_zvfh)
  *out = Norm2RVV(m, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    NORM_FP16_1_AVX512(m, dim, out, std::sqrt)
    return;
  }
#endif
  NORM_FP16_1_AVX(m, dim, out, std::sqrt)
#endif
}

//! Compute the L2-norm of vectors (FP16, M=1)
void SquaredNorm2Matrix<Float16, 1>::Compute(const ValueType *m, size_t dim,
                                             float *out) {
#if defined(__ARM_NEON)
  NORM_FP16_1_NEON(m, dim, out, )
#elif defined(__riscv_zvfh)
  *out = SquaredNorm2RVV(m, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    NORM_FP16_1_AVX512(m, dim, out, )
    return;
  }
#endif
  NORM_FP16_1_AVX(m, dim, out, )
#endif
}
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__) || __riscv_zvfh

}  // namespace ailego
}  // namespace zvec