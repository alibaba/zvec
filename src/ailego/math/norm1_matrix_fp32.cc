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
#include "norm1_matrix.h"
#include "norm_matrix_fp32.i"

namespace zvec {
namespace ailego {

#define NORM_FP32_STEP_GENERAL SA_FP32_GENERAL
#define NORM_FP32_STEP_SSE SA_FP32_SSE
#define NORM_FP32_STEP_AVX SA_FP32_AVX
#define NORM_FP32_STEP_AVX512 SA_FP32_AVX512
#define NORM_FP32_STEP_NEON SA_FP32_NEON

#if defined(__SSE__)
#define ABS_MASK_FP32_SSE _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu))
#endif  // __SSE__

#if defined(__AVX__)
#define ABS_MASK_FP32_AVX _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu))
#endif  // __AVX__

#if defined(__AVX512F__)
#define ABS_MASK_FP32_AVX512 _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffffu))
#endif  // __AVX512F__

//! Calculate sum of absolute (GENERAL)
#define SA_FP32_GENERAL(m, sum) sum += FastAbs(m);

//! Calculate sum of absolute (SSE)
#define SA_FP32_SSE(xmm_m, xmm_sum) \
  xmm_sum = _mm_add_ps(_mm_and_ps(xmm_m, ABS_MASK_FP32_SSE), xmm_sum);

//! Calculate sum of absolute (AVX)
#define SA_FP32_AVX(ymm_m, ymm_sum) \
  ymm_sum = _mm256_add_ps(_mm256_and_ps(ymm_m, ABS_MASK_FP32_AVX), ymm_sum);

//! Calculate sum of absolute (AVX512)
#define SA_FP32_AVX512(zmm_m, zmm_sum) \
  zmm_sum = _mm512_add_ps(_mm512_and_ps(zmm_m, ABS_MASK_FP32_AVX512), zmm_sum);

//! Calculate sum of absolute (NEON)
#define SA_FP32_NEON(v_m, v_sum) v_sum = vaddq_f32(vabsq_f32(v_m), v_sum);

#if defined(__riscv_vector)
//! Compute the L1-norm of vector (RVV)
static inline float Norm1RVV(const float *m, size_t dim) {
  const size_t vlmax = __riscv_vsetvlmax_e32m8();
  vfloat32m8_t v_sum = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (dim != 0) {
    const size_t vl = __riscv_vsetvl_e32m8(dim);
    vfloat32m8_t v_m = __riscv_vle32_v_f32m8(m, vl);
    vfloat32m8_t v_abs = __riscv_vfabs_v_f32m8(v_m, vl);
    v_sum = __riscv_vfadd_vv_f32m8_tu(v_sum, v_sum, v_abs, vl);
    m += vl;
    dim -= vl;
  }

  vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  vfloat32m1_t v_reduce =
      __riscv_vfredusum_vs_f32m8_f32m1(v_sum, v_zero, vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(v_reduce);
}
#endif  // __riscv_vector

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__)) || \
    defined(__riscv_vector)
//! Compute the L1-norm of vectors (FP32, M=1)
void Norm1Matrix<float, 1>::Compute(const ValueType *m, size_t dim,
                                    float *out) {
#if defined(__ARM_NEON)
  NORM_FP32_1_NEON(m, dim, out, )
#elif defined(__riscv_vector)
  *out = Norm1RVV(m, dim);
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    NORM_FP32_1_AVX512(m, dim, out, )
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    NORM_FP32_1_AVX(m, dim, out, )
    return;
  }
#endif
  NORM_FP32_1_SSE(m, dim, out, )
#endif
}
#endif  // __SSE__ || (__ARM_NEON && __aarch64__) || __riscv_vector

}  // namespace ailego
}  // namespace zvec
