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

#include <ailego/internal/cpu_features.h>
#include "norm2_matrix.h"
#include "norm_matrix_fp32.i"

namespace zvec {
namespace ailego {

#define NORM_FP32_STEP_GENERAL SS_FP32_GENERAL
#define NORM_FP32_STEP_SSE SS_FP32_SSE
#define NORM_FP32_STEP_AVX SS_FP32_AVX
#define NORM_FP32_STEP_AVX512 SS_FP32_AVX512
#define NORM_FP32_STEP_NEON SS_FP32_NEON

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

#if defined(__SSE__) || (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the L2-norm of vectors (FP32, M=1)
void Norm2Matrix<float, 1>::Compute(const ValueType *m, size_t dim,
                                    float *out) {
#if defined(__ARM_NEON)
  NORM_FP32_1_NEON(m, dim, out, std::sqrt)
#else
#if defined(__AVX512F__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    NORM_FP32_1_AVX512(m, dim, out, std::sqrt)
    return;
  }
#endif
#if defined(__AVX__)
  if (zvec::ailego::internal::CpuFeatures::static_flags_.AVX) {
    NORM_FP32_1_AVX(m, dim, out, std::sqrt)
    return;
  }
#endif
  NORM_FP32_1_SSE(m, dim, out, std::sqrt)
#endif
}

//! Compute the squared L2-norm of vectors (FP32, M=1)
void SquaredNorm2Matrix<float, 1>::Compute(const ValueType *m, size_t dim,
                                           float *out) {
#if defined(__ARM_NEON)
  NORM_FP32_1_NEON(m, dim, out, )
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
#endif  // __SSE__ || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec
