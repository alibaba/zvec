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
#include "norm1_matrix.h"
#include "norm_matrix_fp16.i"

namespace zvec {
namespace ailego {

#define NORM_FP32_STEP_GENERAL SA_FP32_GENERAL
#define NORM_FP32_STEP_SSE SA_FP32_SSE
#define NORM_FP32_STEP_AVX SA_FP32_AVX
#define NORM_FP32_STEP_AVX512 SA_FP32_AVX512
#define NORM_FP32_STEP_NEON SA_FP32_NEON
#define NORM_FP16_STEP_GENERAL SA_FP16_GENERAL
#define NORM_FP16_STEP_NEON SA_FP16_NEON

#if defined(__SSE__)
static const __m128 ABS_MASK_FP32_SSE =
    _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffu));
#endif  // __SSE__

#if defined(__AVX__)
static const __m256 ABS_MASK_FP32_AVX =
    _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffu));
#endif  // __AVX__

#if defined(__AVX512F__)
static const __m512 ABS_MASK_FP32_AVX512 =
    _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffffu));
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

//! Calculate sum of absolute (GENERAL)
#define SA_FP16_GENERAL(m, sum) sum += Float16::Absolute(m);

//! Calculate sum of absolute (NEON)
#define SA_FP16_NEON(v_m, v_sum) v_sum = vaddq_f16(vabsq_f16(v_m), v_sum);

#if (defined(__F16C__) && defined(__AVX__)) || \
    (defined(__ARM_NEON) && defined(__aarch64__))
//! Compute the L1-norm of vectors (FP16, M=1)
void Norm1Matrix<Float16, 1>::Compute(const ValueType *m, size_t dim,
                                      float *out) {
#if defined(__ARM_NEON)
  NORM_FP16_1_NEON(m, dim, out, )
#elif defined(__AVX512F__)
  NORM_FP16_1_AVX512(m, dim, out, )
#else
  NORM_FP16_1_AVX(m, dim, out, )
#endif
}
#endif  // (__F16C__ && __AVX__) || (__ARM_NEON && __aarch64__)

}  // namespace ailego
}  // namespace zvec