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

#include "normalizer.h"

namespace zvec {
namespace ailego {

#if defined(AILEGO_ARM64_NEON)
static inline void NormalizeNEON(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 3) << 3);

  float32x4_t v_norm = vdupq_n_f32(norm);
  for (; arr != last_aligned; arr += 8) {
    vst1q_f32(arr + 0, vdivq_f32(vld1q_f32(arr + 0), v_norm));
    vst1q_f32(arr + 4, vdivq_f32(vld1q_f32(arr + 4), v_norm));
  }
  if (last >= last_aligned + 4) {
    vst1q_f32(arr, vdivq_f32(vld1q_f32(arr), v_norm));
    arr += 4;
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}

// MSVC ARM64 does not declare `float16_t` unless ARMv8.2 FP16 is enabled
// (`_M_ARM64FP16` / `/arch:armv8.2`). Skip the FP16 NEON variants on MSVC
// ARM64 — the FP16 `Normalizer<Float16>::Compute` dispatch below falls
// through to the scalar conversion path.
#if !defined(_MSC_VER)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static inline void NormalizeNEON(float16_t *arr, size_t dim, float norm) {
  float16_t *last = arr + dim;
  float16_t *last_aligned = arr + ((dim >> 4) << 4);

  float16x8_t v_norm = vdupq_n_f16((float16_t)norm);
  for (; arr != last_aligned; arr += 16) {
    vst1q_f16(arr + 0, vdivq_f16(vld1q_f16(arr + 0), v_norm));
    vst1q_f16(arr + 8, vdivq_f16(vld1q_f16(arr + 8), v_norm));
  }
  if (last >= arr + 8) {
    vst1q_f16(arr, vdivq_f16(vld1q_f16(arr), v_norm));
    arr += 8;
  }
  if (last >= arr + 4) {
    vst1_f16(arr, vdiv_f16(vld1_f16(arr), vget_low_f16(v_norm)));
    arr += 4;
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#else
static inline void NormalizeNEON(float16_t *arr, size_t dim, float norm) {
  float16_t *last = arr + dim;
  float16_t *last_aligned = arr + ((dim >> 4) << 4);

  float32x4_t v_norm = vdupq_n_f32(norm);
  for (; arr != last_aligned; arr += 16) {
    float16x8_t vf16_0 = vld1q_f16(arr + 0);
    float16x8_t vf16_1 = vld1q_f16(arr + 8);
    vf16_0 = vcombine_f16(
        vcvt_f16_f32(vdivq_f32(vcvt_f32_f16(vget_low_f16(vf16_0)), v_norm)),
        vcvt_f16_f32(vdivq_f32(vcvt_high_f32_f16(vf16_0), v_norm)));
    vf16_1 = vcombine_f16(
        vcvt_f16_f32(vdivq_f32(vcvt_f32_f16(vget_low_f16(vf16_1)), v_norm)),
        vcvt_f16_f32(vdivq_f32(vcvt_high_f32_f16(vf16_1), v_norm)));
    vst1q_f16(arr + 0, vf16_0);
    vst1q_f16(arr + 8, vf16_1);
  }
  if (last >= arr + 8) {
    float16x8_t vf16 = vld1q_f16(arr);
    vf16 = vcombine_f16(
        vcvt_f16_f32(vdivq_f32(vcvt_f32_f16(vget_low_f16(vf16)), v_norm)),
        vcvt_f16_f32(vdivq_f32(vcvt_high_f32_f16(vf16), v_norm)));
    vst1q_f16(arr, vf16);
    arr += 8;
  }
  if (last >= arr + 4) {
    vst1_f16(arr, vcvt_f16_f32(vdivq_f32(vcvt_f32_f16(vld1_f16(arr)), v_norm)));
    arr += 4;
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#endif  // !_MSC_VER (FP16 NEON: gcc/clang aarch64 only)
#endif  // AILEGO_ARM64_NEON

#if defined(__AVX__)
#if defined(__AVX512F__)
static inline void NormalizeAVX512(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 4) << 4);

  __m512 zmm_norm = _mm512_set1_ps(norm);
  if (((uintptr_t)arr & 0x3f) == 0) {
    for (; arr != last_aligned; arr += 16) {
      _mm512_store_ps(arr, _mm512_div_ps(_mm512_load_ps(arr), zmm_norm));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm256_store_ps(arr, _mm256_div_ps(_mm256_load_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      _mm512_storeu_ps(arr, _mm512_div_ps(_mm512_loadu_ps(arr), zmm_norm));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm256_storeu_ps(arr, _mm256_div_ps(_mm256_loadu_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __AVX512F__

static inline void NormalizeAVX(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 4) << 4);

  __m256 ymm_norm = _mm256_set1_ps(norm);
  if (((uintptr_t)arr & 0x1f) == 0) {
    for (; arr != last_aligned; arr += 16) {
      _mm256_store_ps(arr + 0,
                      _mm256_div_ps(_mm256_load_ps(arr + 0), ymm_norm));
      _mm256_store_ps(arr + 8,
                      _mm256_div_ps(_mm256_load_ps(arr + 8), ymm_norm));
    }
    if (last >= arr + 8) {
      _mm256_store_ps(arr, _mm256_div_ps(_mm256_load_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      _mm256_storeu_ps(arr + 0,
                       _mm256_div_ps(_mm256_loadu_ps(arr + 0), ymm_norm));
      _mm256_storeu_ps(arr + 8,
                       _mm256_div_ps(_mm256_loadu_ps(arr + 8), ymm_norm));
    }
    if (last >= arr + 8) {
      _mm256_storeu_ps(arr, _mm256_div_ps(_mm256_loadu_ps(arr), ymm_norm));
      arr += 8;
    }
    if (last >= arr + 4) {
      __m128 xmm_norm = _mm_set1_ps(norm);
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __AVX__

#if defined(__AVX__) && defined(__F16C__)
#if defined(__AVX512F__)
static inline void NormalizeAVX512(uint16_t *arr, size_t dim, float norm) {
  uint16_t *last = arr + dim;
  uint16_t *last_aligned = arr + ((dim >> 4) << 4);

  __m512 zmm_norm = _mm512_set1_ps(norm);
  if (((uintptr_t)arr & 0x1f) == 0) {
    for (; arr != last_aligned; arr += 16) {
      _mm256_store_si256(
          (__m256i *)arr,
          _mm512_cvtps_ph(_mm512_div_ps(_mm512_cvtph_ps(_mm256_load_si256(
                                            (const __m256i *)arr)),
                                        zmm_norm),
                          _MM_FROUND_NO_EXC));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm_store_si128(
          (__m128i *)arr,
          _mm256_cvtps_ph(_mm256_div_ps(_mm256_cvtph_ps(_mm_load_si128(
                                            (const __m128i *)arr)),
                                        ymm_norm),
                          _MM_FROUND_NO_EXC));
      arr += 8;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      _mm256_storeu_si256(
          (__m256i *)arr,
          _mm512_cvtps_ph(_mm512_div_ps(_mm512_cvtph_ps(_mm256_loadu_si256(
                                            (const __m256i *)arr)),
                                        zmm_norm),
                          _MM_FROUND_NO_EXC));
    }
    if (last >= arr + 8) {
      __m256 ymm_norm = _mm256_set1_ps(norm);
      _mm_storeu_si128(
          (__m128i *)arr,
          _mm256_cvtps_ph(_mm256_div_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                            (const __m128i *)arr)),
                                        ymm_norm),
                          _MM_FROUND_NO_EXC));
      arr += 8;
    }
  }
  if (last >= arr + 4) {
    __m128 xmm_norm = _mm_set1_ps(norm);
    _mm_storel_epi64(
        (__m128i *)arr,
        _mm_cvtps_ph(
            _mm_div_ps(_mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)arr)),
                       xmm_norm),
            _MM_FROUND_NO_EXC));
    arr += 8;
  }
  switch (last - arr) {
    case 3:
      arr[2] = _cvtss_sh(_cvtsh_ss(arr[2]) / norm, _MM_FROUND_NO_EXC);
      /* FALLTHRU */
    case 2:
      arr[1] = _cvtss_sh(_cvtsh_ss(arr[1]) / norm, _MM_FROUND_NO_EXC);
      /* FALLTHRU */
    case 1:
      arr[0] = _cvtss_sh(_cvtsh_ss(arr[0]) / norm, _MM_FROUND_NO_EXC);
  }
}
#endif  // __AVX512F__

static inline void NormalizeAVX(uint16_t *arr, size_t dim, float norm) {
  uint16_t *last = arr + dim;
  uint16_t *last_aligned = arr + ((dim >> 4) << 4);

  __m256 ymm_norm = _mm256_set1_ps(norm);
  if (((uintptr_t)arr & 0xf) == 0) {
    for (; arr != last_aligned; arr += 16) {
      __m128i xmm_0 = _mm_load_si128((const __m128i *)(arr + 0));
      __m128i xmm_1 = _mm_load_si128((const __m128i *)(arr + 8));
      __m256 ymm_0 = _mm256_div_ps(_mm256_cvtph_ps(xmm_0), ymm_norm);
      __m256 ymm_1 = _mm256_div_ps(_mm256_cvtph_ps(xmm_1), ymm_norm);
      _mm_store_si128((__m128i *)(arr + 0),
                      _mm256_cvtps_ph(ymm_0, _MM_FROUND_NO_EXC));
      _mm_store_si128((__m128i *)(arr + 8),
                      _mm256_cvtps_ph(ymm_1, _MM_FROUND_NO_EXC));
    }
    if (last >= arr + 8) {
      _mm_store_si128(
          (__m128i *)arr,
          _mm256_cvtps_ph(_mm256_div_ps(_mm256_cvtph_ps(_mm_load_si128(
                                            (const __m128i *)arr)),
                                        ymm_norm),
                          _MM_FROUND_NO_EXC));
      arr += 8;
    }
  } else {
    for (; arr != last_aligned; arr += 16) {
      __m128i xmm_0 = _mm_loadu_si128((const __m128i *)(arr + 0));
      __m128i xmm_1 = _mm_loadu_si128((const __m128i *)(arr + 8));
      __m256 ymm_0 = _mm256_div_ps(_mm256_cvtph_ps(xmm_0), ymm_norm);
      __m256 ymm_1 = _mm256_div_ps(_mm256_cvtph_ps(xmm_1), ymm_norm);
      _mm_storeu_si128((__m128i *)(arr + 0),
                       _mm256_cvtps_ph(ymm_0, _MM_FROUND_NO_EXC));
      _mm_storeu_si128((__m128i *)(arr + 8),
                       _mm256_cvtps_ph(ymm_1, _MM_FROUND_NO_EXC));
    }
    if (last >= arr + 8) {
      _mm_storeu_si128(
          (__m128i *)arr,
          _mm256_cvtps_ph(_mm256_div_ps(_mm256_cvtph_ps(_mm_loadu_si128(
                                            (const __m128i *)arr)),
                                        ymm_norm),
                          _MM_FROUND_NO_EXC));
      arr += 8;
    }
  }
  if (last >= arr + 4) {
    __m128 xmm_norm = _mm_set1_ps(norm);
    _mm_storel_epi64(
        (__m128i *)arr,
        _mm_cvtps_ph(
            _mm_div_ps(_mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)arr)),
                       xmm_norm),
            _MM_FROUND_NO_EXC));
    arr += 8;
  }
  switch (last - arr) {
    case 3:
      arr[2] = _cvtss_sh(_cvtsh_ss(arr[2]) / norm, _MM_FROUND_NO_EXC);
      /* FALLTHRU */
    case 2:
      arr[1] = _cvtss_sh(_cvtsh_ss(arr[1]) / norm, _MM_FROUND_NO_EXC);
      /* FALLTHRU */
    case 1:
      arr[0] = _cvtss_sh(_cvtsh_ss(arr[0]) / norm, _MM_FROUND_NO_EXC);
  }
}
#endif  // __AVX__ && __F16C__

#if defined(__SSE__)
static inline void NormalizeSSE(float *arr, size_t dim, float norm) {
  float *last = arr + dim;
  float *last_aligned = arr + ((dim >> 3) << 3);

  __m128 xmm_norm = _mm_set1_ps(norm);
  if (((uintptr_t)arr & 0xf) == 0) {
    for (; arr != last_aligned; arr += 8) {
      _mm_store_ps(arr + 0, _mm_div_ps(_mm_load_ps(arr + 0), xmm_norm));
      _mm_store_ps(arr + 4, _mm_div_ps(_mm_load_ps(arr + 4), xmm_norm));
    }
    if (last >= last_aligned + 4) {
      _mm_store_ps(arr, _mm_div_ps(_mm_load_ps(arr), xmm_norm));
      arr += 4;
    }
  } else {
    for (; arr != last_aligned; arr += 8) {
      _mm_storeu_ps(arr + 0, _mm_div_ps(_mm_loadu_ps(arr + 0), xmm_norm));
      _mm_storeu_ps(arr + 4, _mm_div_ps(_mm_loadu_ps(arr + 4), xmm_norm));
    }
    if (last >= last_aligned + 4) {
      _mm_storeu_ps(arr, _mm_div_ps(_mm_loadu_ps(arr), xmm_norm));
      arr += 4;
    }
  }
  switch (last - arr) {
    case 3:
      arr[2] /= norm;
      /* FALLTHRU */
    case 2:
      arr[1] /= norm;
      /* FALLTHRU */
    case 1:
      arr[0] /= norm;
  }
}
#endif  // __SSE__

#if defined(__SSE__) || defined(AILEGO_ARM64_NEON)
//! Compute the norm of vector
void Normalizer<float>::Compute(ValueType *arr, size_t dim, float norm) {
#if defined(AILEGO_HAVE_NEON)
  NormalizeNEON(arr, dim, norm);
#else
#if defined(__AVX512F__)
  if (dim > 15) {
    NormalizeAVX512(arr, dim, norm);
    return;
  }
#endif  // __AVX512F__
#if defined(__AVX__)
  if (dim > 7) {
    NormalizeAVX(arr, dim, norm);
    return;
  }
#endif  // __AVX__
  NormalizeSSE(arr, dim, norm);
#endif  // AILEGO_HAVE_NEON
}
#endif  // __SSE__ || AILEGO_ARM64_NEON

// MSVC ARM64 lacks `float16_t` without ARMv8.2 FP16, so the FP16
// `NormalizeNEON` variants are not defined there (see top of file).
// Exclude MSVC from this specialization so MSVC ARM64 falls back to the
// generic uint16_t-based Float16 normalize path.
#if (defined(__F16C__) && defined(__AVX__)) || defined(AILEGO_ARM64_GNU_LIKE)
//! Compute the norm of vector
void Normalizer<Float16>::Compute(ValueType *arr, size_t dim, float norm) {
#if defined(AILEGO_ARM64_GNU_LIKE)
  NormalizeNEON(reinterpret_cast<float16_t *>(arr), dim, norm);
#else
#if defined(__AVX512F__)
  if (dim > 31) {
    NormalizeAVX512(reinterpret_cast<uint16_t *>(arr), dim, norm);
    return;
  }
#endif  // __AVX512F__
  NormalizeAVX(reinterpret_cast<uint16_t *>(arr), dim, norm);
#endif  // AILEGO_ARM64_GNU_LIKE
}
#endif  // (__F16C__ && __AVX__) || AILEGO_ARM64_GNU_LIKE

}  // namespace ailego
}  // namespace zvec