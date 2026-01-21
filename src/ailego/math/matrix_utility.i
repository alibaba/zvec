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

#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

//! Absolute value of a float
static inline float FastAbs(float x) {
  uint32_t *p = reinterpret_cast<uint32_t *>(&x);
  *p &= 0x7fffffffu;
  return *reinterpret_cast<float *>(p);
}

#if defined(__SSE__)
static inline float HorizontalMax_FP32_V128(__m128 v) {
  __m128 x1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 3, 2));
  __m128 x2 = _mm_max_ps(v, x1);
  __m128 x3 = _mm_shuffle_ps(x2, x2, _MM_SHUFFLE(0, 0, 0, 1));
  __m128 x4 = _mm_max_ps(x2, x3);
  return _mm_cvtss_f32(x4);
}

static inline float HorizontalAdd_FP32_V128(__m128 v) {
#ifdef __SSE3__
  __m128 x1 = _mm_hadd_ps(v, v);
  __m128 x2 = _mm_hadd_ps(x1, x1);
  return _mm_cvtss_f32(x2);
#else
  __m128 x1 = _mm_movehl_ps(v, v);
  __m128 x2 = _mm_add_ps(v, x1);
  __m128 x3 = _mm_shuffle_ps(x2, x2, 1);
  __m128 x4 = _mm_add_ss(x2, x3);
  return _mm_cvtss_f32(x4);
#endif
}
#endif // __SSE__

#if defined(__SSE2__)
static inline int32_t HorizontalAdd_INT32_V128(__m128i v) {
#ifdef __SSE3__
  __m128i x1 = _mm_hadd_epi32(v, v);
  __m128i x2 = _mm_hadd_epi32(x1, x1);
  return _mm_cvtsi128_si32(x2);
#else
  __m128i x1 = _mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2));
  __m128i x2 = _mm_add_epi32(v, x1);
  __m128i x3 = _mm_shuffle_epi32(x2, _MM_SHUFFLE(0, 0, 0, 1));
  __m128i x4 = _mm_add_epi32(x2, x3);
  return _mm_cvtsi128_si32(x4);
#endif
}

static inline int64_t HorizontalAdd_INT64_V128(__m128i v) {
#ifdef __SSE4_1__
  return (_mm_extract_epi64(v, 0) + _mm_extract_epi64(v, 1));
#else
  return _mm_cvtsi128_si64(
      _mm_add_epi64(_mm_shuffle_epi32(v, _MM_SHUFFLE(0, 0, 3, 2)), v));
#endif
}
#endif // __SSE2__

#if defined(__SSSE3__)
static const __m128i POPCNT_LOOKUP_SSE =
    _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

static inline __m128i VerticalPopCount_INT8_V128(__m128i v) {
#if defined(__AVX512VL__) && defined(__AVX512BITALG__)
  return _mm_popcnt_epi8(v);
#else
  const __m128i low_mask = _mm_set1_epi8(0x0f);
  __m128i lo = _mm_shuffle_epi8(POPCNT_LOOKUP_SSE, _mm_and_si128(v, low_mask));
  __m128i hi = _mm_shuffle_epi8(POPCNT_LOOKUP_SSE,
                                _mm_and_si128(_mm_srli_epi32(v, 4), low_mask));
  return _mm_add_epi8(lo, hi);
#endif // __AVX512VL__ && __AVX512BITALG__
}

static inline __m128i VerticalPopCount_INT16_V128(__m128i v) {
#if defined(__AVX512VL__) && defined(__AVX512BITALG__)
  return _mm_popcnt_epi16(v);
#else
  __m128i total = VerticalPopCount_INT8_V128(v);
  return _mm_add_epi16(_mm_srli_epi16(total, 8),
                       _mm_and_si128(total, _mm_set1_epi16(0xff)));
#endif // __AVX512VL__ && __AVX512BITALG__
}

static inline __m128i VerticalPopCount_INT32_V128(__m128i v) {
#if defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__)
  return _mm_popcnt_epi32(v);
#else
  __m128i total =
      _mm_madd_epi16(VerticalPopCount_INT8_V128(v), _mm_set1_epi16(1));
  return _mm_add_epi32(_mm_srli_epi32(total, 8),
                       _mm_and_si128(total, _mm_set1_epi32(0xff)));
#endif // __AVX512VL__ && __AVX512VPOPCNTDQ__
}

static inline __m128i VerticalPopCount_INT64_V128(__m128i v) {
#if defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__)
  return _mm_popcnt_epi64(v);
#else
  return _mm_sad_epu8(VerticalPopCount_INT8_V128(v), _mm_setzero_si128());
#endif // __AVX512VL__ && __AVX512VPOPCNTDQ__
}
#endif // __SSSE3__

#if defined(__SSE4_1__)
static inline int16_t HorizontalMax_UINT8_V128(__m128i v) {
  v = _mm_max_epu8(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 3, 2)));
  v = _mm_max_epu8(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1)));
  v = _mm_max_epu8(v, _mm_shufflelo_epi16(v, _MM_SHUFFLE(1, 1, 1, 1)));
  v = _mm_max_epu8(v, _mm_srli_epi16(v, 8));
  return static_cast<uint8_t>(_mm_cvtsi128_si32(v));
}
#endif // __SSE4_1__

#if defined(__AVX__)
static inline float HorizontalMax_FP32_V256(__m256 v) {
  __m256 x1 = _mm256_permute_ps(v, _MM_SHUFFLE(0, 0, 3, 2));
  __m256 x2 = _mm256_max_ps(v, x1);
  __m256 x3 = _mm256_permute_ps(x2, _MM_SHUFFLE(0, 0, 0, 1));
  __m256 x4 = _mm256_max_ps(x2, x3);
  __m128 x5 = _mm256_extractf128_ps(x4, 1);
  __m128 x6 = _mm_max_ss(_mm256_castps256_ps128(x4), x5);
  return _mm_cvtss_f32(x6);
}

static inline float HorizontalAdd_FP32_V256(__m256 v) {
  __m256 x1 = _mm256_hadd_ps(v, v);
  __m256 x2 = _mm256_hadd_ps(x1, x1);
  __m128 x3 = _mm256_extractf128_ps(x2, 1);
  __m128 x4 = _mm_add_ss(_mm256_castps256_ps128(x2), x3);
  return _mm_cvtss_f32(x4);
}
#endif // __AVX__

#if defined(__AVX2__)
static const __m256i POPCNT_MASK1_INT8_AVX = _mm256_set1_epi8(0x0f);
static const __m256i POPCNT_MASK1_INT16_AVX = _mm256_set1_epi16(1);
static const __m256i POPCNT_MASK2_INT16_AVX = _mm256_set1_epi16(0xff);
static const __m256i POPCNT_MASK1_INT32_AVX = _mm256_set1_epi32(0xff);
static const __m256i POPCNT_ZERO_AVX = _mm256_setzero_si256();
static const __m256i POPCNT_LOOKUP_AVX =
    _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2,
                     1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);

static inline __m256i VerticalPopCount_INT8_V256(__m256i v) {
#if defined(__AVX512VL__) && defined(__AVX512BITALG__)
  return _mm256_popcnt_epi8(v);
#else
  __m256i lo = _mm256_shuffle_epi8(POPCNT_LOOKUP_AVX,
                                   _mm256_and_si256(v, POPCNT_MASK1_INT8_AVX));
  __m256i hi = _mm256_shuffle_epi8(
      POPCNT_LOOKUP_AVX,
      _mm256_and_si256(_mm256_srli_epi32(v, 4), POPCNT_MASK1_INT8_AVX));
  return _mm256_add_epi8(lo, hi);
#endif // __AVX512VL__ && __AVX512BITALG__
}

static inline __m256i VerticalPopCount_INT16_V256(__m256i v) {
#if defined(__AVX512VL__) && defined(__AVX512BITALG__)
  return _mm256_popcnt_epi16(v);
#else
  __m256i total = VerticalPopCount_INT8_V256(v);
  return _mm256_add_epi16(_mm256_srli_epi16(total, 8),
                          _mm256_and_si256(total, POPCNT_MASK2_INT16_AVX));
#endif // __AVX512VL__ && __AVX512BITALG__
}

static inline __m256i VerticalPopCount_INT32_V256(__m256i v) {
#if defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__)
  return _mm256_popcnt_epi32(v);
#else
  __m256i total =
      _mm256_madd_epi16(VerticalPopCount_INT8_V256(v), POPCNT_MASK1_INT16_AVX);
  return _mm256_add_epi32(_mm256_srli_epi32(total, 8),
                          _mm256_and_si256(total, POPCNT_MASK1_INT32_AVX));
#endif // __AVX512VL__ && __AVX512VPOPCNTDQ__
}

static inline __m256i VerticalPopCount_INT64_V256(__m256i v) {
#if defined(__AVX512VL__) && defined(__AVX512VPOPCNTDQ__)
  return _mm256_popcnt_epi64(v);
#else
  return _mm256_sad_epu8(VerticalPopCount_INT8_V256(v), POPCNT_ZERO_AVX);
#endif // __AVX512VL__ && __AVX512VPOPCNTDQ__
}

static inline int16_t HorizontalMax_UINT8_V256(__m256i v) {
  v = _mm256_max_epu8(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 3, 2)));
  v = _mm256_max_epu8(v, _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 1, 1)));
  v = _mm256_max_epu8(v, _mm256_shufflelo_epi16(v, _MM_SHUFFLE(1, 1, 1, 1)));
  __m128i x =
      _mm_max_epu8(_mm256_castsi256_si128(v), _mm256_extractf128_si256(v, 1));
  x = _mm_max_epu8(x, _mm_srli_epi16(x, 8));
  return static_cast<uint8_t>(_mm_cvtsi128_si32(x));
}

static inline int32_t HorizontalAdd_INT32_V256(__m256i v) {
  __m256i x1 = _mm256_hadd_epi32(v, v);
  __m256i x2 = _mm256_hadd_epi32(x1, x1);
  __m128i x3 = _mm256_extractf128_si256(x2, 1);
  __m128i x4 = _mm_add_epi32(_mm256_castsi256_si128(x2), x3);
  return _mm_cvtsi128_si32(x4);
}

static inline int64_t HorizontalAdd_INT64_V256(__m256i v) {
  __m256i x1 = _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
  __m256i x2 = _mm256_add_epi64(v, x1);
  __m128i x3 = _mm256_extractf128_si256(x2, 1);
  __m128i x4 = _mm_add_epi64(_mm256_extractf128_si256(x2, 0), x3);
  return _mm_cvtsi128_si64(x4);
}
#endif // __AVX2__

#if defined(__AVX512F__)
static inline float HorizontalMax_FP32_V512(__m512 v) {
  __m256 low = _mm512_castps512_ps256(v);
  __m256 high =
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  return HorizontalMax_FP32_V256(_mm256_max_ps(low, high));
}

static inline float HorizontalAdd_FP32_V512(__m512 v) {
  __m256 low = _mm512_castps512_ps256(v);
  __m256 high =
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
  return HorizontalAdd_FP32_V256(_mm256_add_ps(low, high));
}
#endif // __AVX512F__

#if defined(__AVX512FP16__)
static inline float HorizontalMax_FP16_V512(__m512h v) {
  __m512 low = _mm512_cvtxph_ps(_mm512_castph512_ph256(v));
  __m512 high = _mm512_cvtxph_ps(
      _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v), 1)));
  return HorizontalMax_FP32_V512(_mm512_max_ps(low, high));
}

static inline float HorizontalAdd_FP16_V512(__m512h v) {
  __m512 low = _mm512_cvtxph_ps(_mm512_castph512_ph256(v));
  __m512 high = _mm512_cvtxph_ps(
      _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(v), 1)));

  return HorizontalAdd_FP32_V512(_mm512_add_ps(low, high));
}
#endif // __AVX512FP16__

} // namespace ailego
} // namespace zvec