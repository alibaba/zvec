// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "avx512/fp16/fp32_query.h"
#include <cstdint>
#include <zvec/ailego/utility/float_helper.h>
#include "common/fp16_common.h"
#if ZVEC_TURBO_FP16_AVX512
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512 {
namespace {

template <bool Cosine>
void distance_impl(const void *vector, const float *query, size_t dim,
                   float *distance) {
#if ZVEC_TURBO_FP16_AVX512
  const auto *stored = static_cast<const uint16_t *>(vector);
  __m512 accumulator = _mm512_setzero_ps();
  size_t i = 0;
  // Keep the FP32 scalar-distance kernel's accumulation and reduction order.
  // Conversion happens in registers instead of a full temporary vector.
  for (; i + 16 <= dim; i += 16) {
    const __m512 value = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(stored + i)));
    const __m512 q = _mm512_loadu_ps(query + i);
    if constexpr (Cosine) {
      accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(value, q));
    } else {
      const __m512 diff = _mm512_sub_ps(value, q);
      accumulator = _mm512_add_ps(accumulator, _mm512_mul_ps(diff, diff));
    }
  }
  float sum = _mm512_reduce_add_ps(accumulator);
  if (i < dim) {
    // Decode the short tail together so its FP32 loop can retain the same
    // compiler vectorization and rounding as the reference kernel's tail.
    float tail[16];
    const size_t count = dim - i;
    ailego::FloatHelper::ToFP32(stored + i, count, tail);
    const float *q = query + i;
    for (size_t j = 0; j < count; ++j) {
      if constexpr (Cosine) {
        sum += tail[j] * q[j];
      } else {
        const float diff = tail[j] - q[j];
        sum += diff * diff;
      }
    }
  }
  *distance = Cosine ? -sum + 1.f : sum;
#else
  (void)vector;
  (void)query;
  (void)dim;
  (void)distance;
#endif
}

}  // namespace

void cosine_fp16_fp32_distance_avx512(const void *vector, const float *query,
                                      size_t dim, float *distance) {
  distance_impl<true>(vector, query, dim, distance);
}

void squared_euclidean_fp16_fp32_distance_avx512(const void *vector,
                                                 const float *query, size_t dim,
                                                 float *distance) {
  distance_impl<false>(vector, query, dim, distance);
}

}  // namespace zvec::turbo::avx512
