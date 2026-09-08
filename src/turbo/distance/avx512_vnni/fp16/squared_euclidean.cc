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

#include "avx512_vnni/fp16/squared_euclidean.h"
#include <algorithm>
#include <cstdint>
#include <zvec/ailego/utility/float_helper.h>
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512_vnni {
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__F16C__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
namespace {

inline float Reduce(__m512 value) {
  const __m256 low256 = _mm512_castps512_ps256(value);
  const __m256 high256 = _mm512_extractf32x8_ps(value, 1);
  const __m256 sum256 = _mm256_add_ps(low256, high256);
  const __m128 low128 = _mm256_castps256_ps128(sum256);
  const __m128 high128 = _mm256_extractf128_ps(sum256, 1);
  const __m128 sum128 = _mm_add_ps(low128, high128);
  const __m128 pair = _mm_hadd_ps(sum128, sum128);
  return _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55)));
}

inline float HalfToFloat(uint16_t value) {
  return ailego::FloatHelper::ToFP32(value);
}

template <size_t Batch>
void DistanceBatch(const void *const *vectors, const uint16_t *query,
                   size_t dimension, float *distances) {
  __m512 sums0[Batch];
  __m512 sums1[Batch];
  for (size_t lane = 0; lane < Batch; ++lane) {
    sums0[lane] = _mm512_setzero_ps();
    sums1[lane] = _mm512_setzero_ps();
  }

  size_t d = 0;
  for (; d + 32 <= dimension; d += 32) {
    const __m512i qh =
        _mm512_loadu_si512(reinterpret_cast<const __m512i *>(query + d));
    const __m512 q0 = _mm512_cvtph_ps(_mm512_castsi512_si256(qh));
    const __m512 q1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(qh, 1));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512i xh =
          _mm512_loadu_si512(reinterpret_cast<const __m512i *>(row + d));
      const __m512 x0 = _mm512_cvtph_ps(_mm512_castsi512_si256(xh));
      const __m512 x1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(xh, 1));
      const __m512 delta0 = _mm512_sub_ps(q0, x0);
      const __m512 delta1 = _mm512_sub_ps(q1, x1);
      sums0[lane] = _mm512_fmadd_ps(delta0, delta0, sums0[lane]);
      sums1[lane] = _mm512_fmadd_ps(delta1, delta1, sums1[lane]);
    }
  }

  if (d + 16 <= dimension) {
    const __m512 q = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + d)));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint16_t *>(vectors[lane]);
      const __m512 x = _mm512_cvtph_ps(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + d)));
      const __m512 delta = _mm512_sub_ps(q, x);
      sums0[lane] = _mm512_fmadd_ps(delta, delta, sums0[lane]);
    }
    d += 16;
  }

  for (size_t lane = 0; lane < Batch; ++lane) {
    float sum = Reduce(_mm512_add_ps(sums0[lane], sums1[lane]));
    const auto *row = static_cast<const uint16_t *>(vectors[lane]);
    for (size_t tail = d; tail < dimension; ++tail) {
      const float delta = HalfToFloat(query[tail]) - HalfToFloat(row[tail]);
      sum += delta * delta;
    }
    distances[lane] = sum;
  }
}

inline void PrefetchRow(const void *vector) {
  _mm_prefetch(static_cast<const char *>(vector), _MM_HINT_T0);
}

}  // namespace

void squared_euclidean_fp16_distance(const void *lhs, const void *rhs,
                                     size_t dimension, float *distance) {
  const void *rows[] = {lhs};
  DistanceBatch<1>(rows, static_cast<const uint16_t *>(rhs), dimension,
                   distance);
}

void squared_euclidean_fp16_batch_distance(
    const void *const *vectors, const void *query, size_t count,
    size_t dimension, float *distances, const void *const * /*extra_values*/) {
  constexpr size_t kBatch = 4;
  const auto *query_fp16 = static_cast<const uint16_t *>(query);
  for (size_t i = 0; i < std::min(count, kBatch); ++i) {
    PrefetchRow(vectors[i]);
  }

  size_t i = 0;
  for (; i + kBatch <= count; i += kBatch) {
    for (size_t lane = 0; lane < kBatch; ++lane) {
      const size_t next = i + kBatch + lane;
      if (next < count) PrefetchRow(vectors[next]);
    }
    DistanceBatch<kBatch>(vectors + i, query_fp16, dimension, distances + i);
  }
  for (; i < count; ++i) {
    DistanceBatch<1>(vectors + i, query_fp16, dimension, distances + i);
  }
}

#else

void squared_euclidean_fp16_distance(const void *lhs, const void *rhs,
                                     size_t dimension, float *distance) {
  const void *rows[] = {lhs};
  squared_euclidean_fp16_batch_distance(rows, rhs, 1, dimension, distance,
                                        nullptr);
}

void squared_euclidean_fp16_batch_distance(
    const void *const *vectors, const void *query, size_t count,
    size_t dimension, float *distances, const void *const * /*extra_values*/) {
  const auto *query_fp16 = static_cast<const uint16_t *>(query);
  for (size_t row = 0; row < count; ++row) {
    const auto *vector = static_cast<const uint16_t *>(vectors[row]);
    float sum = 0.0F;
    for (size_t d = 0; d < dimension; ++d) {
      const float delta = ailego::FloatHelper::ToFP32(vector[d]) -
                          ailego::FloatHelper::ToFP32(query_fp16[d]);
      sum += delta * delta;
    }
    distances[row] = sum;
  }
}

#endif

}  // namespace zvec::turbo::avx512_vnni
