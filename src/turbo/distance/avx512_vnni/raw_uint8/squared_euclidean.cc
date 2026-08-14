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

#include "avx512_vnni/raw_uint8/squared_euclidean.h"
#include <algorithm>
#include <cstdint>
#if (defined(__AVX512F__) && defined(__AVX512BW__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
#include <immintrin.h>
#endif

namespace zvec::turbo::avx512_vnni {
#if (defined(__AVX512F__) && defined(__AVX512BW__)) || \
    (defined(_MSC_VER) && defined(__AVX512F__))
namespace {

inline float ScalarDistance(const uint8_t *lhs, const uint8_t *rhs,
                            size_t dimension) {
  uint64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int delta = static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]);
    sum += static_cast<uint64_t>(delta * delta);
  }
  return static_cast<float>(sum);
}

template <size_t Batch>
void DistanceBatch(const void *const *vectors, const uint8_t *query,
                   size_t dimension, float *distances) {
  if (dimension > 32768) {
    for (size_t lane = 0; lane < Batch; ++lane) {
      distances[lane] = ScalarDistance(
          static_cast<const uint8_t *>(vectors[lane]), query, dimension);
    }
    return;
  }

  __m512i sums[Batch];
  for (size_t lane = 0; lane < Batch; ++lane) {
    sums[lane] = _mm512_setzero_si512();
  }

  size_t offset = 0;
  for (; offset + 32 <= dimension; offset += 32) {
    const __m512i q = _mm512_cvtepu8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(query + offset)));
    for (size_t lane = 0; lane < Batch; ++lane) {
      const auto *row = static_cast<const uint8_t *>(vectors[lane]);
      const __m512i x = _mm512_cvtepu8_epi16(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + offset)));
      const __m512i delta = _mm512_sub_epi16(x, q);
      sums[lane] =
          _mm512_add_epi32(sums[lane], _mm512_madd_epi16(delta, delta));
    }
  }

  for (size_t lane = 0; lane < Batch; ++lane) {
    int64_t sum = _mm512_reduce_add_epi32(sums[lane]);
    const auto *row = static_cast<const uint8_t *>(vectors[lane]);
    for (size_t i = offset; i < dimension; ++i) {
      const int delta = static_cast<int>(row[i]) - static_cast<int>(query[i]);
      sum += delta * delta;
    }
    distances[lane] = static_cast<float>(sum);
  }
}

inline void PrefetchRow(const void *vector) {
  _mm_prefetch(static_cast<const char *>(vector), _MM_HINT_T0);
}

}  // namespace

void fp32_to_raw_uint8(const float *input, size_t dimension,
                       void *output_buffer) {
  auto *output = static_cast<uint8_t *>(output_buffer);
  size_t i = 0;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 max_uint8 = _mm512_set1_ps(255.0F);
  for (; i + 16 <= dimension; i += 16) {
    // Clamp before conversion so infinities and values outside int32 range
    // saturate consistently with the scalar tail. MAXPS selects its second
    // operand for NaN, which maps non-numeric inputs to zero here.
    const __m512 values = _mm512_min_ps(
        _mm512_max_ps(_mm512_loadu_ps(input + i), zero), max_uint8);
    const __m512i integers = _mm512_cvttps_epi32(values);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),
                     _mm512_cvtusepi32_epi8(integers));
  }
  for (; i < dimension; ++i) {
    const float value = input[i];
    output[i] = !(value > 0.0F)   ? 0
                : value >= 255.0F ? 255
                                  : static_cast<uint8_t>(value);
  }
}

void squared_euclidean_uint8_distance(const void *lhs, const void *rhs,
                                      size_t dimension, float *distance) {
  const void *rows[] = {lhs};
  DistanceBatch<1>(rows, static_cast<const uint8_t *>(rhs), dimension,
                   distance);
}

void squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                            const void *query, size_t count,
                                            size_t dimension,
                                            float *distances) {
  constexpr size_t kBatch = 4;
  const auto *query_uint8 = static_cast<const uint8_t *>(query);
  for (size_t i = 0; i < std::min(count, kBatch); ++i) {
    PrefetchRow(vectors[i]);
  }

  size_t i = 0;
  for (; i + kBatch <= count; i += kBatch) {
    for (size_t lane = 0; lane < kBatch; ++lane) {
      const size_t next = i + kBatch + lane;
      if (next < count) PrefetchRow(vectors[next]);
    }
    DistanceBatch<kBatch>(vectors + i, query_uint8, dimension, distances + i);
  }
  for (; i < count; ++i) {
    DistanceBatch<1>(vectors + i, query_uint8, dimension, distances + i);
  }
}

#else

void fp32_to_raw_uint8(const float *input, size_t dimension,
                       void *output_buffer) {
  auto *output = static_cast<uint8_t *>(output_buffer);
  for (size_t i = 0; i < dimension; ++i) {
    const float value = input[i];
    output[i] = !(value > 0.0F)   ? 0
                : value >= 255.0F ? 255
                                  : static_cast<uint8_t>(value);
  }
}

void squared_euclidean_uint8_distance(const void *lhs, const void *rhs,
                                      size_t dimension, float *distance) {
  const auto *left = static_cast<const uint8_t *>(lhs);
  const auto *right = static_cast<const uint8_t *>(rhs);
  uint64_t sum = 0;
  for (size_t i = 0; i < dimension; ++i) {
    const int delta = static_cast<int>(left[i]) - static_cast<int>(right[i]);
    sum += static_cast<uint64_t>(delta * delta);
  }
  *distance = static_cast<float>(sum);
}

void squared_euclidean_uint8_batch_distance(const void *const *vectors,
                                            const void *query, size_t count,
                                            size_t dimension,
                                            float *distances) {
  for (size_t i = 0; i < count; ++i) {
    squared_euclidean_uint8_distance(vectors[i], query, dimension,
                                     distances + i);
  }
}

#endif

}  // namespace zvec::turbo::avx512_vnni
