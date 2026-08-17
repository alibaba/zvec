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

#include "scalar/pq_quantizer_int8/pq_distance.h"

namespace zvec::turbo::scalar {

void pq_adc_int8_distance(const void *pq_code_v, const void *lut_v,
                          size_t num_chunk, float *out) {
  constexpr size_t kNumCentroids = 256;
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  float sum = 0.0f;
  for (size_t m = 0; m < num_chunk; ++m) {
    sum += lut[m * kNumCentroids + pq_code[m]];
  }
  *out = sum;
}

void pq_sdc_int8_distance(const void *a_v, const void *b_v,
                          const void *dist_table_v, size_t num_chunk,
                          float *out) {
  constexpr size_t kNumCentroids = 256;
  constexpr size_t chunk = kNumCentroids * kNumCentroids;  // 65536
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);
  float sum = 0.0f;
  for (size_t m = 0; m < num_chunk; ++m) {
    size_t idx = m * chunk + static_cast<size_t>(a[m]) * kNumCentroids +
                 static_cast<size_t>(b[m]);
    sum += dist_table[idx];
  }
  *out = sum;
}

void pq_adc_int8_batch_distance(const void **candidates_v, const void *lut_v,
                                size_t num, size_t num_chunk, float *out,
                                const void ** /*extra_values*/) {
  constexpr size_t kNumCentroids = 256;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  // candidates_v is const void**, but we need const uint8_t**
  // Use an intermediate cast through const char** to avoid aliasing issues.
  auto candidates = reinterpret_cast<const uint8_t *const *>(candidates_v);

  size_t i = 0;
  // Main loop: process 4 candidates per iteration.
  // Shared LUT base pointer (tab) is computed once per chunk,
  // reducing redundant pointer arithmetic across the 4 candidates.
  for (; i + 4 <= num; i += 4) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (size_t m = 0; m < num_chunk; ++m) {
      const float *tab = lut + m * kNumCentroids;
      d0 += tab[c0[m]];
      d1 += tab[c1[m]];
      d2 += tab[c2[m]];
      d3 += tab[c3[m]];
    }
    out[i] = d0;
    out[i + 1] = d1;
    out[i + 2] = d2;
    out[i + 3] = d3;
  }
  // Scalar leftover: remaining candidates processed one at a time.
  for (; i < num; ++i) {
    const uint8_t *code = candidates[i];
    float d = 0.0f;
    for (size_t m = 0; m < num_chunk; ++m) {
      d += lut[m * kNumCentroids + code[m]];
    }
    out[i] = d;
  }
}

}  // namespace zvec::turbo::scalar
