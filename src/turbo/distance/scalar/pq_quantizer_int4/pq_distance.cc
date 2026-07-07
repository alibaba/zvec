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

#include "scalar/pq_quantizer_int4/pq_distance.h"

namespace zvec::turbo::scalar {

namespace {

// Decode the 4-bit index for sub-quantizer m from a packed int4 code.
// Layout: byte[m/2] = (code[2*(m/2)+1] << 4) | code[2*(m/2)]
//   even m -> low nibble;  odd m -> high nibble.
inline uint8_t decode_nibble(const uint8_t *code, size_t m) {
  uint8_t byte = code[m >> 1];
  return (m & 1u) ? static_cast<uint8_t>(byte >> 4)
                  : static_cast<uint8_t>(byte & 0x0Fu);
}

}  // namespace

void pq_adc_int4_distance(const void *pq_code_v, const void *lut_v,
                           size_t num_subquantizers, float *out) {
  constexpr size_t kNumCentroids = 16;
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  float sum = 0.0f;
  for (size_t m = 0; m < num_subquantizers; ++m) {
    uint8_t idx = decode_nibble(pq_code, m);
    sum += lut[m * kNumCentroids + idx];
  }
  *out = sum;
}

void pq_sdc_int4_distance(const void *a_v, const void *b_v,
                           const void *dist_table_v,
                           size_t num_subquantizers, float *out) {
  constexpr size_t kNumCentroids = 16;
  constexpr size_t kTablePerSub = kNumCentroids * kNumCentroids;  // 256
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);
  float sum = 0.0f;
  for (size_t m = 0; m < num_subquantizers; ++m) {
    uint8_t ai = decode_nibble(a, m);
    uint8_t bi = decode_nibble(b, m);
    size_t idx = m * kTablePerSub +
                 static_cast<size_t>(ai) * kNumCentroids +
                 static_cast<size_t>(bi);
    sum += dist_table[idx];
  }
  *out = sum;
}

void pq_adc_int4_batch_distance(const void **candidates_v, const void *lut_v,
                                 size_t num, size_t num_subquantizers,
                                 float *out) {
  constexpr size_t kNumCentroids = 16;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  size_t i = 0;
  // Main loop: process 4 candidates per iteration (batch4 ILP).
  for (; i + 4 <= num; i += 4) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (size_t m = 0; m < num_subquantizers; ++m) {
      const float *tab = lut + m * kNumCentroids;
      uint8_t n0 = decode_nibble(c0, m);
      uint8_t n1 = decode_nibble(c1, m);
      uint8_t n2 = decode_nibble(c2, m);
      uint8_t n3 = decode_nibble(c3, m);
      d0 += tab[n0];
      d1 += tab[n1];
      d2 += tab[n2];
      d3 += tab[n3];
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
    for (size_t m = 0; m < num_subquantizers; ++m) {
      uint8_t idx = decode_nibble(code, m);
      d += lut[m * kNumCentroids + idx];
    }
    out[i] = d;
  }
}

}  // namespace zvec::turbo::scalar
