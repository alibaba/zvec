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

// NEON lacks hardware gather instructions (like x86 _mm256_i32gather_ps),
// so nibble unpacking and LUT lookups remain scalar. NEON accelerates the
// accumulation path: 4 floats are loaded into a float32x4_t register and
// pairwise-added into the running accumulator, halving the number of FP add
// operations.

#include "neon/pq_quantizer_int4/pq_distance.h"
#include <zvec/ailego/internal/platform.h>
#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#endif
#include <cstddef>
#include <cstdint>
#include "scalar/pq_quantizer_int4/pq_distance.h"

namespace zvec::turbo::neon {

#if defined(AILEGO_ARM64_NEON)
namespace {

// Horizontal sum of 4 floats in a float32x4_t register via pairwise add.
// [a, b, c, d] → [a+b, c+d, a+b, c+d] → [a+b+c+d, ...]
inline float horizontal_sum_neon(float32x4_t v) {
  float32x2_t lo = vget_low_f32(v);
  float32x2_t hi = vget_high_f32(v);
  float32x2_t sum2 = vadd_f32(lo, hi);             // [a+b, c+d]
  return vget_lane_f32(vpadd_f32(sum2, sum2), 0);  // a+b+c+d
}

// Extract the 4-bit code of subquantizer m from a nibble-packed buffer.
inline uint8_t nibble(const uint8_t *code, size_t m) {
  return static_cast<uint8_t>((code[m >> 1] >> ((m & 1) * 4)) & 0x0F);
}

}  // namespace
#endif

void pq_adc_int4_distance_neon(const void *pq_code_v, const void *lut_v,
                               size_t num_chunk, float *out) {
#if defined(AILEGO_ARM64_NEON)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 4;  // NEON processes 4 floats at once
  const auto *pq_code = reinterpret_cast<const uint8_t *>(pq_code_v);
  const auto *lut = reinterpret_cast<const float *>(lut_v);

  float32x4_t acc = vdupq_n_f32(0.0f);

  size_t m = 0;

  // Main loop: process 4 subquantizers per iteration.
  // Scalar LUT lookups (NEON has no gather), then NEON pairwise accumulation.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    float d0 = lut[(m + 0) * kNumCentroids + nibble(pq_code, m + 0)];
    float d1 = lut[(m + 1) * kNumCentroids + nibble(pq_code, m + 1)];
    float d2 = lut[(m + 2) * kNumCentroids + nibble(pq_code, m + 2)];
    float d3 = lut[(m + 3) * kNumCentroids + nibble(pq_code, m + 3)];
    const float lane[4] = {d0, d1, d2, d3};
    float32x4_t d = vld1q_f32(lane);
    acc = vaddq_f32(acc, d);
  }

  float sum = horizontal_sum_neon(acc);

  // Scalar leftover: process remaining subquantizers.
  for (; m < num_chunk; ++m) {
    sum += lut[m * kNumCentroids + nibble(pq_code, m)];
  }

  *out = sum;
#else
  // Without NEON this translation unit still compiles, so delegate to the
  // scalar kernel. Never leave `out` unwritten: turbo.cc selects these entry
  // points from CpuFeatures flags, and a no-op here would silently return
  // whatever the caller's buffer already held.
  scalar::pq_adc_int4_distance(pq_code_v, lut_v, num_chunk, out);
#endif
}

void pq_sdc_int4_distance_neon(const void *a_v, const void *b_v,
                               const void *dist_table_v, size_t num_chunk,
                               float *out) {
#if defined(AILEGO_ARM64_NEON)
  constexpr int kNumCentroids = 16;
  constexpr int kTablePerSub = kNumCentroids * kNumCentroids;  // 256
  constexpr int kChunkSize = 4;
  const auto *a = reinterpret_cast<const uint8_t *>(a_v);
  const auto *b = reinterpret_cast<const uint8_t *>(b_v);
  const auto *dist_table = reinterpret_cast<const float *>(dist_table_v);

  float32x4_t acc = vdupq_n_f32(0.0f);

  size_t m = 0;

  // Main loop: process 4 subquantizers per iteration.
  for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
    float d0 =
        dist_table[(m + 0) * kTablePerSub +
                   static_cast<size_t>(nibble(a, m + 0)) * kNumCentroids +
                   static_cast<size_t>(nibble(b, m + 0))];
    float d1 =
        dist_table[(m + 1) * kTablePerSub +
                   static_cast<size_t>(nibble(a, m + 1)) * kNumCentroids +
                   static_cast<size_t>(nibble(b, m + 1))];
    float d2 =
        dist_table[(m + 2) * kTablePerSub +
                   static_cast<size_t>(nibble(a, m + 2)) * kNumCentroids +
                   static_cast<size_t>(nibble(b, m + 2))];
    float d3 =
        dist_table[(m + 3) * kTablePerSub +
                   static_cast<size_t>(nibble(a, m + 3)) * kNumCentroids +
                   static_cast<size_t>(nibble(b, m + 3))];
    const float lane[4] = {d0, d1, d2, d3};
    float32x4_t d = vld1q_f32(lane);
    acc = vaddq_f32(acc, d);
  }

  float sum = horizontal_sum_neon(acc);

  // Scalar leftover.
  for (; m < num_chunk; ++m) {
    size_t idx = m * kTablePerSub +
                 static_cast<size_t>(nibble(a, m)) * kNumCentroids +
                 static_cast<size_t>(nibble(b, m));
    sum += dist_table[idx];
  }

  *out = sum;
#else
  scalar::pq_sdc_int4_distance(a_v, b_v, dist_table_v, num_chunk, out);
#endif
}

void pq_adc_int4_batch_distance_neon(const void **candidates_v,
                                     const void *lut_v, size_t num,
                                     size_t num_chunk, float *out) {
#if defined(AILEGO_ARM64_NEON)
  constexpr int kNumCentroids = 16;
  constexpr int kChunkSize = 4;
  constexpr int kBatch = 4;
  const auto *lut = reinterpret_cast<const float *>(lut_v);
  const auto *candidates =
      reinterpret_cast<const uint8_t *const *>(candidates_v);

  size_t i = 0;
  for (; i + kBatch <= num; i += kBatch) {
    const uint8_t *c0 = candidates[i];
    const uint8_t *c1 = candidates[i + 1];
    const uint8_t *c2 = candidates[i + 2];
    const uint8_t *c3 = candidates[i + 3];
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t m = 0;
    for (; m + kChunkSize <= num_chunk; m += kChunkSize) {
      // Each subquantizer position has its own 16-entry LUT row, so the
      // unrolled lanes must not share one hoisted row pointer.
      const float *tab0 = lut + (m + 0) * kNumCentroids;
      const float *tab1 = lut + (m + 1) * kNumCentroids;
      const float *tab2 = lut + (m + 2) * kNumCentroids;
      const float *tab3 = lut + (m + 3) * kNumCentroids;

      const float lane0[4] = {tab0[nibble(c0, m + 0)], tab1[nibble(c0, m + 1)],
                              tab2[nibble(c0, m + 2)], tab3[nibble(c0, m + 3)]};
      const float lane1[4] = {tab0[nibble(c1, m + 0)], tab1[nibble(c1, m + 1)],
                              tab2[nibble(c1, m + 2)], tab3[nibble(c1, m + 3)]};
      const float lane2[4] = {tab0[nibble(c2, m + 0)], tab1[nibble(c2, m + 1)],
                              tab2[nibble(c2, m + 2)], tab3[nibble(c2, m + 3)]};
      const float lane3[4] = {tab0[nibble(c3, m + 0)], tab1[nibble(c3, m + 1)],
                              tab2[nibble(c3, m + 2)], tab3[nibble(c3, m + 3)]};
      float32x4_t d0 = vld1q_f32(lane0);
      float32x4_t d1 = vld1q_f32(lane1);
      float32x4_t d2 = vld1q_f32(lane2);
      float32x4_t d3 = vld1q_f32(lane3);

      acc0 = vaddq_f32(acc0, d0);
      acc1 = vaddq_f32(acc1, d1);
      acc2 = vaddq_f32(acc2, d2);
      acc3 = vaddq_f32(acc3, d3);
    }

    float s0 = horizontal_sum_neon(acc0);
    float s1 = horizontal_sum_neon(acc1);
    float s2 = horizontal_sum_neon(acc2);
    float s3 = horizontal_sum_neon(acc3);

    // Scalar leftover for remaining subquantizers.
    for (; m < num_chunk; ++m) {
      const float *tab = lut + m * kNumCentroids;
      s0 += tab[nibble(c0, m)];
      s1 += tab[nibble(c1, m)];
      s2 += tab[nibble(c2, m)];
      s3 += tab[nibble(c3, m)];
    }
    out[i] = s0;
    out[i + 1] = s1;
    out[i + 2] = s2;
    out[i + 3] = s3;
  }
  // Remaining candidates: use single ADC kernel.
  for (; i < num; ++i) {
    pq_adc_int4_distance_neon(candidates[i], lut, num_chunk, out + i);
  }
#else
  scalar::pq_adc_int4_batch_distance(candidates_v, lut_v, num, num_chunk, out);
#endif
}

}  // namespace zvec::turbo::neon
