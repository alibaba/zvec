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

#include "fht.h"
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "common/fht_common.h"
#include "scalar/rotate/fht/fht.h"

namespace zvec::turbo::neon {

void fht_flip_sign_neon(const uint8_t *flip, float *data, size_t dim) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
  size_t simd_end = dim & ~3u;
  size_t flip_bytes = (dim + 7) / 8;
  for (size_t i = 0; i < simd_end; i += 4) {
    uint16_t bits16;
    size_t byte_pos = i / 8;
    if (byte_pos + 1 < flip_bytes) {
      std::memcpy(&bits16, &flip[byte_pos], sizeof(bits16));
    } else {
      bits16 = flip[byte_pos];
    }
    bits16 >>= (i % 8);
    uint32_t b0 = bits16 & 1u;
    uint32_t b1 = (bits16 >> 1) & 1u;
    uint32_t b2 = (bits16 >> 2) & 1u;
    uint32_t b3 = (bits16 >> 3) & 1u;
    uint32x4_t bit_mask = {b0, b1, b2, b3};
    uint32x4_t sign_mask = vmulq_u32(bit_mask, sign_bit);
    float32x4_t v = vld1q_f32(&data[i]);
    v = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v), sign_mask));
    vst1q_f32(&data[i], v);
  }
  // Scalar tail
  for (size_t i = simd_end; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
#else
  (void)flip;
  (void)data;
  (void)dim;
#endif
}

void fht_kacs_walk_neon(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  size_t half_end = half & ~3u;
  for (size_t i = 0; i < half_end; i += 4) {
    float32x4_t x = vld1q_f32(&data[i]);
    float32x4_t y = vld1q_f32(&data[i + offset]);
    vst1q_f32(&data[i], vaddq_f32(x, y));
    vst1q_f32(&data[i + offset], vsubq_f32(x, y));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float x = data[i];
    float y = data[i + offset];
    data[i] = x + y;
    data[i + offset] = x - y;
  }
  if (base != 0) {
    data[half] *= std::sqrt(2.0f);
  }
#else
  (void)data;
  (void)len;
#endif
}

void fht_inv_kacs_walk_neon(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  if (base != 0) {
    data[half] *= std::sqrt(0.5f);
  }
  size_t half_end = half & ~3u;
  const float32x4_t half_fac = vdupq_n_f32(0.5f);
  for (size_t i = 0; i < half_end; i += 4) {
    float32x4_t a = vld1q_f32(&data[i]);
    float32x4_t b = vld1q_f32(&data[i + offset]);
    vst1q_f32(&data[i], vmulq_f32(vaddq_f32(a, b), half_fac));
    vst1q_f32(&data[i + offset], vmulq_f32(vsubq_f32(a, b), half_fac));
  }
  // Scalar tail
  for (size_t i = half_end; i < half; ++i) {
    float a = data[i];
    float b = data[i + offset];
    data[i] = (a + b) * 0.5f;
    data[i + offset] = (a - b) * 0.5f;
  }
#else
  (void)data;
  (void)len;
#endif
}

void fht_vec_rescale_neon(float *data, size_t n, float factor) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const float32x4_t fac = vdupq_n_f32(factor);
  size_t simd_end = n & ~3u;
  for (size_t i = 0; i < simd_end; i += 4) {
    float32x4_t v = vld1q_f32(&data[i]);
    vst1q_f32(&data[i], vmulq_f32(v, fac));
  }
  // Scalar tail
  for (size_t i = simd_end; i < n; ++i) {
    data[i] *= factor;
  }
#else
  (void)data;
  (void)n;
  (void)factor;
#endif
}

void fht_rotate_neon(const float *in, float *out, size_t in_dim,
                     size_t /*out_dim*/, void *ctx) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  static constexpr FhtPrimitives kPrim = {
      fht_flip_sign_neon, scalar::fht_inplace, fht_kacs_walk_neon,
      fht_inv_kacs_walk_neon, fht_vec_rescale_neon};
  fht_rotate_impl(in, out, in_dim, ctx, kPrim);
#else
  (void)in;
  (void)out;
  (void)in_dim;
  (void)ctx;
#endif
}

void fht_unrotate_neon(const float *in, float *out, size_t in_dim,
                       size_t /*out_dim*/, void *ctx) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  static constexpr FhtPrimitives kPrim = {
      fht_flip_sign_neon, scalar::fht_inplace, fht_kacs_walk_neon,
      fht_inv_kacs_walk_neon, fht_vec_rescale_neon};
  fht_unrotate_impl(in, out, in_dim, ctx, kPrim);
#else
  (void)in;
  (void)out;
  (void)in_dim;
  (void)ctx;
#endif
}

}  // namespace zvec::turbo::neon
