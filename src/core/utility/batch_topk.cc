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
#include <algorithm>
#include <stdexcept>
#include <ailego/internal/cpu_features.h>
#include "batch_topk_internal.h"

#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && \
    (defined(__SSE2__) || defined(_M_X64) ||  \
     (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#define ZVEC_BATCH_TOPK_HAVE_SSE2 1
#endif

namespace zvec {
namespace core {
void BatchTopK::reset(size_t k, float threshold, bool strict_threshold) {
  if (k > (std::numeric_limits<size_t>::max() - 31) / 2) {
    throw std::length_error("BatchTopK capacity overflow");
  }
  const size_t buffer_capacity =
      std::max(size_t{32}, (2 * k + 31) & ~size_t{31});
  if (buffer_capacity > kInlineCapacity) {
    if (buffer_capacity > allocated_capacity_) {
      allocated_entries_.reset(new Candidate[buffer_capacity]);
      allocated_capacity_ = buffer_capacity;
    }
  }
  entries_ = buffer_capacity <= kInlineCapacity ? inline_entries_.data()
                                                : allocated_entries_.get();
  size_ = 0;
  limit_ = k;
  buffer_capacity_ = buffer_capacity;
  threshold_ = threshold;
  strict_threshold_ = strict_threshold;
}

void BatchTopK::compact() {
  if (size_ < limit_ || limit_ == 0) {
    return;
  }
  auto *last = entries_ + limit_ - 1;
  std::nth_element(
      entries_, last, entries_ + size_,
      [](const Candidate &a, const Candidate &b) { return a.score < b.score; });
  size_ = limit_;
  threshold_ = last->score;
  strict_threshold_ = true;
}

BatchTopK::Implementation BatchTopK::select_implementation(bool sse2, bool avx2,
                                                           bool avx512f,
                                                           bool neon,
                                                           bool avx_state,
                                                           bool avx512_state) {
  // Unused on targets where the corresponding implementation is not compiled.
  (void)sse2;
  (void)avx2;
  (void)avx512f;
  (void)neon;
  (void)avx_state;
  (void)avx512_state;
#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR)
#if defined(__AVX512F__) || defined(ZVEC_BATCH_TOPK_HAVE_AVX512)
  if (avx512f && avx2 && avx_state && avx512_state) {
    return Implementation::kAVX512;
  }
#endif
#if defined(__AVX2__) || defined(ZVEC_BATCH_TOPK_HAVE_AVX2)
  if (avx2 && avx_state) {
    return Implementation::kAVX2;
  }
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_SSE2)
  if (sse2) {
    return Implementation::kSSE2;
  }
#endif
#if defined(AILEGO_ARM64_NEON)
  if (neon) {
    return Implementation::kNEON;
  }
#endif
#endif
  return Implementation::kScalar;
}

BatchTopK::Implementation BatchTopK::implementation_for_runtime() {
  const auto &flags = ailego::internal::CpuFeatures::static_flags_;
  // Clang's AVX targets also enable the preceding SSE levels and POPCNT;
  // its AVX512F target additionally enables FMA and F16C. Check those implied
  // features as well, including when a VM exposes only a subset of its host.
  bool avx2 = flags.AVX && flags.AVX2 && flags.SSE2 && flags.SSE3 &&
              flags.SSSE3 && flags.SSE4_1 && flags.SSE4_2 && flags.POPCNT;
  bool avx512f = avx2 && flags.AVX512F && flags.FMA && flags.F16C;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  // Also require the features implied by MSVC's AVX targets.
  avx2 = avx2 && flags.FMA && flags.BMI1 && flags.BMI2 && flags.POPCNT &&
         flags.LZCNT;
  avx512f = avx512f && avx2 && flags.AVX512BW && flags.AVX512DQ &&
            flags.AVX512VL && flags.AVX512CD;
#endif
  return select_implementation(flags.SSE2, avx2, avx512f, flags.NEON,
                               flags.AVX_STATE, flags.AVX512_STATE);
}

BatchTopK::Implementation BatchTopK::implementation() {
  static const Implementation impl = implementation_for_runtime();
  return impl;
}

void BatchTopK::push_block_scalar(const float *distances, size_t count,
                                  uint32_t first_id, float scale) {
  if (limit_ == 0) {
    return;
  }
  append_tail(distances, count, first_id, scale, 0);
}

void BatchTopKImpl<BatchTopK::Implementation::kScalar>::push_block(
    const float *distances, size_t count, uint32_t first_id, float scale) {
  push_block_scalar(distances, count, first_id, scale);
}

void BatchTopKImpl<BatchTopK::Implementation::kSSE2>::push_block(
    const float *distances, size_t count, uint32_t first_id, float scale) {
#if defined(ZVEC_BATCH_TOPK_HAVE_SSE2)
  if (limit_ == 0) {
    return;
  }
  size_t i = 0;
  const __m128 multiplier = _mm_set1_ps(scale);
  for (; count - i >= 4; i += 4) {
    const __m128 scores = _mm_mul_ps(_mm_loadu_ps(distances + i), multiplier);
    const __m128 bound = _mm_set1_ps(threshold_);
    uint32_t mask =
        _mm_movemask_ps(strict_threshold_ ? _mm_cmplt_ps(scores, bound)
                                          : _mm_cmple_ps(scores, bound));
    if (mask == 0) {
      continue;
    }
    alignas(16) float values[4];
    _mm_store_ps(values, scores);
    while (mask != 0) {
      const uint32_t lane = ailego_ctz32(mask);
      append(values[lane], first_id + static_cast<uint32_t>(i) + lane);
      mask &= mask - 1;
    }
  }
  append_tail(distances, count, first_id, scale, i);
#else
  push_block_scalar(distances, count, first_id, scale);
#endif
}

void BatchTopKImpl<BatchTopK::Implementation::kNEON>::push_block(
    const float *distances, size_t count, uint32_t first_id, float scale) {
#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && defined(AILEGO_ARM64_NEON)
  if (limit_ == 0) {
    return;
  }
  size_t i = 0;
  const float32x4_t multiplier = vdupq_n_f32(scale);
  const uint32_t bits[4] = {1, 2, 4, 8};
  const uint32x4_t lane_bits = vld1q_u32(bits);
  for (; count - i >= 4; i += 4) {
    const float32x4_t scores = vmulq_f32(vld1q_f32(distances + i), multiplier);
    const float32x4_t bound = vdupq_n_f32(threshold_);
    const uint32x4_t matches =
        strict_threshold_ ? vcltq_f32(scores, bound) : vcleq_f32(scores, bound);
    uint32_t mask = vaddvq_u32(vandq_u32(matches, lane_bits));
    if (mask == 0) {
      continue;
    }
    float values[4];
    vst1q_f32(values, scores);
    while (mask != 0) {
      const uint32_t lane = ailego_ctz32(mask);
      append(values[lane], first_id + static_cast<uint32_t>(i) + lane);
      mask &= mask - 1;
    }
  }
  append_tail(distances, count, first_id, scale, i);
#else
  push_block_scalar(distances, count, first_id, scale);
#endif
}

void BatchTopK::finish() {
  compact();
}

}  // namespace core
}  // namespace zvec
