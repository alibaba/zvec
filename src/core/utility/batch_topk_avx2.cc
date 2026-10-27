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
#include "batch_topk_internal.h"

namespace zvec {
namespace core {

void BatchTopKImpl<BatchTopK::Implementation::kAVX2>::push_block(
    const float *distances, size_t count, uint32_t first_id, float scale) {
#if !defined(ZVEC_BATCH_TOPK_FORCE_SCALAR) && defined(__AVX2__)
  if (limit_ == 0) {
    return;
  }
  size_t i = 0;
  // Refresh the threshold once per SIMD group. Only passing lanes append
  // candidates and share the common compaction routine.
  const __m256 multiplier = _mm256_set1_ps(scale);
  for (; count - i >= 8; i += 8) {
    const __m256 scores =
        _mm256_mul_ps(_mm256_loadu_ps(distances + i), multiplier);
    const __m256 bound = _mm256_set1_ps(threshold_);
    uint32_t mask = _mm256_movemask_ps(
        strict_threshold_ ? _mm256_cmp_ps(scores, bound, _CMP_LT_OQ)
                          : _mm256_cmp_ps(scores, bound, _CMP_LE_OQ));
    if (mask == 0) {
      continue;
    }
    alignas(32) float values[8];
    _mm256_store_ps(values, scores);
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

}  // namespace core
}  // namespace zvec
