// Copyright 2026-present the zvec project
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

#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
namespace zvec {
namespace turbo {
namespace batch_topk_internal {
namespace {
struct Sse2 {
  static constexpr unsigned kWidth = 4;
  static __m128 Broadcast(float value) {
    return _mm_set1_ps(value);
  }
  static __m128 Load(const float *values) {
    return _mm_loadu_ps(values);
  }
  template <bool Strict>
  static uint32_t Compare(__m128 values, __m128 bound) {
    if constexpr (Strict)
      return static_cast<uint32_t>(
          _mm_movemask_ps(_mm_cmplt_ps(values, bound)));
    else
      return static_cast<uint32_t>(
          _mm_movemask_ps(_mm_cmple_ps(values, bound)));
  }
  static uint32_t Equal(__m128 values, __m128 bound) {
    return static_cast<uint32_t>(_mm_movemask_ps(_mm_cmpeq_ps(values, bound)));
  }
};
}  // namespace
const Ops &Sse2Ops(bool small) {
  return small ? Engine<Sse2, true>::Operations()
               : Engine<Sse2, false>::Operations();
}
}  // namespace batch_topk_internal
}  // namespace turbo
}  // namespace zvec
#endif
