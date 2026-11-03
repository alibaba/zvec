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

#if defined(__AVX2__)
namespace zvec {
namespace core {
namespace batch_topk_internal {
namespace {
struct Avx2 {
  static constexpr unsigned kWidth = 8;
  static __m256 Broadcast(float value) {
    return _mm256_set1_ps(value);
  }
  static __m256 Load(const float *values) {
    return _mm256_loadu_ps(values);
  }
  template <bool Strict>
  static uint32_t Compare(__m256 values, __m256 bound) {
    return static_cast<uint32_t>(_mm256_movemask_ps(
        _mm256_cmp_ps(values, bound, Strict ? _CMP_LT_OQ : _CMP_LE_OQ)));
  }
  static uint32_t Equal(__m256 values, __m256 bound) {
    return static_cast<uint32_t>(
        _mm256_movemask_ps(_mm256_cmp_ps(values, bound, _CMP_EQ_OQ)));
  }
};
}  // namespace
const Ops &Avx2Ops(bool small) {
  return small ? Engine<Avx2, true>::Operations()
               : Engine<Avx2, false>::Operations();
}
}  // namespace batch_topk_internal
}  // namespace core
}  // namespace zvec
#endif
