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

#if defined(AILEGO_ARM64_NEON)
namespace zvec {
namespace core {
namespace batch_topk_internal {
namespace {
struct Neon {
  static constexpr unsigned kWidth = 4;
  static float32x4_t Broadcast(float value) {
    return vdupq_n_f32(value);
  }
  static float32x4_t Load(const float *values) {
    return vld1q_f32(values);
  }
  static uint32_t Mask(uint32x4_t matches) {
    const uint32_t bits[4] = {1, 2, 4, 8};
    return vaddvq_u32(vandq_u32(matches, vld1q_u32(bits)));
  }
  template <bool Strict>
  static uint32_t Compare(float32x4_t values, float32x4_t bound) {
    if constexpr (Strict)
      return Mask(vcltq_f32(values, bound));
    else
      return Mask(vcleq_f32(values, bound));
  }
  static uint32_t Equal(float32x4_t values, float32x4_t bound) {
    return Mask(vceqq_f32(values, bound));
  }
};
}  // namespace
const Ops &NeonOps(bool small) {
  return small ? Engine<Neon, true>::Operations()
               : Engine<Neon, false>::Operations();
}
}  // namespace batch_topk_internal
}  // namespace core
}  // namespace zvec
#endif
