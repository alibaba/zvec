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

// NEON is the byte-shuffle equivalent of the x86 pshufb kernels: the
// 16-entry sub-quantizer LUT is exactly one vqtbl1q_u8 table, so one
// sub-quantizer's 32 look-ups are two TBL instructions (lo / hi nibble).

#include "neon/pq_quantizer_fast/pq_distance.h"
#include <zvec/ailego/internal/platform.h>
#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#endif
#include <cstddef>
#include <cstdint>
#include "common/fast_scan_common.h"
#include "scalar/pq_quantizer_fast/pq_distance.h"

namespace zvec::turbo::neon {

void pq_adc_fast_scan_neon(const void *packed_codes_v, const void *packed_lut_v,
                           size_t num_chunk, int32_t *accu32) {
#if defined(AILEGO_ARM64_NEON)
  constexpr size_t kSpillPeriod = 128;  // sub-quantizers per int32 spill
  const auto *packed_codes = reinterpret_cast<const uint8_t *>(packed_codes_v);
  const auto *packed_lut = reinterpret_cast<const uint8_t *>(packed_lut_v);
  const size_t nsq_even = fast_scan_even_chunk(num_chunk);

  const uint16x8_t u16_mask = vdupq_n_u16(0x00FF);
  const uint8x16_t low_mask = vdupq_n_u8(0x0F);

  // uint16 partial sums, one lane per vector within its 8-vector group.
  // Each slot gains at most 255 per sub-quantizer, so spilling every
  // kSpillPeriod sub-quantizers stays well below 65535.
  uint16x8_t s_a = vdupq_n_u16(0);  // vectors  0..7
  uint16x8_t s_b = vdupq_n_u16(0);  // vectors 16..23
  uint16x8_t s_c = vdupq_n_u16(0);  // vectors  8..15
  uint16x8_t s_d = vdupq_n_u16(0);  // vectors 24..31

  // int32 accumulators: 32 ints = 8 uint32x4_t, group order a, c, b, d.
  uint32x4_t acc_a0 = vdupq_n_u32(0), acc_a1 = vdupq_n_u32(0);
  uint32x4_t acc_b0 = vdupq_n_u32(0), acc_b1 = vdupq_n_u32(0);
  uint32x4_t acc_c0 = vdupq_n_u32(0), acc_c1 = vdupq_n_u32(0);
  uint32x4_t acc_d0 = vdupq_n_u32(0), acc_d1 = vdupq_n_u32(0);

  // Widen one uint16x8 partial sum element-wise into two uint32x4 and add
  // them onto the matching int32 accumulator pair.
  auto spill_into = [](uint16x8_t s, uint32x4_t &lo, uint32x4_t &hi) {
    lo = vaddq_u32(lo, vmovl_u16(vget_low_u16(s)));
    hi = vaddq_u32(hi, vmovl_u16(vget_high_u16(s)));
  };
  auto spill = [&]() {
    spill_into(s_a, acc_a0, acc_a1);
    spill_into(s_b, acc_b0, acc_b1);
    spill_into(s_c, acc_c0, acc_c1);
    spill_into(s_d, acc_d0, acc_d1);
    s_a = vdupq_n_u16(0);
    s_b = vdupq_n_u16(0);
    s_c = vdupq_n_u16(0);
    s_d = vdupq_n_u16(0);
  };

  size_t pending = 0;

  // Main loop: one sub-quantizer per iteration (16-byte code load).
  for (size_t m = 0; m < nsq_even; ++m) {
    const uint8x16_t codes = vld1q_u8(packed_codes + m * 16);
    const uint8x16_t table = vld1q_u8(packed_lut + m * 16);

    // Byte j of r0 / r1 is the LUT entry of vector mapper[2j] / mapper[2j+1].
    const uint8x16_t lo = vandq_u8(codes, low_mask);
    const uint8x16_t hi = vandq_u8(vshrq_n_u8(codes, 4), low_mask);
    const uint16x8_t r0 = vreinterpretq_u16_u8(vqtbl1q_u8(table, lo));
    const uint16x8_t r1 = vreinterpretq_u16_u8(vqtbl1q_u8(table, hi));

    // Splitting even / odd bytes yields four contiguous vector groups (see
    // kFastScanMapper): r0 even -> 0..7, r0 odd -> 16..23,
    //                   r1 even -> 8..15, r1 odd -> 24..31.
    s_a = vaddq_u16(s_a, vandq_u16(r0, u16_mask));
    s_b = vaddq_u16(s_b, vshrq_n_u16(r0, 8));
    s_c = vaddq_u16(s_c, vandq_u16(r1, u16_mask));
    s_d = vaddq_u16(s_d, vshrq_n_u16(r1, 8));

    if (++pending == kSpillPeriod) {
      spill();
      pending = 0;
    }
  }

  // Flush the trailing partials.
  spill();

  // Group order is baked into kFastScanMapper, so no permute is needed.
  // vst1q_u32 wants uint32_t *; the bit pattern matches the int32_t API.
  auto *accu = reinterpret_cast<uint32_t *>(accu32);
  vst1q_u32(accu, acc_a0);
  vst1q_u32(accu + 4, acc_a1);
  vst1q_u32(accu + 8, acc_c0);
  vst1q_u32(accu + 12, acc_c1);
  vst1q_u32(accu + 16, acc_b0);
  vst1q_u32(accu + 20, acc_b1);
  vst1q_u32(accu + 24, acc_d0);
  vst1q_u32(accu + 28, acc_d1);
#else
  // Unlike the float-returning PQ kernels, a no-op stub here would leave
  // accu32 untouched and silently yield zero distances, so forward instead.
  scalar::pq_adc_fast_scan(packed_codes_v, packed_lut_v, num_chunk, accu32);
#endif
}

}  // namespace zvec::turbo::neon
