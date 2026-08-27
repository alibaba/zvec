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

// FastScan block / LUT layout, shared by the packing helpers, the scalar
// kernel and the SIMD kernels.
//
// Packed codes (Package32): a block holds 32 vectors. For subquantizer m the
// 16 bytes at packed_codes[m * 16 .. m * 16 + 15] hold that subquantizer's 32
// 4-bit codes: byte j stores vector kFastScanMapper[2 * j] in its low nibble
// and vector kFastScanMapper[2 * j + 1] in its high nibble.
//
// Packed LUT: subquantizer m's 16 uint8 entries live at packed_lut[m * 16], so
// one 32-byte load covers the same subquantizer pair as the code load, one per
// 128-bit lane -- both lanes of a byte shuffle therefore do useful work.
//
// An odd num_chunk is padded with one all-zero subquantizer so that kernels can
// always consume subquantizers in pairs (32-byte loads).

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace zvec::turbo {

/// Vectors per packed block.
constexpr size_t kFastScanBlockSize = 32;

/// Vector index stored at nibble slot p (p = 2 * byte + nibble).
///
/// Defined as kFastScanMapper[4k + m] = m * 8 + k (m in 0..3, k in 0..7) so
/// that after a kernel splits the shuffle results into even / odd bytes, the
/// four accumulators line up with the contiguous vector groups
/// 0..7 / 8..15 / 16..23 / 24..31 -- no cross-lane permute is needed before
/// storing the result.
constexpr uint8_t kFastScanMapper[32] = {
    0, 8,  16, 24, 1, 9,  17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
    4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31};

/// Nibble slot of vector v (inverse of kFastScanMapper).
inline size_t fast_scan_nibble_slot(size_t v) {
  return 4 * (v & 7) + (v >> 3);
}

/// Number of subquantizers after padding to an even count.
inline size_t fast_scan_even_chunk(size_t num_chunk) {
  return (num_chunk + 1) & ~static_cast<size_t>(1);
}

/// Byte size of one packed block (== 32 * ceil(num_chunk / 2)).
inline size_t fast_scan_packed_block_size(size_t num_chunk) {
  return fast_scan_even_chunk(num_chunk) * 16;
}

/// Byte size of the packed uint8 LUT.
inline size_t fast_scan_packed_lut_size(size_t num_chunk) {
  return fast_scan_even_chunk(num_chunk) * 16;
}

/// Pack up to 32 plain nibble-packed PQ codes (subquantizer m in the low
/// nibble of byte m / 2 when m is even, the high nibble when odd; `stride`
/// bytes apart) into one Package32 block. Vectors beyond `num` and the pad
/// subquantizer of an odd num_chunk are zero-filled. `out` must hold
/// fast_scan_packed_block_size(num_chunk) bytes.
inline void fast_scan_pack_codes(const uint8_t *codes, size_t num,
                                 size_t stride, size_t num_chunk,
                                 uint8_t *out) {
  std::memset(out, 0, fast_scan_packed_block_size(num_chunk));
  if (num > kFastScanBlockSize) {
    num = kFastScanBlockSize;
  }
  for (size_t v = 0; v < num; ++v) {
    const uint8_t *code = codes + v * stride;
    const size_t p = fast_scan_nibble_slot(v);
    const size_t j = p >> 1;
    const unsigned shift = static_cast<unsigned>(p & 1) * 4;
    for (size_t m = 0; m < num_chunk; ++m) {
      uint8_t c = static_cast<uint8_t>(code[m >> 1] >> ((m & 1) * 4)) & 0x0F;
      out[m * 16 + j] |= static_cast<uint8_t>(c << shift);
    }
  }
}

/// Pack a plain uint8 LUT [num_chunk * 16] for kernel consumption: a
/// contiguous copy plus one zeroed 16-byte group when num_chunk is odd.
/// `out` must hold fast_scan_packed_lut_size(num_chunk) bytes.
inline void fast_scan_pack_lut(const uint8_t *lut, size_t num_chunk,
                               uint8_t *out) {
  std::memcpy(out, lut, num_chunk * 16);
  if (num_chunk & 1) {
    std::memset(out + num_chunk * 16, 0, 16);
  }
}

}  // namespace zvec::turbo
