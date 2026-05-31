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

#include "bitpacked_simd_scalar.h"
#include <bitpackinghelpers.h>
#include <algorithm>
#include <cstring>
#include "bitpacked_posting_list.h"

namespace zvec::fts::simd {

// ------------------------------------------------------------
// scalar_max_128
// ------------------------------------------------------------

void scalar_max_128(const uint32_t *deltas, const uint32_t *tfs,
                    const uint32_t *doc_lens, size_t start, uint32_t count,
                    uint32_t &max_delta, uint32_t &max_tf, uint32_t &max_dl) {
  uint32_t md = 0, mt = 0, ml = 0;
  for (uint32_t i = 0; i < count; ++i) {
    md = std::max(md, deltas[start + i]);
    mt = std::max(mt, tfs[start + i]);
    ml = std::max(ml, doc_lens[start + i]);
  }
  max_delta = md;
  max_tf = mt;
  max_dl = ml;
}

// ------------------------------------------------------------
// scalar_pack_uint32_128 / scalar_unpack_uint32_128
// ------------------------------------------------------------
//
// These produce / consume the SAME byte layout as the SSE/AVX2 SIMD packers
// (SIMD_fastpackwithoutmask_32 / SIMD_fastunpack_32), so an index encoded on
// one architecture can be decoded on another. The SIMD layout interleaves the
// 128 values across 4 lanes: lane L (0..3), read across the bitwidth output
// __m128i words, holds the scalar bit-packing of the 32 values
// { in[L], in[4+L], in[8+L], ..., in[124+L] }. We reproduce that exactly by
// packing each lane independently with FastPForLib::fastpackwithoutmask and
// interleaving the resulting 32-bit words at 128-bit (4-lane) granularity.

void scalar_pack_uint32_128(const uint32_t *in, uint8_t bitwidth,
                            uint8_t *out) {
  const size_t total_bytes =
      BitPackedPostingList::simd_packed_byte_size(bitwidth);
  std::memset(out, 0, total_bytes);

  uint32_t *out32 = reinterpret_cast<uint32_t *>(out);
  for (uint32_t lane = 0; lane < 4; ++lane) {
    uint32_t lane_in[32];
    for (uint32_t k = 0; k < 32; ++k) {
      lane_in[k] = in[k * 4 + lane];
    }
    alignas(16) uint32_t lane_packed[32] = {};
    FastPForLib::fastpackwithoutmask(lane_in, lane_packed, bitwidth);
    for (uint32_t j = 0; j < bitwidth; ++j) {
      out32[j * 4 + lane] = lane_packed[j];
    }
  }
}

void scalar_unpack_uint32_128(const uint8_t *in, uint8_t bitwidth,
                              uint32_t *out) {
  const uint32_t *in32 = reinterpret_cast<const uint32_t *>(in);
  for (uint32_t lane = 0; lane < 4; ++lane) {
    alignas(16) uint32_t lane_packed[32] = {};
    for (uint32_t j = 0; j < bitwidth; ++j) {
      lane_packed[j] = in32[j * 4 + lane];
    }
    uint32_t lane_out[32];
    FastPForLib::fastunpack(lane_packed, lane_out, bitwidth);
    for (uint32_t k = 0; k < 32; ++k) {
      out[k * 4 + lane] = lane_out[k];
    }
  }
}

// ------------------------------------------------------------
// scalar_prefix_sum_128
// ------------------------------------------------------------

void scalar_prefix_sum_128(const uint32_t *deltas, uint32_t min_doc_id,
                           uint32_t count, uint32_t *out) {
  // First element: min_doc_id corresponds to deltas[0]
  out[0] = min_doc_id;
  for (uint32_t i = 1; i < count; ++i) {
    out[i] = out[i - 1] + deltas[i];
  }
}

// ------------------------------------------------------------
// scalar_find_first_ge
// ------------------------------------------------------------

size_t scalar_find_first_ge(const uint32_t *arr, uint32_t size, uint32_t target,
                            size_t start) {
  for (size_t i = start; i < size; ++i) {
    if (arr[i] >= target) return i;
  }
  return size;
}

}  // namespace zvec::fts::simd
