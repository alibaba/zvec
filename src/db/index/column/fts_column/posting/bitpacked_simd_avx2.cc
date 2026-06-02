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

#include "bitpacked_simd_avx2.h"

#if defined(__AVX2__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))

#include <immintrin.h>
#include <cstring>
#include "bitpacked_simd_sse41.h"

#ifdef _MSC_VER
#include <intrin.h>
static inline int ctz_u32(unsigned int v) {
  unsigned long index;
  _BitScanForward(&index, v);
  return static_cast<int>(index);
}
#else
static inline int ctz_u32(unsigned int v) {
  return __builtin_ctz(v);
}
#endif

namespace zvec::fts::simd {

// ------------------------------------------------------------
// avx2_max_128
// ------------------------------------------------------------

void avx2_max_128(const uint32_t *deltas, const uint32_t *tfs,
                  const uint32_t *doc_lens, size_t start, uint32_t count,
                  uint32_t &max_delta, uint32_t &max_tf, uint32_t &max_dl) {
  __m256i vmax_delta = _mm256_setzero_si256();
  __m256i vmax_tf = _mm256_setzero_si256();
  __m256i vmax_dl = _mm256_setzero_si256();

  for (uint32_t i = 0; i < count; i += 8) {
    vmax_delta = _mm256_max_epu32(
        vmax_delta, _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(&deltas[start + i])));
    vmax_tf = _mm256_max_epu32(
        vmax_tf,
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&tfs[start + i])));
    vmax_dl = _mm256_max_epu32(
        vmax_dl, _mm256_loadu_si256(
                     reinterpret_cast<const __m256i *>(&doc_lens[start + i])));
  }

  // Horizontal max: reduce 8 lanes to scalar
  auto hmax = [](__m256i v) -> uint32_t {
    // Reduce 256-bit to 128-bit by taking max of high and low halves
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m = _mm_max_epu32(lo, hi);
    // Reduce 128-bit to scalar
    m = _mm_max_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_epu32(m, _mm_shuffle_epi32(m, _MM_SHUFFLE(1, 0, 3, 2)));
    return static_cast<uint32_t>(_mm_extract_epi32(m, 0));
  };

  max_delta = hmax(vmax_delta);
  max_tf = hmax(vmax_tf);
  max_dl = hmax(vmax_dl);
}

// ------------------------------------------------------------
// avx2_pack_uint32_128 — fallback to SSE4.1
// ------------------------------------------------------------

void avx2_pack_uint32_128(const uint32_t *in, uint8_t bitwidth, uint8_t *out) {
  // FastPForLib does not provide AVX2 bitpacking; delegate to SSE4.1.
  sse41_pack_uint32_128(in, bitwidth, out);
}

// ------------------------------------------------------------
// avx2_unpack_uint32_128 — fallback to SSE4.1
// ------------------------------------------------------------

void avx2_unpack_uint32_128(const uint8_t *in, uint8_t bitwidth,
                            uint32_t *out) {
  // FastPForLib does not provide AVX2 bitpacking; delegate to SSE4.1.
  sse41_unpack_uint32_128(in, bitwidth, out);
}

// ------------------------------------------------------------
// avx2_prefix_sum_128
// ------------------------------------------------------------

void avx2_prefix_sum_128(const uint32_t *deltas, uint32_t min_doc_id,
                         uint32_t /*count*/, uint32_t *out) {
  // Process 8 elements per iteration (16 groups of 8 = 128 elements).
  // Within each 256-bit register we compute a prefix-sum, then propagate
  // the carry (last element) to the next group.
  __m256i carry = _mm256_set1_epi32(static_cast<int>(min_doc_id) -
                                    static_cast<int>(deltas[0]));

  for (uint32_t g = 0; g < 16; ++g) {
    __m256i v =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&deltas[g * 8]));

    // In-register prefix-sum for 8 elements (two 128-bit lanes independently,
    // then cross-lane fixup).

    // Step 1: shift by 1 element (4 bytes) within each 128-bit lane
    __m256i shifted1 = _mm256_bslli_epi128(v, 4);
    v = _mm256_add_epi32(v, shifted1);

    // Step 2: shift by 2 elements (8 bytes) within each 128-bit lane
    __m256i shifted2 = _mm256_bslli_epi128(v, 8);
    v = _mm256_add_epi32(v, shifted2);

    // Step 3: cross-lane fixup — high lane needs the sum of the low lane's
    // last element (index 3) added to all its elements.
    // Broadcast low lane's element[3] to all positions of high lane.
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i lo_last = _mm_shuffle_epi32(lo, _MM_SHUFFLE(3, 3, 3, 3));
    __m256i cross = _mm256_set_m128i(lo_last, _mm_setzero_si128());
    v = _mm256_add_epi32(v, cross);

    // Add carry from previous group
    v = _mm256_add_epi32(v, carry);

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&out[g * 8]), v);

    // Broadcast the last element (index 7) as carry for next group.
    // Element 7 is in the high lane at position 3.
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i hi_last = _mm_shuffle_epi32(hi, _MM_SHUFFLE(3, 3, 3, 3));
    carry = _mm256_set_m128i(hi_last, hi_last);
  }
}

// ------------------------------------------------------------
// avx2_find_first_ge
// ------------------------------------------------------------

size_t avx2_find_first_ge(const uint32_t *arr, uint32_t size, uint32_t target,
                          size_t start) {
  const __m256i vtarget = _mm256_set1_epi32(static_cast<int>(target));
  const __m256i sign_bit = _mm256_set1_epi32(static_cast<int>(0x80000000u));
  const __m256i starget = _mm256_xor_si256(vtarget, sign_bit);

  size_t i = start;
  // Scalar until aligned to 4-element boundary (minimum for unaligned AVX2)
  for (; i < size && (i & 3); ++i) {
    if (arr[i] >= target) {
      return i;
    }
  }
  // SIMD scan: 8 elements at a time
  for (; i + 8 <= size; i += 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&arr[i]));
    __m256i sv = _mm256_xor_si256(v, sign_bit);
    // cmpgt: sv < starget means arr[i] < target
    __m256i cmp = _mm256_cmpgt_epi32(starget, sv);
    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
    if (mask != 0xFF) {
      // At least one element >= target
      int first = ctz_u32(static_cast<unsigned int>(~mask & 0xFF));
      return i + first;
    }
  }
  // Scalar tail
  for (; i < size; ++i) {
    if (arr[i] >= target) {
      return i;
    }
  }
  return size;
}

}  // namespace zvec::fts::simd

#else  // !defined(__AVX2__) && !(defined(_MSC_VER) && (defined(_M_X64) ||
       // defined(_M_IX86)))

// Stub implementations when AVX2 is not available at compile time.
// The runtime dispatch layer (bitpacked_simd_dispatch.cc) will never call
// these on non-AVX2 machines, but the linker still needs the symbols.

namespace zvec::fts::simd {

void avx2_max_128(const uint32_t *, const uint32_t *, const uint32_t *, size_t,
                  uint32_t, uint32_t &max_delta, uint32_t &max_tf,
                  uint32_t &max_dl) {
  max_delta = 0;
  max_tf = 0;
  max_dl = 0;
}

void avx2_pack_uint32_128(const uint32_t *, uint8_t, uint8_t *) {}

void avx2_unpack_uint32_128(const uint8_t *, uint8_t, uint32_t *) {}

void avx2_prefix_sum_128(const uint32_t *, uint32_t, uint32_t, uint32_t *) {}

size_t avx2_find_first_ge(const uint32_t *, uint32_t size, uint32_t, size_t) {
  return size;
}

}  // namespace zvec::fts::simd

#endif  // defined(__AVX2__)
