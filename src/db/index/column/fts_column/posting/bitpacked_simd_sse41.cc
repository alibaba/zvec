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

#include "bitpacked_simd_sse41.h"

#if defined(__SSE4_1__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))

#include <bitpackinghelpers.h>
#include <emmintrin.h>  // SSE2
#include <simdbitpacking.h>
#include <smmintrin.h>  // SSE4.1
#include <cstring>
#include "bitpacked_posting_list.h"

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
// sse41_max_128
// ------------------------------------------------------------

void sse41_max_128(const uint32_t *deltas, const uint32_t *tfs,
                   const uint32_t *doc_lens, size_t start, uint32_t count,
                   uint32_t &max_delta, uint32_t &max_tf, uint32_t &max_dl) {
  __m128i vmax_delta = _mm_setzero_si128();
  __m128i vmax_tf = _mm_setzero_si128();
  __m128i vmax_dl = _mm_setzero_si128();
  for (uint32_t i = 0; i < count; i += 4) {
    vmax_delta = _mm_max_epu32(
        vmax_delta,
        _mm_load_si128(reinterpret_cast<const __m128i *>(&deltas[start + i])));
    vmax_tf = _mm_max_epu32(
        vmax_tf,
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&tfs[start + i])));
    vmax_dl = _mm_max_epu32(
        vmax_dl, _mm_loadu_si128(
                     reinterpret_cast<const __m128i *>(&doc_lens[start + i])));
  }
  // Horizontal max: reduce 4 lanes to scalar
  auto hmax = [](__m128i v) -> uint32_t {
    v = _mm_max_epu32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    v = _mm_max_epu32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    return static_cast<uint32_t>(_mm_extract_epi32(v, 0));
  };
  max_delta = hmax(vmax_delta);
  max_tf = hmax(vmax_tf);
  max_dl = hmax(vmax_dl);
}

// ------------------------------------------------------------
// sse41_pack_uint32_128
// ------------------------------------------------------------

void sse41_pack_uint32_128(const uint32_t *in, uint8_t bitwidth, uint8_t *out) {
  const size_t total_bytes =
      BitPackedPostingList::simd_packed_byte_size(bitwidth);
  if ((reinterpret_cast<uintptr_t>(out) & 15) == 0) {
    FastPForLib::SIMD_fastpackwithoutmask_32(
        in, reinterpret_cast<__m128i *>(out), bitwidth);
  } else {
    alignas(16) __m128i simd_out[32];
    FastPForLib::SIMD_fastpackwithoutmask_32(in, simd_out, bitwidth);
    std::memcpy(out, simd_out, total_bytes);
  }
}

// ------------------------------------------------------------
// sse41_unpack_uint32_128
// ------------------------------------------------------------

void sse41_unpack_uint32_128(const uint8_t *in, uint8_t bitwidth,
                             uint32_t *out) {
  if ((reinterpret_cast<uintptr_t>(in) & 15) == 0) {
    FastPForLib::SIMD_fastunpack_32(reinterpret_cast<const __m128i *>(in), out,
                                    bitwidth);
  } else {
    const size_t packed_bytes =
        BitPackedPostingList::simd_packed_byte_size(bitwidth);
    alignas(16) __m128i simd_in[32];
    std::memcpy(simd_in, in, packed_bytes);
    FastPForLib::SIMD_fastunpack_32(simd_in, out, bitwidth);
  }
}

// ------------------------------------------------------------
// sse41_prefix_sum_128
// ------------------------------------------------------------

void sse41_prefix_sum_128(const uint32_t *deltas, uint32_t min_doc_id,
                          uint32_t /*count*/, uint32_t *out) {
  __m128i carry = _mm_set1_epi32(static_cast<int>(min_doc_id) -
                                 static_cast<int>(deltas[0]));

  for (uint32_t g = 0; g < 32; ++g) {
    __m128i v =
        _mm_load_si128(reinterpret_cast<const __m128i *>(&deltas[g * 4]));

    // In-register prefix-sum for 4 elements
    __m128i shifted1 = _mm_slli_si128(v, 4);
    v = _mm_add_epi32(v, shifted1);
    __m128i shifted2 = _mm_slli_si128(v, 8);
    v = _mm_add_epi32(v, shifted2);

    // Add carry from previous group
    v = _mm_add_epi32(v, carry);

    _mm_store_si128(reinterpret_cast<__m128i *>(&out[g * 4]), v);

    // Broadcast the last element as carry for next group
    carry = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 3, 3));
  }
}

// ------------------------------------------------------------
// sse41_find_first_ge
// ------------------------------------------------------------

size_t sse41_find_first_ge(const uint32_t *arr, uint32_t size, uint32_t target,
                           size_t start) {
  const __m128i vtarget = _mm_set1_epi32(static_cast<int>(target));
  const __m128i sign_bit = _mm_set1_epi32(static_cast<int>(0x80000000u));
  const __m128i starget = _mm_xor_si128(vtarget, sign_bit);

  size_t i = start;
  // Scalar until aligned to 4-element boundary
  for (; i < size && (i & 3); ++i) {
    if (arr[i] >= target) return i;
  }
  // SIMD scan: 4 elements at a time
  for (; i + 4 <= size; i += 4) {
    __m128i v = _mm_load_si128(reinterpret_cast<const __m128i *>(&arr[i]));
    __m128i sv = _mm_xor_si128(v, sign_bit);
    __m128i cmp = _mm_cmplt_epi32(sv, starget);
    int mask = _mm_movemask_ps(_mm_castsi128_ps(cmp));
    if (mask != 0xF) {
      int first = ctz_u32(static_cast<unsigned int>(~mask));
      return i + first;
    }
  }
  // Scalar tail
  for (; i < size; ++i) {
    if (arr[i] >= target) return i;
  }
  return size;
}

}  // namespace zvec::fts::simd

#else  // !defined(__SSE4_1__) && !(defined(_MSC_VER) && (defined(_M_X64) ||
       // defined(_M_IX86)))

// Stub implementations when SSE4.1 is not available at compile time.
// The runtime dispatch layer (bitpacked_simd_dispatch.cc) will never call
// these on non-SSE4.1 machines, but the linker still needs the symbols.

namespace zvec::fts::simd {

void sse41_max_128(const uint32_t *, const uint32_t *, const uint32_t *, size_t,
                   uint32_t, uint32_t &max_delta, uint32_t &max_tf,
                   uint32_t &max_dl) {
  max_delta = 0;
  max_tf = 0;
  max_dl = 0;
}

void sse41_pack_uint32_128(const uint32_t *, uint8_t, uint8_t *) {}

void sse41_unpack_uint32_128(const uint8_t *, uint8_t, uint32_t *) {}

void sse41_prefix_sum_128(const uint32_t *, uint32_t, uint32_t, uint32_t *) {}

size_t sse41_find_first_ge(const uint32_t *, uint32_t size, uint32_t, size_t) {
  return size;
}

}  // namespace zvec::fts::simd

#endif  // defined(__SSE4_1__)
