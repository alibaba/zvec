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

#include "bitset_helper.h"
#include <zvec/ailego/internal/platform.h>

#ifndef __SSE4_2__
#define bitset_popcount32 ailego_popcount32
#define bitset_popcount64 ailego_popcount64
#else
#define bitset_popcount32 _mm_popcnt_u32
#define bitset_popcount64 _mm_popcnt_u64
#endif  // !__SSE4_2__

#if defined(__ARM_NEON)
static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    vst1q_u32(lhs, vandq_u32(vld1q_u32(lhs), vld1q_u32(rhs)));
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= rhs[0];
  }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    vst1q_u32(lhs, vbicq_u32(vld1q_u32(lhs), vld1q_u32(rhs)));
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= ~rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= ~rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= ~rhs[0];
  }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    vst1q_u32(lhs, vorrq_u32(vld1q_u32(lhs), vld1q_u32(rhs)));
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] |= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] |= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] |= rhs[0];
  }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    vst1q_u32(lhs, veorq_u32(vld1q_u32(lhs), vld1q_u32(rhs)));
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] ^= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] ^= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] ^= rhs[0];
  }
}

static inline void bitset_not(uint32_t *lhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  static const uint32x4_t v_zero = vdupq_n_u32(0);

  for (; lhs != last_aligned; lhs += 4) {
    vst1q_u32(lhs, vornq_u32(v_zero, vld1q_u32(lhs)));
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] = ~lhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] = ~lhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] = ~lhs[0];
  }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    uint64x2_t vu64 = vld1q_u64((const uint64_t *)lhs);
    if ((vgetq_lane_u64(vu64, 0) & vgetq_lane_u64(vu64, 1)) != (uint64_t)-1) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0xffffffffu) {
        return false;
      }
  }
  return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    uint64x2_t vu64 = vld1q_u64((const uint64_t *)lhs);
    if (vgetq_lane_u64(vu64, 0) | vgetq_lane_u64(vu64, 1)) {
      return true;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return true;
      }
  }
  return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    uint64x2_t vu64 = vld1q_u64((const uint64_t *)lhs);
    if (vgetq_lane_u64(vu64, 0) | vgetq_lane_u64(vu64, 1)) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return false;
      }
  }
  return true;
}

#elif defined(__AVX2__)
static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
      _mm256_store_si256((__m256i *)lhs, _mm256_and_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
      _mm256_storeu_si256((__m256i *)lhs, _mm256_and_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      lhs[2] &= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= rhs[0];
  }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
      _mm256_store_si256((__m256i *)lhs, _mm256_andnot_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
      _mm256_storeu_si256((__m256i *)lhs, _mm256_andnot_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      lhs[2] &= ~rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= ~rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= ~rhs[0];
  }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
      _mm256_store_si256((__m256i *)lhs, _mm256_or_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
      _mm256_storeu_si256((__m256i *)lhs, _mm256_or_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      lhs[2] |= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] |= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] |= rhs[0];
  }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0 && ((uintptr_t)rhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_load_si256((__m256i *)rhs);
      _mm256_store_si256((__m256i *)lhs, _mm256_xor_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8, rhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      __m256i ymm1 = _mm256_loadu_si256((__m256i *)rhs);
      _mm256_storeu_si256((__m256i *)lhs, _mm256_xor_si256(ymm1, ymm0));
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
      lhs += 4;
      rhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      lhs[2] ^= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] ^= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] ^= rhs[0];
  }
}

static inline void bitset_not(uint32_t *lhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  static const __m256i mask_256 = _mm256_set1_epi32(0xffffffffu);
  static const __m128i mask_128 = _mm_set1_epi32(0xffffffffu);

  if (((uintptr_t)lhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8) {
      _mm256_store_si256(
          (__m256i *)lhs,
          _mm256_andnot_si256(_mm256_load_si256((__m256i *)lhs), mask_256));
    }
    if (last >= last_aligned + 4) {
      _mm_store_si128(
          (__m128i *)lhs,
          _mm_andnot_si128(_mm_load_si128((__m128i *)lhs), mask_128));
      lhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8) {
      _mm256_storeu_si256(
          (__m256i *)lhs,
          _mm256_andnot_si256(_mm256_loadu_si256((__m256i *)lhs), mask_256));
    }
    if (last >= last_aligned + 4) {
      _mm_storeu_si128(
          (__m128i *)lhs,
          _mm_andnot_si128(_mm_lddqu_si128((__m128i *)lhs), mask_128));
      lhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      lhs[2] = ~lhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] = ~lhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] = ~lhs[0];
  }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  static const __m256i mask_256 = _mm256_set1_epi32(0xffffffffu);
  static const __m128i mask_128 = _mm_set1_epi32(0xffffffffu);

  if (((uintptr_t)lhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i neq =
          _mm256_xor_si256(_mm256_load_si256((__m256i *)lhs), mask_256);
      if (!_mm256_testz_si256(neq, neq)) {
        return false;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i neq = _mm_xor_si128(_mm_load_si128((__m128i *)lhs), mask_128);
      if (!_mm_testz_si128(neq, neq)) {
        return false;
      }
      lhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i neq =
          _mm256_xor_si256(_mm256_loadu_si256((__m256i *)lhs), mask_256);
      if (!_mm256_testz_si256(neq, neq)) {
        return false;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i neq = _mm_xor_si128(_mm_lddqu_si128((__m128i *)lhs), mask_128);
      if (!_mm_testz_si128(neq, neq)) {
        return false;
      }
      lhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      if (lhs[2] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0xffffffffu) {
        return false;
      }
  }
  return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      if (!_mm256_testz_si256(ymm0, ymm0)) {
        return true;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return true;
      }
      lhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      if (!_mm256_testz_si256(ymm0, ymm0)) {
        return true;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return true;
      }
      lhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      if (lhs[2] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return true;
      }
  }
  return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  if (((uintptr_t)lhs & 0x1f) == 0) {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i ymm0 = _mm256_load_si256((__m256i *)lhs);
      if (!_mm256_testz_si256(ymm0, ymm0)) {
        return false;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return false;
      }
      lhs += 4;
    }
  } else {
    for (; lhs != last_aligned; lhs += 8) {
      __m256i ymm0 = _mm256_loadu_si256((__m256i *)lhs);
      if (!_mm256_testz_si256(ymm0, ymm0)) {
        return false;
      }
    }
    if (last >= last_aligned + 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return false;
      }
      lhs += 4;
    }
  }
  switch (last - lhs) {
    case 3:
      if (lhs[2] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return false;
      }
  }
  return true;
}

#elif defined(__SSE2__)
#ifndef __SSE3__
#define _mm_lddqu_si128 _mm_loadu_si128
#endif  // !__SSE3__

static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
    }
  } else {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_and_si128(xmm1, xmm0));
    }
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= rhs[0];
  }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
    }
  } else {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_andnot_si128(xmm1, xmm0));
    }
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= ~rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= ~rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= ~rhs[0];
  }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
    }
  } else {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_or_si128(xmm1, xmm0));
    }
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] |= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] |= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] |= rhs[0];
  }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  if (((uintptr_t)lhs & 0xf) == 0 && ((uintptr_t)rhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_load_si128((__m128i *)rhs);
      _mm_store_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
    }
  } else {
    for (; lhs != last_aligned; lhs += 4, rhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      __m128i xmm1 = _mm_lddqu_si128((__m128i *)rhs);
      _mm_storeu_si128((__m128i *)lhs, _mm_xor_si128(xmm1, xmm0));
    }
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] ^= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] ^= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] ^= rhs[0];
  }
}

static inline void bitset_not(uint32_t *lhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  static const __m128i mask = _mm_set1_epi32(0xffffffffu);

  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      _mm_store_si128((__m128i *)lhs,
                      _mm_andnot_si128(_mm_load_si128((__m128i *)lhs), mask));
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      _mm_storeu_si128((__m128i *)lhs,
                       _mm_andnot_si128(_mm_lddqu_si128((__m128i *)lhs), mask));
    }
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] = ~lhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] = ~lhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] = ~lhs[0];
  }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  static const __m128i mask = _mm_set1_epi32(0xffffffffu);

#ifndef __SSE4_1__
  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), mask);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return false;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), mask);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return false;
      }
    }
  }
#else
  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i neq = _mm_xor_si128(_mm_load_si128((__m128i *)lhs), mask);
      if (!_mm_testz_si128(neq, neq)) {
        return false;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i neq = _mm_xor_si128(_mm_lddqu_si128((__m128i *)lhs), mask);
      if (!_mm_testz_si128(neq, neq)) {
        return false;
      }
    }
  }
#endif  // !__SSE4_1__

  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0xffffffffu) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0xffffffffu) {
        return false;
      }
  }
  return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

#ifndef __SSE4_1__
  static const __m128i zero = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), zero);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return true;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), zero);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return true;
      }
    }
  }
#else
  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return true;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return true;
      }
    }
  }
#endif  // !__SSE4_1__

  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return true;
      }
  }
  return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

#ifndef __SSE4_1__
  static __m128i zero = _mm_setzero_si128();

  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_load_si128((__m128i *)lhs), zero);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return false;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i eq = _mm_cmpeq_epi32(_mm_lddqu_si128((__m128i *)lhs), zero);
      if (_mm_movemask_epi8(eq) != 0xffffu) {
        return false;
      }
    }
  }
#else
  if (((uintptr_t)lhs & 0xf) == 0) {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i xmm0 = _mm_load_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return false;
      }
    }
  } else {
    for (; lhs != last_aligned; lhs += 4) {
      __m128i xmm0 = _mm_lddqu_si128((__m128i *)lhs);
      if (!_mm_testz_si128(xmm0, xmm0)) {
        return false;
      }
    }
  }
#endif  // !__SSE4_1__

  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return false;
      }
  }
  return true;
}

#else
#if defined(AILEGO_M64)
static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    *(uint64_t *)(&lhs[6]) &= *(uint64_t *)(&rhs[6]);
    *(uint64_t *)(&lhs[4]) &= *(uint64_t *)(&rhs[4]);
    *(uint64_t *)(&lhs[2]) &= *(uint64_t *)(&rhs[2]);
    *(uint64_t *)(&lhs[0]) &= *(uint64_t *)(&rhs[0]);
  }
  switch (last - last_aligned) {
    case 7:
      lhs[6] &= rhs[6];
      /* FALLTHRU */
    case 6:
      lhs[5] &= rhs[5];
      /* FALLTHRU */
    case 5:
      lhs[4] &= rhs[4];
      /* FALLTHRU */
    case 4:
      lhs[3] &= rhs[3];
      /* FALLTHRU */
    case 3:
      lhs[2] &= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= rhs[0];
  }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    *(uint64_t *)(&lhs[6]) &= ~(*(uint64_t *)(&rhs[6]));
    *(uint64_t *)(&lhs[4]) &= ~(*(uint64_t *)(&rhs[4]));
    *(uint64_t *)(&lhs[2]) &= ~(*(uint64_t *)(&rhs[2]));
    *(uint64_t *)(&lhs[0]) &= ~(*(uint64_t *)(&rhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      lhs[6] &= ~rhs[6];
      /* FALLTHRU */
    case 6:
      lhs[5] &= ~rhs[5];
      /* FALLTHRU */
    case 5:
      lhs[4] &= ~rhs[4];
      /* FALLTHRU */
    case 4:
      lhs[3] &= ~rhs[3];
      /* FALLTHRU */
    case 3:
      lhs[2] &= ~rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= ~rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= ~rhs[0];
  }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    *(uint64_t *)(&lhs[6]) |= *(uint64_t *)(&rhs[6]);
    *(uint64_t *)(&lhs[4]) |= *(uint64_t *)(&rhs[4]);
    *(uint64_t *)(&lhs[2]) |= *(uint64_t *)(&rhs[2]);
    *(uint64_t *)(&lhs[0]) |= *(uint64_t *)(&rhs[0]);
  }
  switch (last - last_aligned) {
    case 7:
      lhs[6] |= rhs[6];
      /* FALLTHRU */
    case 6:
      lhs[5] |= rhs[5];
      /* FALLTHRU */
    case 5:
      lhs[4] |= rhs[4];
      /* FALLTHRU */
    case 4:
      lhs[3] |= rhs[3];
      /* FALLTHRU */
    case 3:
      lhs[2] |= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] |= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] |= rhs[0];
  }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    *(uint64_t *)(&lhs[6]) ^= *(uint64_t *)(&rhs[6]);
    *(uint64_t *)(&lhs[4]) ^= *(uint64_t *)(&rhs[4]);
    *(uint64_t *)(&lhs[2]) ^= *(uint64_t *)(&rhs[2]);
    *(uint64_t *)(&lhs[0]) ^= *(uint64_t *)(&rhs[0]);
  }
  switch (last - last_aligned) {
    case 7:
      lhs[6] ^= rhs[6];
      /* FALLTHRU */
    case 6:
      lhs[5] ^= rhs[5];
      /* FALLTHRU */
    case 5:
      lhs[4] ^= rhs[4];
      /* FALLTHRU */
    case 4:
      lhs[3] ^= rhs[3];
      /* FALLTHRU */
    case 3:
      lhs[2] ^= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] ^= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] ^= rhs[0];
  }
}

static inline void bitset_not(uint32_t *lhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8) {
    *(uint64_t *)(&lhs[6]) = ~(*(uint64_t *)(&lhs[6]));
    *(uint64_t *)(&lhs[4]) = ~(*(uint64_t *)(&lhs[4]));
    *(uint64_t *)(&lhs[2]) = ~(*(uint64_t *)(&lhs[2]));
    *(uint64_t *)(&lhs[0]) = ~(*(uint64_t *)(&lhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      lhs[6] = ~lhs[6];
      /* FALLTHRU */
    case 6:
      lhs[5] = ~lhs[5];
      /* FALLTHRU */
    case 5:
      lhs[4] = ~lhs[4];
      /* FALLTHRU */
    case 4:
      lhs[3] = ~lhs[3];
      /* FALLTHRU */
    case 3:
      lhs[2] = ~lhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] = ~lhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] = ~lhs[0];
  }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8) {
    if (*(uint64_t *)(&lhs[6]) != (uint64_t)-1) {
      return false;
    }
    if (*(uint64_t *)(&lhs[4]) != (uint64_t)-1) {
      return false;
    }
    if (*(uint64_t *)(&lhs[2]) != (uint64_t)-1) {
      return false;
    }
    if (*(uint64_t *)(&lhs[0]) != (uint64_t)-1) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 7:
      if (lhs[6] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 6:
      if (lhs[5] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 5:
      if (lhs[4] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 4:
      if (lhs[3] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 3:
      if (lhs[2] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != (uint32_t)-1) {
        return false;
      }
  }
  return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8) {
    if (*(uint64_t *)(&lhs[6]) != 0u) {
      return true;
    }
    if (*(uint64_t *)(&lhs[4]) != 0u) {
      return true;
    }
    if (*(uint64_t *)(&lhs[2]) != 0u) {
      return true;
    }
    if (*(uint64_t *)(&lhs[0]) != 0u) {
      return true;
    }
  }
  switch (last - last_aligned) {
    case 7:
      if (lhs[6] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 6:
      if (lhs[5] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 5:
      if (lhs[4] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 4:
      if (lhs[3] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 3:
      if (lhs[2] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return true;
      }
  }
  return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);

  for (; lhs != last_aligned; lhs += 8) {
    if (*(uint64_t *)(&lhs[6]) != 0u) {
      return false;
    }
    if (*(uint64_t *)(&lhs[4]) != 0u) {
      return false;
    }
    if (*(uint64_t *)(&lhs[2]) != 0u) {
      return false;
    }
    if (*(uint64_t *)(&lhs[0]) != 0u) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 7:
      if (lhs[6] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 6:
      if (lhs[5] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 5:
      if (lhs[4] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 4:
      if (lhs[3] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 3:
      if (lhs[2] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return false;
      }
  }
  return true;
}

#else   // AILEGO_M64
static inline void bitset_and(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    lhs[3] &= rhs[3];
    lhs[2] &= rhs[2];
    lhs[1] &= rhs[1];
    lhs[0] &= rhs[0];
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= rhs[0];
  }
}

static inline void bitset_andnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    lhs[3] &= ~rhs[3];
    lhs[2] &= ~rhs[2];
    lhs[1] &= ~rhs[1];
    lhs[0] &= ~rhs[0];
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] &= ~rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] &= ~rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] &= ~rhs[0];
  }
}

static inline void bitset_or(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    lhs[3] |= rhs[3];
    lhs[2] |= rhs[2];
    lhs[1] |= rhs[1];
    lhs[0] |= rhs[0];
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] |= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] |= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] |= rhs[0];
  }
}

static inline void bitset_xor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    lhs[3] ^= rhs[3];
    lhs[2] ^= rhs[2];
    lhs[1] ^= rhs[1];
    lhs[0] ^= rhs[0];
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] ^= rhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] ^= rhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] ^= rhs[0];
  }
}

static inline void bitset_not(uint32_t *lhs, size_t size) {
  uint32_t *last = lhs + size;
  uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    lhs[3] = ~lhs[3];
    lhs[2] = ~lhs[2];
    lhs[1] = ~lhs[1];
    lhs[0] = ~lhs[0];
  }
  switch (last - last_aligned) {
    case 3:
      lhs[2] = ~lhs[2];
      /* FALLTHRU */
    case 2:
      lhs[1] = ~lhs[1];
      /* FALLTHRU */
    case 1:
      lhs[0] = ~lhs[0];
  }
}

static inline bool bitset_test_all(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    if (lhs[3] != (uint32_t)-1) {
      return false;
    }
    if (lhs[2] != (uint32_t)-1) {
      return false;
    }
    if (lhs[1] != (uint32_t)-1) {
      return false;
    }
    if (lhs[0] != (uint32_t)-1) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != (uint32_t)-1) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != (uint32_t)-1) {
        return false;
      }
  }
  return true;
}

static inline bool bitset_test_any(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    if (lhs[3] != 0u) {
      return true;
    }
    if (lhs[2] != 0u) {
      return true;
    }
    if (lhs[1] != 0u) {
      return true;
    }
    if (lhs[0] != 0u) {
      return true;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return true;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return true;
      }
  }
  return false;
}

static inline bool bitset_test_none(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);

  for (; lhs != last_aligned; lhs += 4) {
    if (lhs[3] != 0u) {
      return false;
    }
    if (lhs[2] != 0u) {
      return false;
    }
    if (lhs[1] != 0u) {
      return false;
    }
    if (lhs[0] != 0u) {
      return false;
    }
  }
  switch (last - last_aligned) {
    case 3:
      if (lhs[2] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 2:
      if (lhs[1] != 0u) {
        return false;
      }
      /* FALLTHRU */
    case 1:
      if (lhs[0] != 0u) {
        return false;
      }
  }
  return true;
}
#endif  // AILEGO_M64
#endif  // __AVX2__

#if (defined(__ARM_NEON) && defined(__aarch64__))
static inline size_t bitset_cardinality(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  while (lhs != last_aligned) {
    const uint32_t *last_stage =
        (last_aligned <= lhs + 124u) ? last_aligned : lhs + 124u;

    uint8x16_t v_count = vdupq_n_u8(0);
    for (; lhs != last_stage; lhs += 4) {
      v_count = vaddq_u8(vcntq_u8(vld1q_u8((const uint8_t *)lhs)), v_count);
    }

    v_count = vreinterpretq_u8_u16(vpaddlq_u8(v_count));
    count += vaddvq_u16(vreinterpretq_u16_u8(v_count));
  }

  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0]);
  }
  return count;
}

static inline size_t bitset_xor_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  while (lhs != last_aligned) {
    const uint32_t *last_stage =
        (last_aligned <= lhs + 124u) ? last_aligned : lhs + 124u;

    uint8x16_t v_count = vdupq_n_u8(0);
    for (; lhs != last_stage; lhs += 4, rhs += 4) {
      v_count = vaddq_u8(vcntq_u8(veorq_u8(vld1q_u8((const uint8_t *)lhs),
                                           vld1q_u8((const uint8_t *)rhs))),
                         v_count);
    }

    v_count = vreinterpretq_u8_u16(vpaddlq_u8(v_count));
    count += vaddvq_u16(vreinterpretq_u16_u8(v_count));
  }

  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}

static inline size_t bitset_and_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  while (lhs != last_aligned) {
    const uint32_t *last_stage =
        (last_aligned <= lhs + 124u) ? last_aligned : lhs + 124u;

    uint8x16_t v_count = vdupq_n_u8(0);
    for (; lhs != last_stage; lhs += 4, rhs += 4) {
      v_count = vaddq_u8(vcntq_u8(vandq_u8(vld1q_u8((const uint8_t *)lhs),
                                           vld1q_u8((const uint8_t *)rhs))),
                         v_count);
    }

    v_count = vreinterpretq_u8_u16(vpaddlq_u8(v_count));
    count += vaddvq_u16(vreinterpretq_u16_u8(v_count));
  }

  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] & rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & rhs[0]);
  }
  return count;
}

static inline size_t bitset_andnot_cardinality(const uint32_t *lhs,
                                               const uint32_t *rhs,
                                               size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  while (lhs != last_aligned) {
    const uint32_t *last_stage =
        (last_aligned <= lhs + 124u) ? last_aligned : lhs + 124u;

    uint8x16_t v_count = vdupq_n_u8(0);
    for (; lhs != last_stage; lhs += 4, rhs += 4) {
      v_count = vaddq_u8(vcntq_u8(vbicq_u8(vld1q_u8((const uint8_t *)lhs),
                                           vld1q_u8((const uint8_t *)rhs))),
                         v_count);
    }

    v_count = vreinterpretq_u8_u16(vpaddlq_u8(v_count));
    count += vaddvq_u16(vreinterpretq_u16_u8(v_count));
  }

  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] & ~rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & ~rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & ~rhs[0]);
  }
  return count;
}

static inline size_t bitset_or_cardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  while (lhs != last_aligned) {
    const uint32_t *last_stage =
        (last_aligned <= lhs + 124u) ? last_aligned : lhs + 124u;

    uint8x16_t v_count = vdupq_n_u8(0);
    for (; lhs != last_stage; lhs += 4, rhs += 4) {
      v_count = vaddq_u8(vcntq_u8(vorrq_u8(vld1q_u8((const uint8_t *)lhs),
                                           vld1q_u8((const uint8_t *)rhs))),
                         v_count);
    }

    v_count = vreinterpretq_u8_u16(vpaddlq_u8(v_count));
    count += vaddvq_u16(vreinterpretq_u16_u8(v_count));
  }

  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] | rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] | rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] | rhs[0]);
  }
  return count;
}

#elif defined(AILEGO_M64)
static inline size_t bitset_cardinality(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8) {
    count += bitset_popcount64(*(uint64_t *)(&lhs[6]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[4]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[2]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      count += bitset_popcount32(lhs[6]);
      /* FALLTHRU */
    case 6:
      count += bitset_popcount32(lhs[5]);
      /* FALLTHRU */
    case 5:
      count += bitset_popcount32(lhs[4]);
      /* FALLTHRU */
    case 4:
      count += bitset_popcount32(lhs[3]);
      /* FALLTHRU */
    case 3:
      count += bitset_popcount32(lhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0]);
  }
  return count;
}

static inline size_t bitset_xor_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    count += bitset_popcount64(*(uint64_t *)(&lhs[6]) ^ *(uint64_t *)(&rhs[6]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[4]) ^ *(uint64_t *)(&rhs[4]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[2]) ^ *(uint64_t *)(&rhs[2]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[0]) ^ *(uint64_t *)(&rhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      count += bitset_popcount32(lhs[6] ^ rhs[6]);
      /* FALLTHRU */
    case 6:
      count += bitset_popcount32(lhs[5] ^ rhs[5]);
      /* FALLTHRU */
    case 5:
      count += bitset_popcount32(lhs[4] ^ rhs[4]);
      /* FALLTHRU */
    case 4:
      count += bitset_popcount32(lhs[3] ^ rhs[3]);
      /* FALLTHRU */
    case 3:
      count += bitset_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}

static inline size_t bitset_and_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    count += bitset_popcount64(*(uint64_t *)(&lhs[6]) & *(uint64_t *)(&rhs[6]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[4]) & *(uint64_t *)(&rhs[4]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[2]) & *(uint64_t *)(&rhs[2]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[0]) & *(uint64_t *)(&rhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      count += bitset_popcount32(lhs[6] & rhs[6]);
      /* FALLTHRU */
    case 6:
      count += bitset_popcount32(lhs[5] & rhs[5]);
      /* FALLTHRU */
    case 5:
      count += bitset_popcount32(lhs[4] & rhs[4]);
      /* FALLTHRU */
    case 4:
      count += bitset_popcount32(lhs[3] & rhs[3]);
      /* FALLTHRU */
    case 3:
      count += bitset_popcount32(lhs[2] & rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & rhs[0]);
  }
  return count;
}

static inline size_t bitset_andnot_cardinality(const uint32_t *lhs,
                                               const uint32_t *rhs,
                                               size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    count +=
        bitset_popcount64(*(uint64_t *)(&lhs[6]) & ~(*(uint64_t *)(&rhs[6])));
    count +=
        bitset_popcount64(*(uint64_t *)(&lhs[4]) & ~(*(uint64_t *)(&rhs[4])));
    count +=
        bitset_popcount64(*(uint64_t *)(&lhs[2]) & ~(*(uint64_t *)(&rhs[2])));
    count +=
        bitset_popcount64(*(uint64_t *)(&lhs[0]) & ~(*(uint64_t *)(&rhs[0])));
  }
  switch (last - last_aligned) {
    case 7:
      count += bitset_popcount32(lhs[6] & ~rhs[6]);
      /* FALLTHRU */
    case 6:
      count += bitset_popcount32(lhs[5] & ~rhs[5]);
      /* FALLTHRU */
    case 5:
      count += bitset_popcount32(lhs[4] & ~rhs[4]);
      /* FALLTHRU */
    case 4:
      count += bitset_popcount32(lhs[3] & ~rhs[3]);
      /* FALLTHRU */
    case 3:
      count += bitset_popcount32(lhs[2] & ~rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & ~rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & ~rhs[0]);
  }
  return count;
}

static inline size_t bitset_or_cardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 3) << 3);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 8, rhs += 8) {
    count += bitset_popcount64(*(uint64_t *)(&lhs[6]) | *(uint64_t *)(&rhs[6]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[4]) | *(uint64_t *)(&rhs[4]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[2]) | *(uint64_t *)(&rhs[2]));
    count += bitset_popcount64(*(uint64_t *)(&lhs[0]) | *(uint64_t *)(&rhs[0]));
  }
  switch (last - last_aligned) {
    case 7:
      count += bitset_popcount32(lhs[6] | rhs[6]);
      /* FALLTHRU */
    case 6:
      count += bitset_popcount32(lhs[5] | rhs[5]);
      /* FALLTHRU */
    case 5:
      count += bitset_popcount32(lhs[4] | rhs[4]);
      /* FALLTHRU */
    case 4:
      count += bitset_popcount32(lhs[3] | rhs[3]);
      /* FALLTHRU */
    case 3:
      count += bitset_popcount32(lhs[2] | rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] | rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] | rhs[0]);
  }
  return count;
}

#else   // !__ARM_NEON && !AILEGO_M64
static inline size_t bitset_cardinality(const uint32_t *lhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4) {
    count += bitset_popcount32(lhs[3]);
    count += bitset_popcount32(lhs[2]);
    count += bitset_popcount32(lhs[1]);
    count += bitset_popcount32(lhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0]);
  }
  return count;
}

static inline size_t bitset_xor_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += bitset_popcount32(lhs[3] ^ rhs[3]);
    count += bitset_popcount32(lhs[2] ^ rhs[2]);
    count += bitset_popcount32(lhs[1] ^ rhs[1]);
    count += bitset_popcount32(lhs[0] ^ rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] ^ rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] ^ rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] ^ rhs[0]);
  }
  return count;
}

static inline size_t bitset_and_cardinality(const uint32_t *lhs,
                                            const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += bitset_popcount32(lhs[3] & rhs[3]);
    count += bitset_popcount32(lhs[2] & rhs[2]);
    count += bitset_popcount32(lhs[1] & rhs[1]);
    count += bitset_popcount32(lhs[0] & rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] & rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & rhs[0]);
  }
  return count;
}

static inline size_t bitset_andnot_cardinality(const uint32_t *lhs,
                                               const uint32_t *rhs,
                                               size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += bitset_popcount32(lhs[3] & ~rhs[3]);
    count += bitset_popcount32(lhs[2] & ~rhs[2]);
    count += bitset_popcount32(lhs[1] & ~rhs[1]);
    count += bitset_popcount32(lhs[0] & ~rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] & ~rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] & ~rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] & ~rhs[0]);
  }
  return count;
}

static inline size_t bitset_or_cardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size) {
  const uint32_t *last = lhs + size;
  const uint32_t *last_aligned = lhs + ((size >> 2) << 2);
  size_t count = 0;

  for (; lhs != last_aligned; lhs += 4, rhs += 4) {
    count += bitset_popcount32(lhs[3] | rhs[3]);
    count += bitset_popcount32(lhs[2] | rhs[2]);
    count += bitset_popcount32(lhs[1] | rhs[1]);
    count += bitset_popcount32(lhs[0] | rhs[0]);
  }
  switch (last - last_aligned) {
    case 3:
      count += bitset_popcount32(lhs[2] | rhs[2]);
      /* FALLTHRU */
    case 2:
      count += bitset_popcount32(lhs[1] | rhs[1]);
      /* FALLTHRU */
    case 1:
      count += bitset_popcount32(lhs[0] | rhs[0]);
  }
  return count;
}
#endif  // __ARM_NEON && __aarch64__

namespace zvec {

namespace ailego {

void BitsetHelper::BitwiseAnd(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  bitset_and(lhs, rhs, size);
}

void BitsetHelper::BitwiseAndnot(uint32_t *lhs, const uint32_t *rhs,
                                 size_t size) {
  bitset_andnot(lhs, rhs, size);
}

void BitsetHelper::BitwiseOr(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  bitset_or(lhs, rhs, size);
}

void BitsetHelper::BitwiseXor(uint32_t *lhs, const uint32_t *rhs, size_t size) {
  bitset_xor(lhs, rhs, size);
}

void BitsetHelper::BitwiseNot(uint32_t *arr, size_t size) {
  bitset_not(arr, size);
}

bool BitsetHelper::TestAll(const uint32_t *arr, size_t size) {
  return bitset_test_all(arr, size);
}

bool BitsetHelper::TestAny(const uint32_t *arr, size_t size) {
  return bitset_test_any(arr, size);
}

bool BitsetHelper::TestNone(const uint32_t *arr, size_t size) {
  return bitset_test_none(arr, size);
}

size_t BitsetHelper::BitwiseAndCardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size) {
  return bitset_and_cardinality(lhs, rhs, size);
}

size_t BitsetHelper::BitwiseOrCardinality(const uint32_t *lhs,
                                          const uint32_t *rhs, size_t size) {
  return bitset_or_cardinality(lhs, rhs, size);
}

size_t BitsetHelper::BitwiseAndnotCardinality(const uint32_t *lhs,
                                              const uint32_t *rhs,
                                              size_t size) {
  return bitset_andnot_cardinality(lhs, rhs, size);
}

size_t BitsetHelper::BitwiseXorCardinality(const uint32_t *lhs,
                                           const uint32_t *rhs, size_t size) {
  return bitset_xor_cardinality(lhs, rhs, size);
}

size_t BitsetHelper::Cardinality(const uint32_t *arr, size_t size) {
  return bitset_cardinality(arr, size);
}

bool BitsetHelper::test_all(void) const {
  return bitset_test_all(array_, size_);
}

bool BitsetHelper::test_any(void) const {
  return bitset_test_any(array_, size_);
}

bool BitsetHelper::test_none(void) const {
  return bitset_test_none(array_, size_);
}

size_t BitsetHelper::cardinality(void) const {
  return bitset_cardinality(array_, size_);
}

}  // namespace ailego

}  // namespace zvec