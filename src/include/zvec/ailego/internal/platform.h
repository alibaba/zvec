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

#pragma once

#if defined(_WIN32) || defined(_WIN64)
#include <sdkddkver.h>
#endif

#include <sys/types.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <strings.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386)
#include <x86intrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef NDEBUG
#define AILEGO_DEBUG
#endif

//! Fixed Intel intrinsics macro in MSVC
#if defined(_MSC_VER)
#if (_M_IX86_FP == 2 || defined(_M_AMD64) || defined(_M_X64))
#define __SSE__ 1
#define __SSE2__ 1
#if _MSC_VER >= 1500
#define __SSE3__ 1
#define __SSSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#endif
#elif _M_IX86_FP == 1
#define __SSE__ 1
#endif
#endif  // _MSC_VER

#if defined(_WIN32) || defined(_WIN64)
#if defined(_WIN64)
#define AILEGO_M64
#else
#define AILEGO_M32
#endif
#endif

#if defined(__GNUC__)
#if defined(__x86_64__) || defined(__aarch64__) || defined(__ppc64__)
#define AILEGO_M64
#else
#define AILEGO_M32
#endif
#endif

#ifndef AILEGO_ALIGNED
#if defined(_MSC_VER)
#define AILEGO_ALIGNED(x) __declspec(align(x))
#define AILEGO_DEPRECATED __declspec(deprecated)
#elif defined(__GNUC__)
#define AILEGO_ALIGNED(x) __attribute__((aligned(x)))
#define AILEGO_DEPRECATED __attribute__((deprecated))
#else
#define AILEGO_ALIGNED(x)
#define AILEGO_DEPRECATED
#endif
#endif

//! Add 'inline' for MSVC
#if defined(_MSC_VER) && !defined(__cplusplus)
#if !defined(inline)
#define inline __inline
#endif
#endif

//! Add 'ssize_t' for MSVC
#if defined(_MSC_VER)
typedef intptr_t ssize_t;
#endif

#if defined(_MSC_VER)
//! Returns the number of trailing 0-bits in x
static inline int ailego_ctz32(uint32_t x) {
  unsigned long r = 0;
  _BitScanForward(&r, x);
  return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int ailego_clz32(uint32_t x) {
  unsigned long r = 0;
  _BitScanReverse(&r, x);
  return (31 - (int)r);
}

#if defined(AILEGO_M64)
//! Returns the number of trailing 0-bits in x
static inline int ailego_ctz64(uint64_t x) {
  unsigned long r = 0;
  _BitScanForward64(&r, x);
  return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int ailego_clz64(uint64_t x) {
  unsigned long r = 0;
  _BitScanReverse64(&r, x);
  return (63 - (int)r);
}
#else
//! Returns the number of trailing 0-bits in x
static inline int ailego_ctz64(uint64_t x) {
  unsigned long r = 0;
  unsigned long m = (unsigned long)x;
  _BitScanForward(&r, m);
  if (r == 0) {
    m = (unsigned long)(x >> 32);
    _BitScanForward(&r, m);
    if (r != 0) {
      r += 32;
    }
  }
  return (int)r;
}

//! Returns the number of leading 0-bits in x
static inline int ailego_clz64(uint64_t x) {
  unsigned long r = 0;
  unsigned long m = (unsigned long)(x >> 32);
  _BitScanReverse(&r, m);
  if (r != 0) {
    return (31 - (int)r);
  }
  m = (unsigned long)x;
  _BitScanReverse(&r, m);
  return (63 - (int)r);
}
#endif  // AILEGO_M64

//! Counts the number of one bits
#define ailego_popcount32(x) (__popcnt(x))
#define ailego_popcount64(x) (__popcnt64(x))
#define ailego_likely(x) (x)
#define ailego_unlikely(x) (x)
#ifdef __SSE__
#define ailego_prefetch(p) _mm_prefetch((p), 0)
#else
#define ailego_prefetch(p) ((void)(p))
#endif
#else  // !_MSC_VER
#define ailego_ctz32(x) (__builtin_ctz(x))
#define ailego_ctz64(x) (__builtin_ctzll(x))
#define ailego_clz32(x) (__builtin_clz(x))
#define ailego_clz64(x) (__builtin_clzll(x))
#define ailego_popcount32(x) (__builtin_popcount(x))
#define ailego_popcount64(x) (__builtin_popcountl(x))
#define ailego_likely(x) (__builtin_expect(!!(x), 1))
#define ailego_unlikely(x) (__builtin_expect(!!(x), 0))
#define ailego_prefetch(p) (__builtin_prefetch((p)))
#endif  // _MSC_VER

#if defined(AILEGO_M64)
#define ailego_ctz ailego_ctz64
#define ailego_clz ailego_clz64
#define ailego_popcount ailego_popcount64
#else
#define ailego_ctz ailego_ctz32
#define ailego_clz ailego_clz32
#define ailego_popcount ailego_popcount32
#endif  // AILEGO_M64

#if defined(__arm__) || defined(__aarch64__)
// ARMv7 Architecture Reference Manual (for YIELD)
// ARM Compiler toolchain Compiler Reference (for __yield() instrinsic)
#if defined(__CC_ARM)
#define ailego_yield() __yield()
#else
#define ailego_yield() __asm__ __volatile__("yield")
#endif  // __CC_ARM
#elif defined(__SSE2__)
#define ailego_yield() _mm_pause()
#else
#define ailego_yield() ((void)0)
#endif  // __arm__ || __aarch64__

#if defined(_MSC_VER)
#define ailego_aligned_malloc(SIZE, ALIGN) \
  _aligned_malloc((size_t)(SIZE), (ALIGN))
#define ailego_aligned_free _aligned_free
#else  // !_MSC_VER
#if defined(_ISOC11_SOURCE)
#define ailego_aligned_malloc(SIZE, ALIGN) \
  aligned_alloc((ALIGN), (size_t)(SIZE))
#else  // !_ISOC11_SOURCE
#define ailego_aligned_malloc(SIZE, ALIGN) \
  ailego_posix_malloc((size_t)(SIZE), (ALIGN))
#endif  // _ISOC11_SOURCE
#define ailego_aligned_free free
#endif  // _MSC_VER

#if !defined(__SANITIZE_ADDRESS__)
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define __SANITIZE_ADDRESS__ 1
#endif  // address_sanitizer
#endif  // __has_feature
#endif  // !__SANITIZE_ADDRESS__

#if !defined(__SANITIZE_ADDRESS__)
#if !defined(ailego_malloc)
#if defined(__AVX512F__)
#define ailego_malloc(SIZE) ailego_aligned_malloc((SIZE), 64)
#elif defined(__AVX__)
#define ailego_malloc(SIZE) ailego_aligned_malloc((SIZE), 32)
#elif defined(__SSE__)
#define ailego_malloc(SIZE) ailego_aligned_malloc((SIZE), 16)
#elif defined(__ARM_NEON)
#define ailego_malloc(SIZE) ailego_aligned_malloc((SIZE), 16)
#endif
#endif  // !ailego_malloc
#if (defined(__SSE__) || defined(__ARM_NEON)) && !defined(ailego_free)
#define ailego_free ailego_aligned_free
#endif
#endif  // !__SANITIZE_ADDRESS__

#ifndef ailego_malloc
#define ailego_malloc(SIZE) malloc((size_t)(SIZE))
#endif
#ifndef ailego_free
#define ailego_free free
#endif

#ifndef ailego_offsetof
#define ailego_offsetof(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#endif

#ifndef ailego_align
#define ailego_align(SIZE, BOUND) (((SIZE) + ((BOUND) - 1)) & ~((BOUND) - 1))
#endif

#ifndef ailego_align8
#define ailego_align8(SIZE) ailego_align(SIZE, 8)
#endif

#ifndef ailego_min
#define ailego_min(A, B) (((A) < (B)) ? (A) : (B))
#endif

#ifndef ailego_max
#define ailego_max(A, B) (((A) > (B)) ? (A) : (B))
#endif

#ifndef ailego_malloc_object
#define ailego_malloc_object(TYPE) ((TYPE *)ailego_malloc(sizeof(TYPE)))
#endif
#ifndef ailego_malloc_array
#define ailego_malloc_array(TYPE, SIZE) \
  ((TYPE *)ailego_malloc(SIZE * sizeof(TYPE)))
#endif

#ifndef ailego_minus_if_ne_zero
#define ailego_minus_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) return (-1)
#endif

#ifndef ailego_zero_if_ne_zero
#define ailego_zero_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) return (0)
#endif

#ifndef ailego_null_if_ne_zero
#define ailego_null_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) return (NULL)
#endif

#ifndef ailego_false_if_ne_zero
#define ailego_false_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) return (false)
#endif

#ifndef ailego_return_if_ne_zero
#define ailego_return_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) return
#endif

#ifndef ailego_break_if_ne_zero
#define ailego_break_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) break
#endif

#ifndef ailego_continue_if_ne_zero
#define ailego_continue_if_ne_zero(COND) \
  if (ailego_unlikely((COND) != 0)) continue
#endif

#ifndef ailego_do_if_ne_zero
#define ailego_do_if_ne_zero(COND) if (ailego_unlikely((COND) != 0))
#endif

#ifndef ailego_minus_if_lt_zero
#define ailego_minus_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) return (-1)
#endif

#ifndef ailego_zero_if_lt_zero
#define ailego_zero_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) return (0)
#endif

#ifndef ailego_null_if_lt_zero
#define ailego_null_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) return (NULL)
#endif

#ifndef ailego_false_if_lt_zero
#define ailego_false_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) return (false)
#endif

#ifndef ailego_return_if_lt_zero
#define ailego_return_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) return
#endif

#ifndef ailego_break_if_lt_zero
#define ailego_break_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) break
#endif

#ifndef ailego_continue_if_lt_zero
#define ailego_continue_if_lt_zero(COND) \
  if (ailego_unlikely((COND) < 0)) continue
#endif

#ifndef ailego_do_if_lt_zero
#define ailego_do_if_lt_zero(COND) if (ailego_unlikely((COND) < 0))
#endif

#ifndef ailego_minus_if_false
#define ailego_minus_if_false(COND) \
  if (ailego_unlikely(!(COND))) return (-1)
#endif

#ifndef ailego_zero_if_false
#define ailego_zero_if_false(COND) \
  if (ailego_unlikely(!(COND))) return (0)
#endif

#ifndef ailego_null_if_false
#define ailego_null_if_false(COND) \
  if (ailego_unlikely(!(COND))) return (NULL)
#endif

#ifndef ailego_false_if_false
#define ailego_false_if_false(COND) \
  if (ailego_unlikely(!(COND))) return (false)
#endif

#ifndef ailego_return_if_false
#define ailego_return_if_false(COND) \
  if (ailego_unlikely(!(COND))) return
#endif

#ifndef ailego_break_if_false
#define ailego_break_if_false(COND) \
  if (ailego_unlikely(!(COND))) break
#endif

#ifndef ailego_continue_if_false
#define ailego_continue_if_false(COND) \
  if (ailego_unlikely(!(COND))) continue
#endif

#ifndef ailego_do_if_false
#define ailego_do_if_false(COND) if (ailego_unlikely(!(COND)))
#endif

#ifndef ailego_compile_assert
#define ailego_compile_assert(COND, MSG) \
  typedef char Static_Assertion_##MSG[(!!(COND)) * 2 - 1]
#endif

#ifndef ailego_static_assert3
#define ailego_static_assert3(COND, LINE) \
  ailego_compile_assert(COND, At_Line_##LINE)
#endif

#ifndef ailego_static_assert2
#define ailego_static_assert2(COND, LINE) ailego_static_assert3(COND, LINE)
#endif

#ifndef ailego_static_assert
#define ailego_static_assert(COND) ailego_static_assert2(COND, __LINE__)
#endif

//! Abort and report if an assertion is failed
#ifndef ailego_assert_abort
#define ailego_assert_abort(COND, MSG)                                         \
  (void)(ailego_likely(COND) || (ailego_assert_report(__FILE__, __FUNCTION__,  \
                                                      __LINE__, #COND, (MSG)), \
                                 abort(), 0))
#endif

#ifdef AILEGO_DEBUG
#ifndef ailego_assert
#define ailego_assert(COND) ailego_assert_abort(COND, "")
#endif
#ifndef ailego_assert_with
#define ailego_assert_with(COND, MSG) ailego_assert_abort(COND, MSG)
#endif
#else  // !AILEGO_DEBUG
#ifndef ailego_assert
#define ailego_assert(COND) ((void)0)
#endif
#ifndef ailego_assert_with
#define ailego_assert_with(COND, MSG) ((void)0)
#endif
#endif  // AILEGO_DEBUG

#ifndef ailego_check
#define ailego_check(COND) ailego_assert_abort(COND, "")
#endif
#ifndef ailego_check_with
#define ailego_check_with(COND, MSG) ailego_assert_abort(COND, MSG)
#endif

#ifndef _MSC_VER
//! Allocates memory on a specified alignment boundary
static inline void *ailego_posix_malloc(size_t size, size_t align) {
  void *ptr;
  ailego_null_if_ne_zero(posix_memalign(&ptr, align, size));
  return ptr;
}
#endif

//! Report an assertion is failed
static inline void ailego_assert_report(const char *file, const char *func,
                                        int line, const char *cond,
                                        const char *msg) {
  fprintf(stderr, "Assertion failed: (%s) in %s(), %s line %d. %s\n", cond,
          func, file, line, msg);
}

// because the behavior of conversion from negative float to unsigned integer
// is undefined (on arm, result will be zero), it's necessary to convert it
// to signed integer firstly
static inline uint8_t static_cast_from_float_to_uint8(float data) {
  return static_cast<uint8_t>(static_cast<int8_t>(data));
}

static inline uint16_t static_cast_from_float_to_uint16(float data) {
  return static_cast<uint16_t>(static_cast<int16_t>(data));
}

#ifdef __cplusplus
} /* extern "C" */
#endif
