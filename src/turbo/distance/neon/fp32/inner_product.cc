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

#include "neon/fp32/inner_product.h"
#include <algorithm>
#include <cstdint>
#include <zvec/ailego/internal/platform.h>

#if defined(AILEGO_ARM64_NEON)
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#else
#include "scalar/fp32/inner_product.h"
#endif

namespace zvec::turbo::neon {

#if defined(AILEGO_ARM64_NEON)
namespace {

#if defined(__clang__)
#define ZVEC_NEON_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define ZVEC_NEON_UNROLL _Pragma("GCC unroll 8")
#else
#define ZVEC_NEON_UNROLL
#endif

constexpr size_t kPrefetchCacheLine = AILEGO_CACHE_LINE_SIZE;

[[maybe_unused]] inline void prefetch_l2(uintptr_t address) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(reinterpret_cast<const void *>(address), 0, 2);
#else
  (void)address;
#endif
}

#if defined(__APPLE__)
inline void prefetch_vector(const void *vector, size_t bytes) {
  uintptr_t next = reinterpret_cast<uintptr_t>(vector);
  const uintptr_t end = reinterpret_cast<uintptr_t>(vector) + bytes;
  if (next < end) {
    prefetch_l2(next);
    next = (next & ~(kPrefetchCacheLine - 1)) + kPrefetchCacheLine;
    for (; next < end; next += kPrefetchCacheLine) {
      prefetch_l2(next);
    }
  }
}

bool supports_batch_prefetch(size_t dim) noexcept {
  // The lead distance was tuned only for 256D vectors on 128-byte Apple
  // hardware. A build-wide cache-line override must not broaden that policy.
  if (dim != 256 || kPrefetchCacheLine != 128) {
    return false;
  }
  // Check the specialization's hardware assumption once, rather than using
  // runtime detection as a separate source of the configured cache-line size.
  static const bool supported = []() noexcept {
    uint64_t cache_line = 0;
    size_t size = sizeof(cache_line);
    return sysctlbyname("hw.cachelinesize", &cache_line, &size, nullptr, 0) ==
               0 &&
           cache_line == kPrefetchCacheLine;
  }();
  return supported;
}

// Sequential scans already benefit from hardware prefetch. Recognize tightly
// packed rows and rows with a small metadata/alignment tail, in either order.
// Use integer addresses: the pointer batch may refer to unrelated allocations.
bool is_sequential_batch(const void *const *vectors, size_t n,
                         size_t dim) noexcept {
  const uintptr_t first = reinterpret_cast<uintptr_t>(vectors[0]);
  const uintptr_t second = reinterpret_cast<uintptr_t>(vectors[1]);
  const uintptr_t spacing = first < second ? second - first : first - second;
  const size_t bytes = dim * sizeof(float);
  if (spacing < bytes || spacing > bytes + kPrefetchCacheLine) {
    return false;
  }
  const uintptr_t step = second - first;
  uintptr_t previous = second;
  for (size_t i = 2; i < n; ++i) {
    const uintptr_t current = reinterpret_cast<uintptr_t>(vectors[i]);
    if (current - previous != step) {
      return false;
    }
    previous = current;
  }
  return true;
}
#endif

// Reuse each query load across several vectors. Independent FMA chains hide
// instruction latency; limiting the largest block to 16 accumulators keeps it
// within AArch64's SIMD register file without spilling to the stack.
template <size_t Batch, size_t Accumulators, bool StreamPrefetch = false>
inline void inner_product_block(const void *const *vectors, const float *query,
                                size_t dim, float *distances,
                                const void *const *future = nullptr,
                                size_t future_count = 0) {
  static_assert(Accumulators == 2 || Accumulators == 4);
  const float *data[Batch];
  float32x4_t sums[Batch][Accumulators];
  [[maybe_unused]] uintptr_t next[Batch], ends[Batch], bases[Batch];
  ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    data[b] = static_cast<const float *>(vectors[b]);
    ZVEC_NEON_UNROLL
    for (size_t u = 0; u < Accumulators; ++u) {
      sums[b][u] = vdupq_n_f32(0.0f);
    }
    if constexpr (StreamPrefetch) {
      bases[b] = b < future_count ? reinterpret_cast<uintptr_t>(future[b]) : 0;
      ends[b] = bases[b] + (b < future_count ? dim * sizeof(float) : 0);
      next[b] = bases[b];
    }
  }

  size_t offset = 0;
  for (; dim - offset >= Accumulators * 4; offset += Accumulators * 4) {
    if constexpr (StreamPrefetch) {
      if ((offset & (kPrefetchCacheLine / sizeof(float) - 1)) == 0) {
        ZVEC_NEON_UNROLL
        for (size_t b = 0; b < Batch; ++b) {
          const uintptr_t stop = std::min(
              ends[b], bases[b] + offset * sizeof(float) + kPrefetchCacheLine);
          while (next[b] < stop) {
            prefetch_l2(next[b]);
            next[b] =
                (next[b] & ~(kPrefetchCacheLine - 1)) + kPrefetchCacheLine;
          }
        }
      }
    }
    float32x4_t q[Accumulators];
    ZVEC_NEON_UNROLL
    for (size_t u = 0; u < Accumulators; ++u) {
      q[u] = vld1q_f32(query + offset + u * 4);
    }
    ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
      ZVEC_NEON_UNROLL
      for (size_t u = 0; u < Accumulators; ++u) {
        sums[b][u] =
            vfmaq_f32(sums[b][u], q[u], vld1q_f32(data[b] + offset + u * 4));
      }
    }
  }
  for (; dim - offset >= 4; offset += 4) {
    const float32x4_t q = vld1q_f32(query + offset);
    ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
      sums[b][0] = vfmaq_f32(sums[b][0], q, vld1q_f32(data[b] + offset));
    }
  }

  float results[Batch];
  ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    float32x4_t sum = vaddq_f32(sums[b][0], sums[b][1]);
    if constexpr (Accumulators == 4) {
      sum = vaddq_f32(sum, vaddq_f32(sums[b][2], sums[b][3]));
    }
    results[b] = vaddvq_f32(sum);
  }
  // Keep the scalar remainder explicitly bounded to three elements. Otherwise
  // compilers may emit a second general vector loop after the SIMD loops.
  const size_t tail = dim & 3;
  ZVEC_NEON_UNROLL
  for (size_t t = 0; t < tail; ++t) {
    const float q = query[offset + t];
    ZVEC_NEON_UNROLL
    for (size_t b = 0; b < Batch; ++b) {
      results[b] += q * data[b][offset + t];
    }
  }
  ZVEC_NEON_UNROLL
  for (size_t b = 0; b < Batch; ++b) {
    distances[b] = -results[b];
  }
}

// Keep the large batch register frame out of the single-vector fast path.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void inner_product_batch(const void *const *vectors, const void *query, size_t n,
                         size_t dim, float *distances) {
  const float *typed_query = static_cast<const float *>(query);
  size_t i = 0;
  for (; n - i >= 8; i += 8) {
    inner_product_block<8, 2>(vectors + i, typed_query, dim, distances + i);
  }
  if (n - i >= 4) {
    inner_product_block<4, 2>(vectors + i, typed_query, dim, distances + i);
    i += 4;
  }
  if (n - i >= 2) {
    inner_product_block<2, 4>(vectors + i, typed_query, dim, distances + i);
    i += 2;
  }
  if (i < n) {
    inner_product_fp32_distance(vectors[i], query, dim, distances + i);
  }
}

#if defined(__APPLE__)
// Warm up three groups of eight vectors, then prefetch the next group while
// computing. The kernel manages its own requests; callers may also prefetch.
__attribute__((noinline)) void inner_product_batch_prefetched(
    const void *const *vectors, const void *query, size_t n, size_t dim,
    float *distances) {
  constexpr size_t kAhead = 3 * 8;
  const float *typed_query = static_cast<const float *>(query);
  const size_t bytes = dim * sizeof(float);
  for (size_t j = 0; j < std::min(n, kAhead); ++j) {
    prefetch_vector(vectors[j], bytes);
  }
  size_t i = 0;
  for (; n - i >= 8; i += 8) {
    const size_t begin = std::min(n, i + kAhead);
    const size_t end = std::min(n, begin + 8);
    inner_product_block<8, 2, true>(vectors + i, typed_query, dim,
                                    distances + i, vectors + begin,
                                    end - begin);
  }
  if (n - i >= 4) {
    inner_product_block<4, 2>(vectors + i, typed_query, dim, distances + i);
    i += 4;
  }
  if (n - i >= 2) {
    inner_product_block<2, 4>(vectors + i, typed_query, dim, distances + i);
    i += 2;
  }
  if (i < n) {
    inner_product_fp32_distance(vectors[i], query, dim, distances + i);
  }
}
#endif

#undef ZVEC_NEON_UNROLL

}  // namespace
#endif

void inner_product_fp32_distance(const void *a, const void *b, size_t dim,
                                 float *distance) {
#if defined(AILEGO_ARM64_NEON)
  // Small vectors cannot amortize four-accumulator setup and reduction.
  if (dim < 32) {
    inner_product_block<1, 2>(&a, static_cast<const float *>(b), dim, distance);
  } else {
    inner_product_block<1, 4>(&a, static_cast<const float *>(b), dim, distance);
  }
#else
  scalar::inner_product_fp32_distance(a, b, dim, distance);
#endif
}

void inner_product_fp32_batch_distance(const void *const *vectors,
                                       const void *query, size_t n, size_t dim,
                                       float *distances,
                                       const void *const * /*extra_values*/) {
  if (n == 0) {
    return;
  }
#if defined(AILEGO_ARM64_NEON)
  if (dim == 0) {
    for (size_t i = 0; i < n; ++i) {
      distances[i] = -0.0f;
    }
    return;
  }
  if (n == 1) {
    inner_product_fp32_distance(vectors[0], query, dim, distances);
    return;
  }

#if defined(__APPLE__)
  if (supports_batch_prefetch(dim) && !is_sequential_batch(vectors, n, dim)) {
    inner_product_batch_prefetched(vectors, query, n, dim, distances);
    return;
  }
#endif
  inner_product_batch(vectors, query, n, dim, distances);
#else
  for (size_t i = 0; i < n; ++i) {
    inner_product_fp32_distance(vectors[i], query, dim, &distances[i]);
  }
#endif
}

}  // namespace zvec::turbo::neon
