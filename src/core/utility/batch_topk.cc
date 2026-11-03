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

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <ailego/internal/cpu_features.h>
#include "batch_topk_internal.h"
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif
#if defined(AILEGO_MACOS_X86_64)
#include <sys/sysctl.h>
#endif

namespace zvec {
namespace core {
namespace batch_topk_internal {
namespace {
struct Capabilities {
  bool sse2{false}, avx2{false}, avx512{false}, neon{false};
};
Capabilities Detect() {
  const auto &f = ailego::internal::CpuFeatures::static_flags_;
  uint64_t xcr0 = 0;
  bool lzcnt = false;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  int regs[4] = {};
  __cpuidex(regs, 0, 0);
  if (regs[0] >= 1) {
    __cpuidex(regs, 1, 0);
    if ((static_cast<uint32_t>(regs[2]) & (3u << 26)) == (3u << 26))
      xcr0 = _xgetbv(0);
  }
  __cpuidex(regs, static_cast<int>(0x80000000u), 0);
  if (static_cast<uint32_t>(regs[0]) >= 0x80000001u) {
    __cpuidex(regs, static_cast<int>(0x80000001u), 0);
    lzcnt = (static_cast<uint32_t>(regs[2]) & (1u << 5)) != 0;
  }
#elif defined(__x86_64__) || defined(__i386__)
  uint32_t eax, ebx, ecx, edx;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) &&
      (ecx & (3u << 26)) == (3u << 26)) {
    // XGETBV is legal only after both XSAVE and OSXSAVE are advertised.
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
  }
#endif
  (void)lzcnt;
  const bool avx_state = (xcr0 & 0x6u) == 0x6u;
  bool avx512_state = (xcr0 & 0xe6u) == 0xe6u;
#if defined(AILEGO_MACOS_X86_64)
  // Darwin enables opmask/ZMM lazily, so an early XCR0 snapshot is incomplete.
  int enabled = 0;
  size_t bytes = sizeof(enabled);
  avx512_state = avx_state &&
                 (avx512_state || (sysctlbyname("hw.optional.avx512f", &enabled,
                                                &bytes, nullptr, 0) == 0 &&
                                   bytes == sizeof(enabled) && enabled != 0));
#endif
  // Include features the compiler may use implicitly for these target flags.
  bool avx2 = avx_state && f.AVX && f.AVX2 && f.SSE2 && f.SSE3 && f.SSSE3 &&
              f.SSE4_1 && f.SSE4_2 && f.POPCNT;
  bool avx512 = avx512_state && avx2 && f.AVX512F && f.FMA && f.F16C;
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  avx2 = avx2 && f.FMA && f.BMI1 && f.BMI2 && lzcnt;
  avx512 =
      avx512 && avx2 && f.AVX512BW && f.AVX512DQ && f.AVX512VL && f.AVX512CD;
#endif
  (void)avx2;
  (void)avx512;
  Capabilities out;
#if defined(ZVEC_BATCH_TOPK_HAVE_SSE2)
  out.sse2 = f.SSE2;
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_AVX2)
  out.avx2 = avx2;
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_AVX512)
  out.avx512 = avx512;
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_NEON)
  out.neon = f.NEON;
#endif
  return out;
}
const Capabilities &RuntimeCapabilities() {
  static const Capabilities caps = Detect();
  return caps;
}
struct Scalar {
  static constexpr unsigned kWidth = 1;
  static float Broadcast(float value) {
    return value;
  }
  static float Load(const float *value) {
    return *value;
  }
  template <bool Strict>
  static uint32_t Compare(float value, float bound) {
    return Strict ? value < bound : value <= bound;
  }
  static uint32_t Equal(float value, float bound) {
    return value == bound;
  }
};
inline void Swap(float *distances, uint64_t *ids, size_t a, size_t b) {
  if (a != b) {
    std::swap(distances[a], distances[b]);
    std::swap(ids[a], ids[b]);
  }
}
}  // namespace

bool Supported(Isa isa) {
  const auto &caps = RuntimeCapabilities();
  switch (isa) {
    case Isa::kSSE2:
      return caps.sse2;
    case Isa::kAVX2:
      return caps.avx2;
    case Isa::kAVX512:
      return caps.avx512;
    case Isa::kNEON:
      return caps.neon;
    default:
      return true;
  }
}
Isa RuntimeIsa() {
  static const Isa isa = [] {
    const auto &c = RuntimeCapabilities();
    if (c.avx512) return Isa::kAVX512;
    if (c.avx2) return Isa::kAVX2;
    if (c.sse2) return Isa::kSSE2;
    if (c.neon) return Isa::kNEON;
    return Isa::kScalar;
  }();
  return isa;
}
const Ops &GetOps(Isa isa, bool small) {
  switch (isa) {
#if defined(ZVEC_BATCH_TOPK_HAVE_SSE2)
    case Isa::kSSE2:
      return Sse2Ops(small);
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_AVX2)
    case Isa::kAVX2:
      return Avx2Ops(small);
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_AVX512)
    case Isa::kAVX512:
      return Avx512Ops(small);
#endif
#if defined(ZVEC_BATCH_TOPK_HAVE_NEON)
    case Isa::kNEON:
      return NeonOps(small);
#endif
    default:
      return ScalarOps(small);
  }
}
const Ops &ScalarOps(bool small) {
  return small ? Engine<Scalar, true>::Operations()
               : Engine<Scalar, false>::Operations();
}
// Keep only matches. Forward writes cannot overwrite an unread selected lane.
// Unlike a full partition, discarded IDs never need to move.
void CompactMasks(float *distances, uint64_t *ids, size_t begin, size_t count,
                  const uint32_t *masks) {
  size_t write = begin;
  for (size_t read = 0; read < count; read += 32) {
    uint32_t mask = masks[read / 32];
    if (mask == UINT32_MAX) {
      if (write != begin + read) {
        std::memmove(distances + write, distances + begin + read,
                     32 * sizeof(float));
        std::memmove(ids + write, ids + begin + read, 32 * sizeof(uint64_t));
      }
      write += 32;
      continue;
    }
    while (mask != 0) {
      const unsigned lane = static_cast<unsigned>(ailego_ctz32(mask));
      distances[write] = distances[begin + read + lane];
      ids[write++] = ids[begin + read + lane];
      mask &= mask - 1;
    }
  }
}

// Swap only a misplaced false on the left with a misplaced true on the
// right. Masks replace unpredictable per-score comparisons and no second
// score scan is necessary. Padding bits in the last mask are never visited.
void PartitionMasks(float *distances, uint64_t *ids, size_t begin, size_t count,
                    size_t matched, const uint32_t *masks) {
  if (matched == 0 || matched == count) return;
  const size_t last_word = (count - 1) / 32;
  const unsigned tail = static_cast<unsigned>(count % 32);
  const uint32_t tail_mask = tail ? (uint32_t{1} << tail) - 1 : UINT32_MAX;
  size_t left_word = 0;
  size_t right_word = last_word;
  uint32_t left_bits = ~masks[left_word];
  if (left_word == last_word) left_bits &= tail_mask;
  uint32_t right_bits = masks[right_word];
  for (;;) {
    while (left_bits == 0) {
      if (left_word == last_word) return;
      left_bits = ~masks[++left_word];
      if (left_word == last_word) left_bits &= tail_mask;
    }
    while (right_bits == 0) {
      if (right_word == 0) return;
      right_bits = masks[--right_word];
    }
    const unsigned left_lane = static_cast<unsigned>(ailego_ctz32(left_bits));
    const unsigned right_lane =
        31u - static_cast<unsigned>(ailego_clz32(right_bits));
    const size_t left = left_word * 32 + left_lane;
    const size_t right = right_word * 32 + right_lane;
    if (left >= right) return;
    Swap(distances, ids, begin + left, begin + right);
    left_bits &= left_bits - 1;
    right_bits &= ~(uint32_t{1} << right_lane);
  }
}

void SiftDown(float *distances, uint64_t *ids, size_t base, size_t root,
              size_t count) {
  while (root < count / 2) {
    size_t child = 2 * root + 1;
    if (child + 1 < count &&
        distances[base + child] < distances[base + child + 1])
      ++child;
    if (!(distances[base + root] < distances[base + child])) return;
    Swap(distances, ids, base + root, base + child);
    root = child;
  }
}

// Introspective escape hatch: pathological pivots cannot make selection
// quadratic or nonterminating. Only the unresolved interval is sorted, and
// score/ID pairs always move together. No temporary AoS or allocation is used.
void HeapSort(float *distances, uint64_t *ids, size_t begin, size_t end) {
  const size_t count = end - begin;
  for (size_t root = count / 2; root != 0; --root)
    SiftDown(distances, ids, begin, root - 1, count);
  for (size_t remaining = count; remaining > 1; --remaining) {
    Swap(distances, ids, begin, begin + remaining - 1);
    SiftDown(distances, ids, begin, 0, remaining - 1);
  }
}


}  // namespace batch_topk_internal

void BatchTopK::reset(size_t k, float threshold, bool strict) {
  reset_impl(k, threshold, strict, batch_topk_internal::RuntimeIsa(),
             k <= kSmallLimit);
}
void BatchTopK::reset_impl(size_t k, float threshold, bool strict,
                           batch_topk_internal::Isa isa, bool small) {
  if (k > (std::numeric_limits<size_t>::max() - 31) / 2)
    throw std::length_error("BatchTopK capacity overflow");
  const size_t capacity = std::max(size_t{32}, (2 * k + 31) & ~size_t{31});
  const size_t item_size = small ? sizeof(Candidate) : sizeof(uint64_t);
  if (capacity > std::numeric_limits<size_t>::max() / item_size)
    throw std::length_error("BatchTopK allocation size overflow");
  if (small) {
    if (capacity > kInlineCapacity && capacity > allocated_aos_capacity_) {
      allocated_entries_.reset(new Candidate[capacity]);
      allocated_aos_capacity_ = capacity;
    }
    entries_ = capacity <= kInlineCapacity ? inline_entries_.data()
                                           : allocated_entries_.get();
  } else {
    if (capacity > kInlineCapacity && capacity > allocated_soa_capacity_) {
      std::unique_ptr<float[]> distances(new float[capacity]);
      std::unique_ptr<uint64_t[]> ids(new uint64_t[capacity]);
      std::unique_ptr<uint32_t[]> masks(new uint32_t[capacity / 32]);
      allocated_distances_ = std::move(distances);
      allocated_ids_ = std::move(ids);
      allocated_masks_ = std::move(masks);
      allocated_soa_capacity_ = capacity;
    }
    distances_ = capacity <= kInlineCapacity ? inline_distances_.data()
                                             : allocated_distances_.get();
    ids_ =
        capacity <= kInlineCapacity ? inline_ids_.data() : allocated_ids_.get();
    masks_ = capacity <= kInlineCapacity ? inline_masks_.data()
                                         : allocated_masks_.get();
  }
  capacity_ = capacity;
  limit_ = k;
  size_ = 0;
  threshold_ = threshold;
  small_ = small;
  needs_finish_ = false;
  ops_ = &batch_topk_internal::GetOps(isa, small);
  push_ = strict ? ops_->strict : ops_->inclusive;
}
void BatchTopK::compact_aos() {
  auto *last = entries_ + limit_ - 1;
  std::nth_element(
      entries_, last, entries_ + size_,
      [](const Candidate &a, const Candidate &b) { return a.score < b.score; });
  size_ = limit_;
  threshold_ = last->score;
}
void BatchTopK::finish() {
  if (ops_ != nullptr) ops_->finish(*this);
}
}  // namespace core
}  // namespace zvec
