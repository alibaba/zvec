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

#pragma once
#include <zvec/ailego/internal/platform.h>
#include "batch_topk.h"

namespace zvec {
namespace core {
namespace batch_topk_internal {
// Private implementation/test surface: it is not a caller-facing ISA API.
enum class Isa : uint8_t { kScalar, kSSE2, kAVX2, kAVX512, kNEON };
struct Ops {
  using Push = void (*)(BatchTopK &, const float *, const uint64_t *, size_t);
  Push inclusive;
  Push strict;
  void (*finish)(BatchTopK &);
};
bool Supported(Isa isa);
Isa RuntimeIsa();
const Ops &GetOps(Isa isa, bool small);
const Ops &ScalarOps(bool small);
const Ops &Sse2Ops(bool small);
const Ops &Avx2Ops(bool small);
const Ops &Avx512Ops(bool small);
const Ops &NeonOps(bool small);

// These memory operations are ISA-independent; SIMD specializes the score
// comparisons that produce masks, not the width or interpretation of IDs.
void CompactMasks(float *distances, uint64_t *ids, size_t begin, size_t count,
                  const uint32_t *masks);
void PartitionMasks(float *distances, uint64_t *ids, size_t begin, size_t count,
                    size_t matched, const uint32_t *masks);
void HeapSort(float *distances, uint64_t *ids, size_t begin, size_t end);
// TU-local helpers cannot be coalesced with a higher-ISA implementation when
// inlining is disabled (for example, in Debug builds).
static inline unsigned CountBits(uint32_t value) {
  // Legal even on SSE2-only CPUs: do not use MSVC's unconditional __popcnt.
  value -= (value >> 1) & 0x55555555u;
  value = (value & 0x33333333u) + ((value >> 2) & 0x33333333u);
  value = (value + (value >> 4)) & 0x0f0f0f0fu;
  return (value * 0x01010101u) >> 24;
}
static inline float MedianOfThree(float a, float b, float c) {
  if (b < a) {
    const float tmp = a;
    a = b;
    b = tmp;
  }
  if (c < b) b = c;
  return a < b ? b : a;
}

template <class Simd, bool Equal>
size_t BuildMasks(const float *distances, size_t count, float pivot,
                  uint32_t *masks) {
  const auto bound = Simd::Broadcast(pivot);
  size_t matched = 0;
  size_t i = 0;
  // Fixed full-word groups let each ISA unroll the common 32-score mask.
  for (; count - i >= 32; i += 32) {
    uint32_t mask = 0;
    for (unsigned lane = 0; lane < 32; lane += Simd::kWidth) {
      const auto values = Simd::Load(distances + i + lane);
      const uint32_t part = Equal ? Simd::Equal(values, bound)
                                  : Simd::template Compare<true>(values, bound);
      mask |= part << lane;
    }
    masks[i / 32] = mask;
    matched += CountBits(mask);
  }
  if (i < count) {
    const unsigned lanes = static_cast<unsigned>(count - i);
    uint32_t mask = 0;
    unsigned lane = 0;
    for (; lanes - lane >= Simd::kWidth; lane += Simd::kWidth) {
      const auto values = Simd::Load(distances + i + lane);
      const uint32_t part = Equal ? Simd::Equal(values, bound)
                                  : Simd::template Compare<true>(values, bound);
      mask |= part << lane;
    }
    for (; lane < lanes; ++lane)
      if (Equal ? distances[i + lane] == pivot : distances[i + lane] < pivot)
        mask |= uint32_t{1} << lane;
    masks[i / 32] = mask;
    matched += CountBits(mask);
  }
  return matched;
}

// ISA and storage policy are compile-time parameters throughout the hot loop.
// The only indirect call is the public entry, bound at reset. Initial <= and
// subsequent < loops are separate instantiations: the first GC continues in
// the strict loop, including every unconsumed lane of the current SIMD group.
template <class Simd, bool Small>
struct Engine {
  static void Collect(BatchTopK &self, size_t keep_min, size_t keep_max) {
    size_t begin = 0;
    size_t end = self.size_;
    unsigned iterations = 4;
    for (size_t n = self.size_; n > 1; n >>= 1) iterations += 2;
    while (iterations-- != 0) {
      const float pivot = MedianOfThree(
          self.distances_[begin], self.distances_[begin + (end - begin) / 2],
          self.distances_[end - 1]);
      const size_t matched = BuildMasks<Simd, false>(
          self.distances_ + begin, end - begin, pivot, self.masks_);
      const size_t less = begin + matched;
      if (less >= keep_min) {
        CompactMasks(self.distances_, self.ids_, begin, end - begin,
                     self.masks_);
        if (less > keep_max) {
          end = less;
          continue;
        }
        self.size_ = less;
        self.threshold_ = pivot;
        return;
      }
      PartitionMasks(self.distances_, self.ids_, begin, end - begin, matched,
                     self.masks_);
      const size_t equal = BuildMasks<Simd, true>(
          self.distances_ + less, end - less, pivot, self.masks_);
      const size_t equal_end = less + equal;
      if (equal_end >= keep_min) {
        CompactMasks(self.distances_, self.ids_, less, end - less, self.masks_);
        self.size_ = keep_min;
        self.threshold_ = pivot;
        return;
      }
      PartitionMasks(self.distances_, self.ids_, less, end - less, equal,
                     self.masks_);
      begin = equal_end;
    }
    // Introspective bound: degenerate pivots cannot cause quadratic work or
    // infinite retries. Only the still-unresolved interval needs sorting.
    HeapSort(self.distances_, self.ids_, begin, end);
    self.size_ = keep_min;
    self.threshold_ = self.distances_[keep_min - 1];
  }

  static bool Append(BatchTopK &self, float score, uint64_t id) {
    if constexpr (Small) {
      self.entries_[self.size_++] = {score, id};
    } else {
      self.distances_[self.size_] = score;
      self.ids_[self.size_++] = id;
    }
    if (self.size_ != self.capacity_) return false;
    if constexpr (Small) {
      self.compact_aos();
    } else {
      const size_t keep_max =
          self.limit_ + (self.capacity_ - self.limit_) / 2 - 1;
      Collect(self, self.limit_, keep_max);
    }
    return true;
  }

  template <bool Strict>
  static void Push(BatchTopK &self, const float *distances, const uint64_t *ids,
                   size_t count) {
    size_t i = 0;
    auto bound = Simd::Broadcast(self.threshold_);
    for (; count - i >= Simd::kWidth; i += Simd::kWidth) {
      const auto values = Simd::Load(distances + i);
      uint32_t mask = Simd::template Compare<Strict>(values, bound);
      while (mask != 0) {
        const unsigned lane = static_cast<unsigned>(ailego_ctz32(mask));
        mask &= mask - 1;
        if (Append(self, distances[i + lane], ids[i + lane])) {
          if constexpr (!Strict) {
            self.push_ = self.ops_->strict;
            const size_t consumed = i + lane + 1;
            Push<true>(self, distances + consumed, ids + consumed,
                       count - consumed);
            return;
          } else {
            bound = Simd::Broadcast(self.threshold_);
            mask &= Simd::template Compare<true>(values, bound);
          }
        }
      }
    }
    for (; i < count; ++i) {
      const float value = distances[i];
      if (Strict ? value < self.threshold_ : value <= self.threshold_) {
        if (Append(self, value, ids[i])) {
          if constexpr (!Strict) {
            self.push_ = self.ops_->strict;
            Push<true>(self, distances + i + 1, ids + i + 1, count - i - 1);
            return;
          }
        }
      }
    }
  }

  static void Finish(BatchTopK &self) {
    if (!self.needs_finish_) return;
    if (self.size_ > self.limit_) {
      if constexpr (Small)
        self.compact_aos();
      else
        Collect(self, self.limit_, self.limit_);
    } else if (self.size_ == self.limit_ && self.size_ != 0) {
      if constexpr (Small) {
        float threshold = self.entries_[0].score;
        for (size_t i = 1; i < self.size_; ++i) {
          if (threshold < self.entries_[i].score)
            threshold = self.entries_[i].score;
        }
        self.threshold_ = threshold;
      } else {
        float threshold = self.distances_[0];
        for (size_t i = 1; i < self.size_; ++i) {
          if (threshold < self.distances_[i]) threshold = self.distances_[i];
        }
        self.threshold_ = threshold;
      }
    }
    if (self.size_ == self.limit_ && self.size_ != 0) {
      self.push_ = self.ops_->strict;
    }
    self.needs_finish_ = false;
  }

  static const Ops &Operations() {
    static const Ops ops{&Push<false>, &Push<true>, &Finish};
    return ops;
  }
};
}  // namespace batch_topk_internal
}  // namespace core
}  // namespace zvec
