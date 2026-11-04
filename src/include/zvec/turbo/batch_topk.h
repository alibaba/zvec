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
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <zvec/export.h>

namespace zvec {
namespace turbo {
namespace batch_topk_internal {
enum class Isa : uint8_t;
struct Ops;
template <class Simd, bool Small>
struct Engine;
}  // namespace batch_topk_internal

// Exact, unsorted top-k over distance batches and opaque 64-bit IDs.
// Equal-score IDs/order are unspecified. Duplicate IDs are not deduplicated.
// NaN scores are rejected; no ID bits are reserved. Buffered selection uses
// O(k) memory and does not maintain a heap or an intermediate ordering. The
// larger-k partition design is inspired by ScaNN FastTopNeighbors (Apache-2.0),
// with independent implementation.
class BatchTopK {
 public:
  struct Candidate {
    float score;
    uint64_t id;
  };

  BatchTopK() noexcept;
  BatchTopK(const BatchTopK &) = delete;
  BatchTopK &operator=(const BatchTopK &) = delete;

  // Reuses allocations. The initial bound is inclusive unless strict is set.
  // k == 0 accepts nothing; small selections need no dynamic allocation.
  ZVEC_TURBO_API void reset(
      size_t k, float threshold = std::numeric_limits<float>::max(),
      bool strict = false);
  // Nonempty inputs must be valid. Accepted values are copied, so callers may
  // reuse the input buffers immediately. IDs need not be ordered or unique.
  void push_block(const float *distances, const uint64_t *ids, size_t count) {
    if (limit_ != 0 && count != 0) {
      // Mark once per batch instead of once per accepted candidate. A wholly
      // rejected batch may make finish revisit unchanged results, which is
      // safe.
      needs_finish_ = true;
      push_(*this, distances, ids, count);
    }
  }
  // Call before consuming size()/result(); further pushes and repeated finish
  // are valid. result(i) requires i < size() after finish().
  ZVEC_TURBO_API void finish();
  size_t size() const {
    return size_;
  }
  Candidate result(size_t i) const {
    return small_ ? entries_[i] : Candidate{distances_[i], ids_[i]};
  }

 private:
  friend class BatchTopKTestAccess;
  template <class Simd, bool Small>
  friend struct batch_topk_internal::Engine;
  using Push = void (*)(BatchTopK &, const float *, const uint64_t *, size_t);
  void reset_impl(size_t k, float threshold, bool strict,
                  batch_topk_internal::Isa isa, bool small);
  void compact_aos();

  static constexpr size_t kInlineCapacity = 128;
  // The crossover is an internal policy, not part of the public contract.
  static constexpr size_t kSmallLimit = 64;
  std::array<Candidate, kInlineCapacity> inline_entries_;
  alignas(64) std::array<float, kInlineCapacity> inline_distances_;
  alignas(64) std::array<uint64_t, kInlineCapacity> inline_ids_;
  std::array<uint32_t, kInlineCapacity / 32> inline_masks_;
  std::unique_ptr<Candidate[]> allocated_entries_;
  std::unique_ptr<float[]> allocated_distances_;
  std::unique_ptr<uint64_t[]> allocated_ids_;
  std::unique_ptr<uint32_t[]> allocated_masks_;
  Candidate *entries_{inline_entries_.data()};
  float *distances_{inline_distances_.data()};
  uint64_t *ids_{inline_ids_.data()};
  uint32_t *masks_{inline_masks_.data()};
  size_t allocated_aos_capacity_{0};
  size_t allocated_soa_capacity_{0};
  size_t capacity_{0};
  size_t limit_{0};
  size_t size_{0};
  float threshold_{std::numeric_limits<float>::max()};
  bool small_{true};
  bool needs_finish_{false};
  const batch_topk_internal::Ops *ops_{nullptr};
  Push push_{nullptr};
};

// Default after the first declaration to keep this constructor user-provided:
// value-initialization must not zero the scratch arrays before reset().
inline BatchTopK::BatchTopK() noexcept = default;
}  // namespace turbo
}  // namespace zvec
