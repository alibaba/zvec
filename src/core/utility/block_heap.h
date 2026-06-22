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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace zvec {
namespace core {

// BlockHeap is a block-insert optimized alternative to LinearPool for graph
// search. It receives candidates in batches (push_block) and maintains a
// distance-sorted prefix of size ef with amortized O(k) bookkeeping per
// batch, replacing LinearPool's one-by-one sorted insert.
//
// Derived from pyglass' BlockHeap (https://github.com/zilliztech/pyglass,
// MIT License; see the NOTICE file and linear_pool.h for the full attribution).
// The graph prefetch is intentionally omitted: the call-site is expected to
// issue the neighbor-array prefetch itself (Vamana's greedy_search already
// does so).
//
// AVX2 requirement
// ----------------
// The implementation uses AVX2 intrinsics in push_block for the common case
// where the pool is full and we need to filter a block of candidates against
// the current distance threshold. The intrinsics are confined to
// block_heap.cc and guarded with `#if defined(__AVX2__)`, so this header is
// always safe to include. Callers MUST gate the invocation of BlockHeap-based
// code paths on a runtime CpuFeatures::AVX2 check to avoid illegal
// instructions when running on a low-arch machine with a binary built on a
// higher-arch host.
struct BlockHeap {
  BlockHeap() = default;
  ~BlockHeap() = default;

  BlockHeap(const BlockHeap &) = delete;
  BlockHeap &operator=(const BlockHeap &) = delete;

  BlockHeap(BlockHeap &&) = default;
  BlockHeap &operator=(BlockHeap &&) = default;

  // Reset the pool state for a new search round.  `capacity` is the retained
  // top-k size, `block_size` is an upper bound on the per-call push_block size
  // (used only for capacity hints).  Visited-node tracking is no longer owned
  // by the pool — the caller passes a VisitFilter reference instead.
  void reset(int32_t capacity, int32_t block_size);

  // Insert a block of candidates. The distance array must have at least
  // `block_size` entries and the id array must have the same length.
  // `block_size` may differ from the value passed to reset(); reset()'s
  // block_size is only a capacity hint.
  void push_block(const float *distances, const uint32_t *nodes,
                  int32_t block_size);

  // Is there an unpopped candidate?
  bool has_next() const {
    return cur_ < data_.size();
  }

  // Pop the closest unpopped candidate id (without the check bit).
  // Caller must ensure has_next() is true.
  uint32_t pop();

  // Retained candidate count.
  int32_t size() const {
    return static_cast<int32_t>(data_.size());
  }

  // Export sorted top-`length` ids (and optionally scores) — data_ is already
  // distance-sorted ascending.
  void to_sorted(uint32_t *ids, float *scores, int32_t length) const;

  // Direct sorted accessors (used by search result copy-out).
  uint32_t id(int32_t i) const {
    return get_id(data_[i].first);
  }
  float dist(int32_t i) const {
    return data_[i].second;
  }

  // Internal check-bit helpers (high bit marks a popped entry).
  static constexpr uint32_t kCheckedBit = 0x80000000u;
  static constexpr uint32_t kIdMask = 0x7FFFFFFFu;

  static void set_checked(uint32_t &id) {
    id |= kCheckedBit;
  }
  static bool is_checked(uint32_t id) {
    return (id & kCheckedBit) != 0u;
  }
  static uint32_t get_id(uint32_t id) {
    return id & kIdMask;
  }

 private:
  std::vector<std::pair<uint32_t, float>> data_;
  std::vector<std::pair<uint32_t, float>> tmp_;
  int32_t ef_{0};
  int32_t block_size_{0};
  size_t cur_{0};
};

}  // namespace core
}  // namespace zvec
