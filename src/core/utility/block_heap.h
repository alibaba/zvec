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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <zvec/ailego/internal/platform.h>

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define ZVEC_BLOCK_HEAP_AVX2_TARGET __attribute__((target("avx2")))
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 1
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#define ZVEC_BLOCK_HEAP_AVX2_TARGET
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 1
#else
#define ZVEC_BLOCK_HEAP_AVX2_TARGET
#define ZVEC_BLOCK_HEAP_AVX2_INTRINSICS 0
#endif

namespace zvec {
namespace core {

// BlockHeap is a block-insert optimized alternative to LinearPool for graph
// search. It receives candidates in batches (push_block) and maintains a
// distance-sorted prefix of size ef with amortized O(k) bookkeeping per
// batch, replacing LinearPool's one-by-one sorted insert.
//
// Derived from pyglass' BlockHeap (https://github.com/zilliztech/pyglass,
// MIT License; see the NOTICE file and linear_pool.h for the full attribution).
// Graph prefetch remains a call-site policy. Callers that need to overlap the
// current and next neighbor-row fetches can explicitly use pop_with_next().
//
// AVX2 requirement
// ----------------
// The implementation uses AVX2 intrinsics in push_block for the common case
// where the pool is full and we need to filter a block of candidates against
// the current distance threshold. On x86, push_block is an AVX2-targeted
// inline definition so baseline translation units can include this header.
// MSVC supports these intrinsics without a per-source /arch flag. Select by
// architecture, not __AVX2__, to keep the inline body identical across TUs.
// Callers MUST gate BlockHeap-based paths on a runtime CpuFeatures::AVX2
// check. Other architectures use the scalar implementation.
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
  void reset(int32_t capacity, int32_t block_size) {
    ef_ = capacity;
    block_size_ = block_size;
    data_.clear();
    const size_t reserve_cnt =
        static_cast<size_t>((std::max)(capacity, block_size)) +
        static_cast<size_t>(block_size);
    data_.reserve(reserve_cnt);
    tmp_.clear();
    tmp_.reserve(static_cast<size_t>(block_size));
    cur_ = 0;
  }

  // Insert a block of candidates. The distance array must have at least
  // `block_size` entries and the id array must have the same length.
  // `block_size` may differ from the value passed to reset(); reset()'s
  // block_size is only a capacity hint.
  ZVEC_BLOCK_HEAP_AVX2_TARGET void push_block(const float *distances,
                                              const uint32_t *nodes,
                                              int32_t block_size) {
    // Phase 1: collect candidates with dist < current threshold into tmp_.
    if (static_cast<int32_t>(data_.size()) == ef_) {
      const float max_dist = data_.back().second;
#if ZVEC_BLOCK_HEAP_AVX2_INTRINSICS
      const __m256 threshold_vec = _mm256_set1_ps(max_dist);
      int32_t i = 0;
      for (; i + 8 <= block_size; i += 8) {
        __m256 d = _mm256_loadu_ps(distances + i);
        __m256 mask = _mm256_cmp_ps(d, threshold_vec, _CMP_LT_OS);
        int bitmask = _mm256_movemask_ps(mask);
        if (bitmask == 0) {
          continue;
        }
        while (bitmask) {
          int tz = ailego_ctz32(bitmask);
          tmp_.emplace_back(nodes[i + tz], distances[i + tz]);
          bitmask &= bitmask - 1;
        }
      }
      for (; i < block_size; ++i) {
        if (distances[i] < max_dist) {
          tmp_.emplace_back(nodes[i], distances[i]);
        }
      }
#else
      for (int32_t i = 0; i < block_size; ++i) {
        if (distances[i] < max_dist) {
          tmp_.emplace_back(nodes[i], distances[i]);
        }
      }
#endif
    } else {
      for (int32_t i = 0; i < block_size; ++i) {
        tmp_.emplace_back(nodes[i], distances[i]);
      }
    }
    if (tmp_.empty()) {
      return;
    }

    // Phase 2: sort tmp_ ascending by distance (with ef truncation).
    auto cmp = [](const std::pair<uint32_t, float> &a,
                  const std::pair<uint32_t, float> &b) {
      return a.second < b.second;
    };
    if (static_cast<int32_t>(tmp_.size()) > ef_) {
      // nth_element + sort of the top-ef slice is O(n) + O(k log k), which is
      // faster than partial_sort's O(n log k) for the hot path.
      std::nth_element(tmp_.begin(), tmp_.begin() + ef_, tmp_.end(), cmp);
      tmp_.resize(static_cast<size_t>(ef_));
    }
    if (tmp_.size() <= 32) {
      // Insertion sort for small arrays — branch-predictor friendly and has
      // lower overhead than std::sort for tiny inputs.
      for (size_t i = 1; i < tmp_.size(); ++i) {
        auto key = tmp_[i];
        int32_t j = static_cast<int32_t>(i) - 1;
        while (j >= 0 && tmp_[j].second > key.second) {
          tmp_[j + 1] = tmp_[j];
          --j;
        }
        tmp_[j + 1] = key;
      }
    } else {
      std::sort(tmp_.begin(), tmp_.end(), cmp);
    }

    // Phase 3: in-place merge (tail-write) data_ and tmp_, truncated at ef_.
    const int32_t old_data_size = static_cast<int32_t>(data_.size());
    const int32_t tmp_size = static_cast<int32_t>(tmp_.size());
    int32_t i = old_data_size - 1;
    int32_t j = tmp_size - 1;
    int32_t write_pos = old_data_size + tmp_size - 1;
    data_.resize((std::min)(static_cast<size_t>(old_data_size + tmp_size),
                            static_cast<size_t>(ef_)));
    // Drop the overflow tail (entries past ef_): advance i/j without writing,
    // since data_[write_pos] would be out of bounds.
    while (write_pos >= ef_) {
      if (data_[i].second > tmp_[j].second) {
        --i;
      } else {
        --j;
      }
      --write_pos;
    }
    // Merge phase: consume the larger of data_[i]/tmp_[j] into
    // data_[write_pos].
    while (i >= 0 && j >= 0) {
      if (data_[i].second > tmp_[j].second) {
        data_[write_pos--] = data_[i--];
      } else {
        data_[write_pos--] = tmp_[j--];
      }
    }
    if (j >= 0) {
      // tmp_ entries remaining at front — copy them and reset cursor so the
      // caller re-scans from the head.
      while (j >= 0) {
        data_[write_pos--] = tmp_[j--];
      }
      cur_ = 0;
    } else {
      // All tmp_ entries consumed; old data_[0..i] are already in place.
      // Move cursor back if new items were inserted ahead of it.
      if (static_cast<size_t>(write_pos + 1) <= cur_) {
        cur_ = static_cast<size_t>(write_pos + 1);
      }
    }
    tmp_.clear();
  }

  // Is there an unpopped candidate?
  bool has_next() const {
    return cur_ < data_.size();
  }

  // Pop the closest unpopped candidate id (without the check bit).
  // Caller must ensure has_next() is true.
  // Keep cursor advancement visible to the search loop without requiring LTO.
  ailego_force_inline uint32_t pop() {
    size_t ret_idx = cur_;
    set_checked(data_[cur_].first);
    while (cur_ < data_.size() && is_checked(data_[cur_].first)) {
      ++cur_;
    }
    return get_id(data_[ret_idx].first);
  }

  // Pop the closest unpopped candidate and expose the next unexpanded id.
  // `next_id` is UINT32_MAX when no candidate remains.
  ailego_force_inline uint32_t pop_with_next(uint32_t *next_id) {
    const uint32_t id = pop();
    if (next_id != nullptr) {
      *next_id = cur_ < data_.size() ? get_id(data_[cur_].first) : UINT32_MAX;
    }
    return id;
  }

  // Retained candidate count.
  int32_t size() const {
    return static_cast<int32_t>(data_.size());
  }

  int32_t capacity() const {
    return ef_;
  }

  // Export sorted top-`length` ids (and optionally scores) — data_ is already
  // distance-sorted ascending.
  void to_sorted(uint32_t *ids, float *scores, int32_t length) const {
    const int32_t n = (std::min)(length, static_cast<int32_t>(data_.size()));
    for (int32_t i = 0; i < n; ++i) {
      ids[i] = get_id(data_[i].first);
      if (scores != nullptr) {
        scores[i] = data_[i].second;
      }
    }
  }

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

#undef ZVEC_BLOCK_HEAP_AVX2_TARGET
#undef ZVEC_BLOCK_HEAP_AVX2_INTRINSICS
