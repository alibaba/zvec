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
//
// This translation unit hosts the AVX2-accelerated push_block implementation
// for BlockHeap. The build system compiles this .cc with an AVX2-capable
// `-march` when the toolchain/host supports it (see src/core/CMakeLists.txt
// and src/core/utility/CMakeLists.txt); otherwise the scalar fallback below
// is used. Callers must still runtime-gate invocation on CpuFeatures::AVX2
// because a binary compiled on an AVX2 host may be deployed on a lower-arch
// machine, in which case the AVX2 code here would fault.

// linear_pool.h (pulled in by block_heap.h) uses printf but does not
// #include <cstdio>; include it here before block_heap.h so the template
// non-dependent name lookup at definition time succeeds.
#include "block_heap.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include "zvec/ailego/internal/platform.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace zvec {
namespace core {

void BlockHeap::reset(int32_t capacity, int32_t block_size) {
  ef_ = capacity;
  block_size_ = block_size;
  data_.clear();
  const size_t reserve_cnt =
      static_cast<size_t>(std::max(capacity, block_size)) +
      static_cast<size_t>(block_size);
  data_.reserve(reserve_cnt);
  tmp_.clear();
  tmp_.reserve(static_cast<size_t>(block_size));
  cur_ = 0;
}

uint32_t BlockHeap::pop() {
  size_t ret_idx = cur_;
  set_checked(data_[cur_].first);
  while (cur_ < data_.size() && is_checked(data_[cur_].first)) {
    ++cur_;
  }
  return get_id(data_[ret_idx].first);
}

void BlockHeap::to_sorted(uint32_t *ids, float *scores, int32_t length) const {
  const int32_t n = std::min(length, static_cast<int32_t>(data_.size()));
  for (int32_t i = 0; i < n; ++i) {
    ids[i] = get_id(data_[i].first);
    if (scores != nullptr) {
      scores[i] = data_[i].second;
    }
  }
}

void BlockHeap::push_block(const float *distances, const uint32_t *nodes,
                           int32_t block_size) {
  // Phase 1: collect candidates with dist < current threshold into tmp_.
  if (static_cast<int32_t>(data_.size()) == ef_) {
    const float max_dist = data_.back().second;
#if defined(__AVX2__)
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
  data_.resize(std::min(static_cast<size_t>(old_data_size + tmp_size),
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
  // Merge phase: consume the larger of data_[i]/tmp_[j] into data_[write_pos].
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

}  // namespace core
}  // namespace zvec
