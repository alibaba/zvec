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
#include <limits>
#include <new>
#include "utility/visit_filter.h"
#include "diskann_read_memory.h"

namespace zvec::core {

// Exact visitation with one bit per vector and one generation byte per word.
// Query workspace is charged to RSS independently of the page cache.
class DiskAnnSearchVisitFilter {
 public:
  void init(uint64_t vectors) {
    const uint64_t words =
        vectors / kBitsPerWord + (vectors % kBitsPerWord != 0);
    if (words > (std::numeric_limits<size_t>::max)()) throw std::bad_alloc();

    // Hold the partial allocation locally so a rejected second allocation
    // releases the first before the caller retries context creation.
    DiskAnnReadVector<uint64_t> bits(static_cast<size_t>(words));
    DiskAnnReadVector<uint8_t> generations(static_cast<size_t>(words));
    bits_.swap(bits);
    generations_.swap(generations);
    generation_ = 1;
  }

  // Callers validate IDs against the document count before marking them.
  void set_visited(uint32_t id) {
    const size_t word = id / kBitsPerWord;
    if (generations_[word] != generation_) {
      bits_[word] = 0;
      generations_[word] = generation_;
    }
    bits_[word] |= uint64_t{1} << (id % kBitsPerWord);
  }

  bool visited(uint32_t id) const {
    const size_t word = id / kBitsPerWord;
    return generations_[word] == generation_ &&
           (bits_[word] & (uint64_t{1} << (id % kBitsPerWord))) != 0;
  }

  void clear() {
    if (++generation_ == 0) {
      std::fill(generations_.begin(), generations_.end(), 0);
      generation_ = 1;
    }
  }

 private:
  static constexpr uint32_t kBitsPerWord = 64;
  DiskAnnReadVector<uint64_t> bits_;
  DiskAnnReadVector<uint8_t> generations_;
  uint8_t generation_{1};
};

}  // namespace zvec::core
