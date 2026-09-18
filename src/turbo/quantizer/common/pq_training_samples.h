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
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>
#include <zvec/core/framework/index_holder.h>

namespace zvec::turbo {

// Preserve legacy sampling order with O(limit * vector_size) copied data.
// Iterators may reuse their buffers. Leave output unchanged on read failure.
inline bool CollectPqTrainingSamples(const core::IndexHolder::Pointer &holder,
                                     size_t limit, size_t vec_bytes,
                                     std::vector<uint8_t> *output) {
  if (!holder || !output || !limit || !vec_bytes) return false;
  if (vec_bytes > holder->element_size()) return false;
  const size_t input_count = holder->count();
  const size_t num = std::min(input_count, limit);
  if (!num || num > std::numeric_limits<size_t>::max() / vec_bytes)
    return false;
  // Perform the same seeded partial Fisher-Yates shuffle on row ordinals,
  // then stream only selected rows into their original training slots. The
  // sparse permutation and data buffer are bounded by the training limit.
  std::vector<std::pair<size_t, size_t>> selected;
  if (input_count > num) {
    selected.reserve(num);
    std::unordered_map<size_t, size_t> permutation;
    permutation.reserve(num);
    auto at = [&](size_t position) {
      auto entry = permutation.find(position);
      return entry == permutation.end() ? position : entry->second;
    };
    std::mt19937 rng(42);
    for (size_t i = 0; i < num; ++i) {
      std::uniform_int_distribution<size_t> dist(i, input_count - 1);
      size_t j = dist(rng);
      selected.emplace_back(at(j), i);
      if (i != j) permutation[j] = at(i);
      permutation.erase(i);
    }
    std::sort(selected.begin(), selected.end());
  }
  std::vector<uint8_t> all_data(num * vec_bytes);
  auto iter = holder->create_iterator();
  if (!iter) return false;
  size_t row = 0, copied = 0;
  for (; iter->is_valid(); iter->next(), ++row) {
    if (row >= input_count) return false;
    if (!selected.empty() && (copied == num || selected[copied].first != row))
      continue;
    const void *data = iter->data();
    if (!data) return false;
    size_t slot = selected.empty() ? row : selected[copied].second;
    std::memcpy(all_data.data() + slot * vec_bytes, data, vec_bytes);
    ++copied;
  }
  if (row != input_count || copied != num) return false;
  iter.reset();

  *output = std::move(all_data);
  return true;
}

}  // namespace zvec::turbo
