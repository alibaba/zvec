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
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include "core/utility/stable_vector_holder.h"

namespace zvec::turbo {

// Shared selection and traversal for copied bytes and borrowed pointers.
// Leave output unchanged on read failure, including sticky iterator errors.
template <typename Buffer, typename Store>
inline int CollectPqTrainingSamplesImpl(
    const core::IndexHolder::Pointer &holder, size_t limit, size_t vec_bytes,
    size_t items_per_sample, Buffer *output, Store store) {
  if (!holder || !output || !limit || !vec_bytes)
    return core::IndexError_InvalidArgument;
  if (vec_bytes > holder->element_size())
    return core::IndexError_InvalidArgument;
  const size_t input_count = holder->count();
  const size_t num = std::min(input_count, limit);
  if (!num || num > std::numeric_limits<size_t>::max() / vec_bytes)
    return core::IndexError_InvalidArgument;
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
  Buffer all_data(num * items_per_sample);
  auto iter = holder->create_iterator();
  if (!iter) return core::IndexError_ReadData;
  size_t row = 0, copied = 0;
  for (; iter->is_valid(); iter->next(), ++row) {
    if (iter->status() != 0) return iter->status();
    if (row >= input_count) return core::IndexError_Mismatch;
    if (!selected.empty() && (copied == num || selected[copied].first != row))
      continue;
    const void *data = iter->data();
    if (iter->status() != 0) return iter->status();
    if (!data) return core::IndexError_ReadData;
    size_t slot = selected.empty() ? row : selected[copied].second;
    store(all_data, slot, data);
    ++copied;
  }
  if (iter->status() != 0) return iter->status();
  if (row != input_count || copied != num) return core::IndexError_Mismatch;
  iter.reset();

  *output = std::move(all_data);
  return 0;
}

// Preserve legacy sampling order with O(limit * vector_size) copied data.
// Iterators may reuse their buffers. Leave output unchanged on read failure.
inline bool CollectPqTrainingSamples(const core::IndexHolder::Pointer &holder,
                                     size_t limit, size_t vec_bytes,
                                     std::vector<uint8_t> *output) {
  return CollectPqTrainingSamplesImpl(
             holder, limit, vec_bytes, vec_bytes, output,
             [vec_bytes](std::vector<uint8_t> &data, size_t slot,
                         const void *row) {
               std::memcpy(data.data() + slot * vec_bytes, row, vec_bytes);
             }) == 0;
}

// Borrow only from sources that explicitly guarantee immutable, stable data.
// NotImplemented permits the copying fallback; other errors must propagate.
// On failure both outputs remain unchanged.
inline int BorrowPqTrainingSamples(const core::IndexHolder::Pointer &holder,
                                   size_t limit, size_t vec_bytes,
                                   std::vector<const void *> *output,
                                   std::shared_ptr<const void> *lifetime) {
  if (!holder || !output || !lifetime || !limit || !vec_bytes)
    return core::IndexError_InvalidArgument;
  auto *stable = dynamic_cast<core::StableVectorHolder *>(holder.get());
  if (!stable) return core::IndexError_NotImplemented;
  std::shared_ptr<const void> owner;
  int ret = stable->retain_stable_data(&owner);
  if (ret != 0) return ret;
  if (!owner) return core::IndexError_Runtime;
  ret = CollectPqTrainingSamplesImpl(
      holder, limit, vec_bytes, 1, output,
      [](std::vector<const void *> &data, size_t slot, const void *row) {
        data[slot] = row;
      });
  if (ret == 0) *lifetime = std::move(owner);
  return ret;
}

}  // namespace zvec::turbo
