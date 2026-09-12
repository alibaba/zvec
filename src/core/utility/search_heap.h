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
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <ailego/internal/cpu_features.h>
#include <zvec/ailego/container/heap.h>
#include "block_heap.h"
#include "linear_pool.h"

namespace zvec {
namespace core {

// Own exactly one active graph-search container. Its variant alternative is
// also the result type; no separate output mode or result-source tag is needed.
class SearchHeap {
 public:
  using TopkHeap = ailego::KeyValueHeap<uint32_t, float>;

  // Select only when the search strategy changes. Repeated selection keeps
  // the existing container and its allocations; the search resets its contents.
  template <typename Heap>
  Heap &select() {
    if (!std::holds_alternative<Heap>(heap_)) {
      heap_.emplace<Heap>();
      if constexpr (std::is_same_v<Heap, TopkHeap>) apply_limit(topk());
    }
    return std::get<Heap>(heap_);
  }

  // A concrete reference for the dual-heap/build paths after selecting
  // TopkHeap.
  TopkHeap &topk() {
    return std::get<TopkHeap>(heap_);
  }

  // Preparation only: select the backend, clear its contents and set capacity.
  // Selecting the same backend preserves its allocated storage. Execution
  // dispatches the prepared alternative without making this decision again.
  template <typename Heap>
  Heap &reset(size_t capacity, int32_t block_size = 0) {
    limit_ = (std::max)(capacity, size_t{1});
    auto &heap = select<Heap>();
    if constexpr (std::is_same_v<Heap, TopkHeap>) {
      heap.clear();
      apply_limit(heap);
    } else {
      heap.reset(static_cast<int32_t>(capacity), block_size);
    }
    return heap;
  }

  // The caller chooses the pool strategy; select its backend using the cached
  // CPU flags. No CPU probing or backend selection occurs in the search loop.
  ailego_force_inline void reset_pool(size_t capacity, int32_t block_size) {
    if (ailego::internal::CpuFeatures::static_flags_.AVX2) {
      reset<BlockHeap>(capacity, block_size);
    } else {
      reset<LinearPool<float>>(capacity, block_size);
    }
  }

  void limit(size_t capacity) {
    limit_ = (std::max)(capacity, size_t{1});
    if (auto *heap = std::get_if<TopkHeap>(&heap_)) apply_limit(*heap);
  }

  size_t size() const {
    return std::visit(
        [](const auto &heap) { return static_cast<size_t>(heap.size()); },
        heap_);
  }

  // Invalidate the previous query's contents without discarding its capacity.
  void clear() {
    dispatch([&](auto &heap) {
      if constexpr (std::is_same_v<std::decay_t<decltype(heap)>, TopkHeap>) {
        heap.clear();
      } else {
        heap.reset(0, 0);
      }
    });
  }

  // Dispatch once at a phase boundary, including future refine-candidate
  // export. Keep references inside the callback; selecting another alternative
  // invalidates references to the current one. Query policy (topk, threshold,
  // padding, ID-to-key mapping) belongs to the result collector, not this
  // owner.
  template <typename Fn>
  ailego_force_inline void dispatch(Fn &&fn) {
    std::visit(std::forward<Fn>(fn), heap_);
  }

  // Visit retained candidates without changing their order (e.g. group-by).
  // Returning false stops iteration. IDs never include pool traversal bits.
  template <typename Fn>
  void for_each(Fn &&fn) {
    dispatch([&](const auto &heap) { visit(heap, limit_, fn); });
  }

  // Terminal result export: sort TopkHeap in place, or read the pool's existing
  // ascending order. Reset before inserting into a sorted TopkHeap again.
  // Pools keep their existing tie order; TopkHeap uses its usual sort.
  template <typename Fn>
  void for_each_sorted(size_t count, Fn &&fn) {
    if (count == 0) return;
    dispatch([&](auto &heap) {
      using Heap = std::decay_t<decltype(heap)>;
      if constexpr (std::is_same_v<Heap, TopkHeap>) heap.sort();
      visit(heap, (std::min)(count, limit_), fn);
    });
  }

  // Concrete-container helpers for optional result padding. These are used
  // inside one dispatch, not through per-candidate runtime type checks.
  template <typename Heap>
  static size_t capacity(const Heap &heap) {
    if constexpr (std::is_same_v<Heap, TopkHeap>) {
      return heap.limit();
    } else {
      return static_cast<size_t>(heap.capacity());
    }
  }

  template <typename Heap>
  static void emplace(Heap &heap, uint32_t id, float distance) {
    if constexpr (std::is_same_v<Heap, TopkHeap>) {
      heap.emplace(id, distance);
    } else {
      heap.push_block(&distance, &id, 1);
    }
  }

 private:
  void apply_limit(TopkHeap &heap) const {
    if (limit_ == std::numeric_limits<size_t>::max()) {
      heap.unlimit();
    } else {
      heap.limit(limit_);
    }
  }

  template <typename Heap, typename Fn>
  static void visit(const Heap &heap, size_t count, Fn &&fn) {
    count = (std::min)(count, static_cast<size_t>(heap.size()));
    for (size_t i = 0; i < count; ++i) {
      if constexpr (std::is_same_v<Heap, TopkHeap>) {
        if (!fn(heap[i].first, heap[i].second)) break;
      } else {
        if (!fn(heap.id(static_cast<int32_t>(i)),
                heap.dist(static_cast<int32_t>(i))))
          break;
      }
    }
  }

  std::variant<TopkHeap, LinearPool<float>, BlockHeap> heap_;
  size_t limit_{std::numeric_limits<size_t>::max()};
};

}  // namespace core
}  // namespace zvec
