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

  // Document output and group-by need the legacy heap ordering. Pool conversion
  // uses private, reusable scratch and never replaces the active search pool.
  template <typename Fn>
  void with_topk(Fn &&fn) {
    dispatch([&](auto &heap) { with_topk(heap, std::forward<Fn>(fn)); });
  }

  // For an explicit operation that changes the search distances themselves,
  // unlike temporary result export. This intentionally replaces the pool.
  TopkHeap &materialize_topk() {
    if (auto *heap = std::get_if<TopkHeap>(&heap_)) return *heap;
    with_topk([](TopkHeap &) {});
    heap_.emplace<TopkHeap>(std::move(fallback_));
    return topk();
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
  void with_topk(Heap &heap, Fn &&fn) {
    if constexpr (std::is_same_v<Heap, TopkHeap>) {
      std::forward<Fn>(fn)(heap);
    } else {
      fallback_.clear();
      apply_limit(fallback_);
      copy_pool_to_topk(heap, fallback_);
      std::forward<Fn>(fn)(fallback_);
    }
  }

  std::variant<TopkHeap, LinearPool<float>, BlockHeap> heap_;
  // Scratch is valid only during with_topk, not an independent
  // search result. Its vector allocates only when heap fallback is needed.
  TopkHeap fallback_;
  size_t limit_{std::numeric_limits<size_t>::max()};
};

}  // namespace core
}  // namespace zvec
