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
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <zvec/ailego/internal/platform.h>

namespace zvec {
namespace core {

//! CRTP base that builds search results from top-k heaps.
//!
//! The single/group result construction logic is identical across the
//! graph-based index contexts. They differ only in the heap element value
//! type and in how a single result document is materialized. Concrete
//! contexts inherit from this base and supply the small set of accessors and
//! hooks listed below (make this base a `friend` if any of them are private):
//!   - HeapT &topk_heap();
//!   - std::map<std::string, HeapT> &group_topk_heaps();
//!   - ResultList &mutable_results();        // std::vector<IndexDocumentList>
//!   - GroupList &mutable_group_results();   // std::vector<IndexGroupDocList>
//!   - uint32_t topk() const;
//!   - uint32_t group_topk() const;
//!   - uint32_t group_num() const;
//!   - float threshold() const;
//!   - bool group_by_search();
//!   - float result_score(const HeapValue &value) const;
//!   - void emplace_result_doc(IndexDocumentList &docs, IdType id,
//!                             float score, const HeapValue &value);
template <typename Derived, typename HeapT>
class TopkResultBuilder {
 public:
  //! Heap element value type (e.g. VectorInfo / dist_t / ResultRecord).
  using HeapValue =
      typename std::decay<decltype(std::declval<HeapT &>()[0].second)>::type;

  //! Dispatch to single or group result construction.
  inline void topk_to_result(uint32_t idx) {
    if (derived().group_by_search()) {
      topk_to_group_result(idx);
    } else {
      topk_to_single_result(idx);
    }
  }

  //! Construct a flat result list from the top-k heap.
  inline void topk_to_single_result(uint32_t idx) {
    HeapT &heap = derived().topk_heap();
    if (ailego_unlikely(heap.size() == 0)) {
      return;
    }

    auto &results = derived().mutable_results();
    ailego_assert_with(idx < results.size(), "invalid idx");
    uint32_t size =
        std::min(derived().topk(), static_cast<uint32_t>(heap.size()));
    heap.sort();
    results[idx].clear();

    for (uint32_t i = 0; i < size; ++i) {
      float score = derived().result_score(heap[i].second);
      if (score > derived().threshold()) {
        break;
      }
      derived().emplace_result_doc(results[idx], heap[i].first, score,
                                   heap[i].second);
    }
  }

  //! Construct grouped result lists from the per-group top-k heaps.
  inline void topk_to_group_result(uint32_t idx) {
    auto &group_results = derived().mutable_group_results();
    ailego_assert_with(idx < group_results.size(), "invalid idx");

    auto &group_heaps = derived().group_topk_heaps();
    group_results[idx].clear();

    std::vector<std::pair<std::string, HeapT>> group_topk_list;
    std::vector<std::pair<std::string, float>> best_score_in_groups;
    for (auto &entry : group_heaps) {
      HeapT &heap = entry.second;
      heap.sort();
      if (heap.size() > 0) {
        best_score_in_groups.emplace_back(
            entry.first, derived().result_score(heap[0].second));
      }
    }

    std::sort(best_score_in_groups.begin(), best_score_in_groups.end(),
              [](const std::pair<std::string, float> &a,
                 const std::pair<std::string, float> &b) {
                return a.second < b.second;
              });

    // truncate to group num
    for (uint32_t i = 0;
         i < derived().group_num() && i < best_score_in_groups.size(); ++i) {
      const std::string &group_id = best_score_in_groups[i].first;
      group_topk_list.emplace_back(group_id, group_heaps[group_id]);
    }

    group_results[idx].resize(group_topk_list.size());
    for (uint32_t i = 0; i < group_topk_list.size(); ++i) {
      const std::string &group_id = group_topk_list[i].first;
      group_results[idx][i].set_group_id(group_id);

      HeapT &heap = group_topk_list[i].second;
      uint32_t size =
          std::min(derived().group_topk(), static_cast<uint32_t>(heap.size()));
      for (uint32_t j = 0; j < size; ++j) {
        float score = derived().result_score(heap[j].second);
        if (score > derived().threshold()) {
          break;
        }
        derived().emplace_result_doc(*group_results[idx][i].mutable_docs(),
                                     heap[j].first, score, heap[j].second);
      }
    }
  }

 private:
  inline Derived &derived() {
    return *static_cast<Derived *>(this);
  }
};

}  // namespace core
}  // namespace zvec
