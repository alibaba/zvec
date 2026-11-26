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

#include <vector>
#include <roaring/roaring.h>

namespace zvec {
namespace core {

struct FilterResultCache {
 public:
  FilterResultCache() {
    bitmap_ = roaring_bitmap_create();
  }

  FilterResultCache(uint32_t capacity_hint) {
    bitmap_ = roaring_bitmap_create_with_capacity(capacity_hint);
  }

  ~FilterResultCache() {
    roaring_bitmap_free(bitmap_);
    bitmap_ = nullptr;
  }

  bool find(uint64_t key) const {
    return !roaring_bitmap_contains(bitmap_, key);
  }

  void set(uint64_t key) const {
    roaring_bitmap_add(bitmap_, key);
  }

  int filter(const std::vector<std::vector<uint64_t>> &id_to_tags_list,
             const std::vector<uint64_t> &query_tag_list,
             const std::vector<uint64_t> &id_to_key_list) {
    for (size_t i = 0; i < id_to_tags_list.size(); ++i) {
      auto &id_tag_list = id_to_tags_list[i];

      size_t t_i = 0;
      size_t q_i = 0;
      while (t_i < id_tag_list.size() && q_i < query_tag_list.size()) {
        if (id_tag_list[t_i] == query_tag_list[q_i]) {
          uint64_t key = id_to_key_list[i];

          set(key);

          break;
        } else if (id_tag_list[t_i] < query_tag_list[q_i]) {
          ++t_i;
        } else {
          ++q_i;
        }
      }
    }

    return 0;
  }

 public:
  roaring_bitmap_t *bitmap_{nullptr};
};

}  // namespace core
}  // namespace zvec
