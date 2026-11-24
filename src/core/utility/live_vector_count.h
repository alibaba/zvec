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
#include <mutex>

namespace zvec::core {

// Entity clones share key writes, but may expose different slot bounds.
class LiveVectorCount {
 public:
  template <typename Entity>
  size_t get(const Entity &entity) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t revision = entity.key_revision();
    const size_t slots = entity.slot_count();
    if (revision_ != revision || slots_ != slots) {
      size_t count = 0;
      for (uint32_t id = 0; id < slots; ++id)
        if (entity.contains_node(id)) ++count;
      count_ = count;
      slots_ = slots;
      revision_ = revision;
    }
    return count_;
  }

 private:
  mutable std::mutex mutex_;
  mutable uint64_t revision_{UINT64_MAX};
  mutable size_t slots_{0};
  mutable size_t count_{0};
};

}  // namespace zvec::core
