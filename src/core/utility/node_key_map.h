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

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sparsehash/dense_hash_map>

namespace zvec::core {

// Optional external-key lookup. Copies share the map with entity clones.
// Identity addressing and node existence are handled by the entity, so an
// identity index allocates no hash table and never performs a hash lookup.
class NodeKeyMap {
 public:
  void set_enabled(bool enabled) {
    enabled_ = enabled;
    if (enabled && !state_) state_ = std::make_shared<State>();
  }

  bool enabled() const {
    return enabled_;
  }

  uint32_t find(uint64_t key) const {
    if (!enabled_ || !state_ || key == UINT64_MAX) return UINT32_MAX;
    std::shared_lock lock(state_->mutex);
    auto it = state_->nodes.find(key);
    return it == state_->nodes.end() ? UINT32_MAX : it->second;
  }

  void insert(uint64_t key, uint32_t id) {
    if (!enabled_ || key == UINT64_MAX) return;
    if (!state_) state_ = std::make_shared<State>();
    std::unique_lock lock(state_->mutex);
    state_->nodes[key] = id;
  }

  void clear() {
    if (!state_) return;
    std::unique_lock lock(state_->mutex);
    state_->nodes.clear();
  }

 private:
  struct State {
    State() {
      nodes.set_empty_key(UINT64_MAX);
    }
    mutable std::shared_mutex mutex;
    google::dense_hash_map<uint64_t, uint32_t> nodes;
  };
  bool enabled_{true};
  std::shared_ptr<State> state_;
};

}  // namespace zvec::core
