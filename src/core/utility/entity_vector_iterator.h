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
#include <zvec/core/framework/index_provider.h>

namespace zvec::core {

template <typename Entity>
class EntityVectorIterator : public IndexProvider::Iterator {
 public:
  EntityVectorIterator(const std::shared_ptr<Entity> &entity,
                       bool include_empty = false)
      : entity_(entity), include_empty_(include_empty) {
    cur_id_ = seek(0);
  }

  const void *data() const override {
    if (!is_valid() || key_ == UINT64_MAX) return nullptr;
    status_ = entity_->get_vector(cur_id_, block_);
    if (status_ == 0 && !block_.data()) status_ = IndexError_ReadData;
    return status_ == 0 ? block_.data() : nullptr;
  }

  bool is_valid() const override {
    return status_ == 0 && cur_id_ < entity_->slot_count();
  }
  uint64_t key() const override {
    return key_;
  }
  int status() const override {
    return status_;
  }
  void next() override {
    if (is_valid()) cur_id_ = seek(cur_id_ + 1);
  }
  void reset() {
    status_ = 0;
    cur_id_ = seek(0);
  }

 private:
  uint32_t seek(uint32_t start_id) {
    block_.reset();
    key_ = UINT64_MAX;
    for (uint32_t id = start_id; id < entity_->slot_count(); ++id) {
      status_ = entity_->read_key(id, &key_);
      if (status_ != 0) return UINT32_MAX;
      if (include_empty_ || key_ != UINT64_MAX) return id;
    }
    return UINT32_MAX;
  }

  const std::shared_ptr<Entity> entity_;
  bool include_empty_{false};
  uint32_t cur_id_{UINT32_MAX};
  uint64_t key_{UINT64_MAX};
  mutable int status_{0};
  mutable IndexStorage::MemoryBlock block_;
};

}  // namespace zvec::core
