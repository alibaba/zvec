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

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_holder.h>
#include "stable_vector_holder.h"

namespace zvec {
namespace core {

// An immutable, contiguous snapshot for sources whose iterator data is only
// valid until the next read. Copy each record before advancing the source.
class OwnedVectorHolder final : public IndexHolder, public StableVectorHolder {
 public:
  using Pointer = std::shared_ptr<OwnedVectorHolder>;

  static int CopyFrom(const IndexHolder::Pointer &source, Pointer *output) {
    if (!source || !output || source->element_size() == 0) {
      return IndexError_InvalidArgument;
    }
    const size_t count = source->count();
    const size_t bytes = source->element_size();
    if (count > std::numeric_limits<size_t>::max() / bytes) {
      return IndexError_NoMemory;
    }
    auto snapshot =
        std::shared_ptr<OwnedVectorHolder>(new OwnedVectorHolder(*source));
    auto &storage = *snapshot->storage_;
    storage.values.reserve(count * bytes);
    storage.keys.reserve(count);
    auto iter = source->create_iterator();
    if (!iter) return IndexError_Runtime;
    for (; iter->is_valid(); iter->next()) {
      if (iter->status() != 0) return iter->status();
      if (storage.keys.size() >= count) return IndexError_Mismatch;
      const uint64_t key = iter->key();
      if (iter->status() != 0) return iter->status();
      const void *data = iter->data();
      if (iter->status() != 0) return iter->status();
      if (!data) return IndexError_ReadData;
      storage.values.append(static_cast<const char *>(data), bytes);
      storage.keys.push_back(key);
    }
    if (iter->status() != 0) return iter->status();
    if (storage.keys.size() != count) return IndexError_Mismatch;
    *output = std::move(snapshot);
    return 0;
  }

  size_t count() const override {
    return storage_->keys.size();
  }
  size_t dimension() const override {
    return dimension_;
  }
  IndexMeta::DataType data_type() const override {
    return data_type_;
  }
  size_t element_size() const override {
    return element_size_;
  }
  bool multipass() const override {
    return true;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    return std::make_unique<Iterator>(storage_, element_size_);
  }
  int retain_stable_data(std::shared_ptr<const void> *lifetime) override {
    if (!lifetime) return IndexError_InvalidArgument;
    *lifetime = storage_;
    return 0;
  }

 private:
  struct Storage {
    std::string values;
    std::vector<uint64_t> keys;
  };

  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(std::shared_ptr<const Storage> storage, size_t element_size)
        : storage_(std::move(storage)), element_size_(element_size) {}
    const void *data() const override {
      return storage_->values.data() + position_ * element_size_;
    }
    bool is_valid() const override {
      return position_ < storage_->keys.size();
    }
    uint64_t key() const override {
      return storage_->keys[position_];
    }
    void next() override {
      ++position_;
    }

   private:
    std::shared_ptr<const Storage> storage_;
    size_t element_size_;
    size_t position_{0};
  };

  explicit OwnedVectorHolder(const IndexHolder &source)
      : dimension_(source.dimension()),
        data_type_(source.data_type()),
        element_size_(source.element_size()) {}

  size_t dimension_;
  IndexMeta::DataType data_type_;
  size_t element_size_;
  std::shared_ptr<Storage> storage_{std::make_shared<Storage>()};
};

}  // namespace core
}  // namespace zvec
