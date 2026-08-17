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
#include <zvec/core/framework/index_holder.h>

namespace zvec {
namespace core {

// A bounded view, preserving the source's iteration order and pointer lifetime.
// Training must copy each selected vector before advancing its iterator.
class PrefixIndexHolder : public IndexHolder {
 public:
  PrefixIndexHolder(IndexHolder::Pointer source, size_t limit, IndexMeta meta)
      : source_(std::move(source)),
        count_(std::min(source_->count(), limit)),
        meta_(std::move(meta)) {}

  size_t count() const override {
    return count_;
  }
  size_t dimension() const override {
    return meta_.dimension();
  }
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }
  size_t element_size() const override {
    return meta_.element_size();
  }
  bool multipass() const override {
    return source_->multipass();
  }

  IndexHolder::Iterator::Pointer create_iterator() override {
    auto iter = source_->create_iterator();
    if (!iter) return nullptr;
    return std::make_unique<Iterator>(source_, std::move(iter), count_);
  }

 private:
  class Iterator : public IndexHolder::Iterator {
   public:
    Iterator(IndexHolder::Pointer source, IndexHolder::Iterator::Pointer iter,
             size_t count)
        : source_(std::move(source)),
          iter_(std::move(iter)),
          remaining_(count) {}
    const void *data() const override {
      return iter_->data();
    }
    bool is_valid() const override {
      return remaining_ && iter_->is_valid();
    }
    uint64_t key() const override {
      return iter_->key();
    }
    void next() override {
      if (remaining_ && --remaining_) iter_->next();
    }

   private:
    IndexHolder::Pointer source_;
    IndexHolder::Iterator::Pointer iter_;
    size_t remaining_;
  };
  IndexHolder::Pointer source_;
  size_t count_;
  IndexMeta meta_;
};

}  // namespace core
}  // namespace zvec
