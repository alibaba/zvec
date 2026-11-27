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


#include <new>
#include <roaring/roaring.h>
#include "db/common/constants.h"
#include "db/index/column/common/index_results.h"


namespace zvec {


class InvertedSearchResult
    : public IndexResults,
      public std::enable_shared_from_this<InvertedSearchResult> {
 public:
  using Ptr = std::shared_ptr<InvertedSearchResult>;


  class Filter : public IndexFilter {
   public:
    explicit Filter(std::shared_ptr<const InvertedSearchResult> result)
        : result_(std::move(result)) {};

    bool is_filtered(uint64_t id) const override {
      return !result_->contains(id);
    }

   private:
    const std::shared_ptr<const InvertedSearchResult> result_{};
  };


  const IndexFilter::Ptr make_filter() const {
    return bitmap() ? std::make_shared<Filter>(shared_from_this()) : nullptr;
  }


  explicit InvertedSearchResult() = default;


  explicit InvertedSearchResult(roaring_bitmap_t *bitmap) : bitmap_(bitmap) {}


  ~InvertedSearchResult() override {
    destroy_bitmap();
  }


  InvertedSearchResult(const InvertedSearchResult &) = delete;
  InvertedSearchResult(InvertedSearchResult &&) = delete;
  InvertedSearchResult &operator=(const InvertedSearchResult &) = delete;
  InvertedSearchResult &operator=(InvertedSearchResult &&) = delete;


  bool contains(uint32_t id) const {
    const auto *data = bitmap();
    return data && roaring_bitmap_contains(data, id);
  }


  size_t count() const override {
    const auto *data = bitmap();
    return data ? roaring_bitmap_get_cardinality(data) : 0;
  }


  class InvertedIndexIterator : public Iterator {
   public:
    explicit InvertedIndexIterator(
        std::shared_ptr<const InvertedSearchResult> result)
        : result_(result) {
      if (const auto *data = result_->bitmap()) {
        iter_ = roaring_create_iterator(data);
      }
    }

    ~InvertedIndexIterator() override {
      if (iter_) {
        roaring_free_uint32_iterator(iter_);
      }
    }

    idx_t doc_id() const override {
      if (!iter_) {
        return INVALID_DOC_ID;
      }
      if (iter_->has_value) {
        return iter_->current_value;
      } else {
        return INVALID_DOC_ID;
      }
    }

    float score() const override {
      return 0.0f;
    }

    void next() override {
      if (iter_ && iter_->has_value) {
        roaring_advance_uint32_iterator(iter_);
      }
    }

    bool valid() const override {
      return iter_ ? iter_->has_value : false;
    }

   private:
    const std::shared_ptr<const InvertedSearchResult> result_{};
    roaring_uint32_iterator_t *iter_{nullptr};
  };


  IteratorUPtr create_iterator() override {
    return std::make_unique<InvertedIndexIterator>(shared_from_this());
  }


  void extract_ids(std::vector<uint32_t> *ids) const {
    if (!ids) {
      LOG_ERROR("Failed to extract ids: ids pointer is null");
      return;
    }
    const auto *data = bitmap();
    if (!data) {
      return;
    }

    ids->reserve(static_cast<size_t>(count()));
    roaring_uint32_iterator_t *iter = roaring_create_iterator(data);
    while (iter->has_value) {
      ids->push_back(iter->current_value);
      roaring_advance_uint32_iterator(iter);
    }
    roaring_free_uint32_iterator(iter);
  }


  void set_and_own_bitmap(roaring_bitmap_t *bitmap) {
    destroy_bitmap();
    bitmap_ = bitmap;
  }


  void set_shared_bitmap(std::shared_ptr<const roaring_bitmap_t> bitmap) {
    destroy_bitmap();
    shared_bitmap_ = std::move(bitmap);
  }


  void destroy_bitmap() {
    if (bitmap_) {
      roaring_bitmap_free(bitmap_);
      bitmap_ = nullptr;
    }
    shared_bitmap_.reset();
  }


  // Set-algebra op; 'and' is a C++ keyword so the method keeps an upper name.
  // NOLINTNEXTLINE(readability-identifier-naming)
  void AND(const InvertedSearchResult &other) {
    const auto *right = other.bitmap();
    if (!bitmap() || !right) {
      destroy_bitmap();
    } else if (this != &other) {
      roaring_bitmap_and_inplace(mutable_bitmap(), right);
    }
  }


  // Set-algebra op; 'or' is a C++ keyword so the method keeps an upper name.
  // NOLINTNEXTLINE(readability-identifier-naming)
  void OR(const InvertedSearchResult &other) {
    const auto *right = other.bitmap();
    if (!right || this == &other) {
      return;
    }
    if (!bitmap()) {
      bitmap_ = roaring_bitmap_copy(right);
      return;
    }
    roaring_bitmap_or_inplace(mutable_bitmap(), right);
  }


 private:
  const roaring_bitmap_t *bitmap() const {
    return shared_bitmap_ ? shared_bitmap_.get() : bitmap_;
  }

  roaring_bitmap_t *mutable_bitmap() {
    if (shared_bitmap_) {
      auto *copy = roaring_bitmap_copy(shared_bitmap_.get());
      if (!copy) {
        throw std::bad_alloc();
      }
      set_and_own_bitmap(copy);
    }
    return bitmap_;
  }

  roaring_bitmap_t *bitmap_{nullptr};
  std::shared_ptr<const roaring_bitmap_t> shared_bitmap_;
};


}  // namespace zvec
