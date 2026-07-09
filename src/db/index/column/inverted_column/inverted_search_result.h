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
    return bitmap_ ? std::make_shared<Filter>(shared_from_this()) : nullptr;
  }


  explicit InvertedSearchResult() {}


  explicit InvertedSearchResult(roaring_bitmap_t *bitmap) : bitmap_(bitmap) {}


  ~InvertedSearchResult() override {
    destroy_bitmap();
  }


  InvertedSearchResult(const InvertedSearchResult &) = delete;
  InvertedSearchResult(InvertedSearchResult &&) = delete;
  InvertedSearchResult &operator=(const InvertedSearchResult &) = delete;
  InvertedSearchResult &operator=(InvertedSearchResult &&) = delete;


  bool contains(uint32_t id) const {
    if (bitmap_) {
      return roaring_bitmap_contains(bitmap_, id);
    } else {
      return false;
    }
  }


  size_t count() const override {
    if (bitmap_) {
      return roaring_bitmap_get_cardinality(bitmap_);
    } else {
      return 0;
    }
  }


  class InvertedIndexIterator : public Iterator {
   public:
    explicit InvertedIndexIterator(
        std::shared_ptr<const InvertedSearchResult> result)
        : result_(result) {
      if (result_->bitmap_) {
        iter_ = roaring_create_iterator(result_->bitmap_);
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
    if (!bitmap_) {
      return;
    }

    ids->reserve(static_cast<size_t>(count()));
    roaring_uint32_iterator_t *iter = roaring_create_iterator(bitmap_);
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


  void destroy_bitmap() {
    if (bitmap_) {
      roaring_bitmap_free(bitmap_);
      bitmap_ = nullptr;
    }
  }


  void AND(const InvertedSearchResult &other) {
    if (!bitmap_ || !other.bitmap_) {
      destroy_bitmap();
    } else {
      roaring_bitmap_and_inplace(bitmap_, other.bitmap_);
    }
  }


  void OR(const InvertedSearchResult &other) {
    if (!other.bitmap_) {
      return;
    }
    if (!bitmap_) {
      bitmap_ = roaring_bitmap_copy(other.bitmap_);
      return;
    }
    roaring_bitmap_or_inplace(bitmap_, other.bitmap_);
  }


 private:
  roaring_bitmap_t *bitmap_{nullptr};
};


}  // namespace zvec