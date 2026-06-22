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
#include <numeric>
#include <vector>
#include <zvec/core/framework/index_searcher.h>
#include "ivf_entity.h"

namespace zvec {
namespace core {

/*! IVF IndexProvider
 */
class IVFIndexProvider : public IndexProvider {
 public:
  IVFIndexProvider(const IndexMeta &meta, const IVFEntity::Pointer &entity,
                   const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  IVFIndexProvider(const IVFIndexProvider &) = delete;
  IVFIndexProvider &operator=(const IVFIndexProvider &) = delete;

 public:
  //! Create a new iterator
  Iterator::Pointer create_iterator(void) override {
    return Iterator::Pointer(new (std::nothrow) SortedIterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return entity_->vector_count();
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const override {
    return meta_.dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type(void) const override {
    return meta_.data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size(void) const override {
    return meta_.element_size();
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  class SortedIterator : public IndexProvider::Iterator {
   public:
    SortedIterator(const IVFEntity::Pointer &entity) : entity_(entity) {
      count_ = entity_->vector_count();
      mapping_ = entity_->get_key_order_mapping();
      if (!mapping_) {
        // Fallback: compute sorting if mapping segment is unavailable
        fallback_.resize(count_);
        std::iota(fallback_.begin(), fallback_.end(), size_t(0));
        std::sort(fallback_.begin(), fallback_.end(), [&](size_t a, size_t b) {
          return entity_->get_key(a) < entity_->get_key(b);
        });
      }
    }

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    const void *data(void) const override {
      return entity_->get_vector(current_local_id());
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return pos_ < count_;
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return entity_->get_key(current_local_id());
    }

    //! Next iterator
    void next(void) override {
      ++pos_;
    }

   private:
    size_t current_local_id() const {
      return mapping_ ? static_cast<size_t>(mapping_[pos_]) : fallback_[pos_];
    }

    //! Members
    IVFEntity::Pointer entity_;
    const uint32_t *mapping_{nullptr};  // points into mapping_ segment data
    std::vector<size_t> fallback_;      // used only if mapping_ unavailable
    size_t count_{0};
    size_t pos_{0};
  };

  //! Original sequential iterator (kept for potential internal use)
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const IVFEntity::Pointer &entity) : entity_(entity) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return entity_->get_vector(index_);
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return index_ < entity_->vector_count();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return entity_->get_key(index_);
    }

    //! Next iterator
    void next(void) override {
      ++index_;
    }

   private:
    //! Members
    IVFEntity::Pointer entity_;
    size_t index_{0};
  };

 private:
  //! Members
  const IndexMeta &meta_;
  IVFEntity::Pointer entity_;
  std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
