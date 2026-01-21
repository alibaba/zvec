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
  virtual Iterator::Pointer create_iterator(void) override {
    return Iterator::Pointer(new (std::nothrow) Iterator(entity_));
  }

  //! Retrieve count of vectors
  virtual size_t count(void) const override {
    return entity_->vector_count();
  }

  //! Retrieve dimension of vector
  virtual size_t dimension(void) const override {
    return meta_.dimension();
  }

  //! Retrieve type of vector
  virtual IndexMeta::DataType data_type(void) const override {
    return meta_.data_type();
  }

  //! Retrieve vector size in bytes
  virtual size_t element_size(void) const override {
    return meta_.element_size();
  }

  //! Retrieve a vector using a primary key
  virtual const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  //! Retrieve the owner class
  virtual const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const IVFEntity::Pointer &entity) : entity_(entity) {}

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    virtual const void *data(void) const override {
      return entity_->get_vector(index_);
    }

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return index_ < entity_->vector_count();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return entity_->get_key(index_);
    }

    //! Next iterator
    virtual void next(void) override {
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
