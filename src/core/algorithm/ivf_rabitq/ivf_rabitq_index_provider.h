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

#include <memory>
#include <new>
#include <string>
#include <utility>
#include "zvec/core/framework/index_provider.h"
#include "ivf_rabitq_entity.h"

namespace zvec {
namespace core {

/*! IVF RaBitQ IndexProvider
 */
class IvfRabitqIndexProvider : public IndexProvider {
 public:
  IvfRabitqIndexProvider(const IndexMeta &, IvfRabitqEntity::Pointer entity,
                         const std::string &owner)
      : entity_(std::move(entity)), owner_class_(owner) {}

  IvfRabitqIndexProvider(const IvfRabitqIndexProvider &) = delete;
  IvfRabitqIndexProvider &operator=(const IvfRabitqIndexProvider &) = delete;

  //! Create a new iterator
  Iterator::Pointer create_iterator() override {
    return Iterator::Pointer(new (std::nothrow) Iterator(entity_));
  }

  Iterator::Pointer create_slot_iterator() override {
    return std::make_unique<Iterator>(entity_, true);
  }
  size_t slot_count() const override {
    return entity_->slot_count();
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return entity_->total_vector_count();
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return entity_->padded_dim();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
    return 0;
  }

  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  //! Retrieve a vector using a primary key
  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    return entity_->get_vector_by_key(key, block);
  }

  //! Retrieve the owner class
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    explicit Iterator(const IvfRabitqEntity::Pointer &entity,
                      bool include_empty = false)
        : entity_(entity), include_empty_(include_empty) {}

    //! Retrieve pointer of data
    const void *data() const override {
      return nullptr;
    }

    //! Test if the iterator is valid
    bool is_valid() const override {
      return status_ == 0 &&
             id_ < (include_empty_ ? entity_->slot_count()
                                   : entity_->total_vector_count());
    }

    //! Retrieve primary key
    uint64_t key() const override {
      uint32_t local_id = id_;
      if (include_empty_) {
        status_ = entity_->slot_to_id(id_, &local_id);
        if (status_ != 0 || local_id == UINT32_MAX) return UINT64_MAX;
      }
      return entity_->get_key(local_id);
    }

    int status() const override {
      return status_;
    }

    //! Next iterator
    void next() override {
      ++id_;
    }

   private:
    IvfRabitqEntity::Pointer entity_;
    size_t id_{0};
    bool include_empty_{false};
    mutable int status_{0};
  };

 private:
  IvfRabitqEntity::Pointer entity_;
  std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
