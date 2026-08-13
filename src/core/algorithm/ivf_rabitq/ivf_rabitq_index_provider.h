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
  Iterator::Pointer create_iterator(void) override {
    return Iterator::Pointer(new (std::nothrow) Iterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return entity_->total_vector_count();
  }

  //! Retrieve dimension of vector
  size_t dimension(void) const override {
    return entity_->padded_dim();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type(void) const override {
    return IndexMeta::DataType::DT_UNDEFINED;
  }

  //! Retrieve vector size in bytes
  size_t element_size(void) const override {
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
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    explicit Iterator(const IvfRabitqEntity::Pointer &entity)
        : entity_(entity) {}

    //! Retrieve pointer of data
    const void *data(void) const override {
      return nullptr;
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return id_ < entity_->total_vector_count();
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return entity_->get_key(id_);
    }

    //! Next iterator
    void next(void) override {
      ++id_;
    }

   private:
    IvfRabitqEntity::Pointer entity_;
    size_t id_{0};
  };

 private:
  IvfRabitqEntity::Pointer entity_;
  std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
