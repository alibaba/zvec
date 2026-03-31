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

#include "zvec/core/framework/index_provider.h"
#include "zvec/core/framework/index_searcher.h"
#include "zvec/core/framework/index_streamer.h"
#include "hnsw_rabitq_entity.h"

namespace zvec {
namespace core {

class HnswRabitqIndexProvider : public IndexProvider {
 public:
  HnswRabitqIndexProvider(const IndexMeta &meta,
                          const HnswRabitqEntity::Pointer &entity,
                          const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  HnswRabitqIndexProvider(const HnswRabitqIndexProvider &) = delete;
  HnswRabitqIndexProvider &operator=(const HnswRabitqIndexProvider &) = delete;

 public:  // holder interface
  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override {
    return HnswRabitqIndexProvider::Iterator::Pointer(new (std::nothrow)
                                                          Iterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return entity_->doc_cnt();
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

 public:  // provider's unique interface
  //! Retrieve a vector using a primary key
  const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

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
    Iterator(const HnswRabitqEntity::Pointer &entity)
        : entity_(entity), cur_id_(0U) {}

    //! Retrieve pointer of data
    //! NOTICE: the vec feature will be changed after iterating to next, so
    //! the caller need to keep a copy of it before iterator to next vector
    virtual const void *data(void) const override {
      return entity_->get_vector(cur_id_);
    }

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return cur_id_ < entity_->doc_cnt();
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      return entity_->get_key(cur_id_);
    }

    //! Next iterator
    virtual void next(void) override {
      // cur_id_ += 1;
      cur_id_ = get_next_valid_id(cur_id_ + 1);
    }

    //! Reset the iterator
    void reset(void) {
      cur_id_ = get_next_valid_id(0);
    }

   private:
    node_id_t get_next_valid_id(node_id_t start_id) {
      for (node_id_t i = start_id; i < entity_->doc_cnt(); i++) {
        if (entity_->get_key(i) != kInvalidNodeId) {
          cur_id_ = i;
          return i;
        }
      }
      return kInvalidNodeId;
    }

   private:
    const HnswRabitqEntity::Pointer entity_;
    node_id_t cur_id_;
  };

 private:
  const IndexMeta &meta_;
  const HnswRabitqEntity::Pointer entity_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
