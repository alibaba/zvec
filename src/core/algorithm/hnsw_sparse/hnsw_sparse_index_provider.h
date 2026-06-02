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

#include "hnsw_sparse_entity.h"

namespace zvec {
namespace core {

class HnswSparseIndexProvider : public IndexSparseProvider {
 public:
  HnswSparseIndexProvider(const IndexMeta &meta,
                          const HnswSparseEntity::Pointer &entity,
                          const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  HnswSparseIndexProvider(const HnswSparseIndexProvider &) = delete;
  HnswSparseIndexProvider &operator=(const HnswSparseIndexProvider &) = delete;

 public:
  //! Create a new iterator
  IndexSparseProvider::Iterator::Pointer create_iterator(void) override {
    return IndexSparseProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_));
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return entity_->doc_cnt();
  }

  size_t total_sparse_count(void) const override {
    return entity_->total_sparse_count();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type(void) const override {
    return meta_.data_type();
  }

  //! Retrieve a vector using a primary key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override {
    return entity_->get_sparse_vector_by_key(
        key, sparse_count, sparse_indices_buffer, sparse_values_buffer);
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexSparseProvider::Iterator {
   public:
    Iterator(const HnswSparseEntity::Pointer &entity)
        : entity_(entity), cur_id_(0U), valid_(false) {
      const void *sparse_data = entity_->get_sparse_data(cur_id_);
      if (sparse_data != nullptr) {
        valid_ = true;

        sparse_indices_buffer_.clear();
        sparse_data_buffer_.clear();

        SparseUtility::ReverseSparseFormat(
            sparse_data, &sparse_count_, &sparse_indices_buffer_,
            &sparse_data_buffer_, entity_->sparse_unit_size());
      }
    }

    //! Retrieve sparse count
    uint32_t sparse_count() const override {
      return sparse_count_;
    }

    //! Retrieve sparse indices
    const uint32_t *sparse_indices() const override {
      return reinterpret_cast<const uint32_t *>(sparse_indices_buffer_.data());
    }

    //! Retrieve sparse data
    const void *sparse_data() const override {
      return reinterpret_cast<const void *>(sparse_data_buffer_.data());
    }

    //! Test if the iterator is valid
    bool is_valid(void) const override {
      return cur_id_ < entity_->doc_cnt() && valid_;
    }

    //! Retrieve primary key
    uint64_t key(void) const override {
      return entity_->get_key(cur_id_);
    }

    //! Next iterator
    void next(void) override {
      cur_id_ = get_next_valid_id(cur_id_ + 1);

      if (cur_id_ < entity_->doc_cnt()) {
        const void *sparse_data = entity_->get_sparse_data(cur_id_);
        if (sparse_data != nullptr) {
          valid_ = true;

          sparse_indices_buffer_.clear();
          sparse_data_buffer_.clear();

          SparseUtility::ReverseSparseFormat(
              sparse_data, &sparse_count_, &sparse_indices_buffer_,
              &sparse_data_buffer_, entity_->sparse_unit_size());
        } else {
          valid_ = false;
        }
      }
    }

    //! Reset the iterator
    void reset(void) {
      cur_id_ = get_next_valid_id(0);
      const void *sparse_data = entity_->get_sparse_data(cur_id_);
      if (sparse_data != nullptr) {
        valid_ = true;

        SparseUtility::ReverseSparseFormat(
            sparse_data, &sparse_count_, &sparse_indices_buffer_,
            &sparse_data_buffer_, entity_->sparse_unit_size());
      }
    }

   private:
    node_id_t get_next_valid_id(node_id_t start_id) {
      for (node_id_t i = start_id; i < entity_->doc_cnt(); i++) {
        if (entity_->get_key(i) != kInvalidNodeId) {
          return i;
        }
      }
      return kInvalidNodeId;
    }

   private:
    const HnswSparseEntity::Pointer entity_;
    node_id_t cur_id_;
    uint32_t sparse_count_{0};
    std::string sparse_indices_buffer_;
    std::string sparse_data_buffer_;
    bool valid_{false};
  };

 private:
  const IndexMeta &meta_;
  const HnswSparseEntity::Pointer entity_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
