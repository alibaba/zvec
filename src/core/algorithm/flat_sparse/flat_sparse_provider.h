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

#include <string>
#include <utility/sparse_utility.h>
#include <zvec/core/framework/index_logger.h>
#include <zvec/core/framework/index_meta.h>
#include "flat_sparse_streamer_entity.h"

namespace zvec {
namespace core {

/*! Brute Force Sparse Streamer Provider
 */
// FlatSparseStreamerEntity or FlatSparseSearcherEntity
template <typename FlatSparseEntityType>
class FlatSparseIndexProvider : public IndexSparseProvider {
 public:
  //! Constructor
  FlatSparseIndexProvider(const std::shared_ptr<FlatSparseEntityType> entity,
                          const IndexMeta &meta, const std::string &owner)
      : entity_(entity), meta_(meta), owner_class_(owner) {}

  //! Create a new iterator
  IndexSparseProvider::Iterator::Pointer create_iterator(void) override {
    return IndexSparseProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_, meta_));
  }

  //! Retrieve count of vectors
  size_t count(void) const override {
    return entity_->doc_cnt();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type(void) const override {
    return meta_.data_type();
  }

  //! Retrieve a vector using a primary key
  int get_sparse_vector(uint64_t key, uint32_t *sparse_count,
                        std::string *sparse_indices_buffer,
                        std::string *sparse_values_buffer) const override {
    std::string sparse_data;

    int ret = entity_->get_sparse_vector_by_key(key, &sparse_data);
    if (ailego_unlikely(ret != 0)) {
      LOG_ERROR("Failed to get sparse vector, key=%zu, ret=%s", (size_t)key,
                IndexError::What(ret));
      return ret;
    }

    SparseUtility::ReverseSparseFormat(sparse_data, sparse_count,
                                       sparse_indices_buffer,
                                       sparse_values_buffer, meta_.unit_size());
    return 0;
  }

  //! Retrieve the owner class
  const std::string &owner_class(void) const override {
    return owner_class_;
  }

  size_t total_sparse_count() const override {
    return entity_->total_sparse_count();
  }

 private:
  class Iterator : public IndexSparseProvider::Iterator {
   public:
    Iterator(const std::shared_ptr<FlatSparseEntityType> &entity,
             const IndexMeta &meta)
        : entity_(entity), meta_(meta), cur_id_(0U), valid_(false) {
      IndexStorage::MemoryBlock sparse_data_block;
      entity_->get_sparse_vector(cur_id_, sparse_data_block);
      const void *sparse_data = sparse_data_block.data();
      if (sparse_data != nullptr) {
        valid_ = true;

        sparse_indices_buffer_.clear();
        sparse_data_buffer_.clear();

        SparseUtility::ReverseSparseFormat(
            sparse_data, &sparse_count_, &sparse_indices_buffer_,
            &sparse_data_buffer_, meta.unit_size());
      }
    }

    //! Retrieve sparse count
    virtual uint32_t sparse_count() const override {
      return sparse_count_;
    }

    //! Retrieve sparse indices
    virtual const uint32_t *sparse_indices() const override {
      return reinterpret_cast<const uint32_t *>(sparse_indices_buffer_.data());
    }

    //! Retrieve sparse data
    virtual const void *sparse_data() const override {
      return reinterpret_cast<const void *>(sparse_data_buffer_.data());
    }

    //! Test if the iterator is valid
    virtual bool is_valid(void) const override {
      return cur_id_ < entity_->doc_cnt() && valid_;
    }

    //! Retrieve primary key
    virtual uint64_t key(void) const override {
      // std::cout << "iter key=" << cur_id_ << std::endl;
      return entity_->get_key(cur_id_);
    }

    //! Next iterator
    virtual void next(void) override {
      cur_id_ = get_next_valid_id(cur_id_ + 1);

      if (cur_id_ < entity_->doc_cnt()) {
        IndexStorage::MemoryBlock sparse_data_block;
        entity_->get_sparse_vector(cur_id_, sparse_data_block);
        const void *sparse_data = sparse_data_block.data();
        if (sparse_data != nullptr) {
          valid_ = true;

          sparse_indices_buffer_.clear();
          sparse_data_buffer_.clear();

          SparseUtility::ReverseSparseFormat(
              sparse_data, &sparse_count_, &sparse_indices_buffer_,
              &sparse_data_buffer_, meta_.unit_size());
        } else {
          valid_ = false;
        }
      }
    }

    //! Reset the iterator
    void reset(void) {
      cur_id_ = get_next_valid_id(0);
      IndexStorage::MemoryBlock sparse_data_block;
      entity_->get_sparse_vector(cur_id_, sparse_data_block);
      const void *sparse_data = sparse_data_block.data();
      if (sparse_data != nullptr) {
        valid_ = true;

        SparseUtility::ReverseSparseFormat(
            sparse_data, &sparse_count_, &sparse_indices_buffer_,
            &sparse_data_buffer_, meta_.unit_size());
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
    const std::shared_ptr<FlatSparseEntityType> entity_{nullptr};
    const IndexMeta &meta_;
    node_id_t cur_id_;
    uint32_t sparse_count_;
    std::string sparse_indices_buffer_;
    std::string sparse_data_buffer_;
    bool valid_{false};
  };

 private:
  const std::shared_ptr<FlatSparseEntityType> entity_{nullptr};
  const IndexMeta &meta_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
