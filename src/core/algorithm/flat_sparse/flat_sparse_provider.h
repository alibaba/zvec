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

#include <mutex>
#include <string>
#include <utility/sparse_utility.h>
#include <zvec/ailego/logger/logger.h>
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
  IndexSparseProvider::Iterator::Pointer create_iterator() override {
    return IndexSparseProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_, meta_));
  }

  //! Retrieve count of vectors
  IndexSparseProvider::Iterator::Pointer create_slot_iterator() override {
    return std::make_unique<Iterator>(entity_, meta_, true);
  }

  size_t count() const override {
    std::call_once(count_once_, [&] {
      for (uint32_t id = 0; id < entity_->doc_cnt(); ++id)
        if (entity_->contains_node(id)) ++vector_count_;
    });
    return vector_count_;
  }

  size_t slot_count() const override {
    return entity_->doc_cnt();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
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
  const std::string &owner_class() const override {
    return owner_class_;
  }

  size_t total_sparse_count() const override {
    return entity_->total_sparse_count();
  }

 private:
  class Iterator : public IndexSparseProvider::Iterator {
   public:
    Iterator(const std::shared_ptr<FlatSparseEntityType> &entity,
             const IndexMeta &meta, bool include_empty = false)
        : include_empty_(include_empty), entity_(entity), meta_(meta) {
      seek(0);
    }

    uint32_t sparse_count() const override {
      return sparse_count_;
    }

    const uint32_t *sparse_indices() const override {
      return reinterpret_cast<const uint32_t *>(sparse_indices_buffer_.data());
    }

    const void *sparse_data() const override {
      return sparse_data_buffer_.data();
    }

    bool is_valid() const override {
      return status_ == 0 && cur_id_ < entity_->doc_cnt();
    }

    int status() const override {
      return status_;
    }

    uint64_t key() const override {
      return key_;
    }

    void next() override {
      if (is_valid()) seek(cur_id_ + 1);
    }

    void reset() {
      status_ = 0;
      seek(0);
    }

   private:
    void seek(node_id_t start_id) {
      sparse_count_ = 0;
      sparse_indices_buffer_.clear();
      sparse_data_buffer_.clear();
      key_ = kInvalidKey;
      for (cur_id_ = start_id; cur_id_ < entity_->doc_cnt(); ++cur_id_) {
        status_ = entity_->read_key(cur_id_, &key_);
        if (status_ != 0) return;
        if (key_ == kInvalidKey) {
          if (include_empty_) return;
          continue;
        }
        IndexStorage::MemoryBlock block;
        status_ = entity_->get_sparse_vector(cur_id_, block);
        if (status_ != 0) return;
        const void *data = block.data();
        if (!data) {
          status_ = IndexError_ReadData;
          return;
        }
        SparseUtility::ReverseSparseFormat(
            data, &sparse_count_, &sparse_indices_buffer_, &sparse_data_buffer_,
            meta_.unit_size());
        return;
      }
    }

    bool include_empty_{false};
    const std::shared_ptr<FlatSparseEntityType> entity_;
    const IndexMeta &meta_;
    node_id_t cur_id_{0};
    uint64_t key_{kInvalidKey};
    int status_{0};
    uint32_t sparse_count_{0};
    std::string sparse_indices_buffer_;
    std::string sparse_data_buffer_;
  };

 private:
  mutable std::once_flag count_once_;
  mutable size_t vector_count_{0};
  const std::shared_ptr<FlatSparseEntityType> entity_{nullptr};
  const IndexMeta &meta_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
