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
#include <zvec/core/framework/index_provider.h>
#include "utility/sparse_utility.h"
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
  IndexSparseProvider::Iterator::Pointer create_iterator() override {
    return IndexSparseProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_));
  }

  //! Retrieve count of vectors
  IndexSparseProvider::Iterator::Pointer create_slot_iterator() override {
    return std::make_unique<Iterator>(entity_, true);
  }

  size_t count() const override {
    std::call_once(count_once_, [&] {
      for (uint32_t id = 0; id < entity_->slot_count(); ++id)
        if (entity_->contains_node(id)) ++vector_count_;
    });
    return vector_count_;
  }

  size_t slot_count() const override {
    return entity_->slot_count();
  }

  size_t total_sparse_count() const override {
    return entity_->total_sparse_count();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
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
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexSparseProvider::Iterator {
   public:
    Iterator(const HnswSparseEntity::Pointer &entity,
             bool include_empty = false)
        : include_empty_(include_empty), entity_(entity) {
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
      return status_ == 0 && cur_id_ < entity_->slot_count();
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
      for (cur_id_ = start_id; cur_id_ < entity_->slot_count(); ++cur_id_) {
        status_ = entity_->read_key(cur_id_, &key_);
        if (status_ != 0) return;
        if (key_ == kInvalidKey) {
          if (include_empty_) return;
          continue;
        }
        const void *data = entity_->get_sparse_data(cur_id_);
        if (!data) {
          status_ = IndexError_ReadData;
          return;
        }
        SparseUtility::ReverseSparseFormat(
            data, &sparse_count_, &sparse_indices_buffer_, &sparse_data_buffer_,
            entity_->sparse_unit_size());
        return;
      }
    }

    bool include_empty_{false};
    const HnswSparseEntity::Pointer entity_;
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
  const IndexMeta &meta_;
  const HnswSparseEntity::Pointer entity_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
