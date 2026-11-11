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
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>
#include "vamana_entity.h"

namespace zvec {
namespace core {

class VamanaIndexProvider : public IndexProvider {
 public:
  VamanaIndexProvider(const IndexMeta &meta,
                      const VamanaEntity::Pointer &entity,
                      const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  VamanaIndexProvider(const VamanaIndexProvider &) = delete;
  VamanaIndexProvider &operator=(const VamanaIndexProvider &) = delete;

 public:
  IndexProvider::Iterator::Pointer create_iterator() override {
    return VamanaIndexProvider::Iterator::Pointer(new (std::nothrow)
                                                      Iterator(entity_));
  }

  IndexProvider::Iterator::Pointer create_slot_iterator() override {
    return std::make_unique<Iterator>(entity_, true);
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

  size_t dimension() const override {
    return meta_.dimension();
  }

  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  size_t element_size() const override {
    return meta_.element_size();
  }

  const void *get_vector(uint64_t key) const override {
    return entity_->get_vector_by_key(key);
  }

  int get_vector(const uint64_t key,
                 IndexStorage::MemoryBlock &block) const override {
    return entity_->get_vector_by_key(key, block);
  }

  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const VamanaEntity::Pointer &entity, bool include_empty = false)
        : include_empty_(include_empty), entity_(entity), cur_id_(0U) {
      cur_id_ = get_next_valid_id(0);
    }

    const void *data() const override {
      if (!is_valid() || key_ == kInvalidKey) return nullptr;
      status_ = entity_->get_vector(cur_id_, block_);
      if (status_ == 0 && !block_.data()) status_ = IndexError_ReadData;
      return status_ == 0 ? block_.data() : nullptr;
    }

    bool is_valid() const override {
      return status_ == 0 && cur_id_ < entity_->doc_cnt();
    }

    uint64_t key() const override {
      return key_;
    }

    int status() const override {
      return status_;
    }

    //! Next iterator
    void next() override {
      if (!is_valid()) return;
      cur_id_ = get_next_valid_id(cur_id_ + 1);
    }

    void reset() {
      status_ = 0;
      cur_id_ = get_next_valid_id(0);
    }

   private:
    node_id_t get_next_valid_id(node_id_t start_id) {
      block_.reset();
      key_ = kInvalidKey;
      for (node_id_t id = start_id; id < entity_->doc_cnt(); ++id) {
        status_ = entity_->read_key(id, &key_);
        if (status_ != 0) return kInvalidNodeId;
        if (include_empty_ || key_ != kInvalidKey) return id;
      }
      return kInvalidNodeId;
    }

    bool include_empty_{false};
    const VamanaEntity::Pointer entity_;
    node_id_t cur_id_;
    key_t key_{kInvalidKey};
    mutable int status_{0};
    mutable IndexStorage::MemoryBlock block_;
  };

  mutable std::once_flag count_once_;
  mutable size_t vector_count_{0};
  const IndexMeta &meta_;
  const VamanaEntity::Pointer entity_;
  const std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
