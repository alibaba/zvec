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

#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>
#include "diskann_entity.h"

namespace zvec {
namespace core {

//! IndexProvider implementation backed by a DiskAnn entity.
//!
//! Used by ``MixedStreamerReducer`` during segment merge: the reducer needs
//! to walk every vector held by a source DiskAnn streamer and feed it into
//! the merge target. Vectors are read on demand from the entity's on-disk
//! vector segment via ``DiskAnnEntity::get_vector(id)``.
class DiskAnnIndexProvider : public IndexProvider {
 public:
  DiskAnnIndexProvider(const IndexMeta &meta,
                       const DiskAnnEntity::Pointer &entity,
                       const std::string &owner)
      : meta_(meta), entity_(entity), owner_class_(owner) {}

  DiskAnnIndexProvider(const DiskAnnIndexProvider &) = delete;
  DiskAnnIndexProvider &operator=(const DiskAnnIndexProvider &) = delete;

 public:
  IndexProvider::Iterator::Pointer create_iterator() override {
    return IndexProvider::Iterator::Pointer(new (std::nothrow)
                                                Iterator(entity_));
  }

  size_t count(void) const override {
    return entity_->doc_cnt();
  }

  size_t dimension(void) const override {
    return meta_.dimension();
  }

  IndexMeta::DataType data_type(void) const override {
    return meta_.data_type();
  }

  size_t element_size(void) const override {
    return meta_.element_size();
  }

  const void *get_vector(uint64_t key) const override {
    diskann_id_t id = entity_->get_id(static_cast<diskann_key_t>(key));
    if (id == kInvalidId) {
      return nullptr;
    }
    return entity_->get_vector(id);
  }

  const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  class Iterator : public IndexProvider::Iterator {
   public:
    explicit Iterator(const DiskAnnEntity::Pointer &entity)
        : entity_(entity), cur_id_(0U) {
      cur_id_ = next_valid_id(0U);
    }

    const void *data(void) const override {
      return entity_->get_vector(cur_id_);
    }

    bool is_valid(void) const override {
      return cur_id_ < static_cast<diskann_id_t>(entity_->doc_cnt());
    }

    uint64_t key(void) const override {
      return static_cast<uint64_t>(entity_->get_key(cur_id_));
    }

    void next(void) override {
      cur_id_ = next_valid_id(cur_id_ + 1);
    }

   private:
    //! Skip ids that map to ``kInvalidKey`` (deleted / never populated slots).
    diskann_id_t next_valid_id(diskann_id_t start_id) const {
      const auto total = static_cast<diskann_id_t>(entity_->doc_cnt());
      for (diskann_id_t i = start_id; i < total; ++i) {
        if (entity_->get_key(i) != kInvalidKey) {
          return i;
        }
      }
      return total;
    }

    DiskAnnEntity::Pointer entity_;
    diskann_id_t cur_id_;
  };

  IndexMeta meta_;
  DiskAnnEntity::Pointer entity_;
  std::string owner_class_;
};

}  // namespace core
}  // namespace zvec
