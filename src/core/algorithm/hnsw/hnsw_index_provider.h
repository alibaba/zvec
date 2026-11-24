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
#include "utility/entity_vector_iterator.h"
#include "utility/live_vector_count.h"
#include "utility/stable_vector_holder.h"
#include "hnsw_entity.h"

namespace zvec {
namespace core {

class HnswIndexProvider : public IndexProvider, public StableVectorHolder {
 public:
  HnswIndexProvider(const IndexMeta &meta, const HnswEntity::Pointer &entity,
                    const std::string &owner,
                    std::shared_ptr<const void> lifetime = nullptr)
      : meta_(meta),
        entity_(entity),
        owner_class_(owner),
        lifetime_(std::move(lifetime)) {}

  HnswIndexProvider(const HnswIndexProvider &) = delete;
  HnswIndexProvider &operator=(const HnswIndexProvider &) = delete;

 public:  // holder interface
  int retain_stable_data(std::shared_ptr<const void> *lifetime) override {
    if (!lifetime) return IndexError_InvalidArgument;
    if (!lifetime_) return IndexError_NotImplemented;
    *lifetime = lifetime_;
    return 0;
  }

  //! Create a new iterator
  IndexProvider::Iterator::Pointer create_iterator() override {
    return HnswIndexProvider::Iterator::Pointer(new (std::nothrow)
                                                    Iterator(entity_));
  }

  IndexProvider::Iterator::Pointer create_slot_iterator() override {
    return std::make_unique<Iterator>(entity_, true);
  }

  //! Retrieve count of vectors
  size_t count() const override {
    return vector_count_.get(*entity_);
  }

  size_t slot_count() const override {
    return entity_->slot_count();
  }

  //! Retrieve dimension of vector
  size_t dimension() const override {
    return meta_.dimension();
  }

  //! Retrieve type of vector
  IndexMeta::DataType data_type() const override {
    return meta_.data_type();
  }

  //! Retrieve vector size in bytes
  size_t element_size() const override {
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
  const std::string &owner_class() const override {
    return owner_class_;
  }

 private:
  using Iterator = EntityVectorIterator<HnswEntity>;

 private:
  mutable LiveVectorCount vector_count_;
  const IndexMeta &meta_;
  const HnswEntity::Pointer entity_;
  const std::string owner_class_;
  std::shared_ptr<const void> lifetime_;
};

}  // namespace core
}  // namespace zvec
