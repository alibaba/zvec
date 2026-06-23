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
#include <string>
#include "db/common/concurrent_roaring_bitmap.h"
#include "index_filter.h"


namespace zvec {


class DeleteStore : public std::enable_shared_from_this<DeleteStore> {
 public:
  using Ptr = std::shared_ptr<DeleteStore>;

  explicit DeleteStore(std::string collection_name)
      : collection_name_(std::move(collection_name)) {};

  ~DeleteStore() {
    LOG_INFO("Closed delete store");
  }

  static Ptr CreateAndLoad(std::string collection_name,
                           const std::string &file_path) {
    if (file_path.empty()) {
      LOG_ERROR("File path is empty");
      return nullptr;
    }
    DeleteStore::Ptr ptr =
        std::make_shared<DeleteStore>(std::move(collection_name));
    if (ptr->load(file_path).ok()) {
      return ptr;
    } else {
      return nullptr;
    }
  }


 private:
  DeleteStore(const DeleteStore &) = delete;
  DeleteStore &operator=(const DeleteStore &) = delete;
  DeleteStore &operator=(DeleteStore &&) = delete;


 public:
  class Filter : public IndexFilter {
   public:
    explicit Filter(std::shared_ptr<const DeleteStore> delete_store)
        : delete_store_(std::move(delete_store)) {}

    bool is_filtered(uint64_t id) const override {
      return delete_store_->is_deleted(id);
    }

   private:
    const std::shared_ptr<const DeleteStore> delete_store_;
  };

  Status load(const std::string &file_path) {
    Status status = bitmap_.deserialize(file_path);
    if (status.ok()) {
      empty_ = bitmap_.cardinality() == 0 ? true : false;
      LOG_INFO("Opened delete store, count[%lu]", bitmap_.cardinality());
    } else {
      LOG_ERROR("Failed to load delete store from file[%s]", file_path.c_str());
    }
    return status;
  }

  Status flush(const std::string &file_path) {
    Status status = bitmap_.serialize(file_path, true);
    if (status.ok()) {
      LOG_DEBUG("Flushed delete store to file[%s]", file_path.c_str());
      modified_since_last_flush_ = false;
    } else {
      LOG_ERROR("Failed to flush delete store to file[%s]", file_path.c_str());
    }
    return status;
  }

  void mark_deleted(uint64_t doc_id) {
    bitmap_.add(doc_id);
    empty_ = false;
    modified_since_last_flush_ = true;
  }

  bool is_deleted(uint64_t doc_id) const {
    return bitmap_.contains(doc_id);
  }

  std::shared_ptr<IndexFilter> make_filter() const {
    return empty_ ? nullptr : std::make_shared<Filter>(shared_from_this());
  };

  size_t storage_size_in_bytes() const {
    return bitmap_.storage_size_in_bytes();
  }

  size_t count() const {
    return bitmap_.cardinality();
  }

  size_t range_count(uint64_t min_doc_id, uint64_t max_doc_id) const {
    return bitmap_.range_cardinality(min_doc_id, max_doc_id);
  }

  const std::string &collection_name() const {
    return collection_name_;
  }

  bool modified_since_last_flush() const {
    return modified_since_last_flush_;
  }

  Ptr clone() const {
    auto ptr = std::make_shared<DeleteStore>(collection_name_);
    ptr->bitmap_ = bitmap_;
    ptr->empty_ = bitmap_.cardinality() == 0 ? true : false;
    ptr->modified_since_last_flush_ = false;
    return ptr;
  }

  bool empty() const {
    return empty_;
  }

 private:
  using FILE = ailego::File;

  const std::string collection_name_{};
  ConcurrentRoaringBitmap64 bitmap_{};
  bool empty_{true};
  bool modified_since_last_flush_{false};
};


}  // namespace zvec