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

#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <zvec/core/framework/index_provider.h>
#include <zvec/core/framework/index_searcher.h>
#include <zvec/core/framework/index_streamer.h>
#include "diskann_context.h"
#include "diskann_indexer.h"

namespace zvec {
namespace core {

class DiskAnnProviderTestPeer;

//! IndexProvider implementation backed by a DiskAnn indexer.
//!
//! Used by ``MixedStreamerReducer`` during segment merge: the reducer needs
//! to walk every vector held by a source DiskAnn streamer and feed it into
//! the merge target. Vectors are read on demand through the same aligned file
//! reader used by DiskAnn search. The provider owns the indexer, its in-memory
//! entity and independent I/O contexts, so it remains valid after its source
//! streamer is closed.
class DiskAnnIndexProvider : public IndexProvider {
  friend class DiskAnnProviderTestPeer;

 private:
  struct ResultBufferOwner {};

  struct ThreadResultBuffer {
    explicit ThreadResultBuffer(
        const std::shared_ptr<ResultBufferOwner> &buffer_owner)
        : owner(buffer_owner) {}

    std::weak_ptr<ResultBufferOwner> owner;
    std::string data;
  };

 public:
  DiskAnnIndexProvider(const IndexMeta &meta,
                       const IndexMetric::Pointer &measure,
                       const DiskAnnEntity::Pointer &entity,
                       const DiskAnnIndexer::Pointer &indexer,
                       const std::string &owner)
      : meta_(meta),
        measure_(measure),
        entity_(entity),
        indexer_(indexer),
        owner_class_(owner) {
    try {
      result_buffer_owner_ = std::make_shared<ResultBufferOwner>();
    } catch (const std::bad_alloc &) {
      LOG_ERROR("Failed to allocate DiskAnn provider result-buffer owner");
      throw;
    }
  }

  DiskAnnIndexProvider(const DiskAnnIndexProvider &) = delete;
  DiskAnnIndexProvider &operator=(const DiskAnnIndexProvider &) = delete;

 public:
  IndexProvider::Iterator::Pointer create_iterator() override {
    std::unique_ptr<Iterator> iterator(
        new (std::nothrow) Iterator(meta_, measure_, entity_, indexer_));
    if (!iterator || !iterator->ready()) {
      return nullptr;
    }
    return IndexProvider::Iterator::Pointer(iterator.release());
  }

  bool ready() const {
    return measure_ && entity_ && indexer_ && result_buffer_owner_;
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
    if (!ready()) {
      return nullptr;
    }

    const diskann_id_t id = indexer_->get_id(static_cast<diskann_key_t>(key));
    if (id == kInvalidId) {
      return nullptr;
    }

    // Serialize the heavyweight I/O context, but keep only the returned bytes
    // in thread-local storage. Buffers are isolated by provider lifetime, so a
    // fetch through another provider or on another thread cannot invalidate
    // this pointer. Expired providers are pruned lazily without retaining any
    // per-thread file/context resources. The returned pointer is valid until
    // this thread's next fetch through this provider, provider destruction, or
    // thread exit.
    try {
      std::string &vector_buffer = thread_result_buffer(result_buffer_owner_);
      std::lock_guard<std::mutex> lock(fetch_mutex_);
      if (!fetch_context_) {
        fetch_context_ =
            DiskAnnContext::create_fetch_context(meta_, measure_, entity_);
      }
      if (!fetch_context_ ||
          indexer_->get_vector(id, fetch_context_, vector_buffer) != 0) {
        return nullptr;
      }
      return vector_buffer.data();
    } catch (const std::bad_alloc &) {
      LOG_ERROR("Failed to allocate DiskAnn provider vector buffer");
      return nullptr;
    }
  }

  const std::string &owner_class(void) const override {
    return owner_class_;
  }

 private:
  static std::string &thread_result_buffer(
      const std::shared_ptr<ResultBufferOwner> &owner) {
    // A list keeps every live provider's string object stable when another
    // provider first fetches on this thread. Only the vector bytes are kept in
    // TLS; heavyweight DiskAnnContext and file handles remain provider-owned.
    static thread_local std::list<ThreadResultBuffer> buffers;
    for (auto it = buffers.begin(); it != buffers.end();) {
      std::shared_ptr<ResultBufferOwner> entry_owner = it->owner.lock();
      if (!entry_owner) {
        it = buffers.erase(it);
        continue;
      }
      if (entry_owner.get() == owner.get()) {
        return it->data;
      }
      ++it;
    }

    buffers.emplace_back(owner);
    return buffers.back().data;
  }

  class Iterator : public IndexProvider::Iterator {
   public:
    Iterator(const IndexMeta &meta, const IndexMetric::Pointer &measure,
             const DiskAnnEntity::Pointer &entity,
             const DiskAnnIndexer::Pointer &indexer)
        : meta_(meta),
          measure_(measure),
          entity_(entity),
          indexer_(indexer),
          cur_id_(0U) {
      cur_id_ = next_valid_id(0U);
    }

    bool ready() const {
      return meta_.element_size() > 0 && measure_ && entity_ && indexer_;
    }

    const void *data(void) const override {
      if (!is_valid() || !ready()) {
        return nullptr;
      }
      if (!data_loaded_) {
        // Context setup owns aligned I/O scratch space and platform resources;
        // iterators that are only inspected should not pay that cost.
        if (!context_) {
          context_ =
              DiskAnnContext::create_fetch_context(meta_, measure_, entity_);
        }
        if (!context_) {
          return nullptr;
        }
        if (indexer_->get_vector(cur_id_, context_, vector_buffer_) != 0) {
          return nullptr;
        }
        data_loaded_ = true;
      }
      return vector_buffer_.data();
    }

    bool is_valid(void) const override {
      return cur_id_ < static_cast<diskann_id_t>(entity_->doc_cnt());
    }

    uint64_t key(void) const override {
      return static_cast<uint64_t>(entity_->get_key(cur_id_));
    }

    void next(void) override {
      cur_id_ = next_valid_id(cur_id_ + 1);
      data_loaded_ = false;
      vector_buffer_.clear();
    }

   private:
    friend class DiskAnnProviderTestPeer;

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

    IndexMeta meta_;
    IndexMetric::Pointer measure_;
    DiskAnnEntity::Pointer entity_;
    DiskAnnIndexer::Pointer indexer_;
    mutable IndexContext::Pointer context_;
    mutable std::string vector_buffer_;
    mutable bool data_loaded_{false};
    diskann_id_t cur_id_;
  };

  IndexMeta meta_;
  IndexMetric::Pointer measure_;
  DiskAnnEntity::Pointer entity_;
  DiskAnnIndexer::Pointer indexer_;
  std::string owner_class_;
  std::shared_ptr<ResultBufferOwner> result_buffer_owner_;
  mutable std::mutex fetch_mutex_;
  mutable IndexContext::Pointer fetch_context_;
};

}  // namespace core
}  // namespace zvec
