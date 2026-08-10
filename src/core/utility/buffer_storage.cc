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

#include <sys/stat.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/time_helper.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {
namespace {

// C++17-compatible atomic access to serialized uint64_t fields.
inline uint64_t bs_load_acquire(const uint64_t *p) {
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#else
  uint64_t v = *static_cast<const volatile uint64_t *>(p);
  std::atomic_thread_fence(std::memory_order_acquire);
  return v;
#endif
}

inline uint64_t bs_load_relaxed(const uint64_t *p) {
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_load_n(p, __ATOMIC_RELAXED);
#else
  return *static_cast<const volatile uint64_t *>(p);
#endif
}

inline void bs_store_release(uint64_t *p, uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
#else
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint64_t *>(p) = v;
#endif
}

inline void bs_store_relaxed(uint64_t *p, uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
  __atomic_store_n(p, v, __ATOMIC_RELAXED);
#else
  *static_cast<volatile uint64_t *>(p) = v;
#endif
}

}  // namespace

// Legacy pointer reads stay valid until close_index(); prefer MemoryBlock for
// bounded ownership.

/*! Buffer Storage
 */
class BufferStorage : public IndexStorage {
 public:
  /*! Index Storage Segment
   */
  class WrappedSegment : public IndexStorage::Segment,
                         public std::enable_shared_from_this<Segment> {
   public:
    //! Index Storage Pointer
    typedef std::shared_ptr<Segment> Pointer;

    //! Constructor.  See segment_info_ for the pointer-stability contract.
    WrappedSegment(BufferStorage *owner, IndexMapping::SegmentInfo *info,
                   size_t segment_id)
        : segment_info_(info),
          owner_(owner),
          segment_id_(segment_id),
          capacity_(static_cast<size_t>(info->segment.meta()->data_size +
                                        info->segment.meta()->padding_size)) {}
    //! Destructor
    ~WrappedSegment(void) override {}

    //! Retrieve size of data. Paired acquire/release operations publish
    //! concurrent write/resize metadata changes to lock-free readers.
    size_t data_size(void) const override {
      return static_cast<size_t>(
          bs_load_acquire(&segment_info_->segment.meta()->data_size));
    }

    size_t data_offset(void) const override {
      return segment_info_->segment_header_start_offset +
             segment_info_->segment_header->content_offset +
             segment_info_->segment.meta()->data_index;
    }

    //! Retrieve crc of data
    uint32_t data_crc(void) const override {
      return segment_info_->segment.meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size(void) const override {
      return static_cast<size_t>(
          bs_load_acquire(&segment_info_->segment.meta()->padding_size));
    }

    //! Retrieve capacity of segment
    size_t capacity(void) const override {
      return capacity_;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::fetch: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      const size_t data_size =
          bs_load_acquire(&segment_info_->segment.meta()->data_size);
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      if (len == 0) {
        return 0;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      if (!owner_->read_range(abs_offset, len, static_cast<char *>(buf))) {
        LOG_ERROR(
            "WrappedSegment::fetch: read_range failed, file[%s], id[%zu], "
            "abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_, abs_offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::read: handle is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        *data = nullptr;
        return 0;
      }
      const size_t data_size =
          bs_load_acquire(&segment_info_->segment.meta()->data_size);
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      if (len == 0) {
        *data = nullptr;
        return 0;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      // Without a page cache, retain a copied result for the legacy pointer
      // lifetime. Cached single-page reads below pin the page until close(),
      // including for writable pools, instead of retaining one copy per read.
      if (!owner_->cache_enabled_) {
        static constexpr size_t kAlign = 4096UL;
        if (ailego_unlikely(len > std::numeric_limits<size_t>::max() -
                                      (kAlign - 1))) {
          *data = nullptr;
          return 0;
        }
        const size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
        char *tmp = nullptr;
        {
          std::lock_guard<std::mutex> tmp_latch(owner_->tmp_buffers_mutex_);
          tmp = owner_->tmp_arena_alloc_locked(alloc_size);
        }
        if (tmp == nullptr || !owner_->read_range(abs_offset, len, tmp)) {
          *data = nullptr;
          return 0;
        }
        *data = tmp;
        return len;
      }
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      bool force_bypass = false;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (raw != nullptr) {
          *data = raw;
          // Legacy pointer reads retain their pin until close_index().
          {
            std::lock_guard<std::mutex> pin_latch(owner_->pinned_pages_mutex_);
            owner_->pinned_pages_.push_back(page_id);
          }
          return len;
        }
        force_bypass = true;
      }
      // Keep scratch buffers 4K-aligned without over-allocating on platforms
      // whose native page size is larger.
      static constexpr size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      // The arena amortizes aligned allocation and avoids fragmented-heap
      // failures observed with many small aligned allocations on Android.
      char *tmp = nullptr;
      {
        std::lock_guard<std::mutex> tmp_latch(owner_->tmp_buffers_mutex_);
        tmp = owner_->tmp_arena_alloc_locked(alloc_size);
      }
      if (!tmp) {
        LOG_ERROR(
            "WrappedSegment::read: cross-page alloc failed, file[%s], "
            "id[%zu], abs_offset=%zu, len=%zu, alloc_size=%zu, align=%zu",
            owner_->file_name_.c_str(), segment_id_, abs_offset, len,
            alloc_size, kAlign);
        *data = nullptr;
        return 0;
      }
      const bool read_ok = force_bypass
                               ? owner_->buffer_pool_handle_->read_range_bypass(
                                     abs_offset, len, tmp)
                               : owner_->read_range(abs_offset, len, tmp);
      if (!read_ok) {
        LOG_ERROR(
            "WrappedSegment::read: cross-page read_range failed, file[%s], "
            "id[%zu], abs_offset=%zu, len=%zu, first_page=%zu, last_page=%zu",
            owner_->file_name_.c_str(), segment_id_, abs_offset, len,
            first_page, last_page);
        // Avoid holding the arena lock across I/O; a failed read loses one
        // temporary slot.
        *data = nullptr;
        return 0;
      }
      *data = tmp;
      return len;
    }

    //! Read data into a bounded-lifetime block.
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read(MemoryBlock&): handle is null, file[%s], "
            "id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      const size_t data_size =
          bs_load_acquire(&segment_info_->segment.meta()->data_size);
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      if (len == 0) {
        data.reset();
        return 0;
      }
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      if (!owner_->cache_enabled_ || owner_->buffer_pool_->writable()) {
        static constexpr size_t kAlign = 4096UL;
        if (ailego_unlikely(len > std::numeric_limits<size_t>::max() -
                                      (kAlign - 1))) {
          return 0;
        }
        const size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
        char *tmp =
            static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
        if (tmp == nullptr) {
          return 0;
        }
        if (!owner_->read_range(abs_offset, len, tmp)) {
          ailego_free(tmp);
          return 0;
        }
        data = MemoryBlock::MakeOwned(tmp, len);
        return len;
      }
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      bool force_bypass = false;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (raw != nullptr) {
          data.reset(owner_->buffer_pool_handle_, page_id, raw);
          return len;
        }
        force_bypass = true;
      }
      // aligned_alloc requires a size that is a multiple of the alignment.
      static constexpr size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      char *tmp =
          static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
      if (!tmp) {
        LOG_ERROR("read error (alloc cross-page temp buffer failed).");
        return 0;
      }
      const bool read_ok = force_bypass
                               ? owner_->buffer_pool_handle_->read_range_bypass(
                                     abs_offset, len, tmp)
                               : owner_->read_range(abs_offset, len, tmp);
      if (!read_ok) {
        ailego_free(tmp);
        LOG_ERROR("read error (cross-page read_range failed).");
        return 0;
      }
      data = MemoryBlock::MakeOwned(tmp, len);
      return len;
    }

    //! Borrowed read: the caller keeps the Segment alive until block release.
    size_t read_borrowed(size_t offset, MemoryBlock &data,
                         size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read_borrowed: handle is null, file[%s], "
            "id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      const size_t data_size =
          bs_load_acquire(&segment_info_->segment.meta()->data_size);
      if (ailego_unlikely(offset > data_size || len > data_size - offset)) {
        if (offset > data_size) {
          offset = data_size;
        }
        len = data_size - offset;
      }
      if (len == 0) {
        data.reset();
        return 0;
      }

      const size_t abs_offset = segment_info_->segment_header_start_offset +
                                segment_info_->segment_header->content_offset +
                                segment_info_->segment.meta()->data_index +
                                offset;
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      if (owner_->cache_enabled_ && !owner_->buffer_pool_->writable() &&
          len <= ailego::kVectorPageSize - offset_in_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (raw != nullptr) {
          data.reset(owner_->buffer_pool_handle_.get(), page_id, raw);
          return len;
        }
      }

      return read(offset, data, len);
    }

    bool prefer_borrowed_batch() const override {
      return owner_->cache_enabled_ && owner_->buffer_pool_ != nullptr &&
             !owner_->buffer_pool_->writable() &&
             owner_->buffer_pool_->has_evicted();
    }

    bool read_borrowed_batch(BorrowedRead *reads, size_t count) override {
      if (count == 0) {
        return true;
      }
      if (reads == nullptr || owner_->buffer_pool_handle_ == nullptr ||
          owner_->buffer_pool_ == nullptr) {
        return false;
      }

      // Writable and bypass-only pools retain the scalar ownership rules.
      if (!owner_->cache_enabled_ || owner_->buffer_pool_->writable()) {
        return IndexStorage::Segment::read_borrowed_batch(reads, count);
      }

      struct BatchState {
        BorrowedRead *request{nullptr};
        size_t abs_offset{0};
        size_t first_page_index{0};
        size_t page_count{0};
        char *owned{nullptr};
      };
      struct BatchScratch {
        std::vector<BatchState> states;
        std::vector<ailego::block_id_t> page_ids;
        std::vector<char *> pages;
      };
      static thread_local BatchScratch scratch;

      scratch.states.clear();
      scratch.page_ids.clear();
      scratch.states.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        if (reads[i].block != nullptr) {
          reads[i].block->reset();
        }
      }

      auto cleanup_owned = []() {
        for (BatchState &state : scratch.states) {
          if (state.owned != nullptr) {
            ailego_free(state.owned);
            state.owned = nullptr;
          }
        }
      };
      auto fail = [&]() {
        cleanup_owned();
        for (size_t i = 0; i < count; ++i) {
          if (reads[i].block != nullptr) {
            reads[i].block->reset();
          }
        }
        return false;
      };

      size_t total_pages = 0;
      for (size_t i = 0; i < count; ++i) {
        BorrowedRead &request = reads[i];
        auto *segment = dynamic_cast<WrappedSegment *>(request.segment);
        if (segment == nullptr || segment->owner_ != owner_ ||
            request.block == nullptr) {
          cleanup_owned();
          return IndexStorage::Segment::read_borrowed_batch(reads, count);
        }

        const size_t data_size =
            bs_load_acquire(&segment->segment_info_->segment.meta()->data_size);
        if (request.offset > data_size ||
            request.length > data_size - request.offset) {
          return fail();
        }

        BatchState state;
        state.request = &request;
        if (request.length == 0) {
          scratch.states.emplace_back(state);
          continue;
        }

        const size_t data_offset =
            segment->segment_info_->segment_header_start_offset +
            segment->segment_info_->segment_header->content_offset +
            segment->segment_info_->segment.meta()->data_index;
        if (data_offset > std::numeric_limits<size_t>::max() - request.offset) {
          return fail();
        }
        state.abs_offset = data_offset + request.offset;
        if (state.abs_offset >
            std::numeric_limits<size_t>::max() - (request.length - 1)) {
          return fail();
        }
        const size_t first_page = state.abs_offset / ailego::kVectorPageSize;
        const size_t last_page =
            (state.abs_offset + request.length - 1) / ailego::kVectorPageSize;
        state.first_page_index = total_pages;
        state.page_count = last_page - first_page + 1;
        if (state.page_count >
            std::numeric_limits<size_t>::max() - total_pages) {
          return fail();
        }
        total_pages += state.page_count;
        scratch.states.emplace_back(state);
        for (size_t page = first_page; page <= last_page; ++page) {
          scratch.page_ids.emplace_back(page);
        }
      }

      // Cross-page values need an owned contiguous result, but their source
      // pages still participate in the same batched AIO submission.
      static constexpr size_t kAlignment = 4096UL;
      for (BatchState &state : scratch.states) {
        if (state.page_count <= 1) {
          continue;
        }
        const size_t length = state.request->length;
        if (length > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
          return fail();
        }
        const size_t alloc_size =
            (length + (kAlignment - 1)) & ~(kAlignment - 1);
        state.owned =
            static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlignment));
        if (state.owned == nullptr) {
          return fail();
        }
      }

      scratch.pages.assign(total_pages, nullptr);
      if (total_pages != 0 &&
          !owner_->buffer_pool_handle_->acquire_pages(
              scratch.page_ids.data(), total_pages, scratch.pages.data())) {
        return fail();
      }

      for (BatchState &state : scratch.states) {
        if (state.page_count == 0) {
          state.request->block->reset();
          continue;
        }
        const size_t first = state.first_page_index;
        if (state.page_count == 1) {
          const size_t offset_in_page =
              state.abs_offset % ailego::kVectorPageSize;
          state.request->block->reset(owner_->buffer_pool_handle_.get(),
                                      scratch.page_ids[first],
                                      scratch.pages[first] + offset_in_page);
          // The MemoryBlock now owns this pin.
          scratch.pages[first] = nullptr;
          continue;
        }

        size_t remaining = state.request->length;
        size_t copied = 0;
        size_t offset_in_page = state.abs_offset % ailego::kVectorPageSize;
        for (size_t j = 0; j < state.page_count; ++j) {
          const size_t chunk =
              std::min(remaining, ailego::kVectorPageSize - offset_in_page);
          std::memcpy(state.owned + copied,
                      scratch.pages[first + j] + offset_in_page, chunk);
          owner_->buffer_pool_handle_->release_one(scratch.page_ids[first + j]);
          scratch.pages[first + j] = nullptr;
          copied += chunk;
          remaining -= chunk;
          offset_in_page = 0;
        }
        *state.request->block =
            MemoryBlock::MakeOwned(state.owned, state.request->length);
        state.owned = nullptr;
      }
      return true;
    }

    //! Write data into the storage with offset. The shard latch excludes
    //! flush; meta_mtx_ serializes metadata changes within this segment.
    size_t write(size_t offset, const void *data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_)) {
        LOG_ERROR("WrappedSegment::write: pool is null, file[%s], id[%zu]",
                  owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::write: storage is marked corrupted, refusing "
            "write, file[%s], id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      // In read-only mode the write is a silent no-op so that callers that
      // unconditionally write (e.g. CRC updates) do not return an error.
      if (!owner_->buffer_pool_->writable()) {
        return len;
      }
      if (ailego_unlikely(offset > capacity_ || len > capacity_ - offset)) {
        LOG_ERROR(
            "write() exceeds segment capacity: offset=%zu len=%zu cap=%zu",
            offset, len, capacity_);
        return 0;
      }
      auto meta = segment_info_->segment.meta();
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          meta->data_index + offset;
      // Publish data_size only after the page bytes are visible to readers.
      if (owner_->buffer_pool_handle_->write_range(
              abs_offset, len, static_cast<const char *>(data)) != 0) {
        LOG_ERROR("write() page-cache write_range failed at abs_offset=%zu",
                  abs_offset);
        return 0;
      }
      {
        std::lock_guard<std::mutex> meta_latch(meta_mtx_);
        uint64_t cur = bs_load_relaxed(&meta->data_size);
        if (offset + len > cur) {
          uint64_t new_size = offset + len;
          // Publish padding before data_size to keep the pair consistent.
          bs_store_relaxed(&meta->padding_size, capacity_ - new_size);
          bs_store_release(&meta->data_size, new_size);
        }
      }
      // Fixed-size rewrites are dirty even when data_size is unchanged.
      owner_->set_as_dirty();
      return len;
    }

    //! Resize size of data.  See write() for the locking contract.
    size_t resize(size_t size) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::resize: storage is marked corrupted, refusing "
            "resize, file[%s], id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return 0;
      }
      auto meta = segment_info_->segment.meta();
      bool changed = false;
      {
        std::lock_guard<std::mutex> meta_latch(meta_mtx_);
        uint64_t cur = bs_load_relaxed(&meta->data_size);
        if (cur != size) {
          if (size > capacity_) {
            size = capacity_;
          }
          // Match write(): padding first, then release-store data_size.
          bs_store_relaxed(&meta->padding_size, capacity_ - size);
          bs_store_release(&meta->data_size, size);
          changed = true;
        }
      }
      if (changed) {
        owner_->set_as_dirty();
      }
      return size;
    }

    //! Update crc of data.  See write() for the locking contract.
    void update_data_crc(uint32_t crc) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::update_data_crc: storage is marked corrupted, "
            "refusing CRC update, file[%s], id[%zu]",
            owner_->file_name_.c_str(), segment_id_);
        return;
      }
      {
        std::lock_guard<std::mutex> meta_latch(meta_mtx_);
        segment_info_->segment.meta()->data_crc = crc;
      }
      owner_->set_as_dirty();
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone(void) override {
      return shared_from_this();
    }

    //! Preload a read-only range and attach its eviction priority. Writable
    //! storage deliberately skips this hint: construction has a different
    //! access pattern and dirty-page lifetime must drive admission there.
    void prefetch(size_t offset, size_t len,
                  CachePriority priority = CachePriority::kLow) override {
      if (!owner_->cache_enabled_ || !owner_->buffer_pool_ ||
          !owner_->buffer_pool_handle_ || owner_->buffer_pool_->writable()) {
        return;
      }
      const size_t data_size =
          bs_load_acquire(&segment_info_->segment.meta()->data_size);
      if (offset >= data_size || len == 0) {
        return;
      }
      len = std::min(len, data_size - offset);
      const size_t abs_offset = data_offset() + offset;
      owner_->buffer_pool_handle_->prefetch_range(
          abs_offset, len, static_cast<uint8_t>(priority));
    }

   protected:
    friend BufferStorage;
    // Stable unordered_map value address; reparses update this object in place.
    IndexMapping::SegmentInfo *segment_info_{nullptr};
    // Serializes metadata writers within this segment.
    mutable std::mutex meta_mtx_{};

   private:
    BufferStorage *owner_{nullptr};
    size_t segment_id_{};
    size_t capacity_{};
  };

  //! Destructor
  ~BufferStorage(void) override {
    this->cleanup();
  }

  //! Retrieve the memory block type of this storage
  MemoryBlock::MemoryBlockType memory_block_type(void) const override {
    return MemoryBlock::MBT_BUFFERPOOL;
  }

  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool(void) const override {
    return cache_enabled_ ? buffer_pool_ : nullptr;
  }

  //! Initialize storage
  int init(const ailego::Params &params) override {
    uint32_t val = params.get_as_uint32(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY);
    if (val != 0) {
      segment_meta_capacity_ = val;
    }
    return 0;
  }

  //! Cleanup storage
  int cleanup(void) override {
    this->close_index();
    return 0;
  }

  //! Open storage
  int open(const std::string &path, bool create_if_missing) override {
    file_name_ = path;
    if (!ailego::File::IsExist(path) && create_if_missing) {
      size_t last_slash = path.rfind('/');
      if (last_slash != std::string::npos) {
        ailego::File::MakePath(path.substr(0, last_slash));
      }
      int error_code = this->init_index(path);
      if (error_code != 0) {
        LOG_ERROR("init_index failed for %s, errno=%d", path.c_str(),
                  error_code);
        return error_code;
      }
    }

    // create_if_missing also indicates write intent, matching MMapFileStorage.
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
        path, /*writable=*/create_if_missing);
    buffer_pool_handle_ =
        std::make_shared<ailego::VecBufferPoolHandle>(buffer_pool_);
    int ret = ParseToMapping();
    if (ret != 0) {
      this->close_index();
      return ret;
    }
    const size_t file_size = buffer_pool_->file_size();
    const size_t page_count =
        file_size == 0 ? 0 : (file_size - 1) / ailego::kVectorPageSize + 1;
    const size_t metadata_bytes =
        ailego::VecBufferPool::metadata_bytes_for_page_count(page_count);
    const size_t available =
        ailego::MemoryLimitPool::get_instance().available();
    const bool cache_can_fit =
        metadata_bytes != std::numeric_limits<size_t>::max() &&
        metadata_bytes <= available &&
        ailego::kVectorPageSize <= available - metadata_bytes;
    ret = cache_can_fit ? buffer_pool_->init() : -1;
    if (ret != 0 && create_if_missing) {
      this->close_index();
      return IndexError_NoMemory;
    }
    cache_enabled_ = ret == 0;
    if (!cache_enabled_) {
      LOG_INFO(
          "Read-only BufferStorage opened in bypass-only mode: file=%s "
          "available=%zu metadata=%zu page_size=%zu",
          path.c_str(), available, metadata_bytes, ailego::kVectorPageSize);
    }
    LOG_INFO(
        "BufferStorage opened: file=%s, writable=%d, max_segment_size=%" PRIu64
        ", segment_count=%zu",
        file_name_.c_str(), static_cast<int>(create_if_missing),
        max_segment_size_, segments_.size());
    return 0;
  }

  // Called from single-threaded open or under AllShardsExclusiveLatch.
  int ParseHeader(size_t offset, IndexFormat::MetaHeader *out) {
    constexpr size_t kHeaderSize = sizeof(IndexFormat::MetaHeader);
    std::unique_ptr<char[]> buffer(new char[kHeaderSize]);
    if (buffer_pool_handle_->get_meta(offset, kHeaderSize, buffer.get()) != 0) {
      LOG_ERROR("Get segment header failed.");
      return IndexError_Runtime;
    }
    memcpy(out, buffer.get(), kHeaderSize);
    if (out->meta_header_size != kHeaderSize) {
      LOG_ERROR("Header meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(out, kHeaderSize, out->header_crc) !=
        out->header_crc) {
      LOG_ERROR("Header meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    return 0;
  }

  int ParseFooter(size_t offset) {
    std::unique_ptr<char[]> buffer(new char[sizeof(footer_)]);
    if (buffer_pool_handle_->get_meta(offset, sizeof(footer_), buffer.get()) !=
        0) {
      LOG_ERROR("Get segment footer failed.");
      return IndexError_Runtime;
    }
    uint8_t *footer_ptr = reinterpret_cast<uint8_t *>(buffer.get());
    memcpy(&footer_, footer_ptr, sizeof(footer_));
    if (offset < (size_t)footer_.segments_meta_size) {
      LOG_ERROR("Footer meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(&footer_, sizeof(footer_), footer_.footer_crc) !=
        footer_.footer_crc) {
      LOG_ERROR("Footer meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    return 0;
  }

  int ParseSegment(size_t offset, IndexFormat::MetaHeader *chain_header,
                   uint32_t *out_segment_ids_offset) {
    std::unique_ptr<char[]> segment_buffer =
        std::make_unique<char[]>(footer_.segments_meta_size);
    if (buffer_pool_handle_->get_meta(offset, footer_.segments_meta_size,
                                      segment_buffer.get()) != 0) {
      LOG_ERROR("Get segment meta failed.");
      return IndexError_Runtime;
    }
    if (ailego::Crc32c::Hash(segment_buffer.get(), footer_.segments_meta_size,
                             0u) != footer_.segments_meta_crc) {
      LOG_ERROR("Index segments meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    IndexFormat::SegmentMeta *segment_start =
        reinterpret_cast<IndexFormat::SegmentMeta *>(segment_buffer.get());
    uint32_t segment_ids_offset = footer_.segments_meta_size;
    for (IndexFormat::SegmentMeta *iter = segment_start,
                                  *end = segment_start + footer_.segment_count;
         iter != end; ++iter) {
      if (iter->segment_id_offset >= footer_.segments_meta_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index > footer_.content_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index + iter->data_size > footer_.content_size) {
        return IndexError_InvalidLength;
      }

      if (iter->segment_id_offset < segment_ids_offset) {
        segment_ids_offset = iter->segment_id_offset;
      }
      // segments_ retains stable entries across appends; id_hash_ is the
      // authoritative id sequence. Bound ID parsing to the metadata buffer.
      const char *seg_name_start =
          reinterpret_cast<const char *>(segment_start) +
          iter->segment_id_offset;
      const size_t seg_name_max =
          footer_.segments_meta_size - iter->segment_id_offset;
      const size_t seg_name_len = ::strnlen(seg_name_start, seg_name_max);
      if (seg_name_len == seg_name_max) {
        LOG_ERROR("ParseSegment: segment_id missing NUL terminator, file[%s]",
                  file_name_.c_str());
        return IndexError_InvalidValue;
      }
      const std::string seg_name(seg_name_start, seg_name_len);
      const size_t seg_id = id_hash_.size();
      id_hash_[seg_name] = seg_id;
      // Update in place so existing WrappedSegment pointers remain valid.
      segments_[seg_name] =
          IndexMapping::SegmentInfo{IndexMapping::Segment{iter},
                                    current_header_start_offset_, chain_header};
      max_segment_size_ =
          std::max(max_segment_size_, iter->data_size + iter->padding_size);
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
    }
    buffer_pool_buffers_.push_back(std::move(segment_buffer));
    if (out_segment_ids_offset) {
      *out_segment_ids_offset = segment_ids_offset;
    }
    return 0;
  }

  int ParseToMapping() {
    while (true) {
      int ret;
      // Per-chain owning MetaHeader; see chain_headers_ field comment.
      chain_headers_.emplace_back(std::make_unique<IndexFormat::MetaHeader>());
      IndexFormat::MetaHeader *chain_header = chain_headers_.back().get();
      ret = ParseHeader(current_header_start_offset_, chain_header);
      if (ret != 0) {
        LOG_ERROR("Failed to parse header, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      switch (chain_header->version) {
        case IndexFormat::FORMAT_VERSION:
          break;
        default:
          LOG_ERROR("Unsupported index version: %u", chain_header->version);
          return IndexError_Unsupported;
      }

      // Unpack footer
      if (chain_header->meta_footer_size != sizeof(IndexFormat::MetaFooter)) {
        return IndexError_InvalidLength;
      }
      if ((int32_t)chain_header->meta_footer_offset < 0) {
        return IndexError_Unsupported;
      }
      uint64_t footer_offset =
          chain_header->meta_footer_offset + current_header_start_offset_;
      // Reject uint64 wrap-around and offsets past file_size.
      if (footer_offset < current_header_start_offset_ ||
          footer_offset + sizeof(IndexFormat::MetaFooter) >
              buffer_pool_->file_size()) {
        LOG_ERROR("ParseToMapping: invalid footer_offset=%" PRIu64
                  " (header=%" PRIu64 ", file_size=%zu), file[%s]",
                  footer_offset, current_header_start_offset_,
                  buffer_pool_->file_size(), file_name_.c_str());
        return IndexError_InvalidValue;
      }
      ret = ParseFooter(footer_offset);
      if (ret != 0) {
        LOG_ERROR("Failed to parse footer, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Unpack segment table
      if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count >
          footer_.segments_meta_size) {
        return IndexError_InvalidLength;
      }
      const uint64_t segment_start_offset =
          footer_offset - footer_.segments_meta_size;
      uint32_t segment_ids_offset = footer_.segments_meta_size;
      ret =
          ParseSegment(segment_start_offset, chain_header, &segment_ids_offset);
      if (ret != 0) {
        LOG_ERROR("Failed to parse segment, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Retain the offsets needed to flush each metadata chain.
      meta_chains_.push_back({current_header_start_offset_, footer_offset,
                              segment_start_offset, footer_.segments_meta_size,
                              segment_ids_offset, footer_});

      if (footer_.next_meta_header_offset == 0) {
        break;
      }
      // Reject invalid links before following the metadata chain.
      const uint64_t next_off = footer_.next_meta_header_offset;
      if (next_off <= current_header_start_offset_ ||
          next_off + sizeof(IndexFormat::MetaHeader) >
              buffer_pool_->file_size()) {
        LOG_ERROR("ParseToMapping: invalid next_meta_header_offset=%" PRIu64
                  " (current=%" PRIu64 ", file_size=%zu), file[%s]",
                  next_off, current_header_start_offset_,
                  buffer_pool_->file_size(), file_name_.c_str());
        return IndexError_InvalidValue;
      }
      // Bound corrupted metadata chains.
      constexpr size_t kMaxChains = 1024;
      if (chain_headers_.size() >= kMaxChains) {
        LOG_ERROR(
            "ParseToMapping: chain count exceeds limit %zu, file[%s] may "
            "be corrupted",
            kMaxChains, file_name_.c_str());
        return IndexError_InvalidLength;
      }
      current_header_start_offset_ = next_off;
    }
    return 0;
  }

  //! Flush storage
  int flush(void) override {
    return this->flush_index();
  }

  //! Close storage
  int close(void) override {
    this->close_index();
    return 0;
  }

  //! Append a segment into storage
  int append(const std::string &id, size_t size) override {
    return this->append_segment(id, size);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh(uint64_t chkp) override {
    this->refresh_index(chkp);
  }

  //! Retrieve check point of storage
  uint64_t check_point(void) const override {
    return footer_.check_point;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    auto seg_iter = segments_.find(id);
    if (seg_iter == segments_.end()) {
      return WrappedSegment::Pointer{};
    }
    auto id_iter = id_hash_.find(id);
    if (id_iter == id_hash_.end()) {
      return WrappedSegment::Pointer{};
    }
    return std::make_shared<WrappedSegment>(this, &seg_iter->second,
                                            id_iter->second);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    return this->has_segment(id);
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    if (chain_headers_.empty()) {
      return 0u;
    }
    return chain_headers_.front()->magic;
  }

 protected:
  //! Write the version segment while the new mapping is open.
  int init_version_segment(IndexMapping &mapping) {
    size_t data_size = std::strlen(IndexVersion::Details());
    int error_code = mapping.append(INDEX_VERSION_SEGMENT_NAME, data_size);
    if (error_code != 0) {
      return error_code;
    }
    IndexMapping::Segment *segment =
        mapping.map(INDEX_VERSION_SEGMENT_NAME, false, false);
    if (!segment) {
      return IndexError_MMapFile;
    }
    auto meta = segment->meta();
    size_t capacity = static_cast<size_t>(meta->padding_size + meta->data_size);
    memcpy(segment->data(), IndexVersion::Details(), data_size);
    segment->set_dirty();
    set_as_dirty();
    meta->data_crc = ailego::Crc32c::Hash(segment->data(), data_size, 0);
    meta->data_size = data_size;
    meta->padding_size = capacity - data_size;
    return 0;
  }

  //! Create an index compatible with the mmap storage format.
  int init_index(const std::string &path) {
    IndexMapping mapping;
    int ret = mapping.create(path, segment_meta_capacity_);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to create index file: path[%s], errno[%d]",
          path.c_str(), ret);
      return ret;
    }
    ret = this->init_version_segment(mapping);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to append version segment: path[%s], errno[%d]",
          path.c_str(), ret);
      mapping.close();
      return ret;
    }
    mapping.refresh(0);
    ret = mapping.flush();
    mapping.close();
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to flush new index file: path[%s], errno[%d]",
          path.c_str(), ret);
    }
    return ret;
  }

  bool is_dirty(void) const override {
    return index_dirty_.load(std::memory_order_relaxed);
  }

  //! Publish dirty unconditionally to avoid racing a concurrent flush.
  void set_as_dirty(void) {
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    // Checkpoints are monotonic; the flush latch provides synchronization.
    if (chkp != 0) {
      uint64_t cur = pending_check_point_.load(std::memory_order_relaxed);
      while (chkp > cur) {
        if (pending_check_point_.compare_exchange_weak(
                cur, chkp, std::memory_order_relaxed)) {
          break;
        }
      }
    }
    // Set dirty unconditionally even if our chkp lost the CAS race: the
    // winning larger chkp must still be flushed.
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Flush index storage.
  int flush_index(void) {
    if (!index_dirty_.load(std::memory_order_relaxed)) {
      return 0;
    }
    // Exclude metadata mutation while hashing and persisting it.
    AllShardsExclusiveLatch latch(mapping_shards_);
    return flush_index_locked();
  }

  //! Requires AllShardsExclusiveLatch.
  int flush_index_locked(void) {
    // close_index() may call this before open or after close.
    if (!buffer_pool_ || !buffer_pool_handle_) {
      index_dirty_.store(false, std::memory_order_relaxed);
      return 0;
    }
    if (corrupted_.load(std::memory_order_acquire)) {
      LOG_ERROR(
          "BufferStorage::flush_index skipped: storage is marked corrupted, "
          "file[%s]",
          file_name_.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->writable()) {
      // Read-only pool: nothing to flush.
      index_dirty_.store(false, std::memory_order_relaxed);
      return 0;
    }
    // Claim the current dirty generation; concurrent writes start the next one.
    bool expected_dirty = true;
    if (!index_dirty_.compare_exchange_strong(expected_dirty, false,
                                              std::memory_order_relaxed)) {
      // Another thread already claimed it.
      return 0;
    }
    // Snapshot after claiming dirty so newer checkpoints survive the final CAS.
    const uint64_t consumed_chkp =
        pending_check_point_.load(std::memory_order_relaxed);
    // Restore on failure without overwriting a newer checkpoint.
    auto restore_chkp_on_failure = [this, consumed_chkp]() {
      if (consumed_chkp == 0) return;
      uint64_t cur = pending_check_point_.load(std::memory_order_relaxed);
      while (consumed_chkp > cur) {
        if (pending_check_point_.compare_exchange_weak(
                cur, consumed_chkp, std::memory_order_relaxed)) {
          break;
        }
      }
    };
    // Flush dirty data blocks first.
    if (buffer_pool_handle_->flush_all() != 0) {
      index_dirty_.store(true, std::memory_order_relaxed);
      restore_chkp_on_failure();
      LOG_ERROR("flush_all data blocks failed: file[%s]", file_name_.c_str());
      return IndexError_WriteData;
    }
    // Refresh and persist metadata for each chain.
    for (size_t ci = 0;
         ci < meta_chains_.size() && ci < buffer_pool_buffers_.size(); ++ci) {
      MetaChain &mchain = meta_chains_[ci];
      const char *seg_buf = buffer_pool_buffers_[ci].get();
      mchain.footer.segments_meta_crc =
          ailego::Crc32c::Hash(seg_buf, mchain.segment_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&mchain.footer, consumed_chkp);
      if (buffer_pool_handle_->write_meta(mchain.segment_meta_file_offset,
                                          mchain.segment_meta_size,
                                          seg_buf) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        restore_chkp_on_failure();
        return IndexError_WriteData;
      }
      if (buffer_pool_handle_->write_meta(
              mchain.footer_file_offset, sizeof(mchain.footer),
              reinterpret_cast<const char *>(&mchain.footer)) != 0) {
        LOG_ERROR("Failed to write footer: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        restore_chkp_on_failure();
        return IndexError_WriteData;
      }
    }
    if (!meta_chains_.empty()) {
      footer_ = meta_chains_.back().footer;
    }
    // Consume only the checkpoint observed by this flush.
    uint64_t expected_chkp = consumed_chkp;
    pending_check_point_.compare_exchange_strong(expected_chkp, 0,
                                                 std::memory_order_relaxed);
    return 0;
  }

  //! Close index storage
  void close_index(void) {
    // Keep writers excluded across both flush and teardown.
    AllShardsExclusiveLatch latch(mapping_shards_);
    flush_index_locked();
    file_name_.clear();
    id_hash_.clear();
    segments_.clear();
    chain_headers_.clear();
    memset(&footer_, 0, sizeof(footer_));
    {
      std::lock_guard<std::mutex> tmp_latch(tmp_buffers_mutex_);
      for (const ArenaBlock &b : tmp_buffers_) {
        if (b.base) {
          ailego_free(b.base);
        }
      }
      tmp_buffers_.clear();
    }
    // Drop persistent pointer-read pins before destroying the pool.
    {
      std::lock_guard<std::mutex> pin_latch(pinned_pages_mutex_);
      if (buffer_pool_handle_) {
        for (size_t pid : pinned_pages_) {
          buffer_pool_handle_->release_one(pid);
        }
      }
      pinned_pages_.clear();
    }
    buffer_pool_handle_.reset();
    buffer_pool_.reset();
    cache_enabled_ = false;
    max_segment_size_ = 0;
    buffer_pool_buffers_.clear();
    meta_chains_.clear();
    current_header_start_offset_ = 0;
    pending_check_point_.store(0, std::memory_order_relaxed);
    index_dirty_.store(false, std::memory_order_relaxed);
    corrupted_.store(false, std::memory_order_relaxed);
  }

  //! Append a segment into storage.
  int append_segment(const std::string &id, size_t size) {
    // Persist pending metadata before re-hashing it.
    this->flush_index();

    AllShardsExclusiveLatch latch(mapping_shards_);

    if (!buffer_pool_ || !buffer_pool_handle_) {
      LOG_ERROR("append_segment: pool not ready, file[%s]", file_name_.c_str());
      return IndexError_Runtime;
    }
    if (corrupted_.load(std::memory_order_acquire)) {
      LOG_ERROR(
          "append_segment: storage is marked corrupted, refusing to append, "
          "file[%s], id[%s]",
          file_name_.c_str(), id.c_str());
      return IndexError_Runtime;
    }
    if (!buffer_pool_->writable()) {
      LOG_ERROR("append_segment: pool is read-only, file[%s]",
                file_name_.c_str());
      return IndexError_Runtime;
    }
    if (size == 0) {
      return IndexError_InvalidArgument;
    }
    if (segments_.find(id) != segments_.end()) {
      return IndexError_Duplicate;
    }
    if (meta_chains_.empty() || chain_headers_.empty() ||
        buffer_pool_buffers_.empty()) {
      LOG_ERROR("append_segment: invalid state, file[%s]", file_name_.c_str());
      return IndexError_Runtime;
    }

    // Page-aligned padded size; matches IndexMapping::CalcPageAlignedSize().
    const size_t page_size = ailego::kVectorPageSize;
    const size_t padded_size = (size + page_size - 1) / page_size * page_size;

    // The current last chain owns footer_ (overwritten by ParseFooter).
    size_t id_size = id.length() + 1;
    size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
    MetaChain *chain = &meta_chains_.back();
    IndexFormat::MetaHeader *header = chain_headers_.back().get();
    char *meta_buf = buffer_pool_buffers_.back().get();

    // Set after a chain split so a later append failure can unlink it.
    std::function<void()> rollback_step1 = []() {};

    // Step 1: split a full metadata chain.
    if (sizeof(IndexFormat::SegmentMeta) * footer_.segment_count + need_size >
        chain->segment_ids_offset) {
      size_t new_chain_start = buffer_pool_->file_size();
      new_chain_start =
          (new_chain_start + page_size - 1) / page_size * page_size;
      size_t new_meta_total =
          (segment_meta_capacity_ + sizeof(IndexFormat::MetaHeader) +
           sizeof(IndexFormat::MetaFooter) + page_size - 1) /
          page_size * page_size;
      uint32_t new_segments_meta_size = static_cast<uint32_t>(
          new_meta_total - sizeof(IndexFormat::MetaHeader) -
          sizeof(IndexFormat::MetaFooter));

      // Stage the linked old footer without mutating footer_ yet.
      const auto saved_footer_before_split = footer_;
      IndexFormat::MetaFooter linked_footer = footer_;
      linked_footer.next_meta_header_offset = new_chain_start;
      IndexFormat::UpdateMetaFooter(&linked_footer, 0);

      if (buffer_pool_handle_->write_meta(
              chain->footer_file_offset, sizeof(linked_footer),
              reinterpret_cast<const char *>(&linked_footer)) != 0) {
        LOG_ERROR("append_segment: write old footer failed, file[%s]",
                  file_name_.c_str());
        return IndexError_WriteData;
      }

      // Restore the old link after a failed split; reject writes if rollback
      // also fails.
      auto undo_old_footer = [this, chain, &saved_footer_before_split]() {
        if (buffer_pool_handle_->write_meta(
                chain->footer_file_offset, sizeof(saved_footer_before_split),
                reinterpret_cast<const char *>(&saved_footer_before_split)) !=
            0) {
          LOG_ERROR(
              "append_segment: rollback write of old footer FAILED, file[%s] "
              "is now in an inconsistent state -- marking storage as "
              "corrupted; further writes will be rejected.",
              file_name_.c_str());
          corrupted_.store(true, std::memory_order_release);
        }
      };

      // Extend the file and write the new chain's header + (zero) footer.
      // The segment_meta region is zero-filled by ftruncate.
      if (!buffer_pool_->extend_file(new_chain_start + new_meta_total)) {
        undo_old_footer();
        return IndexError_Runtime;
      }

      auto new_header = std::make_unique<IndexFormat::MetaHeader>();
      IndexFormat::SetupMetaHeader(
          new_header.get(),
          static_cast<uint32_t>(new_meta_total -
                                sizeof(IndexFormat::MetaFooter)),
          static_cast<uint32_t>(new_meta_total));

      auto new_meta_buf = std::make_unique<char[]>(new_segments_meta_size);
      std::memset(new_meta_buf.get(), 0, new_segments_meta_size);

      IndexFormat::MetaFooter new_footer;
      IndexFormat::SetupMetaFooter(&new_footer);
      new_footer.segments_meta_size = new_segments_meta_size;
      new_footer.total_size = new_meta_total;
      new_footer.segments_meta_crc =
          ailego::Crc32c::Hash(new_meta_buf.get(), new_segments_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&new_footer, 0);

      if (buffer_pool_handle_->write_meta(
              new_chain_start, sizeof(IndexFormat::MetaHeader),
              reinterpret_cast<const char *>(new_header.get())) != 0) {
        undo_old_footer();
        return IndexError_WriteData;
      }
      uint64_t new_segment_meta_file_offset =
          new_chain_start + sizeof(IndexFormat::MetaHeader);
      uint64_t new_footer_file_offset =
          new_chain_start + new_header->meta_footer_offset;
      if (buffer_pool_handle_->write_meta(
              new_footer_file_offset, sizeof(new_footer),
              reinterpret_cast<const char *>(&new_footer)) != 0) {
        undo_old_footer();
        return IndexError_WriteData;
      }

      // Snapshot the old chain before reassigning `chain`.
      const auto saved_old_chain_footer = chain->footer;
      const uint64_t saved_old_footer_file_offset = chain->footer_file_offset;
      const uint64_t saved_current_header_start = current_header_start_offset_;

      // Reserve all parallel vectors before committing the split.
      try {
        chain_headers_.reserve(chain_headers_.size() + 1);
        buffer_pool_buffers_.reserve(buffer_pool_buffers_.size() + 1);
        meta_chains_.reserve(meta_chains_.size() + 1);
      } catch (const std::bad_alloc &) {
        LOG_ERROR(
            "append_segment: reserve for chain-split commit failed, file[%s]",
            file_name_.c_str());
        undo_old_footer();
        return IndexError_Runtime;
      }
      chain = &meta_chains_.back();
      chain->footer = linked_footer;  // old chain keeps linked footer
      chain_headers_.push_back(std::move(new_header));
      buffer_pool_buffers_.push_back(std::move(new_meta_buf));
      meta_chains_.push_back(MetaChain{
          new_chain_start, new_footer_file_offset, new_segment_meta_file_offset,
          new_segments_meta_size, new_segments_meta_size, new_footer});
      footer_ = new_footer;
      current_header_start_offset_ = new_chain_start;

      chain = &meta_chains_.back();
      header = chain_headers_.back().get();
      meta_buf = buffer_pool_buffers_.back().get();

      // Capture rollback state by value before local pointers change.
      rollback_step1 = [this, saved_footer_before_split, saved_old_chain_footer,
                        saved_old_footer_file_offset,
                        saved_current_header_start]() {
        // Unlink the new chain before dropping its in-memory state.
        if (buffer_pool_handle_->write_meta(
                saved_old_footer_file_offset, sizeof(saved_footer_before_split),
                reinterpret_cast<const char *>(&saved_footer_before_split)) !=
            0) {
          LOG_ERROR(
              "append_segment: rollback_step1 write of old footer FAILED, "
              "file[%s] is now in an inconsistent state -- marking storage "
              "as corrupted; further writes will be rejected.",
              file_name_.c_str());
          corrupted_.store(true, std::memory_order_release);
        }
        if (!meta_chains_.empty()) meta_chains_.pop_back();
        if (!chain_headers_.empty()) chain_headers_.pop_back();
        if (!buffer_pool_buffers_.empty()) buffer_pool_buffers_.pop_back();
        if (!meta_chains_.empty()) {
          meta_chains_.back().footer = saved_old_chain_footer;
        }
        // The unreachable file tail is reusable; shrinking is unnecessary.
        footer_ = saved_footer_before_split;
        current_header_start_offset_ = saved_current_header_start;
      };
    }

    // Step 2: append the segment metadata and persist the last chain.
    uint64_t new_data_index = footer_.content_size;
    uint64_t new_seg_abs_offset =
        chain->header_start_offset + header->content_offset + new_data_index;
    uint64_t new_file_size = new_seg_abs_offset + padded_size;
    if (new_file_size > buffer_pool_->file_size()) {
      if (!buffer_pool_->extend_file(new_file_size)) {
        return IndexError_Runtime;
      }
    }

    // Snapshot every overwritten metadata region for rollback.
    const auto saved_footer = footer_;
    const auto saved_chain_footer = chain->footer;
    const auto saved_segment_ids_offset = chain->segment_ids_offset;
    const size_t meta_entry_off =
        sizeof(IndexFormat::SegmentMeta) * footer_.segment_count;
    const uint32_t new_ids_off =
        chain->segment_ids_offset - static_cast<uint32_t>(id_size);
    char saved_meta_entry[sizeof(IndexFormat::SegmentMeta)];
    std::memcpy(saved_meta_entry, meta_buf + meta_entry_off,
                sizeof(IndexFormat::SegmentMeta));
    std::unique_ptr<char[]> saved_id_bytes(new char[id_size]);
    std::memcpy(saved_id_bytes.get(), meta_buf + new_ids_off, id_size);

    chain->segment_ids_offset -= static_cast<uint32_t>(id_size);
    IndexFormat::SegmentMeta *new_seg =
        reinterpret_cast<IndexFormat::SegmentMeta *>(meta_buf) +
        footer_.segment_count;
    new_seg->segment_id_offset = chain->segment_ids_offset;
    new_seg->data_index = new_data_index;
    new_seg->data_size = 0;
    new_seg->data_crc = 0;
    new_seg->padding_size = padded_size;
    std::memcpy(meta_buf + chain->segment_ids_offset, id.c_str(), id_size);

    footer_.segment_count += 1;
    footer_.content_size += padded_size;
    footer_.total_size += padded_size;
    footer_.segments_meta_crc =
        ailego::Crc32c::Hash(meta_buf, chain->segment_meta_size, 0u);
    IndexFormat::UpdateMetaFooter(&footer_, 0);
    chain->footer = footer_;  // sync in-memory copy for flush_index

    // Restore memory and disk together; a failed restore marks the file bad.
    auto rollback_step2 = [&]() {
      std::memcpy(meta_buf + meta_entry_off, saved_meta_entry,
                  sizeof(IndexFormat::SegmentMeta));
      std::memcpy(meta_buf + new_ids_off, saved_id_bytes.get(), id_size);
      footer_ = saved_footer;
      chain->footer = saved_chain_footer;
      chain->segment_ids_offset = saved_segment_ids_offset;

      const int rc_meta = buffer_pool_handle_->write_meta(
          chain->segment_meta_file_offset, chain->segment_meta_size, meta_buf);
      const int rc_footer = buffer_pool_handle_->write_meta(
          chain->footer_file_offset, sizeof(footer_),
          reinterpret_cast<const char *>(&footer_));
      if (rc_meta != 0 || rc_footer != 0) {
        LOG_ERROR(
            "append_segment: rollback_step2 disk rewrite FAILED "
            "(rc_meta=%d, rc_footer=%d), file[%s] is now in an "
            "inconsistent state -- marking storage as corrupted; further "
            "writes will be rejected.",
            rc_meta, rc_footer, file_name_.c_str());
        corrupted_.store(true, std::memory_order_release);
      }
    };

    if (buffer_pool_handle_->write_meta(chain->segment_meta_file_offset,
                                        chain->segment_meta_size,
                                        meta_buf) != 0) {
      LOG_ERROR("append_segment: write segment_meta failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_step1();
      return IndexError_WriteData;
    }
    if (buffer_pool_handle_->write_meta(
            chain->footer_file_offset, sizeof(footer_),
            reinterpret_cast<const char *>(&footer_)) != 0) {
      LOG_ERROR("append_segment: write footer failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_step1();
      return IndexError_WriteData;
    }

    // Commit both maps transactionally while preserving value addresses.
    auto seg_ins = segments_.end();
    bool seg_inserted = false;
    try {
      auto ins = segments_.emplace(
          id, IndexMapping::SegmentInfo{IndexMapping::Segment{new_seg},
                                        chain->header_start_offset, header});
      if (!ins.second) {
        // Defensive: the exclusive latch should make this unreachable.
        LOG_ERROR(
            "append_segment: duplicate id appeared after commit, file[%s], "
            "id[%s]",
            file_name_.c_str(), id.c_str());
        rollback_step2();
        rollback_step1();
        return IndexError_Duplicate;
      }
      seg_ins = ins.first;
      seg_inserted = true;
      const size_t new_id = id_hash_.size();
      id_hash_.emplace(id, new_id);
    } catch (const std::bad_alloc &) {
      LOG_ERROR(
          "append_segment: in-memory commit OOM, rolling back, file[%s], "
          "id[%s]",
          file_name_.c_str(), id.c_str());
      if (seg_inserted) {
        segments_.erase(seg_ins);
      }
      rollback_step2();
      rollback_step1();
      return IndexError_Runtime;
    }
    max_segment_size_ = std::max<uint64_t>(max_segment_size_, padded_size);
    // extend_file() already grew the page table in place.
    return 0;
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    return (segments_.find(id) != segments_.end());
  }

 private:
  bool read_range(size_t offset, size_t len, char *out) const {
    return cache_enabled_
               ? buffer_pool_handle_->read_range(offset, len, out)
               : buffer_pool_handle_->read_range_bypass(offset, len, out);
  }

  std::atomic<bool> index_dirty_{false};
  std::atomic<uint64_t> pending_check_point_{0};
  // Raised after an unrecoverable rollback; blocks writes until close.
  std::atomic<bool> corrupted_{false};

  // Readers hash across shards; metadata writers lock every shard.
  static constexpr size_t kMappingMutexShards = 32;
  struct alignas(64) MutexShard {
    std::shared_mutex mtx;
  };
  mutable MutexShard mapping_shards_[kMappingMutexShards]{};

  // Mix the thread and instance identities to distribute reader latches.
  size_t mapping_shard_id() const {
    size_t seed = std::hash<std::thread::id>()(std::this_thread::get_id());
    size_t inst = std::hash<const void *>()(static_cast<const void *>(this));
    seed ^= inst + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed % kMappingMutexShards;
  }

  // Exclusive guard for all metadata shards.
  struct AllShardsExclusiveLatch {
    MutexShard *shards_;
    AllShardsExclusiveLatch(MutexShard *shards) : shards_(shards) {
      for (size_t i = 0; i < kMappingMutexShards; ++i) shards_[i].mtx.lock();
    }
    ~AllShardsExclusiveLatch() {
      for (size_t i = 0; i < kMappingMutexShards; ++i) shards_[i].mtx.unlock();
    }
    AllShardsExclusiveLatch(const AllShardsExclusiveLatch &) = delete;
    AllShardsExclusiveLatch &operator=(const AllShardsExclusiveLatch &) =
        delete;
  };

  // Aligned arenas retain legacy pointer-read results until close_index().
  struct ArenaBlock {
    char *base{nullptr};
    size_t size{0};  // Total bytes in this arena (4K-aligned).
    size_t used{0};  // Bytes already handed out (4K-aligned).
  };
  // Requires tmp_buffers_mutex_; alloc_size must be 4K-aligned.
  char *tmp_arena_alloc_locked(size_t alloc_size) {
    static constexpr size_t kAlign = 4096UL;
    static constexpr size_t kArenaSize = 1UL << 20;  // 1 MiB
    if (!tmp_buffers_.empty()) {
      ArenaBlock &back = tmp_buffers_.back();
      if (back.base && back.size - back.used >= alloc_size) {
        char *out = back.base + back.used;
        back.used += alloc_size;
        return out;
      }
    }
    size_t new_size = alloc_size > kArenaSize ? alloc_size : kArenaSize;
    char *p = static_cast<char *>(ailego_aligned_malloc(new_size, kAlign));
    if (!p) {
      return nullptr;
    }
    tmp_buffers_.push_back(ArenaBlock{p, new_size, alloc_size});
    return p;
  }
  std::vector<ArenaBlock> tmp_buffers_{};
  mutable std::mutex tmp_buffers_mutex_{};

  // One entry per legacy pointer-read pin; duplicates are intentional.
  std::vector<size_t> pinned_pages_{};
  mutable std::mutex pinned_pages_mutex_{};

  // buffer manager
  std::string file_name_;
  // Own every chain header referenced by segments_.
  std::vector<std::unique_ptr<IndexFormat::MetaHeader>> chain_headers_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  bool cache_enabled_{false};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};

  // Segment metadata capacity written by init_index().
  uint32_t segment_meta_capacity_{4096u};

  // Per-chain state used by flush_index() and append_segment().
  struct MetaChain {
    uint64_t header_start_offset;
    uint64_t footer_file_offset;
    uint64_t segment_meta_file_offset;
    uint32_t segment_meta_size;
    // Lowest ID-string offset; detects metadata exhaustion.
    uint32_t segment_ids_offset;
    // In-memory footer synchronized by flush and append.
    IndexFormat::MetaFooter footer;
  };
  std::vector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
