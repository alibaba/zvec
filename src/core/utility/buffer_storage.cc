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

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "utility_params.h"

namespace zvec {
namespace core {
namespace {

constexpr size_t kBufferAlignment = 4096UL;

inline bool AlignBufferSize(size_t size, size_t *aligned_size) {
  if (size > std::numeric_limits<size_t>::max() - (kBufferAlignment - 1)) {
    return false;
  }
  *aligned_size = (size + (kBufferAlignment - 1)) & ~(kBufferAlignment - 1);
  return true;
}

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
    //! Constructor.  See segment_info_ for the pointer-stability contract.
    WrappedSegment(BufferStorage *owner, IndexMapping::SegmentInfo *info,
                   const std::string *segment_id)
        : segment_info_(info),
          owner_(owner),
          segment_id_(segment_id),
          capacity_(static_cast<size_t>(info->segment.meta()->data_size +
                                        info->segment.meta()->padding_size)) {}
    //! Destructor
    ~WrappedSegment(void) override = default;

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
        LOG_ERROR("WrappedSegment::fetch: handle is null, file[%s], id[%s]",
                  owner_->file_name_.c_str(), segment_id_->c_str());
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        return 0;
      }
      const size_t abs_offset = absolute_offset(offset);
      if (!owner_->read_range(abs_offset, len, static_cast<char *>(buf))) {
        LOG_ERROR(
            "WrappedSegment::fetch: read_range failed, file[%s], id[%s], "
            "abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_->c_str(), abs_offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment
    size_t read(size_t offset, const void **data, size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::read: handle is null, file[%s], id[%s]",
                  owner_->file_name_.c_str(), segment_id_->c_str());
        *data = nullptr;
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        *data = nullptr;
        return 0;
      }
      const size_t abs_offset = absolute_offset(offset);
      // Without a page cache, retain a copied result for the legacy pointer
      // lifetime. Cached single-page reads below pin the page until close(),
      // including for writable pools, instead of retaining one copy per read.
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (abs_offset + len - 1) / ailego::kVectorPageSize;
      bool force_bypass = !owner_->cache_enabled_;
      if (owner_->cache_enabled_ && first_page == last_page) {
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
      size_t alloc_size = 0;
      if (ailego_unlikely(!AlignBufferSize(len, &alloc_size))) {
        *data = nullptr;
        return 0;
      }
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
            "id[%s], abs_offset=%zu, len=%zu, alloc_size=%zu, align=%zu",
            owner_->file_name_.c_str(), segment_id_->c_str(), abs_offset, len,
            alloc_size, kBufferAlignment);
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
            "id[%s], abs_offset=%zu, len=%zu, first_page=%zu, last_page=%zu",
            owner_->file_name_.c_str(), segment_id_->c_str(), abs_offset, len,
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
            "id[%s]",
            owner_->file_name_.c_str(), segment_id_->c_str());
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*borrow_handle=*/false,
                               /*immutable=*/false);
    }

    size_t read_immutable(size_t offset, MemoryBlock &data,
                          size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*borrow_handle=*/false,
                               /*immutable=*/true);
    }

    //! Borrowed read: the caller keeps the Segment alive until block release.
    size_t read_borrowed(size_t offset, MemoryBlock &data,
                         size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read_borrowed: handle is null, file[%s], "
            "id[%s]",
            owner_->file_name_.c_str(), segment_id_->c_str());
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*borrow_handle=*/true,
                               /*immutable=*/false);
    }

    size_t read_borrowed_immutable(size_t offset, MemoryBlock &data,
                                   size_t len) override {
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*borrow_handle=*/true,
                               /*immutable=*/true);
    }

    bool prefer_borrowed_batch() const override {
      return owner_->cache_enabled_ && owner_->buffer_pool_ != nullptr &&
             !owner_->buffer_pool_->writable() &&
             owner_->buffer_pool_->has_evicted();
    }

    bool prefer_borrowed_batch_for(size_t value_size) const override {
      if (!owner_->cache_enabled_ || owner_->buffer_pool_ == nullptr) {
        return false;
      }
      if (!owner_->buffer_pool_->writable()) {
        return owner_->buffer_pool_->has_evicted();
      }
      // With uniformly distributed records, crossing a page costs roughly
      // value_size/page_size. Below one eighth, scalar pins are cheaper than
      // constructing and resolving a batch (notably 512 B on macOS 16 KiB).
      return value_size >= std::max<size_t>(1, ailego::kVectorPageSize / 8);
    }

    bool read_borrowed_batch(BorrowedRead *reads, size_t count) override {
      return read_borrowed_batch_impl(reads, count, /*immutable=*/false);
    }

    bool read_borrowed_batch_immutable(BorrowedRead *reads,
                                       size_t count) override {
      return read_borrowed_batch_impl(reads, count, /*immutable=*/true);
    }

    bool read_borrowed_batch_impl(BorrowedRead *reads, size_t count,
                                  bool immutable) {
      if (count == 0) {
        return true;
      }
      if (reads == nullptr || owner_->buffer_pool_handle_ == nullptr ||
          owner_->buffer_pool_ == nullptr) {
        return false;
      }

      // Mutable reads from writable pools retain snapshot ownership rules.
      if (!owner_->cache_enabled_ ||
          (owner_->buffer_pool_->writable() && !immutable)) {
        return immutable
                   ? IndexStorage::Segment::read_borrowed_batch_immutable(reads,
                                                                          count)
                   : IndexStorage::Segment::read_borrowed_batch(reads, count);
      }

      // Cross-page vectors need a contiguous copy. A thread-local scratch arena
      // is reused across calls instead of a per-vector malloc/free: the arena
      // is safe to recycle because the previous hop's blocks are destroyed
      // before this call runs (see fast_search_neighbors_buffer). Cross-page
      // results become non-owning views into the arena.
      // Cap the per-thread arena: a batch needing more falls back to malloc so
      // an abnormally wide batch cannot pin unbounded out-of-pool RSS. HNSW
      // hops need only about max_degree * vector_size (tens of KiB).
      static constexpr size_t kMaxArenaBytes = 2UL << 20;  // 2 MiB / thread
      bool batch_arena = false;

      struct BatchState {
        BorrowedRead *request{nullptr};
        size_t abs_offset{0};
        size_t first_page_index{0};
        size_t page_count{0};
        bool copy_result{false};
        char *owned{nullptr};
      };
      struct PageUse {
        size_t unique_index{0};
        size_t state_index{0};
        size_t source_offset{0};
        size_t destination_offset{0};
        size_t length{0};
      };
      struct BatchScratch {
        std::vector<BatchState> states;
        std::vector<ailego::block_id_t> page_ids;
        std::vector<ailego::block_id_t> unique_page_ids;
        std::vector<char *> pages;
        std::vector<PageUse> page_uses;
        std::vector<ailego::block_id_t> admitted_ids;
        std::vector<size_t> admitted_indices;
        std::vector<char *> admitted_pages;
        std::vector<char> arena;  // reused cross-page scratch (arena mode)
        std::vector<char> bypass_page;
      };
      static thread_local BatchScratch scratch;

      scratch.states.clear();
      scratch.page_ids.clear();
      scratch.unique_page_ids.clear();
      scratch.pages.clear();
      scratch.page_uses.clear();
      scratch.admitted_ids.clear();
      scratch.admitted_indices.clear();
      scratch.admitted_pages.clear();
      scratch.states.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        if (reads[i].block != nullptr) {
          reads[i].block->reset();
        }
      }

      auto cleanup_owned = [&]() {
        // Arena slices are non-owning; only malloc-mode buffers are freed.
        for (BatchState &state : scratch.states) {
          if (!batch_arena && state.owned != nullptr) {
            ailego_free(state.owned);
          }
          state.owned = nullptr;
        }
      };
      auto release_pages = [&]() {
        for (size_t i = 0; i < scratch.pages.size(); ++i) {
          if (scratch.pages[i] != nullptr) {
            owner_->buffer_pool_handle_->release_one(
                scratch.unique_page_ids[i]);
            scratch.pages[i] = nullptr;
          }
        }
      };
      auto fail = [&]() {
        cleanup_owned();
        release_pages();
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
          return immutable
                     ? IndexStorage::Segment::read_borrowed_batch_immutable(
                           reads, count)
                     : IndexStorage::Segment::read_borrowed_batch(reads, count);
        }

        const size_t data_size = segment->data_size();
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

        const size_t data_offset = segment->data_offset();
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

      // Pin unique pages rather than one pin per vector occurrence. Besides
      // removing duplicate work, this lets a pressured batch preserve all
      // resident hits and fall back only for pages that really cannot be
      // admitted. MemoryLimitPool itself enforces the shared non-page reserve;
      // checking available bytes here used to turn the entire batch into
      // per-vector bypass reads as soon as the cache reached that reserve.
      scratch.unique_page_ids = scratch.page_ids;
      std::sort(scratch.unique_page_ids.begin(), scratch.unique_page_ids.end());
      scratch.unique_page_ids.erase(std::unique(scratch.unique_page_ids.begin(),
                                                scratch.unique_page_ids.end()),
                                    scratch.unique_page_ids.end());
      scratch.pages.assign(scratch.unique_page_ids.size(), nullptr);

      // Take resident hits without I/O first, then admit only unique misses
      // selected by the compact frequency policy. One-off cold pages bypass
      // the cache, while repeated HNSW graph pages become resident without
      // continuously replacing useful members of the working set.
      for (size_t i = 0; i < scratch.unique_page_ids.size(); ++i) {
        scratch.pages[i] = owner_->buffer_pool_->try_acquire_buffer(
            scratch.unique_page_ids[i]);
        if (scratch.pages[i] == nullptr) {
          // Under pressure, a first-touch HNSW page is usually a one-off
          // random candidate. Bypass it once; a repeated touch is admitted by
          // the pool's compact frequency policy. This avoids replacing one
          // useful resident page for every miss when the working set is much
          // larger than the budget.
          if (owner_->buffer_pool_->should_admit_page(
                  scratch.unique_page_ids[i])) {
            scratch.admitted_ids.push_back(scratch.unique_page_ids[i]);
            scratch.admitted_indices.push_back(i);
          }
        }
      }

      // Bound each acquisition so an oversized diagnostic batch cannot hold
      // the entire cache pinned. Linux retains batched AIO within each chunk;
      // on macOS, failed capacity resolution stops after one chunk instead of
      // rescanning the same pinned cache for every remaining page.
      static constexpr size_t kAdmissionBatchPages = 64;
      scratch.admitted_pages.resize(scratch.admitted_ids.size(), nullptr);
      size_t admitted_begin = 0;
      while (admitted_begin < scratch.admitted_ids.size()) {
        const size_t admitted_count = std::min(
            kAdmissionBatchPages, scratch.admitted_ids.size() - admitted_begin);
        const bool acquired = owner_->buffer_pool_handle_->acquire_pages(
            scratch.admitted_ids.data() + admitted_begin, admitted_count,
            scratch.admitted_pages.data() + admitted_begin);
        if (acquired) {
          for (size_t j = 0; j < admitted_count; ++j) {
            scratch.pages[scratch.admitted_indices[admitted_begin + j]] =
                scratch.admitted_pages[admitted_begin + j];
          }
          admitted_begin += admitted_count;
          continue;
        }

        // acquire_pages() rolls its pins back on failure, but it may have
        // populated part of the chunk before capacity ran out. Re-pin those
        // pages once so that useful work and concurrent single-flight loads
        // are not discarded, then bypass the remaining cold pages.
        for (size_t j = 0; j < admitted_count; ++j) {
          const size_t admitted_index = admitted_begin + j;
          const size_t unique_index = scratch.admitted_indices[admitted_index];
          scratch.pages[unique_index] =
              owner_->buffer_pool_->try_acquire_buffer(
                  scratch.admitted_ids[admitted_index]);
        }
        break;
      }

      // Close races with other query threads after admission and AIO. This is
      // a hit-only probe; it never turns a bypass candidate into new I/O.
      for (size_t i = 0; i < scratch.unique_page_ids.size(); ++i) {
        if (scratch.pages[i] == nullptr) {
          scratch.pages[i] = owner_->buffer_pool_->try_acquire_buffer(
              scratch.unique_page_ids[i]);
        }
      }

      auto unique_index_for = [&](ailego::block_id_t page_id) -> size_t {
        auto it = std::lower_bound(scratch.unique_page_ids.begin(),
                                   scratch.unique_page_ids.end(), page_id);
        return static_cast<size_t>(it - scratch.unique_page_ids.begin());
      };

      // Cross-page values always need a contiguous copy. A single-page value
      // only needs one when that page is on the bypass side of the hybrid
      // batch; resident single-page values keep their normal zero-copy pin.
      for (BatchState &state : scratch.states) {
        state.copy_result = state.page_count > 1;
        for (size_t j = 0; j < state.page_count && !state.copy_result; ++j) {
          const size_t unique_index =
              unique_index_for(scratch.page_ids[state.first_page_index + j]);
          state.copy_result = scratch.pages[unique_index] == nullptr;
        }
      }

      // Reserve copied values once, then hand out bump slices. This includes
      // single-page bypass values, so the direct-read page scratch can be
      // immediately reused for the next unique miss.
      static constexpr size_t kArenaAlign = 64;
      size_t total = 0;
      for (const BatchState &state : scratch.states) {
        if (!state.copy_result) {
          continue;
        }
        const size_t length = state.request->length;
        if (length > std::numeric_limits<size_t>::max() - (kArenaAlign - 1)) {
          return fail();
        }
        const size_t aligned = (length + kArenaAlign - 1) & ~(kArenaAlign - 1);
        if (aligned > kMaxArenaBytes - total) {
          total = kMaxArenaBytes + 1;
          break;
        }
        total += aligned;
      }
      batch_arena = total <= kMaxArenaBytes;
      if (batch_arena) {
        if (scratch.arena.size() < total) {
          scratch.arena.resize(total);
        }
        size_t off = 0;
        for (BatchState &state : scratch.states) {
          if (!state.copy_result) {
            continue;
          }
          state.owned = scratch.arena.data() + off;
          off += (state.request->length + kArenaAlign - 1) & ~(kArenaAlign - 1);
        }
      }
      if (!batch_arena) {
        for (BatchState &state : scratch.states) {
          if (!state.copy_result) {
            continue;
          }
          const size_t length = state.request->length;
          size_t alloc_size = 0;
          if (!AlignBufferSize(length, &alloc_size)) {
            return fail();
          }
          state.owned = static_cast<char *>(
              ailego_aligned_malloc(alloc_size, kBufferAlignment));
          if (state.owned == nullptr) {
            return fail();
          }
        }
      }

      // Describe every copied fragment, sort by unique source page, and read
      // each bypass page exactly once before scattering it to all vectors that
      // overlap that page.
      scratch.page_uses.reserve(total_pages);
      for (size_t state_index = 0; state_index < scratch.states.size();
           ++state_index) {
        const BatchState &state = scratch.states[state_index];
        if (!state.copy_result) {
          continue;
        }
        size_t remaining = state.request->length;
        size_t destination_offset = 0;
        size_t source_offset = state.abs_offset % ailego::kVectorPageSize;
        for (size_t j = 0; j < state.page_count; ++j) {
          const size_t length =
              std::min(remaining, ailego::kVectorPageSize - source_offset);
          scratch.page_uses.push_back(PageUse{
              unique_index_for(scratch.page_ids[state.first_page_index + j]),
              state_index, source_offset, destination_offset, length});
          destination_offset += length;
          remaining -= length;
          source_offset = 0;
        }
      }
      std::sort(scratch.page_uses.begin(), scratch.page_uses.end(),
                [](const PageUse &lhs, const PageUse &rhs) {
                  return lhs.unique_index < rhs.unique_index;
                });

      if (scratch.bypass_page.size() < ailego::kVectorPageSize) {
        scratch.bypass_page.resize(ailego::kVectorPageSize);
      }
      size_t use_begin = 0;
      while (use_begin < scratch.page_uses.size()) {
        const size_t unique_index = scratch.page_uses[use_begin].unique_index;
        const char *source = scratch.pages[unique_index];
        if (source == nullptr) {
          const size_t page_offset =
              scratch.unique_page_ids[unique_index] * ailego::kVectorPageSize;
          const size_t read_length =
              std::min(ailego::kVectorPageSize,
                       owner_->buffer_pool_->file_size() - page_offset);
          if (!owner_->buffer_pool_handle_->read_range_bypass(
                  page_offset, read_length, scratch.bypass_page.data())) {
            return fail();
          }
          source = scratch.bypass_page.data();
        }
        size_t use_end = use_begin;
        while (use_end < scratch.page_uses.size() &&
               scratch.page_uses[use_end].unique_index == unique_index) {
          const PageUse &use = scratch.page_uses[use_end];
          BatchState &state = scratch.states[use.state_index];
          std::memcpy(state.owned + use.destination_offset,
                      source + use.source_offset, use.length);
          ++use_end;
        }
        use_begin = use_end;
      }

      for (BatchState &state : scratch.states) {
        if (state.page_count == 0) {
          state.request->block->reset();
          continue;
        }
        if (!state.copy_result) {
          const size_t unique_index =
              unique_index_for(scratch.page_ids[state.first_page_index]);
          const size_t offset_in_page =
              state.abs_offset % ailego::kVectorPageSize;
          owner_->buffer_pool_handle_->acquire_one(
              scratch.unique_page_ids[unique_index]);
          state.request->block->reset(
              owner_->buffer_pool_handle_.get(),
              scratch.unique_page_ids[unique_index],
              scratch.pages[unique_index] + offset_in_page);
          continue;
        }
        *state.request->block =
            batch_arena
                ? MemoryBlock::MakeBorrowedView(state.owned)
                : MemoryBlock::MakeOwned(state.owned, state.request->length);
        state.owned = nullptr;
      }
      release_pages();
      return true;
    }

    //! Write data into the storage with offset. The shard latch excludes
    //! flush; meta_mtx_ serializes metadata changes within this segment.
    size_t write(size_t offset, const void *data, size_t len) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_)) {
        LOG_ERROR("WrappedSegment::write: pool is null, file[%s], id[%s]",
                  owner_->file_name_.c_str(), segment_id_->c_str());
        return 0;
      }
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::write: storage is marked corrupted, refusing "
            "write, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->c_str());
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
      const uint64_t write_end = offset + len;
      if (write_end > bs_load_acquire(&meta->data_size)) {
        std::lock_guard<std::mutex> meta_latch(meta_mtx_);
        uint64_t cur = bs_load_relaxed(&meta->data_size);
        if (write_end > cur) {
          // Publish padding before data_size to keep the pair consistent.
          bs_store_relaxed(&meta->padding_size, capacity_ - write_end);
          bs_store_release(&meta->data_size, write_end);
        }
      }
      // Fixed-size rewrites are dirty even when data_size is unchanged.
      owner_->set_as_dirty();
      return len;
    }

    bool write_batch(const SegmentData *writes, size_t count) override {
      if (count == 0) {
        return true;
      }
      if (writes == nullptr || count > kMaxCoalescedWrites) {
        return IndexStorage::Segment::write_batch(writes, count);
      }

      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_ ||
                          owner_->corrupted_.load(std::memory_order_acquire))) {
        return false;
      }
      if (!owner_->buffer_pool_->writable()) {
        return true;
      }

      auto meta = segment_info_->segment.meta();
      const size_t data_base = segment_info_->segment_header_start_offset +
                               segment_info_->segment_header->content_offset +
                               meta->data_index;
      std::array<ailego::VecBufferWriteFragment, kMaxCoalescedWrites>
          fragments{};
      size_t page_id = std::numeric_limits<size_t>::max();
      uint64_t write_end = 0;
      bool has_data = false;
      for (size_t i = 0; i < count; ++i) {
        const auto &write = writes[i];
        if (ailego_unlikely(write.offset > capacity_ ||
                            write.length > capacity_ - write.offset ||
                            (write.length != 0 && write.data == nullptr))) {
          return false;
        }
        if (write.length == 0) {
          continue;
        }
        const size_t abs_offset = data_base + write.offset;
        fragments[i] = {abs_offset, write.length,
                        static_cast<const char *>(write.data)};
        write_end = std::max<uint64_t>(write_end, write.offset + write.length);
        const size_t write_page = abs_offset / ailego::kVectorPageSize;
        const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
        if (write.length > ailego::kVectorPageSize - offset_in_page ||
            (has_data && write_page != page_id)) {
          latch.unlock();
          return IndexStorage::Segment::write_batch(writes, count);
        }
        page_id = write_page;
        has_data = true;
      }

      if (has_data && owner_->buffer_pool_handle_->write_fragments(
                          fragments.data(), count) != 0) {
        return false;
      }
      if (write_end > bs_load_acquire(&meta->data_size)) {
        std::lock_guard<std::mutex> meta_latch(meta_mtx_);
        const uint64_t current = bs_load_relaxed(&meta->data_size);
        if (write_end > current) {
          bs_store_relaxed(&meta->padding_size, capacity_ - write_end);
          bs_store_release(&meta->data_size, write_end);
        }
      }
      owner_->set_as_dirty();
      return true;
    }

    //! Resize size of data.  See write() for the locking contract.
    size_t resize(size_t size) override {
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::resize: storage is marked corrupted, refusing "
            "resize, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->c_str());
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
            "refusing CRC update, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->c_str());
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

   private:
    static constexpr size_t kMaxCoalescedWrites = 8;
    // Stable unordered_map value address; reparses update this object in place.
    IndexMapping::SegmentInfo *segment_info_{nullptr};
    // Serializes metadata writers within this segment.
    mutable std::mutex meta_mtx_{};

    size_t clamp_length(size_t *offset, size_t len) const {
      const size_t current_size = data_size();
      if (ailego_unlikely(*offset > current_size)) {
        *offset = current_size;
        return 0;
      }
      return std::min(len, current_size - *offset);
    }

    size_t absolute_offset(size_t offset) const {
      return data_offset() + offset;
    }

    size_t read_memory_block(size_t abs_offset, MemoryBlock &data, size_t len,
                             bool borrow_handle, bool immutable) const {
      const bool can_pin = owner_->cache_enabled_ &&
                           (immutable || !owner_->buffer_pool_->writable());
      bool force_bypass = !owner_->cache_enabled_;
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      if (can_pin && len <= ailego::kVectorPageSize - offset_in_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (raw != nullptr) {
          if (borrow_handle) {
            data.reset(owner_->buffer_pool_handle_.get(), page_id, raw);
          } else {
            data.reset(owner_->buffer_pool_handle_, page_id, raw);
          }
          return len;
        }
        force_bypass = true;
      }

      // Mutable graph snapshots are typically only a few hundred bytes. They
      // are copied from cached pages and are never submitted to O_DIRECT, so a
      // page-aligned allocation only adds allocator and RSS overhead. Keep
      // alignment for larger/vector snapshots whose distance kernels benefit
      // from it, but let the allocator's thread cache handle small objects.
      static constexpr size_t kSmallSnapshotBytes = 512;
      size_t alloc_size = len;
      char *tmp = nullptr;
      if (!immutable && len <= kSmallSnapshotBytes) {
        tmp = static_cast<char *>(ailego_malloc(len));
      } else {
        if (ailego_unlikely(!AlignBufferSize(len, &alloc_size))) {
          return 0;
        }
        tmp = static_cast<char *>(
            ailego_aligned_malloc(alloc_size, kBufferAlignment));
      }
      if (tmp == nullptr) {
        LOG_ERROR(
            "WrappedSegment::read: owned buffer allocation failed, file[%s], "
            "id[%s], abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_->c_str(), abs_offset, len);
        return 0;
      }
      bool read_ok =
          force_bypass ? owner_->buffer_pool_handle_->read_range_bypass(
                             abs_offset, len, tmp)
          : immutable ? owner_->buffer_pool_handle_->read_range_immutable(
                            abs_offset, len, tmp)
                      : owner_->read_range(abs_offset, len, tmp);
      if (!read_ok && immutable && !force_bypass) {
        // A cross-page immutable read can lose a capacity race after choosing
        // the cache path. The bytes are immutable, so retry directly instead
        // of failing the HNSW expansion.
        read_ok = owner_->buffer_pool_handle_->read_range_bypass(abs_offset,
                                                                 len, tmp);
      }
      if (!read_ok) {
        ailego_free(tmp);
        LOG_ERROR(
            "WrappedSegment::read: owned read failed, file[%s], id[%s], "
            "abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_->c_str(), abs_offset, len);
        return 0;
      }
      data = MemoryBlock::MakeOwned(tmp, len);
      return len;
    }

    BufferStorage *owner_{nullptr};
    // Stable alongside segment_info_; unordered_map rehash preserves element
    // references and pointers.
    const std::string *segment_id_{nullptr};
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
        ailego::VecBufferPool::metadata_bytes_for_page_count(
            page_count, /*writable=*/create_if_missing);
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
    LOG_INFO("BufferStorage opened: file=%s, writable=%d, segment_count=%zu",
             file_name_.c_str(), static_cast<int>(create_if_missing),
             segments_.size());
    return 0;
  }

  // Called from single-threaded open or under AllShardsExclusiveLatch.
  int ParseHeader(size_t offset, IndexFormat::MetaHeader *out) {
    constexpr size_t kHeaderSize = sizeof(IndexFormat::MetaHeader);
    if (buffer_pool_handle_->get_meta(offset, kHeaderSize,
                                      reinterpret_cast<char *>(out)) != 0) {
      LOG_ERROR("Get segment header failed.");
      return IndexError_Runtime;
    }
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

  int ParseFooter(size_t offset, IndexFormat::MetaFooter *footer) {
    if (buffer_pool_handle_->get_meta(offset, sizeof(*footer),
                                      reinterpret_cast<char *>(footer)) != 0) {
      LOG_ERROR("Get segment footer failed.");
      return IndexError_Runtime;
    }
    if (offset < static_cast<size_t>(footer->segments_meta_size)) {
      LOG_ERROR("Footer meta size is invalid.");
      return IndexError_InvalidLength;
    }
    if (ailego::Crc32c::Hash(footer, sizeof(*footer), footer->footer_crc) !=
        footer->footer_crc) {
      LOG_ERROR("Footer meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    return 0;
  }

  int ParseSegment(size_t offset, uint64_t header_start_offset,
                   IndexFormat::MetaHeader *chain_header,
                   const IndexFormat::MetaFooter &footer,
                   uint32_t &out_segment_ids_offset,
                   std::unique_ptr<char[]> &segment_buffer) {
    segment_buffer = std::make_unique<char[]>(footer.segments_meta_size);
    if (buffer_pool_handle_->get_meta(offset, footer.segments_meta_size,
                                      segment_buffer.get()) != 0) {
      LOG_ERROR("Get segment meta failed.");
      return IndexError_Runtime;
    }
    if (ailego::Crc32c::Hash(segment_buffer.get(), footer.segments_meta_size,
                             0u) != footer.segments_meta_crc) {
      LOG_ERROR("Index segments meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    if (sizeof(IndexFormat::SegmentMeta) * footer.segment_count >
        footer.segments_meta_size) {
      return IndexError_InvalidLength;
    }
    IndexFormat::SegmentMeta *segment_start =
        reinterpret_cast<IndexFormat::SegmentMeta *>(segment_buffer.get());
    uint32_t segment_ids_offset = footer.segments_meta_size;
    for (IndexFormat::SegmentMeta *iter = segment_start,
                                  *end = segment_start + footer.segment_count;
         iter != end; ++iter) {
      if (iter->segment_id_offset >= footer.segments_meta_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index > footer.content_size) {
        return IndexError_InvalidValue;
      }
      if (iter->data_index + iter->data_size > footer.content_size) {
        return IndexError_InvalidLength;
      }

      if (iter->segment_id_offset < segment_ids_offset) {
        segment_ids_offset = iter->segment_id_offset;
      }
      // Bound ID parsing to the metadata buffer.
      const char *seg_name_start =
          reinterpret_cast<const char *>(segment_start) +
          iter->segment_id_offset;
      const size_t seg_name_max =
          footer.segments_meta_size - iter->segment_id_offset;
      const size_t seg_name_len = ::strnlen(seg_name_start, seg_name_max);
      if (seg_name_len == seg_name_max) {
        LOG_ERROR("ParseSegment: segment_id missing NUL terminator, file[%s]",
                  file_name_.c_str());
        return IndexError_InvalidValue;
      }
      const std::string seg_name(seg_name_start, seg_name_len);
      // Update in place so existing WrappedSegment pointers remain valid.
      segments_[seg_name] = IndexMapping::SegmentInfo{
          IndexMapping::Segment{iter}, header_start_offset, chain_header};
    }
    out_segment_ids_offset = segment_ids_offset;
    return 0;
  }

  int ParseToMapping() {
    uint64_t header_start_offset = 0;
    while (true) {
      int ret;
      auto header = std::make_unique<IndexFormat::MetaHeader>();
      IndexFormat::MetaHeader *chain_header = header.get();
      ret = ParseHeader(header_start_offset, chain_header);
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
          chain_header->meta_footer_offset + header_start_offset;
      // Reject uint64 wrap-around and offsets past file_size.
      if (footer_offset < header_start_offset ||
          footer_offset + sizeof(IndexFormat::MetaFooter) >
              buffer_pool_->file_size()) {
        LOG_ERROR("ParseToMapping: invalid footer_offset=%" PRIu64
                  " (header=%" PRIu64 ", file_size=%zu), file[%s]",
                  footer_offset, header_start_offset, buffer_pool_->file_size(),
                  file_name_.c_str());
        return IndexError_InvalidValue;
      }
      IndexFormat::MetaFooter footer{};
      ret = ParseFooter(footer_offset, &footer);
      if (ret != 0) {
        LOG_ERROR("Failed to parse footer, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Unpack segment table
      const uint64_t segment_start_offset =
          footer_offset - footer.segments_meta_size;
      uint32_t segment_ids_offset = footer.segments_meta_size;
      std::unique_ptr<char[]> segment_buffer;
      ret =
          ParseSegment(segment_start_offset, header_start_offset, chain_header,
                       footer, segment_ids_offset, segment_buffer);
      if (ret != 0) {
        LOG_ERROR("Failed to parse segment, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Own all state referenced by this metadata chain in one object.
      meta_chains_.push_back({std::move(header), std::move(segment_buffer),
                              header_start_offset, segment_ids_offset, footer});

      if (footer.next_meta_header_offset == 0) {
        break;
      }
      // Reject invalid links before following the metadata chain.
      const uint64_t next_off = footer.next_meta_header_offset;
      if (next_off <= header_start_offset ||
          next_off + sizeof(IndexFormat::MetaHeader) >
              buffer_pool_->file_size()) {
        LOG_ERROR("ParseToMapping: invalid next_meta_header_offset=%" PRIu64
                  " (current=%" PRIu64 ", file_size=%zu), file[%s]",
                  next_off, header_start_offset, buffer_pool_->file_size(),
                  file_name_.c_str());
        return IndexError_InvalidValue;
      }
      // Bound corrupted metadata chains.
      constexpr size_t kMaxChains = 1024;
      if (meta_chains_.size() >= kMaxChains) {
        LOG_ERROR(
            "ParseToMapping: chain count exceeds limit %zu, file[%s] may "
            "be corrupted",
            kMaxChains, file_name_.c_str());
        return IndexError_InvalidLength;
      }
      header_start_offset = next_off;
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
    return meta_chains_.empty() ? 0 : meta_chains_.back().footer.check_point;
  }

  //! Retrieve a segment by id
  IndexStorage::Segment::Pointer get(const std::string &id, int) override {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    auto seg_iter = segments_.find(id);
    if (seg_iter == segments_.end()) {
      return {};
    }
    return std::make_shared<WrappedSegment>(this, &seg_iter->second,
                                            &seg_iter->first);
  }

  //! Test if it a segment exists
  bool has(const std::string &id) const override {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    return segments_.find(id) != segments_.end();
  }

  //! Retrieve magic number of index
  uint32_t magic(void) const override {
    if (meta_chains_.empty()) {
      return 0u;
    }
    return meta_chains_.front().header->magic;
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
    for (size_t ci = 0; ci < meta_chains_.size(); ++ci) {
      MetaChain &mchain = meta_chains_[ci];
      const char *seg_buf = mchain.segment_meta.get();
      mchain.footer.segments_meta_crc =
          ailego::Crc32c::Hash(seg_buf, mchain.footer.segments_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&mchain.footer, consumed_chkp);
      if (buffer_pool_handle_->write_meta(mchain.segment_meta_file_offset(),
                                          mchain.footer.segments_meta_size,
                                          seg_buf) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        restore_chkp_on_failure();
        return IndexError_WriteData;
      }
      if (buffer_pool_handle_->write_meta(
              mchain.footer_file_offset(), sizeof(mchain.footer),
              reinterpret_cast<const char *>(&mchain.footer)) != 0) {
        LOG_ERROR("Failed to write footer: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        index_dirty_.store(true, std::memory_order_relaxed);
        restore_chkp_on_failure();
        return IndexError_WriteData;
      }
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
    segments_.clear();
    meta_chains_.clear();
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
    if (meta_chains_.empty()) {
      LOG_ERROR("append_segment: invalid state, file[%s]", file_name_.c_str());
      return IndexError_Runtime;
    }

    // Page-aligned padded size; matches IndexMapping::CalcPageAlignedSize().
    const size_t page_size = ailego::kVectorPageSize;
    const size_t padded_size = (size + page_size - 1) / page_size * page_size;

    size_t id_size = id.length() + 1;
    size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;
    MetaChain *chain = &meta_chains_.back();
    IndexFormat::MetaHeader *header = chain->header.get();
    char *meta_buf = chain->segment_meta.get();
    IndexFormat::MetaFooter *footer = &chain->footer;

    const auto footer_before_split = *footer;
    const uint64_t old_footer_file_offset = chain->footer_file_offset();
    bool chain_split = false;

    // Step 1: split a full metadata chain.
    if (sizeof(IndexFormat::SegmentMeta) * footer->segment_count + need_size >
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

      // Stage the linked old footer without mutating the chain yet.
      IndexFormat::MetaFooter linked_footer = footer_before_split;
      linked_footer.next_meta_header_offset = new_chain_start;
      IndexFormat::UpdateMetaFooter(&linked_footer, 0);

      if (buffer_pool_handle_->write_meta(
              chain->footer_file_offset(), sizeof(linked_footer),
              reinterpret_cast<const char *>(&linked_footer)) != 0) {
        LOG_ERROR("append_segment: write old footer failed, file[%s]",
                  file_name_.c_str());
        return IndexError_WriteData;
      }

      // Restore the old link after a failed split; reject writes if rollback
      // also fails.
      auto undo_old_footer = [this, chain, &footer_before_split]() {
        if (buffer_pool_handle_->write_meta(
                chain->footer_file_offset(), sizeof(footer_before_split),
                reinterpret_cast<const char *>(&footer_before_split)) != 0) {
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
      uint64_t new_footer_file_offset =
          new_chain_start + new_header->meta_footer_offset;
      if (buffer_pool_handle_->write_meta(
              new_footer_file_offset, sizeof(new_footer),
              reinterpret_cast<const char *>(&new_footer)) != 0) {
        undo_old_footer();
        return IndexError_WriteData;
      }

      // Reserve before committing the split so pointer updates cannot fail.
      try {
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
      meta_chains_.push_back(MetaChain{std::move(new_header),
                                       std::move(new_meta_buf), new_chain_start,
                                       new_segments_meta_size, new_footer});

      chain = &meta_chains_.back();
      header = chain->header.get();
      meta_buf = chain->segment_meta.get();
      footer = &chain->footer;
      chain_split = true;
    }

    auto rollback_chain_split = [&]() {
      if (chain_split) {
        // Unlink the new chain before dropping its in-memory state.
        if (buffer_pool_handle_->write_meta(
                old_footer_file_offset, sizeof(footer_before_split),
                reinterpret_cast<const char *>(&footer_before_split)) != 0) {
          LOG_ERROR(
              "append_segment: chain-split rollback write of old footer "
              "FAILED, "
              "file[%s] is now in an inconsistent state -- marking storage "
              "as corrupted; further writes will be rejected.",
              file_name_.c_str());
          corrupted_.store(true, std::memory_order_release);
        }
        meta_chains_.pop_back();
        meta_chains_.back().footer = footer_before_split;
        // The unreachable file tail is reusable; shrinking is unnecessary.
      }
    };

    // Step 2: append the segment metadata and persist the last chain.
    uint64_t new_data_index = footer->content_size;
    uint64_t new_seg_abs_offset =
        chain->header_start_offset + header->content_offset + new_data_index;
    uint64_t new_file_size = new_seg_abs_offset + padded_size;
    if (new_file_size > buffer_pool_->file_size()) {
      if (!buffer_pool_->extend_file(new_file_size)) {
        rollback_chain_split();
        return IndexError_Runtime;
      }
    }

    // Snapshot every overwritten metadata region for rollback.
    const auto saved_footer = *footer;
    const auto saved_segment_ids_offset = chain->segment_ids_offset;
    const size_t meta_entry_off =
        sizeof(IndexFormat::SegmentMeta) * footer->segment_count;
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
        footer->segment_count;
    new_seg->segment_id_offset = chain->segment_ids_offset;
    new_seg->data_index = new_data_index;
    new_seg->data_size = 0;
    new_seg->data_crc = 0;
    new_seg->padding_size = padded_size;
    std::memcpy(meta_buf + chain->segment_ids_offset, id.c_str(), id_size);

    footer->segment_count += 1;
    footer->content_size += padded_size;
    footer->total_size += padded_size;
    footer->segments_meta_crc =
        ailego::Crc32c::Hash(meta_buf, footer->segments_meta_size, 0u);
    IndexFormat::UpdateMetaFooter(footer, 0);

    // Restore memory and disk together; a failed restore marks the file bad.
    auto rollback_step2 = [&]() {
      std::memcpy(meta_buf + meta_entry_off, saved_meta_entry,
                  sizeof(IndexFormat::SegmentMeta));
      std::memcpy(meta_buf + new_ids_off, saved_id_bytes.get(), id_size);
      *footer = saved_footer;
      chain->segment_ids_offset = saved_segment_ids_offset;

      const int rc_meta =
          buffer_pool_handle_->write_meta(chain->segment_meta_file_offset(),
                                          footer->segments_meta_size, meta_buf);
      const int rc_footer = buffer_pool_handle_->write_meta(
          chain->footer_file_offset(), sizeof(*footer),
          reinterpret_cast<const char *>(footer));
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

    if (buffer_pool_handle_->write_meta(chain->segment_meta_file_offset(),
                                        footer->segments_meta_size,
                                        meta_buf) != 0) {
      LOG_ERROR("append_segment: write segment_meta failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_chain_split();
      return IndexError_WriteData;
    }
    if (buffer_pool_handle_->write_meta(
            chain->footer_file_offset(), sizeof(*footer),
            reinterpret_cast<const char *>(footer)) != 0) {
      LOG_ERROR("append_segment: write footer failed, file[%s]",
                file_name_.c_str());
      rollback_step2();
      rollback_chain_split();
      return IndexError_WriteData;
    }

    // Publish the segment only after its metadata is durable.
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
        rollback_chain_split();
        return IndexError_Duplicate;
      }
    } catch (const std::bad_alloc &) {
      LOG_ERROR(
          "append_segment: in-memory commit OOM, rolling back, file[%s], "
          "id[%s]",
          file_name_.c_str(), id.c_str());
      rollback_step2();
      rollback_chain_split();
      return IndexError_Runtime;
    }
    // extend_file() already grew the page table in place.
    return 0;
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
    static thread_local const size_t thread_seed =
        std::hash<std::thread::id>()(std::this_thread::get_id());
    size_t seed = thread_seed;
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
    char *p =
        static_cast<char *>(ailego_aligned_malloc(new_size, kBufferAlignment));
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
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  bool cache_enabled_{false};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};

  // Segment metadata capacity written by init_index().
  uint32_t segment_meta_capacity_{4096u};

  // Per-chain ownership and state used by parsing, flushing, and appending.
  struct MetaChain {
    // Pointees stay stable across vector moves and are referenced by segments_.
    std::unique_ptr<IndexFormat::MetaHeader> header;
    std::unique_ptr<char[]> segment_meta;
    uint64_t header_start_offset;
    // Lowest ID-string offset; detects metadata exhaustion.
    uint32_t segment_ids_offset;
    // Authoritative in-memory footer for this chain.
    IndexFormat::MetaFooter footer;

    uint64_t footer_file_offset() const {
      return header_start_offset + header->meta_footer_offset;
    }

    uint64_t segment_meta_file_offset() const {
      return footer_file_offset() - footer.segments_meta_size;
    }
  };
  std::vector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
