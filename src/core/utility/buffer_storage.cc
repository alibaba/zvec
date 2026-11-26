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
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <ailego/pattern/defer.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/container/lazy_map.h>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_mapping.h>
#include <zvec/core/framework/index_version.h>
#include "reusable_read_scratch.h"
#include "utility_params.h"

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <io.h>
#else
#include <sys/stat.h>
#endif

namespace zvec {
namespace core {
namespace {

constexpr size_t kBufferAlignment = 4096UL;
thread_local int buffer_storage_read_error = IndexError_ReadData;
thread_local int buffer_storage_write_error = IndexError_WriteData;

inline bool AlignBufferSize(size_t size, size_t *aligned_size,
                            size_t alignment = kBufferAlignment) {
  if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    return false;
  }
  *aligned_size = (size + (alignment - 1)) & ~(alignment - 1);
  return true;
}

ailego::File::Identity FileIdentity(int fd) {
#if defined(_MSC_VER)
  return ailego::File::Identify(reinterpret_cast<HANDLE>(_get_osfhandle(fd)));
#else
  return ailego::File::Identify(fd);
#endif
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

/*! Buffer Storage
 */
class BufferStorage : public IndexStorage {
  struct DirectoryGeneration {
    std::atomic<bool> active{true};
  };

  struct PreparedWriteState {
    explicit PreparedWriteState(BufferStorage *storage) : owner(storage) {}
    std::mutex lifetime_mutex;
    BufferStorage *owner;
    std::atomic<size_t> active{0};
  };

  struct SegmentState {
    IndexMapping::SegmentInfo info{};
    size_t capacity{0};
    std::mutex metadata_mutex;
  };


 public:
  /*! Index Storage Segment
   */
  class WrappedSegment : public IndexStorage::Segment,
                         public std::enable_shared_from_this<Segment> {
   public:
    WrappedSegment(BufferStorage *owner, SegmentState *state,
                   const std::string_view *segment_id)
        : state_(state),
          owner_(owner),
          segment_id_(segment_id),
          generation_(owner->directory_generation_) {}
    //! Destructor
    ~WrappedSegment() override = default;

    //! Retrieve size of data
    //!
    //! data_size / padding_size are mutated lock-free by concurrent
    //! writers (write/resize) and observed by concurrent readers on the
    //! lock-free hot path.  Use acquire/release ordering so weakly-ordered
    //! ARM (e.g. Android arm64) cannot see stale values that would cause
    //! read() to truncate len to 0.
    size_t data_size() const override {
      if (!active()) return 0;
      return static_cast<size_t>(
          bs_load_acquire(&state_->info.segment.meta()->data_size));
    }

    size_t data_offset() const override {
      if (!active()) return 0;
      return state_->info.segment_header_start_offset +
             state_->info.segment_header->content_offset +
             state_->info.segment.meta()->data_index;
    }

    //! Retrieve crc of data
    uint32_t data_crc() const override {
      if (!active()) return 0;
      std::lock_guard<std::mutex> lock(state_->metadata_mutex);
      return state_->info.segment.meta()->data_crc;
    }

    //! Retrieve size of padding
    size_t padding_size() const override {
      if (!active()) return 0;
      return static_cast<size_t>(
          bs_load_acquire(&state_->info.segment.meta()->padding_size));
    }

    //! Retrieve capacity of segment
    size_t capacity() const override {
      return active() ? state_->capacity : 0;
    }

    int read_error() const override {
      return buffer_storage_read_error;
    }

    //! Fetch data from segment (with own buffer)
    size_t fetch(size_t offset, void *buf, size_t len) const override {
      buffer_storage_read_error = IndexError_ReadData;
      if (!active()) return 0;
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR("WrappedSegment::fetch: handle is null, file[%s], id[%s]",
                  owner_->file_name_.c_str(), segment_id_->data());
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
            owner_->file_name_.c_str(), segment_id_->data(), abs_offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment
    //! Read data into a bounded-lifetime block.
    size_t read(size_t offset, MemoryBlock &data, size_t len) override {
      if (!check_read(data)) return 0;
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        LOG_ERROR(
            "WrappedSegment::read(MemoryBlock&): handle is null, file[%s], "
            "id[%s]",
            owner_->file_name_.c_str(), segment_id_->data());
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*immutable=*/false);
    }

    size_t read_immutable(size_t offset, MemoryBlock &data,
                          size_t len) override {
      if (!check_read(data)) return 0;
      if (ailego_unlikely(!owner_->buffer_pool_handle_)) {
        return 0;
      }
      len = clamp_length(&offset, len);
      if (len == 0) {
        data.reset();
        return 0;
      }
      return read_memory_block(absolute_offset(offset), data, len,
                               /*immutable=*/true);
    }

    bool prefer_borrowed_batch() const override {
      return active() && owner_->cache_enabled_ &&
             owner_->buffer_pool_ != nullptr &&
             !owner_->buffer_pool_->writable() &&
             owner_->buffer_pool_->has_evicted();
    }

    bool prefer_borrowed_batch_for(size_t value_size) const override {
      if (!active() || !owner_->cache_enabled_ ||
          owner_->buffer_pool_ == nullptr) {
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

    static constexpr size_t kResidentBatch = 32;

    bool try_read_resident_immutable(BorrowedRead *reads, size_t count) {
      // Keep scratch bounded without imposing a limit on resident requests.
      // If a later chunk misses, the caller resets every earlier result before
      // entering the general path, including aliased output handles.
      for (size_t begin = 0; begin < count;) {
        const size_t chunk = std::min(kResidentBatch, count - begin);
        if (!try_read_resident_immutable_chunk(reads + begin, chunk))
          return false;
        begin += chunk;
      }
      return true;
    }

    bool try_read_resident_immutable_chunk(BorrowedRead *reads, size_t count) {
      // Retain source pins until each output owns its pin. An output reset
      // must not invalidate a source used by another request in this chunk.
      assert(count <= kResidentBatch);
      std::array<ailego::block_id_t, kResidentBatch> ids;
      std::array<char *, kResidentBatch> pages;
      std::array<size_t, kResidentBatch> indices, offsets;
      size_t pinned = 0;
      AILEGO_DEFER(
          [&] { owner_->buffer_pool_->release_pages(ids.data(), pinned); });
      for (size_t i = 0; i < count; ++i) {
        const auto &request = reads[i];
        auto *segment = dynamic_cast<WrappedSegment *>(request.segment);
        if (!segment || segment->generation_ != generation_ || !request.block)
          return false;
        const size_t data_size = segment->data_size();
        if (request.offset > data_size ||
            request.length > data_size - request.offset)
          return false;
        indices[i] = kResidentBatch;
        if (!request.length) continue;
        const size_t start = segment->data_offset();
        if (request.offset > std::numeric_limits<size_t>::max() - start)
          return false;
        const size_t absolute = start + request.offset;
        const size_t offset = absolute % ailego::kVectorPageSize;
        if (request.length > ailego::kVectorPageSize - offset) return false;
        const auto id = absolute / ailego::kVectorPageSize;
        size_t index = 0;
        while (index < pinned && ids[index] != id) ++index;
        if (index == pinned) {
          char *page = owner_->buffer_pool_->try_acquire_buffer(id);
          if (!page) return false;
          ids[pinned] = id;
          pages[pinned++] = page;
        }
        indices[i] = index;
        offsets[i] = offset;
      }
      for (size_t i = 0; i < count; ++i) {
        const size_t index = indices[i];
        if (index == kResidentBatch) {
          reads[i].block->reset();
          continue;
        }
        owner_->buffer_pool_handle_->acquire_one(ids[index]);
        auto *segment = static_cast<WrappedSegment *>(reads[i].segment);
        segment->reset_pinned_block(*reads[i].block, ids[index],
                                    pages[index] + offsets[i]);
      }
      return true;
    }

    bool read_borrowed_batch_impl(BorrowedRead *reads, size_t count,
                                  bool immutable) try {
      buffer_storage_read_error = IndexError_ReadData;
      if (count == 0) {
        return true;
      }
      if (!active() || reads == nullptr ||
          owner_->buffer_pool_handle_ == nullptr ||
          owner_->buffer_pool_ == nullptr) {
        return false;
      }

      // Mutable reads still publish owned snapshots, not writable page pins.
      const bool mutable_snapshot =
          owner_->buffer_pool_->writable() && !immutable;
      if (!owner_->cache_enabled_) {
        return immutable
                   ? IndexStorage::Segment::read_borrowed_batch_immutable(reads,
                                                                          count)
                   : IndexStorage::Segment::read_borrowed_batch(reads, count);
      }

      if ((immutable || !owner_->buffer_pool_->writable()) &&
          try_read_resident_immutable(reads, count))
        return true;

      // Copied results share one bounded allocation. Only returned slices
      // retain its budget; an idle thread must not keep result bytes alive.
      // Larger batches use individually owned buffers.
      static constexpr size_t kMaxArenaBytes = 2UL
                                               << 20;  // 2 MiB shared / batch
      static constexpr size_t kArenaAlign = AILEGO_CACHE_LINE_SIZE;
#if defined(__APPLE__) || (defined(__linux__) && !defined(__ANDROID__))
      // Both read-only backends can submit cold ranges together. A one-page
      // window would prevent Linux's batch AIO path from ever being reached.
      static constexpr size_t kBypassBatchPages = 16;
#else
      static constexpr size_t kBypassBatchPages = 1;
#endif
      bool batch_arena = false;

      struct BatchState {
        BorrowedRead *request{nullptr};
        size_t abs_offset{0};
        size_t first_page_index{0};
        size_t page_count{0};
        bool copy_result{false};
        char *owned{nullptr};
        MemoryBlock owned_block;
      };
      struct PageUse {
        size_t unique_index{0};
        size_t state_index{0};
        size_t source_offset{0};
        size_t destination_offset{0};
        size_t length{0};
      };
      using BatchArena = ailego::BudgetedAlignedBuffer;
      struct BatchScratch {
        std::vector<BatchState, ailego::BudgetedAllocator<BatchState>> states;
        std::vector<ailego::block_id_t,
                    ailego::BudgetedAllocator<ailego::block_id_t>>
            page_ids, unique_page_ids, admitted_ids;
        std::vector<char *, ailego::BudgetedAllocator<char *>> pages,
            admitted_pages;
        std::vector<PageUse, ailego::BudgetedAllocator<PageUse>> page_uses;
        std::vector<size_t, ailego::BudgetedAllocator<size_t>> admitted_indices;
        std::shared_ptr<BatchArena> arena;
        // O_DIRECT must read into this already-reserved page buffer without
        // allocating another bounce page after the cache has been pinned.
        ailego::BudgetedMappedBuffer bypass_page;
      };
      // Descriptors scale with request count and touched pages. Own and
      // charge them per operation, rather than retaining each thread's peak.
      BatchScratch scratch;
      for (size_t i = 0; i < count; ++i) {
        if (reads[i].block != nullptr) {
          reads[i].block->reset();
        }
      }

      auto cleanup_owned = [&]() {
        // Arena slices share their backing owner; individual snapshots
        // release their own heap allocation or mapped backing.
        for (BatchState &state : scratch.states) {
          if (!batch_arena && state.owned != nullptr) {
            state.owned_block.reset();
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
      // Allocation failures must release acquired pages and any earlier
      // malloc fallback buffers just like an explicit read failure does.
      struct Cleanup {
        decltype(cleanup_owned) &owned;
        decltype(release_pages) &pages;
        ~Cleanup() {
          owned();
          pages();
        }
      } cleanup{cleanup_owned, release_pages};
      scratch.states.reserve(count);
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
      size_t copy_upper_bound = 0;
      for (size_t i = 0; i < count; ++i) {
        BorrowedRead &request = reads[i];
        auto *segment = dynamic_cast<WrappedSegment *>(request.segment);
        if (segment == nullptr || segment->generation_ != generation_ ||
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
        if (request.length >
                std::numeric_limits<size_t>::max() - (kArenaAlign - 1) ||
            ((request.length + kArenaAlign - 1) & ~(kArenaAlign - 1)) >
                std::numeric_limits<size_t>::max() - copy_upper_bound)
          return fail();
        copy_upper_bound +=
            (request.length + kArenaAlign - 1) & ~(kArenaAlign - 1);
        scratch.states.emplace_back(state);
        for (size_t page = first_page; page <= last_page; ++page) {
          scratch.page_ids.emplace_back(page);
        }
      }

      // Reserve bounded fallback space before pinning pages. Otherwise a
      // large batch can pin the entire budget and prevent its own fallback.
      if (copy_upper_bound && copy_upper_bound <= kMaxArenaBytes) {
        // This worst-case arena is optional. Avoid waiting/reclaiming for an
        // obviously oversized reservation when scalar pins/copies may fit.
        auto &memory = ailego::MemoryLimitPool::get_instance();
        if (memory.capacity() && copy_upper_bound > memory.available())
          throw std::bad_alloc();
        scratch.arena = std::allocate_shared<BatchArena>(
            ailego::BudgetedAllocator<BatchArena>{}, copy_upper_bound,
            kArenaAlign);
      }
      // Reserve before pinning any cache page, including the bounded cold-read
      // batch. Writable paths still use synchronized scalar bypass reads.
      size_t bypass_batch_pages =
          std::min(owner_->buffer_pool_->writable() ||
                           !owner_->buffer_pool_->has_evicted()
                       ? size_t{1}
                       : kBypassBatchPages,
                   total_pages);
      while (bypass_batch_pages != 0) {
        try {
          scratch.bypass_page.resize_or_throw(bypass_batch_pages *
                                              ailego::kVectorPageSize);
          break;
        } catch (const std::bad_alloc &) {
          // Batching is optional: retain the original one-page fallback when
          // the remaining budget cannot accommodate a larger I/O window.
          if (bypass_batch_pages == 1) throw;
          bypass_batch_pages /= 2;
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
      // All descriptors must fit before taking any pins. Otherwise the
      // pinned batch could consume the capacity needed to describe fallback.
      scratch.admitted_ids.reserve(scratch.unique_page_ids.size());
      scratch.admitted_indices.reserve(scratch.unique_page_ids.size());
      scratch.admitted_pages.reserve(scratch.unique_page_ids.size());
      scratch.page_uses.reserve(total_pages);

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

      // Mutable values always own a snapshot. Immutable single-page values
      // retain a zero-copy pin when resident; other values need a copy.
      for (BatchState &state : scratch.states) {
        state.copy_result = mutable_snapshot || state.page_count > 1;
        for (size_t j = 0; j < state.page_count && !state.copy_result; ++j) {
          const size_t unique_index =
              unique_index_for(scratch.page_ids[state.first_page_index + j]);
          state.copy_result = scratch.pages[unique_index] == nullptr;
        }
      }

      // Reserve copied values once, then hand out bump slices. This includes
      // single-page bypass values, so the direct-read page scratch can be
      // immediately reused for the next unique miss.
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
      std::shared_ptr<BatchArena> arena_owner;
      if (batch_arena && total != 0) {
        if (!scratch.arena || scratch.arena.use_count() != 1) {
          scratch.arena = std::allocate_shared<BatchArena>(
              ailego::BudgetedAllocator<BatchArena>{});
        }
        if (scratch.arena->size() < total) {
          scratch.arena->resize(total, kArenaAlign);
        }
        arena_owner = scratch.arena;
        size_t off = 0;
        for (BatchState &state : scratch.states) {
          if (!state.copy_result) {
            continue;
          }
          state.owned = reinterpret_cast<char *>(arena_owner->data()) + off;
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
          state.owned_block = allocate_snapshot(alloc_size);
          state.owned_block.scratch_size_ = length;
          state.owned = static_cast<char *>(state.owned_block.data_);
          if (state.owned == nullptr) {
            return fail();
          }
        }
      }

      // Describe every copied fragment, sort by unique source page, and read
      // each bypass page exactly once before scattering it to all vectors that
      // overlap that page.
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

      size_t use_begin = 0;
      while (use_begin < scratch.page_uses.size()) {
        std::array<ailego::VecBufferPool::ReadRange, kBypassBatchPages> ranges;
        std::array<size_t, kBypassBatchPages> starts{}, ends{};
        std::array<const char *, kBypassBatchPages> sources{};
        size_t groups = 0;
        size_t range_count = 0;
        while (groups < bypass_batch_pages &&
               use_begin < scratch.page_uses.size()) {
          const size_t unique_index = scratch.page_uses[use_begin].unique_index;
          size_t use_end = use_begin;
          size_t first_byte = ailego::kVectorPageSize;
          size_t end_byte = 0;
          while (use_end < scratch.page_uses.size() &&
                 scratch.page_uses[use_end].unique_index == unique_index) {
            const PageUse &use = scratch.page_uses[use_end];
            first_byte = std::min(first_byte, use.source_offset);
            end_byte = std::max(end_byte, use.source_offset + use.length);
            ++use_end;
          }
          const char *source = scratch.pages[unique_index];
          if (source == nullptr) {
            const size_t page_offset =
                scratch.unique_page_ids[unique_index] * ailego::kVectorPageSize;
            // F_NOCACHE needs only requested fragments. O_DIRECT retains
            // whole-page-aligned ranges, including when batch AIO is used.
            if (owner_->buffer_pool_->requires_aligned_io()) {
              first_byte = 0;
              end_byte =
                  std::min(ailego::kVectorPageSize,
                           owner_->buffer_pool_->file_size() - page_offset);
            }
            char *destination = scratch.bypass_page.data() +
                                range_count * ailego::kVectorPageSize;
            ranges[range_count++] = {page_offset + first_byte,
                                     end_byte - first_byte,
                                     destination + first_byte};
            source = destination;
          }
          starts[groups] = use_begin;
          ends[groups] = use_end;
          sources[groups++] = source;
          use_begin = use_end;
        }
        if (!owner_->buffer_pool_->read_ranges_bypass(ranges.data(),
                                                      range_count))
          return fail();
        for (size_t group = 0; group < groups; ++group) {
          const size_t unique_index =
              scratch.page_uses[starts[group]].unique_index;
          std::shared_lock<std::shared_mutex> page_lock;
          if (mutable_snapshot && scratch.pages[unique_index] != nullptr) {
            // Pins prevent eviction, not writes. Hold the writer's page latch
            // through every copy from this resident page. Bypass sources are
            // already synchronized snapshots owned by this operation.
            page_lock = owner_->buffer_pool_handle_->lock_pinned_page_for_read(
                scratch.unique_page_ids[unique_index]);
          }
          for (size_t i = starts[group]; i < ends[group]; ++i) {
            const PageUse &use = scratch.page_uses[i];
            BatchState &state = scratch.states[use.state_index];
            std::memcpy(state.owned + use.destination_offset,
                        sources[group] + use.source_offset, use.length);
          }
        }
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
          auto *segment = static_cast<WrappedSegment *>(state.request->segment);
          segment->reset_pinned_block(
              *state.request->block, scratch.unique_page_ids[unique_index],
              scratch.pages[unique_index] + offset_in_page);
          continue;
        }
        *state.request->block =
            batch_arena ? MemoryBlock::MakeSharedView(state.owned, arena_owner)
                        : std::move(state.owned_block);
        state.owned = nullptr;
      }
      release_pages();
      return true;
    } catch (const std::bad_alloc &) {
      for (size_t i = 0; reads && i < count; ++i) {
        if (reads[i].block) reads[i].block->reset();
      }
      // Batch descriptors and the worst-case arena are optional for every
      // entry, including immutable and read-only reads. Their pins and scratch
      // have unwound before retrying the scalar implementation.
      buffer_storage_read_error = IndexError_ReadData;
      try {
        return immutable
                   ? IndexStorage::Segment::read_borrowed_batch_immutable(reads,
                                                                          count)
                   : IndexStorage::Segment::read_borrowed_batch(reads, count);
      } catch (const std::bad_alloc &) {
        for (size_t i = 0; reads && i < count; ++i) {
          if (reads[i].block) reads[i].block->reset();
        }
        buffer_storage_read_error = IndexError_NoMemory;
        return false;
      }
    }

    int write_error() const override {
      return buffer_storage_write_error;
    }

    //! Write data into the storage with offset. The shard latch excludes
    //! flush; all handles for this segment share the metadata mutex.
    size_t write(size_t offset, const void *data, size_t len) override {
      buffer_storage_write_error = IndexError_WriteData;
      if (!active()) return 0;
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (!active()) return 0;
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_)) {
        LOG_ERROR("WrappedSegment::write: pool is null, file[%s], id[%s]",
                  owner_->file_name_.c_str(), segment_id_->data());
        return 0;
      }
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::write: storage is marked corrupted, refusing "
            "write, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->data());
        return 0;
      }
      // In read-only mode the write is a silent no-op so that callers that
      // unconditionally write (e.g. CRC updates) do not return an error.
      if (!owner_->buffer_pool_->writable()) {
        return len;
      }
      if (ailego_unlikely(offset > capacity() || len > capacity() - offset)) {
        LOG_ERROR(
            "write() exceeds segment capacity: offset=%zu len=%zu cap=%zu",
            offset, len, capacity());
        return 0;
      }
      const size_t abs_offset = absolute_offset(offset);
      size_t written = 0;
      // Publish data_size only after the page bytes are visible to readers.
      const auto status = owner_->buffer_pool_handle_->write_range_status(
          abs_offset, len, static_cast<const char *>(data), &written);
      if (status == ailego::PageWriteStatus::kNoMemory) {
        buffer_storage_write_error = IndexError_NoMemory;
      }
      if (status == ailego::PageWriteStatus::kOk) {
        publish_written_size(offset + written);
      } else if (written != 0) {
        // Persist the copied bytes without advancing the record cursor.
        owner_->set_as_dirty();
      }
      return written;
    }

    bool write_batch(const SegmentData *writes, size_t count) override {
      buffer_storage_write_error = IndexError_WriteData;
      if (!active()) return false;
      if (count == 0) {
        return true;
      }
      if (writes == nullptr || count > kMaxCoalescedWrites) {
        return IndexStorage::Segment::write_batch(writes, count);
      }

      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (!active()) return false;
      if (ailego_unlikely(!owner_->buffer_pool_handle_ ||
                          !owner_->buffer_pool_ ||
                          owner_->corrupted_.load(std::memory_order_acquire))) {
        return false;
      }
      if (!owner_->buffer_pool_->writable()) {
        return true;
      }

      std::array<ailego::VecBufferWriteFragment, kMaxCoalescedWrites>
          fragments{};
      uint64_t write_end = 0;
      if (!make_write_fragments(writes, count, fragments.data(), write_end))
        return false;

      if (write_end != 0) {
        const auto status = owner_->buffer_pool_handle_->write_fragments_status(
            fragments.data(), count);
        if (status != ailego::PageWriteStatus::kOk) {
          if (status == ailego::PageWriteStatus::kNoMemory)
            buffer_storage_write_error = IndexError_NoMemory;
          return false;
        }
      }
      publish_written_size(write_end);
      return true;
    }

    int prepare_write_batch(const SegmentData *writes, size_t count,
                            std::unique_ptr<PreparedWrite> *out) override try {
      if (!out || *out || (!writes && count)) return IndexError_InvalidArgument;
      if (!active()) return IndexError_WriteData;
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (!active() || !owner_->buffer_pool_ ||
          !owner_->buffer_pool_->writable() ||
          owner_->corrupted_.load(std::memory_order_acquire))
        return IndexError_WriteData;

      ailego::MemoryReservation descriptor;
      std::unique_ptr<PreparedBatch> prepared;
      descriptor.add_for_allocation(sizeof(PreparedBatch), [&] {
        prepared = std::make_unique<PreparedBatch>(
            std::static_pointer_cast<WrappedSegment>(shared_from_this()),
            owner_->buffer_pool_);
      });
      prepared->memory = std::move(descriptor);
      const int ret = prepared->prepare(writes, count);
      if (ret != 0) return ret;
      prepared->hold_storage(owner_->prepared_write_state());
      *out = std::move(prepared);
      return ret;
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }

    //! Resize size of data.  See write() for the locking contract.
    size_t resize(size_t size) override {
      if (!active()) return 0;
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (!active()) return 0;
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::resize: storage is marked corrupted, refusing "
            "resize, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->data());
        return 0;
      }
      auto meta = state_->info.segment.meta();
      bool changed = false;
      {
        std::lock_guard<std::mutex> meta_latch(state_->metadata_mutex);
        uint64_t cur = bs_load_relaxed(&meta->data_size);
        if (cur != size) {
          if (size > capacity()) {
            size = capacity();
          }
          // Match write(): padding first, then release-store data_size.
          bs_store_relaxed(&meta->padding_size, capacity() - size);
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
      if (!active()) return;
      std::shared_lock<std::shared_mutex> latch(
          owner_->mapping_shards_[owner_->mapping_shard_id()].mtx);
      if (!active()) return;
      if (ailego_unlikely(owner_->corrupted_.load(std::memory_order_acquire))) {
        LOG_ERROR(
            "WrappedSegment::update_data_crc: storage is marked corrupted, "
            "refusing CRC update, file[%s], id[%s]",
            owner_->file_name_.c_str(), segment_id_->data());
        return;
      }
      {
        std::lock_guard<std::mutex> meta_latch(state_->metadata_mutex);
        state_->info.segment.meta()->data_crc = crc;
      }
      owner_->set_as_dirty();
    }

    //! Clone the segment
    IndexStorage::Segment::Pointer clone() override {
      return active() ? shared_from_this() : nullptr;
    }

    //! Preload a read-only range and attach its eviction priority. Writable
    //! storage deliberately skips this hint: construction has a different
    //! access pattern and dirty-page lifetime must drive admission there.
    void prefetch(size_t offset, size_t len,
                  CachePriority priority = CachePriority::kLow) override {
      if (!active() || !owner_->cache_enabled_ || !owner_->buffer_pool_ ||
          !owner_->buffer_pool_handle_ || owner_->buffer_pool_->writable()) {
        return;
      }
      const size_t data_size =
          bs_load_acquire(&state_->info.segment.meta()->data_size);
      if (offset >= data_size || len == 0) {
        return;
      }
      len = std::min(len, data_size - offset);
      const size_t abs_offset = data_offset() + offset;
      owner_->buffer_pool_handle_->prefetch_range(
          abs_offset, len, static_cast<uint8_t>(priority));
    }

   private:
    bool active() const {
      return generation_->active.load(std::memory_order_acquire);
    }

    bool check_read(MemoryBlock &block) const {
      if (active()) return true;
      buffer_storage_read_error = IndexError_ReadData;
      block.reset();
      return false;
    }

    void reset_pinned_block(MemoryBlock &block, ailego::block_id_t id,
                            char *data) const {
      // A block can outlive a close or reopen. Retain its pool for just the
      // lifetime of that block so an idle Segment cannot keep a file open.
      block.reset(owner_->buffer_pool_handle_, id, data);
    }

    class PreparedBatch final : public PreparedWrite {
     public:
      PreparedBatch(std::shared_ptr<WrappedSegment> segment,
                    ailego::VecBufferPool::Pointer pool)
          : segment_(std::move(segment)), pool_(std::move(pool)) {}

      ~PreparedBatch() override {
        pins_.reset();
        if (state_) state_->active.fetch_sub(1, std::memory_order_release);
      }

      void hold_storage(std::shared_ptr<PreparedWriteState> state) {
        state->active.fetch_add(1, std::memory_order_relaxed);
        state_ = std::move(state);
      }

      int prepare(const SegmentData *writes, size_t count) {
        fragments_.resize(count);
        if (!segment_->make_write_fragments(writes, count, fragments_.data(),
                                            write_end_))
          return IndexError_InvalidArgument;
        size_t total = 0;
        for (const auto &fragment : fragments_) {
          if (fragment.length > std::numeric_limits<size_t>::max() - total)
            return IndexError_NoMemory;
          total += fragment.length;
        }
        payload_.resize(total);
        size_t offset = 0;
        for (auto &fragment : fragments_) {
          if (!fragment.length) continue;
          std::memcpy(payload_.data() + offset, fragment.src, fragment.length);
          fragment.src = payload_.data() + offset;
          offset += fragment.length;
        }
        const auto status =
            pool_->prepare_write_fragments(fragments_.data(), count, pins_);
        if (status == ailego::PageWriteStatus::kOk) return 0;
        return status == ailego::PageWriteStatus::kNoMemory
                   ? IndexError_NoMemory
                   : IndexError_WriteData;
      }

      void commit() noexcept override {
        if (write_end_ == 0) return;
        std::lock_guard<std::mutex> life(state_->lifetime_mutex);
        auto *owner = state_->owner;
        if (!owner) return;
        std::shared_lock<std::shared_mutex> latch(
            owner->mapping_shards_[owner->mapping_shard_id()].mtx);
        pool_->commit_write_fragments(fragments_.data(), fragments_.size(),
                                      pins_);
        segment_->publish_written_size(write_end_);
        write_end_ = 0;
      }

      ailego::MemoryReservation memory;

     private:
      std::shared_ptr<PreparedWriteState> state_;
      std::shared_ptr<WrappedSegment> segment_;
      ailego::VecBufferPool::Pointer pool_;
      ailego::BudgetedVector<ailego::VecBufferWriteFragment> fragments_;
      ailego::BudgetedVector<char> payload_;
      ailego::VecBufferPool::PreparedWritePages pins_;
      uint64_t write_end_{0};
    };

    bool make_write_fragments(const SegmentData *writes, size_t count,
                              ailego::VecBufferWriteFragment *fragments,
                              uint64_t &write_end) const {
      const size_t data_base = data_offset();
      for (size_t i = 0; i < count; ++i) {
        const auto &write = writes[i];
        if (write.offset > capacity() ||
            write.length > capacity() - write.offset ||
            (write.length != 0 && write.data == nullptr))
          return false;
        if (!write.length) continue;
        fragments[i] = {data_base + write.offset, write.length,
                        static_cast<const char *>(write.data)};
        write_end = std::max<uint64_t>(write_end, write.offset + write.length);
      }
      return true;
    }

    void publish_written_size(uint64_t write_end) {
      auto *meta = state_->info.segment.meta();
      if (write_end > bs_load_acquire(&meta->data_size)) {
        std::lock_guard<std::mutex> meta_latch(state_->metadata_mutex);
        if (write_end > bs_load_relaxed(&meta->data_size)) {
          // Publish data bytes and padding before the logical length.
          bs_store_relaxed(&meta->padding_size, state_->capacity - write_end);
          bs_store_release(&meta->data_size, write_end);
        }
      }
      // Fixed-size rewrites are dirty even when data_size is unchanged.
      owner_->set_as_dirty();
    }

    static constexpr size_t kMaxCoalescedWrites = 8;
    // The directory map keeps the state at a stable address across appends.
    SegmentState *state_{nullptr};

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

    static MemoryBlock allocate_snapshot(size_t bytes,
                                         size_t alignment = kBufferAlignment) {
      if (bytes < ailego::BudgetedAlignedBuffer::kRetainedThreshold) {
        return MemoryBlock::AllocateOwned(bytes, alignment);
      }
      auto owner = std::allocate_shared<ailego::BudgetedAlignedBuffer>(
          ailego::BudgetedAllocator<ailego::BudgetedAlignedBuffer>{}, bytes,
          alignment);
      return MemoryBlock::MakeSharedView(owner->data(), owner,
                                         /*deep_copy=*/true);
    }

    size_t read_memory_block(size_t abs_offset, MemoryBlock &data, size_t len,
                             bool immutable) const try {
      buffer_storage_read_error = IndexError_ReadData;
      const bool can_pin = owner_->cache_enabled_ &&
                           (immutable || !owner_->buffer_pool_->writable());
      bool force_bypass = !owner_->cache_enabled_;
      const size_t offset_in_page = abs_offset % ailego::kVectorPageSize;
      if (can_pin && len <= ailego::kVectorPageSize - offset_in_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (raw != nullptr) {
          reset_pinned_block(data, page_id, raw);
          return len;
        }
        force_bypass = true;
      }

      // Mutable graph snapshots are typically only a few hundred bytes. They
      // are copied from cached pages and are never submitted to O_DIRECT, so a
      // page-aligned allocation only adds allocator and RSS overhead. Keep
      // SIMD alignment for small vector snapshots; sub-page destinations are
      // copied through the I/O layer's aligned staging buffer on cache misses.
      // Whole-page destinations retain their direct-I/O alignment.
      static constexpr size_t kSmallSnapshotBytes = 512;
      size_t alloc_size = len;
      const size_t alignment =
          !immutable && len <= kSmallSnapshotBytes
              ? 0
              : (len < kBufferAlignment ? AILEGO_CACHE_LINE_SIZE
                                        : kBufferAlignment);
      if (alignment) {
        if (ailego_unlikely(!AlignBufferSize(len, &alloc_size, alignment))) {
          return 0;
        }
      }
      const bool reuse = data.can_reuse_scratch(len, alignment);
      MemoryBlock owned;
      if (!reuse) {
        auto allocate = [&] {
          return alignment ? allocate_snapshot(alloc_size, alignment)
                           : MemoryBlock::AllocateOwned(alloc_size);
        };
        owned = owner_->cache_enabled_
                    ? owner_->read_scratch_.acquire(alloc_size, alignment, true,
                                                    allocate)
                    : allocate();
      }
      auto *tmp = static_cast<char *>(reuse ? data.data_ : owned.data_);
      if (tmp == nullptr) {
        LOG_ERROR(
            "WrappedSegment::read: owned buffer allocation failed, file[%s], "
            "id[%s], abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_->data(), abs_offset, len);
        return 0;
      }
      bool read_ok = force_bypass ? owner_->read_bypass(abs_offset, len, tmp)
                     : immutable
                         ? owner_->buffer_pool_handle_->read_range_immutable(
                               abs_offset, len, tmp)
                         : owner_->read_range(abs_offset, len, tmp);
      if (!read_ok && immutable && !force_bypass) {
        // A cross-page immutable read can lose a capacity race after choosing
        // the cache path. Retry without admission; writable bypass reads still
        // preserve resident dirty bytes and synchronize with page transitions.
        read_ok = owner_->read_bypass(abs_offset, len, tmp);
      }
      if (!read_ok) {
        if (reuse) data.reset();
        LOG_ERROR(
            "WrappedSegment::read: owned read failed, file[%s], id[%s], "
            "abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_->data(), abs_offset, len);
        return 0;
      }
      if (!reuse) data = std::move(owned);
      data.scratch_size_ = len;
      return len;
    } catch (const std::bad_alloc &) {
      buffer_storage_read_error = IndexError_NoMemory;
      return 0;
    }

    BufferStorage *owner_{nullptr};
    // Stable alongside state_; unordered_map rehash preserves element
    // references and pointers.
    const std::string_view *segment_id_{nullptr};
    std::shared_ptr<DirectoryGeneration> generation_;
  };

  //! Destructor
  ~BufferStorage() override {
    if (prepared_writes_) {
      std::lock_guard<std::mutex> life(prepared_writes_->lifetime_mutex);
      prepared_writes_->owner = nullptr;
    }
    close_index(/*force=*/true);
  }

  //! Retrieve the memory block type of this storage
  MemoryBlock::MemoryBlockType memory_block_type() const override {
    return MemoryBlock::MBT_BUFFERPOOL;
  }

  std::shared_ptr<ailego::VecBufferPool> vec_buffer_pool() const override {
    return buffer_pool_;
  }

  std::string file_path() const override {
    return file_name_;
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
  int cleanup() override {
    return this->close_index();
  }

  //! Open storage
  int open(const std::string &path, bool create_if_missing) override {
    // Preserve the access mode of callers using the legacy boolean API.
    return open_with_options(
        path, OpenOptions{create_if_missing, /*read_only=*/!create_if_missing});
  }

  int open_with_options(const std::string &path,
                        const OpenOptions &options) override try {
    BufferStorage candidate;
    candidate.segment_meta_capacity_ = segment_meta_capacity_;
    AllShardsExclusiveLatch latch(mapping_shards_);
    if (has_prepared_writes()) return IndexError_Runtime;
    // A second open may read the same file. Publish pending writes before
    // parsing it, and keep the current directory until admission succeeds.
    int ret = flush_index_locked();
    if (ret != 0) return ret;
    ailego::File::Identity created_file;
    bool published = false;
    AILEGO_DEFER([&] {
      if (created_file.valid && !published) {
        candidate.close_index(/*force=*/true);
        if (!ailego::File::DeleteIfSame(path, created_file)) {
          LOG_ERROR("Failed to remove incomplete index file %s", path.c_str());
        }
      }
    });
    ret = candidate.open_unpublished(path, options, &created_file);
    if (ret != 0) return ret;
    if (!options.defer_publication) {
      const bool staged = static_cast<bool>(candidate.staged_file_);
      ret = candidate.publish_creation();
      if (ret != 0) return ret;
      if (staged)
        created_file = FileIdentity(candidate.buffer_pool_->file_descriptor());
    }

    const bool preserve_handles = can_preserve_handles(candidate);
    read_scratch_.clear();
    // Opening and closing are serialized with readers by the storage caller.
    // The latch also keeps page writes and directory append from interleaving
    // with publication of the newly admitted mapping.
    if (preserve_handles) {
      transfer_directory_nodes(candidate);
    } else {
      invalidate_directory();
      directory_generation_ = std::move(candidate.directory_generation_);
    }
    file_name_.swap(candidate.file_name_);
    staged_file_.swap(candidate.staged_file_);
    buffer_pool_.swap(candidate.buffer_pool_);
    buffer_pool_handle_.swap(candidate.buffer_pool_handle_);
    segments_.swap(candidate.segments_);
    meta_chains_.swap(candidate.meta_chains_);
    std::swap(cache_enabled_, candidate.cache_enabled_);
    {
      std::scoped_lock checkpoint_lock(checkpoint_mutex_,
                                       candidate.checkpoint_mutex_);
      std::swap(accept_checkpoint_requests_,
                candidate.accept_checkpoint_requests_);
      std::swap(pending_checkpoint_, candidate.pending_checkpoint_);
      std::swap(refreshed_during_writeback_,
                candidate.refreshed_during_writeback_);
      const auto previous_visible =
          visible_checkpoint_.load(std::memory_order_acquire);
      visible_checkpoint_.store(
          candidate.visible_checkpoint_.load(std::memory_order_acquire),
          std::memory_order_release);
      candidate.visible_checkpoint_.store(previous_visible,
                                          std::memory_order_release);
      const bool previous_dirty = index_dirty_.load(std::memory_order_relaxed);
      index_dirty_.store(candidate.index_dirty_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
      candidate.index_dirty_.store(previous_dirty, std::memory_order_relaxed);
    }
    published = true;
    if (options.created_file) *options.created_file = created_file;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  int publish_creation() override {
    if (!staged_file_) return 0;
    const int ret = flush_index();
    if (ret != 0) return ret;
    if (!staged_file_->publish()) {
      LOG_ERROR("Failed to publish new index %s", file_name_.c_str());
      return IndexError_CreateFile;
    }
    staged_file_.reset();
    return 0;
  }

 private:
  bool can_preserve_handles(const BufferStorage &candidate) const {
    if (!buffer_pool_ ||
        !(FileIdentity(buffer_pool_->file_descriptor()) ==
          FileIdentity(candidate.buffer_pool_->file_descriptor())))
      return false;

    for (const auto &previous : segments_) {
      auto next = candidate.segments_.find(previous.first);
      if (next == candidate.segments_.end()) return false;
      const auto &old_info = previous.second.info;
      const auto &new_info = next->second.info;
      const auto *old_meta = old_info.segment.meta();
      const auto *new_meta = new_info.segment.meta();
      const auto old_offset = old_info.segment_header_start_offset +
                              old_info.segment_header->content_offset +
                              old_meta->data_index;
      const auto new_offset = new_info.segment_header_start_offset +
                              new_info.segment_header->content_offset +
                              new_meta->data_index;
      if (old_offset != new_offset ||
          old_meta->data_size + old_meta->padding_size !=
              new_meta->data_size + new_meta->padding_size)
        return false;
    }
    return true;
  }

  void transfer_directory_nodes(BufferStorage &candidate) {
    // Preflight guarantees a matching candidate entry. Replace it with the
    // extracted original node, whose addresses are still held by Segments.
    // Restoring the same entry count cannot rehash or require admission.
    while (!segments_.empty()) {
      auto previous = segments_.begin();
      auto incoming = candidate.segments_.find(previous->first);
      auto fresh = candidate.segments_.extract(incoming);
      auto kept = segments_.extract(previous);
      kept.key() = fresh.key();
      kept.mapped().info = fresh.mapped().info;
      kept.mapped().capacity = fresh.mapped().capacity;
      candidate.segments_.insert(std::move(kept));
    }
  }

  void invalidate_directory() {
    if (directory_generation_) {
      directory_generation_->active.store(false, std::memory_order_release);
      directory_generation_.reset();
    }
  }

  int open_unpublished(const std::string &path, const OpenOptions &options,
                       ailego::File::Identity *created_file) {
    const bool writable = !options.read_only;
    if (!ailego::File::IsExist(path) && options.create_if_missing) {
      ailego::File::Identity private_identity;
      if (ailego::StagedPath::supported()) {
        staged_file_ = std::make_unique<ailego::StagedPath>();
        if (!staged_file_->prepare(path)) return IndexError_CreateFile;
      }
      if (!staged_file_) {
        const size_t last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
          ailego::File::MakePath(path.substr(0, last_slash));
        }
      }
      int error_code = staged_file_ ? this->init_index(staged_file_->path(),
                                                       &private_identity)
                                    : this->init_index(path, created_file);
      if (error_code != 0) {
        LOG_ERROR("init_index failed for %s, errno=%d", path.c_str(),
                  error_code);
        return error_code;
      }
    }

    try {
      const auto &io_path = staged_file_ ? staged_file_->path() : path;
      auto candidate_pool = std::make_shared<ailego::VecBufferPool>(
          io_path, writable, staged_file_ != nullptr);
      if (created_file->valid &&
          !(FileIdentity(candidate_pool->file_descriptor()) == *created_file)) {
        LOG_ERROR("Index file was replaced during creation %s", path.c_str());
        return IndexError_OpenFile;
      }
      auto candidate_handle =
          std::make_shared<ailego::VecBufferPoolHandle>(candidate_pool);
      auto generation = std::allocate_shared<DirectoryGeneration>(
          ailego::BudgetedAllocator<DirectoryGeneration>{});
      file_name_ = path;
      buffer_pool_ = std::move(candidate_pool);
      buffer_pool_handle_ = std::move(candidate_handle);
      directory_generation_ = std::move(generation);
    } catch (const std::bad_alloc &) {
      LOG_ERROR("Out of memory opening BufferStorage: %s", path.c_str());
      return IndexError_NoMemory;
    } catch (const std::runtime_error &error) {
      LOG_ERROR("Failed to open BufferStorage file %s: %s", path.c_str(),
                error.what());
      return IndexError_OpenFile;
    }
    int ret;
    try {
      ret = parse_to_mapping();
      if (ret == 0 && writable) {
        for (const auto &chain : meta_chains_) {
          buffer_pool_->register_metadata_range(chain.header_start_offset,
                                                chain.header->content_offset);
        }
      }
    } catch (const std::bad_alloc &) {
      ret = IndexError_NoMemory;
    }
    if (ret != 0) {
      this->close_index(/*force=*/true, /*flush_pending=*/false);
      return ret;
    }
    // Parsing bounds every non-tail chain by the start of the following
    // chain. Only bytes past the tail's allocation can be discarded.
    const uint64_t used_end = meta_chains_.back().allocated_file_end();
    if (writable && used_end < buffer_pool_->file_size() &&
        !buffer_pool_->truncate_unused_tail(used_end)) {
      this->close_index(/*force=*/true, /*flush_pending=*/false);
      return IndexError_TruncateFile;
    }
    const size_t file_size = buffer_pool_->file_size();
    const size_t page_count =
        file_size == 0 ? 0 : (file_size - 1) / ailego::kVectorPageSize + 1;
    const size_t metadata_bytes =
        ailego::VecBufferPool::metadata_bytes_for_page_count(page_count,
                                                             writable);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    const size_t available = pool.available();
    size_t cache_budget = available;
    if (writable) {
      // Writable metadata is mandatory. Resident pages can be reclaimed by
      // init()'s actual reservation, so do not reject a new store merely
      // because other stores have filled their caches. Fixed reservations
      // still count against the same shared budget.
      cache_budget = pool.capacity();
      for (size_t fixed : {pool.metadata_used(), pool.external_used()}) {
        cache_budget = fixed >= cache_budget ? 0 : cache_budget - fixed;
      }
    }
    const bool cache_can_fit =
        metadata_bytes != std::numeric_limits<size_t>::max() &&
        metadata_bytes <= cache_budget &&
        ailego::kVectorPageSize <= cache_budget - metadata_bytes;
    ret = cache_can_fit ? buffer_pool_->init() : -1;
    if (ret != 0 && writable) {
      this->close_index(/*force=*/true, /*flush_pending=*/false);
      return IndexError_NoMemory;
    }
    // A legacy footer can understate the allocation even when no segment
    // has been modified since open. Schedule the repair on normal writeback.
    if (writable && std::any_of(meta_chains_.begin(), meta_chains_.end(),
                                [](const auto &chain) {
                                  return chain.needs_footer_extent_repair();
                                })) {
      set_as_dirty();
    }
    {
      std::lock_guard<std::mutex> lock(checkpoint_mutex_);
      const uint64_t checkpoint = meta_chains_.back().footer.check_point;
      visible_checkpoint_.store(checkpoint, std::memory_order_release);
      accept_checkpoint_requests_ = writable;
    }
    cache_enabled_ = ret == 0;
    if (!cache_enabled_) {
      LOG_INFO(
          "Read-only BufferStorage opened in bypass-only mode: file=%s "
          "available=%zu metadata=%zu page_size=%zu",
          path.c_str(), available, metadata_bytes, ailego::kVectorPageSize);
    }
    LOG_INFO("BufferStorage opened: file=%s, writable=%d, segment_count=%zu",
             file_name_.c_str(), static_cast<int>(writable), segments_.size());
    return 0;
  }

  // Called from single-threaded open or under AllShardsExclusiveLatch.
  int parse_header(size_t offset, IndexFormat::MetaHeader *out) {
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

  int parse_footer(size_t offset, IndexFormat::MetaFooter *footer) {
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

  int parse_segment(size_t offset, uint64_t header_start_offset,
                    IndexFormat::MetaHeader *chain_header,
                    const IndexFormat::MetaFooter &footer,
                    uint64_t available_content,
                    uint32_t &out_segment_ids_offset,
                    uint64_t &out_segment_extent,
                    ailego::BudgetedVector<char> &segment_buffer) {
    segment_buffer.resize(footer.segments_meta_size);
    if (buffer_pool_handle_->get_meta(offset, footer.segments_meta_size,
                                      segment_buffer.data()) != 0) {
      LOG_ERROR("Get segment meta failed.");
      return IndexError_Runtime;
    }
    if (ailego::Crc32c::Hash(segment_buffer.data(), footer.segments_meta_size,
                             0u) != footer.segments_meta_crc) {
      LOG_ERROR("Index segments meta checksum is invalid.");
      return IndexError_InvalidChecksum;
    }
    if (sizeof(IndexFormat::SegmentMeta) * footer.segment_count >
        footer.segments_meta_size) {
      return IndexError_InvalidLength;
    }
    IndexFormat::SegmentMeta *segment_start =
        reinterpret_cast<IndexFormat::SegmentMeta *>(segment_buffer.data());
    uint32_t segment_ids_offset = footer.segments_meta_size;
    for (IndexFormat::SegmentMeta *iter = segment_start,
                                  *end = segment_start + footer.segment_count;
         iter != end; ++iter) {
      if (iter->segment_id_offset >= footer.segments_meta_size) {
        return IndexError_InvalidValue;
      }
      // Older mmap writers could leave the footer short of the actual
      // allocation. Validate the full reserved capacity against the file.
      if (iter->data_index > available_content ||
          iter->data_size > available_content - iter->data_index ||
          iter->padding_size >
              available_content - iter->data_index - iter->data_size) {
        return IndexError_InvalidLength;
      }
      out_segment_extent =
          std::max(out_segment_extent,
                   iter->data_index + iter->data_size + iter->padding_size);

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
      const std::string_view seg_name(seg_name_start, seg_name_len);
      auto &state = segments_[seg_name];
      state.info = IndexMapping::SegmentInfo{IndexMapping::Segment{iter},
                                             header_start_offset, chain_header};
      state.capacity = iter->data_size + iter->padding_size;
    }
    out_segment_ids_offset = segment_ids_offset;
    return 0;
  }

  int parse_to_mapping() try {
    const uint64_t file_size = buffer_pool_->file_size();
    uint64_t header_start_offset = 0;
    while (true) {
      int ret;
      auto header = std::allocate_shared<IndexFormat::MetaHeader>(
          ailego::BudgetedAllocator<IndexFormat::MetaHeader>{});
      IndexFormat::MetaHeader *chain_header = header.get();
      ret = parse_header(header_start_offset, chain_header);
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
      const uint64_t footer_end = uint64_t{chain_header->meta_footer_offset} +
                                  sizeof(IndexFormat::MetaFooter);
      if (footer_end > file_size - header_start_offset) {
        LOG_ERROR("ParseToMapping: invalid meta_footer_offset=%" PRIu32
                  " (header=%" PRIu64 ", file_size=%zu), file[%s]",
                  chain_header->meta_footer_offset, header_start_offset,
                  buffer_pool_->file_size(), file_name_.c_str());
        return IndexError_InvalidValue;
      }
      const uint64_t footer_offset =
          header_start_offset + chain_header->meta_footer_offset;
      IndexFormat::MetaFooter footer{};
      ret = parse_footer(footer_offset, &footer);
      if (ret != 0) {
        LOG_ERROR("Failed to parse footer, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      const uint64_t next_off = footer.next_meta_header_offset;
      if (next_off != 0 &&
          (next_off <= header_start_offset ||
           next_off > file_size - sizeof(IndexFormat::MetaHeader))) {
        LOG_ERROR("ParseToMapping: invalid next_meta_header_offset=%" PRIu64
                  " (current=%" PRIu64 ", file_size=%zu), file[%s]",
                  next_off, header_start_offset, buffer_pool_->file_size(),
                  file_name_.c_str());
        return IndexError_InvalidValue;
      }
      const uint64_t chain_size =
          (next_off == 0 ? file_size : next_off) - header_start_offset;
      const uint64_t content_offset = chain_header->content_offset;
      if (footer.segments_meta_size > chain_header->meta_footer_offset ||
          content_offset < footer_end || content_offset > chain_size ||
          footer.total_size > chain_size ||
          footer.content_size > chain_size - content_offset ||
          footer.content_padding_size >
              chain_size - content_offset - footer.content_size) {
        return IndexError_InvalidLength;
      }

      // Unpack segment table
      const uint64_t segment_start_offset =
          footer_offset - footer.segments_meta_size;
      uint32_t segment_ids_offset = footer.segments_meta_size;
      uint64_t segment_extent = 0;
      ailego::BudgetedVector<char> segment_buffer;
      ret = parse_segment(segment_start_offset, header_start_offset,
                          chain_header, footer, chain_size - content_offset,
                          segment_ids_offset, segment_extent, segment_buffer);
      if (ret != 0) {
        LOG_ERROR("Failed to parse segment, errno %d, %s", ret,
                  IndexError::What(ret));
        return ret;
      }

      // Own all state referenced by this metadata chain in one object.
      meta_chains_.push_back({std::move(header), std::move(segment_buffer),
                              header_start_offset, segment_ids_offset,
                              segment_extent, footer});

      if (next_off == 0) {
        if (buffer_pool_->writable()) {
          auto &last = meta_chains_.back();
          last.append_meta.resize(footer.segments_meta_size);
          std::memcpy(last.append_meta.data(), last.segment_meta.data(),
                      footer.segments_meta_size);
        }
        break;
      }
      // The strictly increasing, file-bounded link guarantees termination.
      header_start_offset = next_off;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

 public:
  //! Flush storage
  int flush() override {
    return this->flush_index();
  }

  //! Close storage
  int close() override {
    return this->close_index();
  }

  //! Append a segment into storage
  int append(const std::string &id, size_t size) override {
    return this->append_segment(id, size);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh(uint64_t chkp) override {
    this->refresh_index(chkp);
  }

  //! Retrieve the logical checkpoint of the current tail.
  uint64_t check_point() const override {
    return visible_checkpoint_.load(std::memory_order_acquire);
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
  uint32_t magic() const override {
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
  int init_index(const std::string &path,
                 ailego::File::Identity *created_file) {
    IndexMapping mapping;
    int ret = mapping.create_exclusive(path, segment_meta_capacity_);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to create index file: path[%s], errno[%d]",
          path.c_str(), ret);
      return ret;
    }
    *created_file = mapping.file_identity();
    AILEGO_DEFER([&] { mapping.close(); });
    ret = this->init_version_segment(mapping);
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to append version segment: path[%s], errno[%d]",
          path.c_str(), ret);
      return ret;
    }
    mapping.refresh(0);
    ret = mapping.flush();
    if (ret != 0) {
      LOG_ERROR(
          "BufferStorage failed to flush new index file: path[%s], errno[%d]",
          path.c_str(), ret);
    }
    return ret;
  }

  bool is_dirty() const override {
    return index_dirty_.load(std::memory_order_relaxed);
  }

  //! The shard latch serializes segment mutations with writeback completion.
  void set_as_dirty() {
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);
    if (!accept_checkpoint_requests_) return;
    if (chkp != 0) {
      pending_checkpoint_ = chkp;
      visible_checkpoint_.store(chkp, std::memory_order_release);
    }
    refreshed_during_writeback_ = true;
    set_as_dirty();
  }

  //! Flush index storage.
  int flush_index() {
    // Wait for any current writeback before checking dirty or failed state.
    AllShardsExclusiveLatch latch(mapping_shards_);
    return flush_index_locked();
  }

  //! PRECONDITION: caller holds AllShardsExclusiveLatch. Used by explicit
  //! flush, append and close.
  int flush_index_locked() try {
    // No-op on never-opened / already-closed storage: close_index()
    // unconditionally calls us during teardown.
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
    uint64_t checkpoint;
    {
      std::lock_guard<std::mutex> lock(checkpoint_mutex_);
      if (!index_dirty_.load(std::memory_order_relaxed)) return 0;
      refreshed_during_writeback_ = false;
      checkpoint = std::exchange(pending_checkpoint_, 0);
    }
    bool persisted = false;
    AILEGO_DEFER([&]() {
      std::lock_guard<std::mutex> lock(checkpoint_mutex_);
      if (persisted) {
        // The shard latch excludes segment mutations. Only refresh can have
        // requested another writeback while this one was in progress.
        index_dirty_.store(refreshed_during_writeback_,
                           std::memory_order_relaxed);
      } else if (pending_checkpoint_ == 0) {
        pending_checkpoint_ = checkpoint;
      }
    });
    // Flush dirty data blocks first.
    if (buffer_pool_handle_->flush_all() != 0) {
      LOG_ERROR("flush_all data blocks failed: file[%s]", file_name_.c_str());
      return IndexError_WriteData;
    }
    // A short write can leave any chain's table and footer out of sync.
    metadata_writeback_incomplete_ = true;
    for (size_t ci = 0; ci < meta_chains_.size(); ++ci) {
      MetaChain &mchain = meta_chains_[ci];
      const char *seg_buf = mchain.segment_meta.data();
      if (mchain.needs_footer_extent_repair()) {
        mchain.footer.content_size = mchain.allocated_content_size();
        mchain.footer.content_padding_size = 0;
        mchain.footer.total_size =
            mchain.header->content_offset + mchain.footer.content_size;
      }
      mchain.footer.segments_meta_crc =
          ailego::Crc32c::Hash(seg_buf, mchain.footer.segments_meta_size, 0u);
      IndexFormat::UpdateMetaFooter(&mchain.footer, 0);
      if (!mchain.append_meta.empty()) {
        // The page writeback succeeded, so this table is now safe to publish.
        std::memcpy(mchain.append_meta.data(), seg_buf,
                    mchain.footer.segments_meta_size);
      }
      if (buffer_pool_handle_->write_meta(mchain.segment_meta_file_offset(),
                                          mchain.footer.segments_meta_size,
                                          seg_buf) != 0) {
        LOG_ERROR("Failed to write segment meta: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_WriteData;
      }
      auto footer = mchain.footer;
      const bool changes_checkpoint =
          checkpoint != 0 && checkpoint != footer.check_point;
      if (changes_checkpoint) {
        IndexFormat::UpdateMetaFooter(&footer, checkpoint);
      }
      // A short write can already contain the new checkpoint and CRC.
      // Restore the old checkpoint against the table that just reached disk.
      bool footer_written = false;
      AILEGO_DEFER([&]() {
        if (!changes_checkpoint || footer_written) return;
        // Retry the restore once with the retained old footer. If the disk
        // still rejects it, the checkpoint outcome is uncertain and must be
        // reported instead of silently dropping the restore result.
        if (!restore_metadata(mchain.footer_file_offset(),
                              sizeof(mchain.footer),
                              reinterpret_cast<const char *>(&mchain.footer)) &&
            !restore_metadata(mchain.footer_file_offset(),
                              sizeof(mchain.footer),
                              reinterpret_cast<const char *>(&mchain.footer))) {
          schedule_metadata_writeback();
          try {
            LOG_ERROR(
                "flush_index: cannot restore footer after a short write; "
                "checkpoint commit outcome is uncertain: file[%s], chain[%zu]",
                file_name_.c_str(), ci);
          } catch (...) {
            // The original write error still propagates if logging fails.
          }
        }
      });
      if (buffer_pool_handle_->write_meta(
              mchain.footer_file_offset(), sizeof(footer),
              reinterpret_cast<const char *>(&footer)) != 0) {
        LOG_ERROR("Failed to write footer: file[%s], chain[%zu]",
                  file_name_.c_str(), ci);
        return IndexError_WriteData;
      }
      footer_written = true;
      mchain.footer = footer;
    }
    metadata_writeback_incomplete_ = false;
    // Every on-disk link now matches the live chains. Ranges retained by a
    // failed link rollback are unreachable and can be dropped safely.
    for (const size_t start : stale_metadata_ranges_) {
      buffer_pool_->unregister_metadata_range(start);
    }
    stale_metadata_ranges_.clear();
    // A rolled-back split can now return the grown tail and its page-table
    // entries, but never below the live directory's allocated end: a later
    // committed append must not be cut by a stale truncation record.
    if (pending_tail_truncation_) {
      size_t allocated_end = 0;
      for (const auto &chain : meta_chains_) {
        allocated_end =
            std::max<uint64_t>(allocated_end, chain.allocated_file_end());
      }
      const size_t target = std::max(*pending_tail_truncation_, allocated_end);
      if (!buffer_pool_->truncate_unused_tail(target)) {
        LOG_WARN(
            "flush_index: cannot truncate rolled-back tail yet: file[%s], "
            "target_eof[%zu]",
            file_name_.c_str(), target);
      }
      pending_tail_truncation_.reset();
    }
    persisted = true;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  //! Close index storage
  int close_index(bool force = false, bool flush_pending = true) {
    // Hold ONE continuous all-shards latch across flush + teardown so no
    // writer can slip in between (which would dirty meta_buf only to have
    // the page table reset under it, dropping the modification).
    AllShardsExclusiveLatch latch(mapping_shards_);
    // Keep retryable failures open. Terminal failures still release the
    // unusable generation and report the failure to the caller.
    const bool discard = force || corrupted_.load(std::memory_order_acquire);
    if (!discard && has_prepared_writes()) return IndexError_Runtime;
    bool accepted_checkpoint_requests;
    {
      // Include all requests accepted before close in its final writeback.
      std::lock_guard<std::mutex> lock(checkpoint_mutex_);
      accepted_checkpoint_requests =
          std::exchange(accept_checkpoint_requests_, false);
    }
    const int status = flush_pending ? flush_index_locked() : 0;
    std::lock_guard<std::mutex> lock(checkpoint_mutex_);
    if (status != 0 && !discard) {
      accept_checkpoint_requests_ = accepted_checkpoint_requests;
      return status;
    }
    read_scratch_.clear();
    invalidate_directory();
    file_name_.clear();
    segments_.reset();
    decltype(meta_chains_)().swap(meta_chains_);
    buffer_pool_handle_.reset();
    buffer_pool_.reset();
    staged_file_.reset();
    cache_enabled_ = false;
    prepared_writes_.reset();
    refreshed_during_writeback_ = false;
    pending_checkpoint_ = 0;
    visible_checkpoint_.store(0, std::memory_order_release);
    index_dirty_.store(false, std::memory_order_relaxed);
    metadata_writeback_incomplete_ = false;
    decltype(stale_metadata_ranges_){}.swap(stale_metadata_ranges_);
    pending_tail_truncation_.reset();
    corrupted_.store(false, std::memory_order_relaxed);
    return status;
  }

  // The live directory still contains the metadata needed to repair the file.
  void schedule_metadata_writeback() {
    metadata_writeback_incomplete_ = true;
    set_as_dirty();
  }

  // Rollback can run while handling an exception from the I/O path.
  bool restore_metadata(size_t offset, size_t size, const char *data) noexcept {
    try {
      return buffer_pool_handle_->write_meta(offset, size, data) == 0;
    } catch (...) {
      return false;
    }
  }

  //! Append a segment into storage.
  int append_segment(const std::string &id, size_t size) try {
    AllShardsExclusiveLatch latch(mapping_shards_);
    if (id.find('\0') != std::string::npos) return IndexError_InvalidArgument;

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
    if (size == 0 || id.find('\0') != std::string::npos) {
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
    if (size > std::numeric_limits<size_t>::max() - (page_size - 1)) {
      return IndexError_InvalidLength;
    }
    const size_t padded_size = (size + page_size - 1) / page_size * page_size;

    size_t id_size = id.length() + 1;
    size_t need_size = sizeof(IndexFormat::SegmentMeta) + id_size;

    // Allocate rollback bookkeeping on the fallible path BEFORE inserting
    // the pending entry: an allocation failure must not leave an entry
    // behind. A later push_back during unwinding must never allocate.
    stale_metadata_ranges_.reserve(stale_metadata_ranges_.size() + 1);

    // Insert into the real directory map before any metadata is published.
    // The exclusive latch keeps the pending entry private until commit.
    auto pending = segments_.try_emplace(std::string_view(id));
    if (!pending.second) return IndexError_Duplicate;
    bool committed = false;
    size_t unpublished_directory_start = 0;
    size_t original_file_size = buffer_pool_->file_size();
    // Establish the rollback guard before any fallible operation: an
    // allocation failure must never leave a pending segment in the map.
    AILEGO_DEFER([&]() {
      if (committed) return;
      if (pending.first != segments_.end()) segments_.erase(pending.first);
      if (unpublished_directory_start != 0) {
        buffer_pool_->unregister_metadata_range(unpublished_directory_start);
      }
      // The directory rollback guards run first. Keep the grown bytes if
      // restoring the directory failed and they may still be referenced.
      if (metadata_writeback_incomplete_ ||
          corrupted_.load(std::memory_order_acquire) ||
          original_file_size >= buffer_pool_->file_size()) {
        return;
      }
      try {
        if (buffer_pool_->truncate_unused_tail(original_file_size)) {
          // The immediate reclaim already returned the tail; the deferred
          // record must not fire again after a later committed split.
          pending_tail_truncation_.reset();
          return;
        }
      } catch (...) {
      }
      corrupted_.store(true, std::memory_order_release);
    });
    MetaChain *chain = &meta_chains_.back();
    bool chain_split = false;
    bool link_attempted = false;

    // Step 1: prepare an unlinked directory chain when the current one is full.
    if (sizeof(IndexFormat::SegmentMeta) * chain->footer.segment_count +
            need_size >
        chain->segment_ids_offset) {
      size_t new_chain_start = chain->allocated_file_end();
      if (new_chain_start >
          std::numeric_limits<size_t>::max() - (page_size - 1)) {
        return IndexError_InvalidLength;
      }
      new_chain_start =
          (new_chain_start + page_size - 1) / page_size * page_size;
      // Existing mmap readers use the root directory's mapping size for
      // every link. The initialization parameter only sizes a new file.
      const auto &root_header = *meta_chains_.front().header;
      const size_t new_meta_total = root_header.content_offset;
      if (root_header.meta_footer_offset < sizeof(IndexFormat::MetaHeader)) {
        return IndexError_InvalidLength;
      }
      const uint32_t new_segments_meta_size = static_cast<uint32_t>(
          root_header.meta_footer_offset - sizeof(IndexFormat::MetaHeader));
      if (need_size > new_segments_meta_size ||
          new_meta_total >
              std::numeric_limits<size_t>::max() - new_chain_start) {
        return IndexError_InvalidLength;
      }

      auto new_header = std::allocate_shared<IndexFormat::MetaHeader>(
          ailego::BudgetedAllocator<IndexFormat::MetaHeader>{});
      IndexFormat::SetupMetaHeader(new_header.get(),
                                   root_header.meta_footer_offset,
                                   root_header.content_offset);

      ailego::BudgetedVector<char> new_meta_buf(new_segments_meta_size);
      ailego::BudgetedVector<char> new_append_meta(new_segments_meta_size);

      IndexFormat::MetaFooter new_footer;
      IndexFormat::SetupMetaFooter(&new_footer);
      new_footer.segments_meta_size = new_segments_meta_size;
      new_footer.total_size = new_meta_total;
      if (meta_chains_.size() == meta_chains_.capacity()) {
        meta_chains_.reserve(2 * meta_chains_.size());
      }

      // A reader can follow the new link as soon as it is written. Persist
      // the existing chains and their checkpoint before exposing that link.
      const int status = flush_index_locked();
      if (status != 0) return status;
      original_file_size = buffer_pool_->file_size();

      // Extend the file and write the new chain's header.
      if (!buffer_pool_->extend_file(new_chain_start + new_meta_total)) {
        return IndexError_Runtime;
      }
      buffer_pool_->register_metadata_range(new_chain_start, new_meta_total);
      unpublished_directory_start = new_chain_start;
      // A retry after a failed link rollback revives this range: it belongs
      // to the live chain again, not to the deferred cleanup list.
      stale_metadata_ranges_.erase(
          std::remove(stale_metadata_ranges_.begin(),
                      stale_metadata_ranges_.end(), new_chain_start),
          stale_metadata_ranges_.end());

      if (buffer_pool_handle_->write_meta(
              new_chain_start, sizeof(IndexFormat::MetaHeader),
              reinterpret_cast<const char *>(new_header.get())) != 0) {
        return IndexError_WriteData;
      }
      meta_chains_.push_back(MetaChain{
          std::move(new_header), std::move(new_meta_buf), new_chain_start,
          new_segments_meta_size, 0, new_footer, std::move(new_append_meta)});

      chain = &meta_chains_.back();
      chain_split = true;
    } else if (metadata_writeback_incomplete_) {
      // An in-place append only writes the tail. Finish interrupted metadata
      // writeback before publishing it, including any affected older chain.
      const int status = flush_index_locked();
      if (status != 0) return status;
      original_file_size = buffer_pool_->file_size();
    }

    AILEGO_DEFER([&]() {
      if (committed || !chain_split) return;
      meta_chains_.pop_back();
      // Once the old chain's link is repaired, the grown tail beyond the
      // original EOF and its page-table entries can be reclaimed.
      pending_tail_truncation_ = original_file_size;
      const auto &old_chain = meta_chains_.back();
      if (link_attempted &&
          !restore_metadata(
              old_chain.footer_file_offset(), sizeof(old_chain.footer),
              reinterpret_cast<const char *>(&old_chain.footer))) {
        // The on-disk link may still reference the new chain. Keep its range
        // protected until a successful writeback unlinks it, then clean up.
        stale_metadata_ranges_.push_back(unpublished_directory_start);
        unpublished_directory_start = 0;
        schedule_metadata_writeback();
      }
    });

    // Step 2: append the segment metadata and persist the last chain.
    auto footer = chain->footer;
    auto *header = chain->header.get();
    char *meta_buf = chain->append_meta.data();
    char *live_meta = chain->segment_meta.data();
    const uint64_t new_data_index = chain->allocated_content_size();
    const uint64_t new_seg_abs_offset = chain->allocated_file_end();
    if (padded_size > std::numeric_limits<size_t>::max() - new_seg_abs_offset) {
      return IndexError_InvalidLength;
    }
    uint64_t new_file_size = new_seg_abs_offset + padded_size;
    if (new_file_size > buffer_pool_->file_size()) {
      if (!buffer_pool_->extend_file(new_file_size)) {
        return IndexError_Runtime;
      }
    }

    const size_t meta_entry_off =
        sizeof(IndexFormat::SegmentMeta) * footer.segment_count;
    const uint32_t new_ids_off =
        chain->segment_ids_offset - static_cast<uint32_t>(id_size);
    const uint64_t meta_file_offset = chain->segment_meta_file_offset();
    const uint64_t entry_file_offset = meta_file_offset + meta_entry_off;
    const uint64_t id_file_offset = meta_file_offset + new_ids_off;

    // The unused regions match in both tables. Leave the live table intact
    // until publication succeeds so it can also supply the rollback bytes.
    AILEGO_DEFER([&]() {
      if (committed || chain_split) return;
      std::memcpy(meta_buf + meta_entry_off, live_meta + meta_entry_off,
                  sizeof(IndexFormat::SegmentMeta));
      std::memcpy(meta_buf + new_ids_off, live_meta + new_ids_off, id_size);

      const bool entry_restored =
          restore_metadata(entry_file_offset, sizeof(IndexFormat::SegmentMeta),
                           meta_buf + meta_entry_off);
      const bool id_restored =
          restore_metadata(id_file_offset, id_size, meta_buf + new_ids_off);
      const bool footer_restored =
          restore_metadata(chain->footer_file_offset(), sizeof(chain->footer),
                           reinterpret_cast<const char *>(&chain->footer));
      if (!entry_restored || !id_restored || !footer_restored) {
        schedule_metadata_writeback();
      }
    });

    auto *new_seg =
        reinterpret_cast<IndexFormat::SegmentMeta *>(meta_buf + meta_entry_off);
    new_seg->segment_id_offset = new_ids_off;
    new_seg->data_index = new_data_index;
    new_seg->data_size = 0;
    new_seg->data_crc = 0;
    new_seg->padding_size = padded_size;
    std::memcpy(meta_buf + new_ids_off, id.c_str(), id_size);

    footer.segment_count += 1;
    footer.content_size = new_data_index + padded_size;
    footer.content_padding_size = 0;
    footer.total_size = header->content_offset + footer.content_size;
    footer.segments_meta_crc =
        ailego::Crc32c::Hash(meta_buf, footer.segments_meta_size, 0u);
    IndexFormat::UpdateMetaFooter(&footer, 0);

    // In-place appends only write the new record and name. A new chain
    // needs the full table written before its link can be published.
    bool meta_written;
    if (chain_split) {
      meta_written =
          buffer_pool_handle_->write_meta(
              meta_file_offset, footer.segments_meta_size, meta_buf) == 0;
    } else {
      meta_written =
          buffer_pool_handle_->write_meta(entry_file_offset, sizeof(*new_seg),
                                          meta_buf + meta_entry_off) == 0 &&
          buffer_pool_handle_->write_meta(id_file_offset, id_size,
                                          meta_buf + new_ids_off) == 0;
    }
    if (!meta_written) {
      LOG_ERROR("append_segment: write segment_meta failed, file[%s]",
                file_name_.c_str());
      return IndexError_WriteData;
    }
    if (buffer_pool_handle_->write_meta(
            chain->footer_file_offset(), sizeof(footer),
            reinterpret_cast<const char *>(&footer)) != 0) {
      LOG_ERROR("append_segment: write footer failed, file[%s]",
                file_name_.c_str());
      return IndexError_WriteData;
    }

    // Link the chain only after the new segment's metadata and footer exist.
    if (chain_split) {
      auto &old_chain = meta_chains_[meta_chains_.size() - 2];
      auto linked_footer = old_chain.footer;
      linked_footer.next_meta_header_offset = chain->header_start_offset;
      IndexFormat::UpdateMetaFooter(&linked_footer, 0);
      link_attempted = true;
      if (buffer_pool_handle_->write_meta(
              old_chain.footer_file_offset(), sizeof(linked_footer),
              reinterpret_cast<const char *>(&linked_footer)) != 0) {
        return IndexError_WriteData;
      }
      old_chain.footer = linked_footer;
      decltype(old_chain.append_meta){}.swap(old_chain.append_meta);
    }

    auto *live_seg = reinterpret_cast<IndexFormat::SegmentMeta *>(
        live_meta + meta_entry_off);
    std::memcpy(live_seg, new_seg, sizeof(*live_seg));
    std::memcpy(live_meta + new_ids_off, id.c_str(), id_size);
    chain->footer = footer;
    chain->segment_ids_offset = new_ids_off;
    chain->segment_extent = footer.content_size;
    // Retain the node allocation and rebind its key to owned live metadata.
    auto node = segments_.extract(pending.first);
    pending.first = segments_.end();
    node.key() = std::string_view(live_meta + new_ids_off, id.size());
    node.mapped().info = IndexMapping::SegmentInfo{
        IndexMapping::Segment{live_seg}, chain->header_start_offset, header};
    node.mapped().capacity = padded_size;
    if (!segments_.insert(std::move(node)).inserted)
      return IndexError_Duplicate;
    if (chain_split) {
      std::lock_guard<std::mutex> lock(checkpoint_mutex_);
      visible_checkpoint_.store(pending_checkpoint_, std::memory_order_release);
    }
    // A committed append makes any earlier rollback's tail record
    // obsolete: truncating now would cut the freshly committed chain.
    pending_tail_truncation_.reset();
    committed = true;
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

 private:
  std::shared_ptr<PreparedWriteState> prepared_write_state() {
    auto state =
        std::atomic_load_explicit(&prepared_writes_, std::memory_order_acquire);
    if (state) return state;
    auto candidate = std::allocate_shared<PreparedWriteState>(
        ailego::BudgetedAllocator<PreparedWriteState>{}, this);
    if (std::atomic_compare_exchange_strong_explicit(
            &prepared_writes_, &state, candidate, std::memory_order_acq_rel,
            std::memory_order_acquire))
      return candidate;
    return state;
  }

  // Called with all mapping shards held. Every prepare operation acquires a
  // shard before publishing a handle, so no new hold can race replacement.
  bool has_prepared_writes() const {
    return prepared_writes_ &&
           prepared_writes_->active.load(std::memory_order_acquire) != 0;
  }

  bool read_bypass(size_t offset, size_t len, char *out) const {
    bool allocation_failed = false;
    const bool ok = buffer_pool_handle_->read_range_bypass(offset, len, out,
                                                           allocation_failed);
    if (allocation_failed) buffer_storage_read_error = IndexError_NoMemory;
    return ok;
  }

  bool read_range(size_t offset, size_t len, char *out) const {
    if (!cache_enabled_) return read_bypass(offset, len, out);
    bool allocation_failed = false;
    const bool ok =
        buffer_pool_handle_->read_range(offset, len, out, allocation_failed);
    if (allocation_failed) buffer_storage_read_error = IndexError_NoMemory;
    return ok;
  }

  std::atomic<bool> index_dirty_{false};
  ReusableReadScratch read_scratch_;
  std::shared_ptr<PreparedWriteState> prepared_writes_;
  std::mutex checkpoint_mutex_;
  bool accept_checkpoint_requests_{false};
  bool refreshed_during_writeback_{false};
  uint64_t pending_checkpoint_{0};
  std::atomic<uint64_t> visible_checkpoint_{0};
  // Set after page writeback until every metadata table and footer is written.
  // Accessed only under AllShardsExclusiveLatch.
  bool metadata_writeback_incomplete_{false};
  // Ranges kept alive by a failed link rollback; cleared after the next
  // successful metadata writeback. Same latch discipline as above.
  ailego::BudgetedVector<size_t> stale_metadata_ranges_{};
  // Original EOF recorded by a rolled-back split, reclaimed after the old
  // chain's link is repaired. Optional keeps the record allocation-free.
  std::optional<size_t> pending_tail_truncation_{};
  // A failed file-growth rollback blocks further writes.
  std::atomic<bool> corrupted_{false};

  // Readers hash across shards; metadata writers lock every shard.
  static constexpr size_t kMappingMutexShards = 32;
  struct alignas(AILEGO_CACHE_LINE_SIZE) MutexShard {
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

  // buffer manager
  std::string file_name_;
  std::unique_ptr<ailego::StagedPath> staged_file_;
  using SegmentMap =
      std::unordered_map<std::string_view, SegmentState,
                         std::hash<std::string_view>,
                         std::equal_to<std::string_view>,
                         ailego::BudgetedAllocator<
                             std::pair<const std::string_view, SegmentState>>>;
  ailego::LazyMap<SegmentMap> segments_;
  std::shared_ptr<DirectoryGeneration> directory_generation_;

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  bool cache_enabled_{false};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};

  // Segment metadata capacity written by init_index().
  uint32_t segment_meta_capacity_{4096u};

  // Per-chain ownership and state used by parsing, flushing, and appending.
  struct MetaChain {
    // Pointees stay stable across vector moves and are referenced by segments_.
    std::shared_ptr<IndexFormat::MetaHeader> header;
    ailego::BudgetedVector<char> segment_meta;
    uint64_t header_start_offset;
    // Lowest ID-string offset; detects metadata exhaustion.
    uint32_t segment_ids_offset;
    // Furthest end of any segment's full capacity, relative to content.
    uint64_t segment_extent;
    // Authoritative in-memory footer for this chain.
    IndexFormat::MetaFooter footer;
    // Only the last writable chain needs a table that append can publish
    // without exposing segment metadata whose data pages are still dirty.
    ailego::BudgetedVector<char> append_meta{};

    uint64_t footer_file_offset() const {
      return header_start_offset + header->meta_footer_offset;
    }

    uint64_t segment_meta_file_offset() const {
      return footer_file_offset() - footer.segments_meta_size;
    }

    bool needs_footer_extent_repair() const {
      return segment_extent > footer.content_size;
    }

    uint64_t allocated_content_size() const {
      const uint64_t declared_extent =
          std::max(footer.total_size, header->content_offset) -
          header->content_offset;
      return std::max({segment_extent, declared_extent,
                       footer.content_size + footer.content_padding_size});
    }

    uint64_t allocated_file_end() const {
      return header_start_offset + header->content_offset +
             allocated_content_size();
    }
  };
  ailego::BudgetedVector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
