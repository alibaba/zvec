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
#include <cstring>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
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

// Cross-compiler helpers for lock-free 64-bit acquire/release access
// to SegmentMeta::data_size / padding_size.
//
// These fields are POD (uint64_t) inside a serialised struct so we cannot
// change their type to std::atomic<>; std::atomic_ref is C++20 and the
// project targets C++17.  GCC/Clang have native __atomic_* builtins that
// emit single ldar/stlr on arm64 and plain mov on x86_64.  MSVC lacks
// these builtins, so we fall back to volatile load/store paired with a
// std::atomic_thread_fence, which is correct on all targets MSVC ships
// (x86_64 / arm64 desktop) and equivalent in cost.
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

// The legacy read(const void**) overload guarantees the returned pointer
// stays valid until close_index().  Single-page reads pin the page
// (never released); cross-page reads allocate a temp buffer owned by
// tmp_buffers_ (freed in close_index()).  Callers wanting bounded
// lifetime should use the read(MemoryBlock&) overload.

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

    //! Retrieve size of data
    //!
    //! data_size / padding_size are mutated lock-free by concurrent
    //! writers (write/resize) and observed by concurrent readers on the
    //! lock-free hot path.  Use acquire/release ordering so weakly-ordered
    //! ARM (e.g. Android arm64) cannot see stale values that would cause
    //! read() to truncate len to 0.
    size_t data_size(void) const override {
      return static_cast<size_t>(
          bs_load_acquire(&segment_info_->segment.meta()->data_size));
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
    //!
    //! C1: pool/handle are stable for the lifetime of the index
    //! (no retire/rebuild), so no lock is needed on the hot path.
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
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len,
                                                   static_cast<char *>(buf))) {
        LOG_ERROR(
            "WrappedSegment::fetch: read_range failed, file[%s], id[%zu], "
            "abs_offset=%zu, len=%zu",
            owner_->file_name_.c_str(), segment_id_, abs_offset, len);
        return 0;
      }
      return len;
    }

    //! Read data from segment
    //! C1: lock-free hot path (pool/handle never change during operation).
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
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (!raw) {
          LOG_ERROR(
              "WrappedSegment::read: single-page acquire failed, file[%s], "
              "id[%zu], abs_offset=%zu, len=%zu, page=%zu",
              owner_->file_name_.c_str(), segment_id_, abs_offset, len,
              first_page);
          *data = nullptr;
          return 0;
        }
        *data = raw;
        // Pin held until close_index() per the never-released contract
        // of this overload. Record the page so close_index() can release
        // the pin before tearing down the pool; otherwise the lingering
        // ref_count would trip ~VecBufferPool's "all blocks released"
        // assertion.
        {
          std::lock_guard<std::mutex> pin_latch(owner_->pinned_pages_mutex_);
          owner_->pinned_pages_.push_back(page_id);
        }
        return len;
      }
      // Cross-page path: see file-level banner.  C11 aligned_alloc requires
      // size to be a multiple of alignment, and alignment must be a power
      // of two.  Use a fixed 4096-byte alignment for the dst buffer: 4K is
      // the minimum page granularity across all supported platforms
      // (always a divisor of the 16K/64K page sizes used on Apple Silicon
      // and some Android arm64 configurations) and is sufficient for the
      // downstream SIMD/DMA-friendly access contract.  Pinning kAlign to
      // 4096 also avoids over-allocating 16KB per cross-page read on
      // large-page platforms.
      static constexpr size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      // Allocate a 4K-aligned slot from the per-storage arena pool.
      // This batches page-aligned allocation: under heap fragmentation
      // (notably Android Bionic scudo), one large posix_memalign per
      // arena via the secondary (mmap-backed) allocator is far more
      // reliable than many independent posix_memalign(4K, 4K) calls.
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
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        LOG_ERROR(
            "WrappedSegment::read: cross-page read_range failed, file[%s], "
            "id[%zu], abs_offset=%zu, len=%zu, first_page=%zu, last_page=%zu",
            owner_->file_name_.c_str(), segment_id_, abs_offset, len,
            first_page, last_page);
        // The arena slot is intentionally not rolled back: rolling back
        // would require holding the arena lock across read_range, while
        // the worst-case leak per failed read is one slot (alloc_size).
        *data = nullptr;
        return 0;
      }
      *data = tmp;
      return len;
    }

    //! C1: lock-free hot path (pool/handle never change during operation).
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
      size_t abs_offset = segment_info_->segment_header_start_offset +
                          segment_info_->segment_header->content_offset +
                          segment_info_->segment.meta()->data_index + offset;
      size_t first_page = abs_offset / ailego::kVectorPageSize;
      size_t last_page = (len == 0)
                             ? first_page
                             : (abs_offset + len - 1) / ailego::kVectorPageSize;
      if (first_page == last_page) {
        size_t page_id = 0;
        char *raw = owner_->buffer_pool_handle_->get_single_page(abs_offset,
                                                                 len, page_id);
        if (!raw) {
          LOG_ERROR("read error (single-page acquire failed).");
          return 0;
        }
        data.reset(owner_->buffer_pool_handle_.get(), page_id, raw);
        return len;
      }
      // C11 aligned_alloc requires the requested size to be a multiple of
      // the alignment, and alignment must be a power of two.  See the
      // sibling read(const void**) overload above for the rationale of
      // pinning kAlign to a fixed 4096 instead of sysconf(_SC_PAGESIZE).
      static constexpr size_t kAlign = 4096UL;
      size_t alloc_size = (len + (kAlign - 1UL)) & ~(kAlign - 1UL);
      char *tmp =
          static_cast<char *>(ailego_aligned_malloc(alloc_size, kAlign));
      if (!tmp) {
        LOG_ERROR("read error (alloc cross-page temp buffer failed).");
        return 0;
      }
      if (!owner_->buffer_pool_handle_->read_range(abs_offset, len, tmp)) {
        ailego_free(tmp);
        LOG_ERROR("read error (cross-page read_range failed).");
        return 0;
      }
      data = MemoryBlock::MakeOwned(tmp, len);
      return len;
    }

    //! Write data into the storage with offset.
    //!
    //! Locking: shared shard latch pairs with flush_index()'s exclusive
    //! all-shards latch -- excludes CRC compute over meta_buf while we
    //! mutate (data_size, padding_size).  meta_mtx_ additionally
    //! serialises concurrent writers on the SAME segment so the pair
    //! stays consistent (sum == capacity_).
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
      // Write the bytes BEFORE publishing the new data_size to readers.
      // Lock-free readers observe data_size with acquire ordering; the
      // release-store below establishes happens-before with the page
      // contents written above.  Publishing data_size first (the previous
      // ordering) allowed a reader on weakly-ordered ARM to see the new
      // length but still read stale page contents -- or, in the inverse
      // direction, see a stale length and truncate len to 0
      // (root cause of "Read sparse vector failed ... ret=0").
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
          // padding_size is paired with data_size; publish it first
          // (relaxed) so readers that acquire data_size see a
          // consistent (data_size + padding_size == capacity_) pair.
          bs_store_relaxed(&meta->padding_size, capacity_ - new_size);
          bs_store_release(&meta->data_size, new_size);
        }
      }
      // Mark dirty unconditionally even when data_size did not grow:
      // fixed-size in-place rewrites (e.g. chunk_meta_segment) must still
      // trigger flush_all() before the next append_segment().
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
          // See write() for the publish ordering rationale: padding first
          // (relaxed), then release-store data_size so concurrent lock-free
          // readers observe a consistent pair.
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

   protected:
    friend BufferStorage;
    // Pointer into BufferStorage::segments_ (unordered_map mapped value).
    // The address is stable across map insertions, so re-parses after
    // append_segment() are picked up without recreating WrappedSegment.
    IndexMapping::SegmentInfo *segment_info_{nullptr};
    // Serialises hot-path writers on the SAME segment so
    // (data_size, padding_size, data_crc) updates do not interleave.
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

    // Open in writable mode when the caller expects to modify the index
    // (create_if_missing=true implies write intent, same as MMapFileStorage).
    buffer_pool_ = std::make_shared<ailego::VecBufferPool>(
        path, /*writable=*/create_if_missing);
    buffer_pool_handle_ = std::make_shared<ailego::VecBufferPoolHandle>(
        buffer_pool_->get_handle());
    int ret = ParseToMapping();
    if (ret != 0) {
      this->close_index();
      return ret;
    }
    ret = buffer_pool_->init();
    if (ret != 0) {
      this->close_index();
      return ret;
    }
    LOG_INFO(
        "BufferStorage opened: file=%s, writable=%d, max_segment_size=%lu, "
        "segment_count=%zu",
        file_name_.c_str(), static_cast<int>(create_if_missing),
        max_segment_size_, segments_.size());
    return 0;
  }

  // PRECONDITION (also for ParseFooter/ParseSegment/ParseToMapping):
  // caller holds either single-threaded open() or AllShardsExclusiveLatch.
  // Do NOT add an internal lock here -- std::shared_mutex is not reentrant.
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
      // Use id_hash_.size() (not segments_.size()) for the block_id:
      // segments_ is intentionally NOT cleared between appends to keep
      // existing WrappedSegment pointers valid, so it carries stale entries.
      //
      // Bound the C-string scan to the segments_meta buffer so a missing
      // NUL terminator cannot walk past the buffer end (defence against
      // crafted-CRC inputs; CRC already covers benign bit flips).
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
      // In-place update so existing WrappedSegment pointers see the
      // refreshed meta_ptr_ after re-parse.  chain_header MUST be the
      // per-chain owning copy (not a shared &header_) -- see
      // chain_headers_ field comment.
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
        LOG_ERROR(
            "ParseToMapping: invalid footer_offset=%lu (header=%lu, "
            "file_size=%lu), file[%s]",
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

      // Record per-chain metadata offsets so flush_index() can write
      // updated segment metas and footers back to the backing file.
      meta_chains_.push_back({current_header_start_offset_, footer_offset,
                              segment_start_offset, footer_.segments_meta_size,
                              segment_ids_offset, footer_});

      if (footer_.next_meta_header_offset == 0) {
        break;
      }
      // Reject self-reference / backward jumps and offsets past file_size:
      // such a corrupted next_meta_header_offset would otherwise drive the
      // loop into infinite chain growth -> OOM.
      const uint64_t next_off = footer_.next_meta_header_offset;
      if (next_off <= current_header_start_offset_ ||
          next_off + sizeof(IndexFormat::MetaHeader) >
              buffer_pool_->file_size()) {
        LOG_ERROR(
            "ParseToMapping: invalid next_meta_header_offset=%lu "
            "(current=%lu, file_size=%lu), file[%s]",
            next_off, current_header_start_offset_, buffer_pool_->file_size(),
            file_name_.c_str());
        return IndexError_InvalidValue;
      }
      // Bound chain count: 1024 chains @ default 1MB segment_meta_capacity
      // covers >1GB of metadata, far above realistic load.
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
  //! Initialize index version segment (writes content into an IndexMapping).
  //! Only intended to be called from init_index() while `mapping` is still
  //! open in create-mode.
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

  //! Create the initial on-disk index structure and write the mandatory
  //! version segment.  Uses IndexMapping (the same engine as MMapFileStorage)
  //! so the produced file is fully compatible with both storage backends.
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

  //! Mark the index as dirty.  HOT PATH: store(true) unconditionally --
  //! a load-then-store guard could let a stale cached `true` skip the
  //! store after flush_index() CAS'd dirty=false on another core, losing
  //! the writer's modification.
  void set_as_dirty(void) {
    index_dirty_.store(true, std::memory_order_relaxed);
  }

  //! Refresh meta information (checksum, update time, etc.)
  void refresh_index(uint64_t chkp) {
    // CAS-loop max: callers may invoke refresh() out of order, and the
    // persisted check_point must be non-decreasing.  Relaxed ordering is
    // sufficient because flush_index() takes AllShardsExclusiveLatch which
    // establishes the necessary happens-before for the disk write.
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
    // Exclusive all-shards latch excludes the lock-free hot path while we
    // hash meta_buf and pwrite footer; without it segments_meta_crc would
    // not match the bytes on disk.
    AllShardsExclusiveLatch latch(mapping_shards_);
    return flush_index_locked();
  }

  //! PRECONDITION: caller holds AllShardsExclusiveLatch.  Used by
  //! flush_index() (acquires the latch) and close_index() (must flush
  //! and tear down under one continuous latch hold).
  int flush_index_locked(void) {
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
    // Claim dirty atomically AT THE START so any concurrent write() that
    // lands during this flush re-sets dirty=true and is picked up by the
    // next flush; an unconditional store(false) at the end would silently
    // swallow it.
    bool expected_dirty = true;
    if (!index_dirty_.compare_exchange_strong(expected_dirty, false,
                                              std::memory_order_relaxed)) {
      // Another thread already claimed; bail out.
      return 0;
    }
    // Snapshot pending_check_point_ AFTER claiming dirty: any newer chkp
    // stored by a concurrent refresh_index() will be preserved by the
    // CAS-reset at the end (and refresh_index() will have re-set dirty).
    const uint64_t consumed_chkp =
        pending_check_point_.load(std::memory_order_relaxed);
    // Restore consumed_chkp on failure paths (CAS-loop max, same as
    // refresh_index()) so a concurrent larger chkp wins.
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
    // Per-chain: recompute segments_meta CRC, refresh footer, pwrite both.
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
    // CAS-reset pending: only consume the chkp we observed.  A concurrent
    // larger chkp survives and will be flushed next round (refresh_index()
    // also re-set dirty).
    uint64_t expected_chkp = consumed_chkp;
    pending_check_point_.compare_exchange_strong(expected_chkp, 0,
                                                 std::memory_order_relaxed);
    return 0;
  }

  //! Close index storage
  void close_index(void) {
    // Hold ONE continuous all-shards latch across flush + teardown so no
    // writer can slip in between (which would dirty meta_buf only to have
    // the page table reset under it, dropping the modification).
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
    // Release every page pinned by the single-page read(const void**)
    // overload (the never-released contract holds the pin until here).
    // Each pin incremented the page table ref_count, so we must drop the
    // matching reference before resetting the pool, otherwise
    // ~VecBufferPool asserts that all blocks were released.
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
    max_segment_size_ = 0;
    buffer_pool_buffers_.clear();
    meta_chains_.clear();
    current_header_start_offset_ = 0;
    pending_check_point_.store(0, std::memory_order_relaxed);
    index_dirty_.store(false, std::memory_order_relaxed);
    corrupted_.store(false, std::memory_order_relaxed);
  }

  //! Append a segment into storage.  C1: page table extends in-place;
  //! latch held only briefly to protect segments_/id_hash_ insertion.
  int append_segment(const std::string &id, size_t size) {
    // Persist any pending data_size/padding/CRC mutations from prior
    // write()/resize() before we re-hash and rewrite the segment_meta.
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

    // Rollback handle for an in-memory-committed chain split.  Default
    // no-op; populated only after Step 1 commits, so a Step 2 failure
    // can fully undo the split (otherwise an orphan empty chain would
    // remain linked in the file).
    std::function<void()> rollback_step1 = []() {};

    // ---- Step 1: chain split if current chain has no meta capacity left.
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

      // Best-effort restore of the old footer if any subsequent write in
      // this split block fails.  If the restore itself fails, mark the
      // storage corrupted -- on-disk old footer now points at a partial
      // new chain region.
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

      // Snapshot the OLD chain's pre-commit state for rollback_step1
      // (captured by value: `chain` is reassigned below).
      const auto saved_old_chain_footer = chain->footer;
      const uint64_t saved_old_footer_file_offset = chain->footer_file_offset;
      const uint64_t saved_current_header_start = current_header_start_offset_;

      // Strong exception guarantee: reserve() FIRST so the three
      // push_back's cannot throw mid-way and leave
      // chain_headers_/buffer_pool_buffers_/meta_chains_ at mismatched
      // sizes (which flush_index_locked() would silently skip while
      // ParseToMapping() on next open follows the on-disk forward link).
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

      // Install rollback for the committed split.  Captures by value so
      // later reassignment of chain/header/meta_buf does not corrupt the
      // closure.
      rollback_step1 = [this, saved_footer_before_split, saved_old_chain_footer,
                        saved_old_footer_file_offset,
                        saved_current_header_start]() {
        // 1. Drop the forward link on the old footer.  If this fails the
        //    on-disk old footer still points at the popped new chain
        //    region -- mark corrupted.
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
        // 2. Pop the freshly-pushed new chain (releases its unique_ptrs).
        if (!meta_chains_.empty()) meta_chains_.pop_back();
        if (!chain_headers_.empty()) chain_headers_.pop_back();
        if (!buffer_pool_buffers_.empty()) buffer_pool_buffers_.pop_back();
        // 3. Restore the old chain's in-memory footer (forward link cleared).
        if (!meta_chains_.empty()) {
          meta_chains_.back().footer = saved_old_chain_footer;
        }
        // 4. Restore footer_ + current_header_start_offset_.  The on-disk
        //    file size is intentionally NOT shrunk: the orphan region is
        //    unreachable (step 1 cleared the link) and reusable by the
        //    next split via file_size() realignment.
        footer_ = saved_footer_before_split;
        current_header_start_offset_ = saved_current_header_start;
      };
    }

    // ---- Step 2: append SegmentMeta + ID into the (possibly new) last
    //              chain, then persist meta_buf and footer.
    uint64_t new_data_index = footer_.content_size;
    uint64_t new_seg_abs_offset =
        chain->header_start_offset + header->content_offset + new_data_index;
    uint64_t new_file_size = new_seg_abs_offset + padded_size;
    if (new_file_size > buffer_pool_->file_size()) {
      if (!buffer_pool_->extend_file(new_file_size)) {
        return IndexError_Runtime;
      }
    }

    // Save mutable state for rollback if a Step 2 disk write fails.  The
    // meta_buf regions that get overwritten (SegmentMeta entry + ID
    // string) are also snapshotted so they can be restored exactly,
    // keeping CRC consistent for a later flush_index().
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

    // Rollback for Step 2: restore in-memory state AND best-effort
    // rewrite the OLD segments_meta + footer back to disk.  Without the
    // disk rewrite, a write_meta(footer) failure (or post-write OOM)
    // would tell the caller the append failed yet leave on-disk bytes
    // describing the failed append -- ParseToMapping() on next open
    // would surface a ghost segment with no entry in segments_/id_hash_.
    //
    // If the rewrite itself fails the file is unrepairable from here:
    // raise corrupted_ so subsequent writers refuse to proceed.
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

    // Strong exception guarantee for the in-memory commit: emplace into
    // segments_ and id_hash_ as one transactional unit -- if id_hash_
    // throws after segments_ succeeded, undo segments_ before
    // propagating.  unordered_map::emplace() leaves existing element
    // addresses stable, so WrappedSegment instances pointing into
    // segments_ remain valid.
    auto seg_ins = segments_.end();
    bool seg_inserted = false;
    try {
      auto ins = segments_.emplace(
          id, IndexMapping::SegmentInfo{IndexMapping::Segment{new_seg},
                                        chain->header_start_offset, header});
      if (!ins.second) {
        // Cannot happen under the exclusive latch we hold (find() above
        // checked), but be defensive.
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
    // C1: extend_file() already extended the page table in-place; no pool
    // rotation or flush_all needed.
    return 0;
  }

  //! Test if a segment exists
  bool has_segment(const std::string &id) const {
    std::shared_lock<std::shared_mutex> latch(
        mapping_shards_[mapping_shard_id()].mtx);
    return (segments_.find(id) != segments_.end());
  }

 private:
  std::atomic<bool> index_dirty_{false};
  std::atomic<uint64_t> pending_check_point_{0};
  // Set when an append_segment() rollback fails to restore on-disk state.
  // Once set, all writers (write/append_segment/flush_index_locked) refuse
  // to proceed.  Only ever raised; cleared only by close_index().
  std::atomic<bool> corrupted_{false};

  // Sharded reader-writer lock: each reader hashes to its own shard to
  // avoid cache-line ping-pong on the reader counter; writers lock all
  // shards.
  static constexpr size_t kMappingMutexShards = 32;
  struct alignas(64) MutexShard {
    std::shared_mutex mtx;
  };
  mutable MutexShard mapping_shards_[kMappingMutexShards]{};

  // Per-(thread, instance) shard selection.  Combining thread::id with
  // `this` ensures two BufferStorage instances on the same thread map to
  // different shards (a thread_local-only id collapses them onto one
  // shard).  boost-style hash_combine disperses skewed thread::id
  // distributions across the 32 shards.
  size_t mapping_shard_id() const {
    size_t seed = std::hash<std::thread::id>()(std::this_thread::get_id());
    size_t inst = std::hash<const void *>()(static_cast<const void *>(this));
    // boost::hash_combine(seed, inst)
    seed ^= inst + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed % kMappingMutexShards;
  }

  // RAII guard that locks ALL shards exclusively (for writers).
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

  // Arena slab for cross-page temp buffers handed out by
  // WrappedSegment::read(const void**).  The legacy contract requires
  // every returned pointer to stay valid until close_index(), so slots
  // are never freed individually -- they are carved out of large
  // 4K-aligned arenas which are released in bulk.
  //
  // Why an arena instead of one posix_memalign(4K, 4K) per read:
  // Android Bionic scudo's small-class chunk pool is prone to large-
  // alignment starvation under fragmentation (we observed sporadic
  // posix_memalign(4096, 4096) returning ENOMEM even with plenty of
  // free memory).  A single large request (>= kArenaSize) is served
  // from scudo's secondary allocator (mmap-backed), which is reliable
  // up to the true OOM boundary.
  struct ArenaBlock {
    char *base{nullptr};
    size_t size{0};  // Total bytes in this arena (4K-aligned).
    size_t used{0};  // Bytes already handed out (4K-aligned).
  };
  // Caller MUST hold tmp_buffers_mutex_.  alloc_size MUST be a
  // multiple of 4096.  Returns nullptr only if scudo cannot satisfy a
  // fresh arena allocation, i.e. effectively true OOM.
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

  // Page ids pinned by the single-page read(const void**) overload, which
  // keeps the pin alive until close_index() (the never-released contract).
  // Each pin increments the page table ref_count, so close_index() must
  // release every recorded pin before tearing down the pool; otherwise
  // ~VecBufferPool fires its "all blocks released" assertion. Duplicates are
  // allowed: repeated single-page reads of the same page each take a pin and
  // therefore each need a matching release.
  std::vector<size_t> pinned_pages_{};
  mutable std::mutex pinned_pages_mutex_{};

  // buffer manager
  std::string file_name_;
  // Per-chain owning copies of MetaHeader.  segments_[name].segment_header
  // points into one of these; using a single shared header_ would let the
  // next chain's ParseHeader overwrite earlier-chain content_offset.
  std::vector<std::unique_ptr<IndexFormat::MetaHeader>> chain_headers_{};
  IndexFormat::MetaFooter footer_{};
  std::unordered_map<std::string, IndexMapping::SegmentInfo> segments_{};
  std::unordered_map<std::string, size_t> id_hash_{};
  uint64_t max_segment_size_{0};
  std::vector<std::unique_ptr<char[]>> buffer_pool_buffers_{};

  ailego::VecBufferPool::Pointer buffer_pool_{nullptr};
  ailego::VecBufferPoolHandle::Pointer buffer_pool_handle_{nullptr};
  uint64_t current_header_start_offset_{0u};

  // Capacity (in bytes) of the segment metadata section written by
  // init_index().
  uint32_t segment_meta_capacity_{4096u};

  // Per-header-chain file offsets used by flush_index() and append_segment().
  struct MetaChain {
    uint64_t header_start_offset;
    uint64_t footer_file_offset;
    uint64_t segment_meta_file_offset;
    uint32_t segment_meta_size;
    // Lowest segment-ID-string offset within segment_meta; equals
    // segment_meta_size when empty, decreases by strlen(id)+1 per append.
    // Used to detect when a chain split is needed.
    uint32_t segment_ids_offset;
    // In-memory copy of this chain's MetaFooter, kept in sync with disk by
    // flush_index() and append_segment() to avoid a pread per chain.
    IndexFormat::MetaFooter footer;
  };
  std::vector<MetaChain> meta_chains_{};
};

INDEX_FACTORY_REGISTER_STORAGE(BufferStorage);

}  // namespace core
}  // namespace zvec
