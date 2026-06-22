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
#include <chrono>
#include <cstring>
#include <thread>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/logger/logger.h>

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
static ssize_t zvec_pread(int fd, void *buf, size_t count, size_t offset) {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov = {};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_read = 0;
  if (!ReadFile(handle, buf, static_cast<DWORD>(count), &bytes_read, &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_read);
}
static ssize_t zvec_pwrite(int fd, const void *buf, size_t count,
                           size_t offset) {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) return -1;
  OVERLAPPED ov = {};
  ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
  ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD bytes_written = 0;
  if (!WriteFile(handle, buf, static_cast<DWORD>(count), &bytes_written, &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_written);
}
#else
#include <unistd.h>
static inline ssize_t zvec_pread(int fd, void *buf, size_t count,
                                 size_t offset) {
  return ::pread(fd, buf, count, static_cast<off_t>(offset));
}
static inline ssize_t zvec_pwrite(int fd, const void *buf, size_t count,
                                  size_t offset) {
  return ::pwrite(fd, buf, count, static_cast<off_t>(offset));
}
#endif

namespace zvec {
namespace ailego {

const size_t kVectorPageSize = MemoryHelper::PageSize();

bool VectorPageTable::init(size_t entry_num) {
  size_t need_segments = (entry_num + kSegmentSize - 1) / kSegmentSize;
  if (need_segments > kMaxSegments) {
    LOG_ERROR(
        "VectorPageTable::init: entry_num=%zu exceeds capacity "
        "(kMaxEntries=%zu, need_segments=%zu, kMaxSegments=%zu); "
        "refusing to init.",
        entry_num, kMaxEntries, need_segments, kMaxSegments);
    return false;
  }
  // Free old segments if any.  init() is only called from VecBufferPool::init
  // which is single-threaded with respect to other accesses, so a relaxed
  // load of segment_count_ is sufficient here.
  size_t old_count = segment_count_.load(std::memory_order_relaxed);
  for (size_t i = 0; i < old_count; ++i) {
    delete[] segments_[i];
    segments_[i] = nullptr;
  }
  for (size_t s = 0; s < need_segments; ++s) {
    segments_[s] = new Entry[kSegmentSize];
    for (size_t i = 0; i < kSegmentSize; ++i) {
      segments_[s][i].ref_count.store(std::numeric_limits<int>::min());
      segments_[s][i].in_evict_queue.store(false);
      segments_[s][i].is_dirty.store(false);
      segments_[s][i].buffer = nullptr;
      segments_[s][i].file_offset = 0;
    }
  }
  // Publish new segments to readers.  segment_count_ is published first
  // (release) so that a reader that acquire-loads segment_count_ before
  // entry_num_ also sees a consistent segment table; entry_num_ is the
  // primary synchronization point used by callers via entry_num().
  segment_count_.store(need_segments, std::memory_order_release);
  entry_num_.store(entry_num, std::memory_order_release);
  return true;
}

bool VectorPageTable::extend(size_t new_entry_num) {
  // Relaxed read is fine: extend() is serialized by the caller (extend_file
  // is invoked under the BufferStorage write latch).  No other writer races
  // with us on entry_num_ / segment_count_.
  if (new_entry_num <= entry_num_.load(std::memory_order_relaxed)) {
    return true;
  }
  size_t new_segment_count = (new_entry_num + kSegmentSize - 1) / kSegmentSize;
  if (new_segment_count > kMaxSegments) {
    LOG_ERROR(
        "VectorPageTable::extend: new_entry_num=%zu exceeds capacity "
        "(kMaxEntries=%zu, new_segment_count=%zu, kMaxSegments=%zu); "
        "refusing to extend.",
        new_entry_num, kMaxEntries, new_segment_count, kMaxSegments);
    return false;
  }
  size_t old_count = segment_count_.load(std::memory_order_relaxed);
  for (size_t s = old_count; s < new_segment_count; ++s) {
    segments_[s] = new Entry[kSegmentSize];
    for (size_t i = 0; i < kSegmentSize; ++i) {
      segments_[s][i].ref_count.store(std::numeric_limits<int>::min());
      segments_[s][i].in_evict_queue.store(false);
      segments_[s][i].is_dirty.store(false);
      segments_[s][i].buffer = nullptr;
      segments_[s][i].file_offset = 0;
    }
  }
  // Publish in the same order as init(): segment_count_ first, entry_num_
  // last.  Both are release-stores so that the prior segment allocation /
  // Entry initialization is visible to any reader that acquire-loads either
  // counter (typically via entry_num()).
  segment_count_.store(new_segment_count, std::memory_order_release);
  entry_num_.store(new_entry_num, std::memory_order_release);
  return true;
}

char *VectorPageTable::acquire_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  while (true) {
    int current_count = e.ref_count.load(std::memory_order_acquire);
    if (current_count < 0) {
      return nullptr;
    }
    if (e.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
      return e.buffer;
    }
  }
}

void VectorPageTable::release_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);

  if (e.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    bool expected = false;
    if (e.in_evict_queue.compare_exchange_strong(expected, true,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_relaxed)) {
      BlockEvictionQueue::BlockType block;
      block.owner = this;
      block.owner_key = block_id;
      block.version = 0;
      BlockEvictionQueue::get_instance().add_single_block(block, 0);
    }
  }
}

void VectorPageTable::evict_block(block_id_t block_id) {
  assert(block_id < entry_num_.load(std::memory_order_relaxed));
  Entry &e = entry_at(block_id);
  int expected = 0;
  // Two-phase eviction to prevent data race on e.buffer with
  // set_block_acquired.  We first CAS to kEvicting (-1), which causes
  // set_block_acquired to spin-wait; then do the actual work (flush, free,
  // null buffer); finally store INT_MIN ("evicted") which unblocks
  // set_block_acquired.
  static constexpr int kEvicting = -1;
  if (e.ref_count.compare_exchange_strong(expected, kEvicting)) {
    char *buffer = e.buffer;
    if (buffer && e.is_dirty.load(std::memory_order_relaxed) &&
        flush_callback_) {
      flush_callback_(block_id, buffer, kVectorPageSize, e.file_offset);
      e.is_dirty.store(false, std::memory_order_relaxed);
    }
    if (buffer) {
      e.buffer = nullptr;
      MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    }
    // Transition to fully-evicted state.  Use release so that the
    // set_block_acquired acquire-load sees e.buffer == nullptr.
    e.ref_count.store(std::numeric_limits<int>::min(),
                      std::memory_order_release);
  }
  e.in_evict_queue.store(false, std::memory_order_relaxed);
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t file_offset) {
  assert(block_id < entry_num_.load(std::memory_order_acquire));
  Entry &e = entry_at(block_id);
  // Diagnostics for the kEvicting wait. The wait itself never gives up:
  // the only thread that can transition kEvicting -> INT_MIN is the
  // evict_block() owner, so abandoning the spin here would orphan the
  // entry in kEvicting forever. Instead, we use bounded backoff and emit
  // tiered logs so a stuck eviction is observable.
  using clock = std::chrono::steady_clock;
  const auto wait_start = clock::now();
  auto last_log = wait_start;
  unsigned spin_count = 0;
  bool warned = false;
  while (true) {
    int current_count = e.ref_count.load(std::memory_order_acquire);
    if (current_count >= 0) {
      if (e.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                            std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
        return e.buffer;
      }
    } else if (current_count == std::numeric_limits<int>::min()) {
      // Fully evicted — safe to claim this entry for our new buffer.
      e.buffer = buffer;
      e.file_offset = file_offset;
      e.in_evict_queue.store(false, std::memory_order_relaxed);
      e.is_dirty.store(false, std::memory_order_relaxed);
      e.ref_count.store(1, std::memory_order_release);
      return e.buffer;
    } else {
      // kEvicting (-1): eviction is in progress on this entry.
      // Tiered backoff: hot spin first, then short sleep, then longer sleep.
      ++spin_count;
      if (spin_count < 64) {
        // Pure busy wait for the common ~μs case.
      } else if (spin_count < 1024) {
        std::this_thread::yield();
      } else if (spin_count < 8192) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      // Tiered diagnostics: warn once after 100ms, error every 1s after 1s.
      const auto now = clock::now();
      const auto elapsed = now - wait_start;
      if (!warned && elapsed >= std::chrono::milliseconds(100)) {
        LOG_WARN(
            "set_block_acquired: long kEvicting wait on block_id=%zu "
            "(>=100ms); evict_block may be slow",
            static_cast<size_t>(block_id));
        warned = true;
      }
      if (elapsed >= std::chrono::seconds(1) &&
          (now - last_log) >= std::chrono::seconds(1)) {
        const auto secs =
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        LOG_ERROR(
            "set_block_acquired: stuck in kEvicting on block_id=%zu for "
            "%lld s; evict_block owner may be hung or starved",
            static_cast<size_t>(block_id), static_cast<long long>(secs));
        last_log = now;
      }
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename, bool writable) {
  file_name_ = filename;
  writable_ = writable;
#if defined(_MSC_VER)
  int flags = writable_ ? (O_RDWR | _O_BINARY) : (O_RDONLY | _O_BINARY);
  fd_ = _open(filename.c_str(), flags, 0644);
#else
  int flags = writable_ ? O_RDWR : O_RDONLY;
  fd_ = ::open(filename.c_str(), flags, 0644);
#endif
  if (fd_ < 0) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
#if defined(_MSC_VER)
  struct _stat64 st;
  if (_fstat64(fd_, &st) < 0) {
    _close(fd_);
#else
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    ::close(fd_);
#endif
    throw std::runtime_error("Failed to stat file: " + filename);
  }
  file_size_ = st.st_size;
}

int VecBufferPool::init() {
  size_t block_num = (file_size_ + kVectorPageSize - 1) / kVectorPageSize;
  if (!page_table_.init(block_num)) {
    LOG_ERROR(
        "VecBufferPool::init: page_table_ init failed for file[%s], "
        "file_size=%zu, block_num=%zu (exceeds "
        "VectorPageTable::kMaxEntries=%zu)",
        file_name_.c_str(), file_size_, block_num,
        VectorPageTable::kMaxEntries);
    return -1;
  }
  block_mutexes_ =
      std::make_unique<std::mutex[]>(VecBufferPool::kMutexBucketCount);
  LOG_DEBUG("entry num: %zu, file_size: %zu", page_table_.entry_num(),
            file_size_);

  // In writable mode, inject a flush callback into the page table so that
  // evict_block()/flush_block()/flush_all() can pwrite dirty blocks back to
  // the backing file without needing to know about fd_ directly.
  if (writable_) {
    int fd = fd_;
    const std::string &name = file_name_;
    page_table_.set_flush_callback([fd, &name](block_id_t /*block_id*/,
                                               char *buf, size_t sz,
                                               size_t off) -> int {
      ssize_t w = zvec_pwrite(fd, buf, sz, off);
      if (w != static_cast<ssize_t>(sz)) {
        LOG_ERROR(
            "Buffer pool flush failed: file[%s], offset[%zu], "
            "expected[%zu], got[%zd]",
            name.c_str(), off, sz, w);
        return -1;
      }
      return 0;
    });
  }
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

char *VecBufferPool::acquire_buffer(block_id_t page_id, int retry) {
  assert(page_id < page_table_.entry_num());
  char *buffer = page_table_.acquire_block(page_id);
  if (buffer) {
    return buffer;
  }
  std::lock_guard<std::mutex> lock(
      block_mutexes_[page_id % VecBufferPool::kMutexBucketCount]);
  buffer = page_table_.acquire_block(page_id);
  if (buffer) {
    return buffer;
  }
  {
    bool found = MemoryLimitPool::get_instance().try_acquire_buffer(
        kVectorPageSize, buffer);
    if (!found) {
      for (int i = 0; i < retry; i++) {
        BlockEvictionQueue::get_instance().recycle();
        found = MemoryLimitPool::get_instance().try_acquire_buffer(
            kVectorPageSize, buffer);
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      LOG_ERROR("Buffer pool failed to get free buffer: file[%s], page_id[%zu]",
                file_name_.c_str(), page_id);
      return nullptr;
    }
  }

  size_t page_offset = page_id * kVectorPageSize;
  size_t expected_bytes = std::min(kVectorPageSize, file_size_ - page_offset);
  if (expected_bytes < kVectorPageSize) {
    std::memset(buffer + expected_bytes, 0, kVectorPageSize - expected_bytes);
  }
  ssize_t read_bytes = zvec_pread(fd_, buffer, expected_bytes, page_offset);
  if (read_bytes != static_cast<ssize_t>(expected_bytes)) {
    LOG_ERROR(
        "Buffer pool failed to read file at offset: file[%s], page_id[%zu], "
        "offset[%zu], expected[%zu], got[%zd]",
        file_name_.c_str(), page_id, page_offset, expected_bytes, read_bytes);
    MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    return nullptr;
  }
  return page_table_.set_block_acquired(page_id, buffer, page_offset);
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
  ssize_t read_bytes = zvec_pread(fd_, buffer, length, offset);
  if (read_bytes != static_cast<ssize_t>(length)) {
    LOG_ERROR(
        "Buffer pool failed to read file at offset: file[%s], offset[%zu], "
        "length[%zu]",
        file_name_.c_str(), offset, length);
    return -1;
  }
  return 0;
}

int VecBufferPool::write_range(size_t file_offset, size_t length,
                               const char *src) {
  if (!writable_) {
    LOG_ERROR("write_range called on read-only pool: file[%s]",
              file_name_.c_str());
    return -1;
  }
  if (length == 0) {
    return 0;
  }
  size_t first_page = file_offset / kVectorPageSize;
  size_t last_page = (file_offset + length - 1) / kVectorPageSize;
  size_t remaining = length;
  size_t src_cursor = 0;
  for (size_t pg = first_page; pg <= last_page; ++pg) {
    // Loading the page ensures we do not clobber unrelated bytes within the
    // same page when the write is not page-aligned. acquire_buffer() pre-fills
    // from the backing file (or zero-pads beyond EOF).
    char *page = this->acquire_buffer(pg, 50);
    if (!page) {
      LOG_ERROR("write_range acquire failed: file[%s], page[%zu]",
                file_name_.c_str(), pg);
      return -1;
    }
    size_t page_start = pg * kVectorPageSize;
    size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
    size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
    std::memcpy(page + intra_offset, src + src_cursor, chunk);
    page_table_.mark_dirty(pg);
    page_table_.release_block(pg);
    src_cursor += chunk;
    remaining -= chunk;
  }
  return 0;
}

int VecBufferPool::write_meta(size_t offset, size_t length,
                              const char *buffer) {
  if (!writable_) {
    LOG_ERROR("write_meta called on read-only pool: file[%s]",
              file_name_.c_str());
    return -1;
  }
  ssize_t w = zvec_pwrite(fd_, buffer, length, offset);
  if (w != static_cast<ssize_t>(length)) {
    LOG_ERROR(
        "Buffer pool failed to write meta: file[%s], offset[%zu], "
        "length[%zu], got[%zd]",
        file_name_.c_str(), offset, length, w);
    return -1;
  }
  return 0;
}

int VecBufferPool::flush_all() {
  if (!writable_) {
    return 0;
  }
  int rc = 0;
  size_t total_dirty = 0;
  size_t fail_count = 0;
  for (size_t i = 0; i < page_table_.entry_num(); ++i) {
    if (page_table_.is_block_dirty(i)) {
      ++total_dirty;
      int r = page_table_.flush_block(i);
      if (r != 0) {
        rc = r;
        ++fail_count;
      }
    }
  }
  if (fail_count != 0) {
    // Aggregated diagnostic so that callers (notably ~VecBufferPool, which
    // discards the return value) cannot silently lose dirty pages: any
    // unflushed page at this point means the on-disk image is now stale.
    LOG_ERROR(
        "VecBufferPool::flush_all: %zu/%zu dirty page(s) failed to flush, "
        "file[%s] last_rc=%d -- on-disk data may be stale.",
        fail_count, total_dirty, file_name_.c_str(), rc);
  }
  return rc;
}

bool VecBufferPool::extend_file(size_t new_size) {
  if (!writable_) {
    LOG_ERROR("extend_file called on read-only pool: file[%s]",
              file_name_.c_str());
    return false;
  }
  if (new_size <= file_size_) {
    return true;
  }
  // Pre-validate against the page table's static capacity BEFORE mutating
  // any on-disk state.  Otherwise a successful ftruncate followed by a
  // failed page_table_.extend() would leave the file size and the page
  // table out of sync (file grew, but no Entry slots cover the new range).
  size_t new_entry_num = (new_size + kVectorPageSize - 1) / kVectorPageSize;
  if (new_entry_num > VectorPageTable::kMaxEntries) {
    LOG_ERROR(
        "extend_file: requested new_size=%zu would require %zu page entries, "
        "exceeding VectorPageTable::kMaxEntries=%zu (file=%s).",
        new_size, new_entry_num, VectorPageTable::kMaxEntries,
        file_name_.c_str());
    return false;
  }
#if defined(_MSC_VER)
  if (_chsize_s(fd_, static_cast<int64_t>(new_size)) != 0) {
    LOG_ERROR("extend_file _chsize_s failed: file[%s], new_size[%zu]",
              file_name_.c_str(), new_size);
    return false;
  }
#else
  if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
    LOG_ERROR("extend_file ftruncate failed: file[%s], new_size[%zu]",
              file_name_.c_str(), new_size);
    return false;
  }
#endif
  file_size_ = new_size;
  // Extend the page table to cover the new file range.  Existing entries
  // stay at their original addresses so concurrent readers are unaffected.
  // Capacity has already been validated above, so this should never fail;
  // a failure here would indicate a programming error and is logged.
  if (new_entry_num > page_table_.entry_num()) {
    if (!page_table_.extend(new_entry_num)) {
      LOG_ERROR(
          "extend_file: page_table_.extend(%zu) failed unexpectedly after "
          "capacity pre-check (file=%s, new_size=%zu).",
          new_entry_num, file_name_.c_str(), new_size);
      return false;
    }
  }
  return true;
}

char *VecBufferPoolHandle::get_single_page(size_t file_offset, size_t len,
                                           size_t &out_page_id) {
  size_t first_page = file_offset / kVectorPageSize;
  assert(len == 0 || (file_offset + len - 1) / kVectorPageSize == first_page);
  out_page_id = first_page;
  char *page = pool_.acquire_buffer(first_page, 50);
  if (!page) {
    LOG_ERROR(
        "VecBufferPoolHandle::get_single_page: acquire_buffer failed, "
        "file_offset=%zu, len=%zu, page=%zu, page_size=%zu",
        file_offset, len, first_page, kVectorPageSize);
    return nullptr;
  }
  return page + (file_offset - first_page * kVectorPageSize);
}

bool VecBufferPoolHandle::read_range(size_t file_offset, size_t len,
                                     char *out) {
  if (len == 0) {
    return true;
  }
  size_t first_page = file_offset / kVectorPageSize;
  size_t last_page = (file_offset + len - 1) / kVectorPageSize;
  size_t remaining = len;
  size_t dst_cursor = 0;
  for (size_t pg = first_page; pg <= last_page; ++pg) {
    char *page = pool_.acquire_buffer(pg, 50);
    if (!page) {
      LOG_ERROR(
          "VecBufferPoolHandle::read_range: acquire_buffer failed, "
          "file_offset=%zu, len=%zu, page=%zu, first_page=%zu, last_page=%zu, "
          "page_size=%zu",
          file_offset, len, pg, first_page, last_page, kVectorPageSize);
      return false;
    }
    size_t page_start = pg * kVectorPageSize;
    size_t intra_offset = (pg == first_page) ? (file_offset - page_start) : 0;
    size_t chunk = std::min(kVectorPageSize - intra_offset, remaining);
    std::memcpy(out + dst_cursor, page + intra_offset, chunk);
    pool_.page_table_.release_block(pg);
    dst_cursor += chunk;
    remaining -= chunk;
  }
  return true;
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
  return pool_.get_meta(offset, length, buffer);
}

int VecBufferPoolHandle::write_range(size_t file_offset, size_t len,
                                     const char *src) {
  return pool_.write_range(file_offset, len, src);
}

int VecBufferPoolHandle::write_meta(size_t offset, size_t length,
                                    const char *buffer) {
  return pool_.write_meta(offset, length, buffer);
}

int VecBufferPoolHandle::flush_all() {
  return pool_.flush_all();
}

bool VecBufferPoolHandle::writable() const {
  return pool_.writable();
}

void VecBufferPoolHandle::release_one(block_id_t block_id) {
  pool_.page_table_.release_block(block_id);
}

void VecBufferPoolHandle::acquire_one(block_id_t block_id) {
  // The caller must guarantee the block is already loaded before calling
  // acquire_one(). The return value of acquire_block() is intentionally
  // ignored here, as a null return would indicate a contract violation.
  pool_.page_table_.acquire_block(block_id);
}

}  // namespace ailego
}  // namespace zvec
