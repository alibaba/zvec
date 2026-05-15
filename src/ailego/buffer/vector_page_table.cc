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
#include <cstring>
#include <ailego/utility/memory_helper.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_logger.h>

#if !defined(_MSC_VER)
#include <unistd.h>
#endif

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
#endif

namespace zvec {
namespace ailego {

const size_t kVectorPageSize = MemoryHelper::PageSize();

void VectorPageTable::init(size_t entry_num) {
  if (entries_) {
    delete[] entries_;
  }
  entry_num_ = entry_num;
  entries_ = new Entry[entry_num_];
  for (size_t i = 0; i < entry_num_; i++) {
    entries_[i].ref_count.store(std::numeric_limits<int>::min());
    entries_[i].in_evict_queue.store(false);
    entries_[i].is_dirty.store(false);
    entries_[i].buffer = nullptr;
    entries_[i].file_offset = 0;
  }
}

char *VectorPageTable::acquire_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_acquire);
    if (current_count < 0) {
      return nullptr;
    }
    if (entry.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return entry.buffer;
    }
  }
}

void VectorPageTable::release_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];

  if (entry.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    // Attempt to transition in_evict_queue from false -> true.  The CAS ensures
    // only one thread enqueues this block even if multiple threads race here.
    bool expected = false;
    if (entry.in_evict_queue.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      BlockEvictionQueue::BlockType block;
      block.page_table = this;
      block.vector_block.first = block_id;
      block.vector_block.second = 0;
      BlockEvictionQueue::get_instance().add_single_block(block, 0);
    }
    // else: block is already in the eviction queue; do not add a duplicate
    // entry.
  }
}

void VectorPageTable::evict_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  char *buffer = entry.buffer;
  int expected = 0;
  if (entry.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    // If the block is dirty, flush it to disk before freeing the memory so
    // that no modified data is silently lost during eviction.
    if (buffer && entry.is_dirty.load(std::memory_order_relaxed) &&
        flush_callback_) {
      flush_callback_(block_id, buffer, kVectorPageSize, entry.file_offset);
      entry.is_dirty.store(false, std::memory_order_relaxed);
    }
    if (buffer) {
      MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
    }
  }
  // Always reset in_evict_queue regardless of whether the CAS succeeded:
  //  - On success: the block is evicted; future releases should re-register it.
  //  - On failure: the block was re-acquired by another thread between the
  //    ref-count check and this call.  Clearing in_evict_queue lets the next
  //    release_block() re-enqueue it so it is not silently lost.
  entry.in_evict_queue.store(false, std::memory_order_relaxed);
}

char *VectorPageTable::set_block_acquired(block_id_t block_id, char *buffer,
                                          size_t file_offset) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        MemoryLimitPool::get_instance().release_buffer(buffer, kVectorPageSize);
        return entry.buffer;
      }
    } else {
      entry.buffer = buffer;
      entry.file_offset = file_offset;
      entry.in_evict_queue.store(false, std::memory_order_relaxed);
      // A freshly loaded block is clean (memory matches disk).
      entry.is_dirty.store(false, std::memory_order_relaxed);
      entry.ref_count.store(1, std::memory_order_release);
      return entry.buffer;
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename, bool writable,
                             bool create) {
  file_name_ = filename;
  writable_ = writable || create;
#if defined(_MSC_VER)
  int flags =
      writable_
          ? (create ? (O_RDWR | O_CREAT | O_TRUNC | _O_BINARY)
                    : (O_RDWR | _O_BINARY))
          : (O_RDONLY | _O_BINARY);
  fd_ = _open(filename.c_str(), flags, 0644);
#else
  int flags = writable_
                  ? (create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDWR)
                  : O_RDONLY;
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
  page_table_.init(block_num);
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
    page_table_.set_flush_callback(
        [fd, &name](block_id_t /*block_id*/, char *buf, size_t sz,
                    size_t off) -> int {
#if defined(_MSC_VER)
          ssize_t w = zvec_pwrite(fd, buf, sz, off);
#else
          ssize_t w = ::pwrite(fd, buf, sz, off);
#endif
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
#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, expected_bytes, page_offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, expected_bytes, page_offset);
#endif
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
#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, length, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, length, offset);
#endif
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
    size_t intra_offset =
        (pg == first_page) ? (file_offset - page_start) : 0;
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
#if defined(_MSC_VER)
  ssize_t w = zvec_pwrite(fd_, buffer, length, offset);
#else
  ssize_t w = ::pwrite(fd_, buffer, length, offset);
#endif
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
  for (size_t i = 0; i < page_table_.entry_num(); ++i) {
    if (page_table_.is_block_dirty(i)) {
      int r = page_table_.flush_block(i);
      if (r != 0) {
        rc = r;
      }
    }
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
  return true;
}

char *VecBufferPoolHandle::get_single_page(size_t file_offset, size_t len,
                                           size_t &out_page_id) {
  size_t first_page = file_offset / kVectorPageSize;
  assert(len == 0 || (file_offset + len - 1) / kVectorPageSize == first_page);
  out_page_id = first_page;
  char *page = pool_.acquire_buffer(first_page, 50);
  if (!page) {
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