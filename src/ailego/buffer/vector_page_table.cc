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
  if (!WriteFile(handle, buf, static_cast<DWORD>(count), &bytes_written,
                 &ov)) {
    return -1;
  }
  return static_cast<ssize_t>(bytes_written);
}
#endif

namespace zvec {
namespace ailego {

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
  size_t size = entry.size;
  int expected = 0;
  if (entry.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    // If the block is dirty, flush it to disk before freeing the memory so
    // that no modified data is silently lost during eviction.
    if (buffer && entry.is_dirty.load(std::memory_order_relaxed) &&
        flush_callback_) {
      flush_callback_(block_id, buffer, size, entry.file_offset);
      entry.is_dirty.store(false, std::memory_order_relaxed);
    }
    if (buffer) {
      MemoryLimitPool::get_instance().release_buffer(buffer, size);
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
                                          size_t size, size_t file_offset) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      // Defensive branch: in practice this path should never be reached.
      // set_block_acquired() is always called under block_mutexes_[block_id],
      // and the caller (acquire_buffer) re-checks acquire_block() inside the
      // same lock before invoking this function. Therefore, if we get here,
      // ref_count must still be negative (unloaded). This branch is retained
      // as a safety net in case the locking contract is violated in the future,
      // e.g. if set_block_acquired is called from an unlocked context.
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        MemoryLimitPool::get_instance().release_buffer(buffer, size);
        return entry.buffer;
      }
    } else {
      entry.buffer = buffer;
      entry.size = size;
      entry.file_offset = file_offset;
      // Ensure in_evict_queue is cleared when the block is freshly loaded so
      // that the first release_block() after loading can register it in the
      // eviction queue.
      entry.in_evict_queue.store(false, std::memory_order_relaxed);
      // A freshly loaded block is clean (data matches disk).
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
  int flags = writable_
                  ? (create ? (O_RDWR | O_CREAT | O_TRUNC | _O_BINARY)
                            : (O_RDWR | _O_BINARY))
                  : (O_RDONLY | _O_BINARY);
  fd_ = _open(filename.c_str(), flags, 0644);
#else
  int flags = writable_ ? (create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDWR)
                        : O_RDONLY;
  fd_ = open(filename.c_str(), flags, 0644);
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

int VecBufferPool::init(size_t segment_count) {
  size_t block_num = segment_count + 10;
  page_table_.init(block_num);
  // Allocate all mutexes in a single contiguous array so that the cold-path
  // lock in acquire_buffer() accesses cache-friendly memory instead of
  // chasing 31K+ independent heap pointers.
  block_mutexes_ = std::make_unique<std::mutex[]>(block_num);
  block_mutexes_count_ = block_num;
  // In writable mode, inject a flush callback into the page table so that
  // evict_block() can pwrite dirty blocks back to the backing file.
  if (writable_) {
    page_table_.set_flush_callback(
        [this](block_id_t /*block_id*/, char *buf, size_t sz,
               size_t off) -> int {
#if defined(_MSC_VER)
          ssize_t w = zvec_pwrite(fd_, buf, sz, off);
#else
          ssize_t w = pwrite(fd_, buf, sz, off);
#endif
          if (w != static_cast<ssize_t>(sz)) {
            LOG_ERROR(
                "Buffer pool flush failed: file[%s], offset[%zu], size[%zu]",
                file_name_.c_str(), off, sz);
            return -1;
          }
          return 0;
        });
  }
  LOG_DEBUG("entry num: %zu", page_table_.entry_num());
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

char *VecBufferPool::acquire_buffer(block_id_t block_id, size_t offset,
                                    size_t size, int retry) {
  assert(block_id < block_mutexes_count_);
  char *buffer = page_table_.acquire_block(block_id);
  if (buffer) {
    return buffer;
  }
  std::lock_guard<std::mutex> lock(block_mutexes_[block_id]);
  buffer = page_table_.acquire_block(block_id);
  if (buffer) {
    return buffer;
  }
  {
    bool found =
        MemoryLimitPool::get_instance().try_acquire_buffer(size, buffer);
    if (!found) {
      for (int i = 0; i < retry; i++) {
        BlockEvictionQueue::get_instance().recycle();
        found =
            MemoryLimitPool::get_instance().try_acquire_buffer(size, buffer);
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      LOG_ERROR(
          "Buffer pool failed to get free buffer: file[%s], block_id[%zu], "
          "offset[%zu], size[%zu]",
          file_name_.c_str(), block_id, offset, size);
      return nullptr;
    }
  }

#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, size, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, size, offset);
#endif
  if (read_bytes != static_cast<ssize_t>(size)) {
    LOG_ERROR(
        "Buffer pool failed to read file at offset: file[%s], block_id[%zu], "
        "offset[%zu], size[%zu]",
        file_name_.c_str(), block_id, offset, size);
    MemoryLimitPool::get_instance().release_buffer(buffer, size);
    return nullptr;
  }
  return page_table_.set_block_acquired(block_id, buffer, size, offset);
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

int VecBufferPool::write_block(block_id_t block_id, size_t file_offset,
                               const char *data, size_t size) {
  if (!writable_) {
    LOG_ERROR(
        "write_block called on read-only pool: file[%s], block_id[%zu]",
        file_name_.c_str(), block_id);
    return -1;
  }
  assert(block_id < block_mutexes_count_);
  // Persist to disk first so the data is safe regardless of cache eviction.
#if defined(_MSC_VER)
  ssize_t written = zvec_pwrite(fd_, data, size, file_offset);
#else
  ssize_t written = pwrite(fd_, data, size, file_offset);
#endif
  if (written != static_cast<ssize_t>(size)) {
    LOG_ERROR(
        "write_block failed to write file: file[%s], block_id[%zu], "
        "offset[%zu], size[%zu]",
        file_name_.c_str(), block_id, file_offset, size);
    return -1;
  }
  // Optionally populate the page-table cache so subsequent reads are fast.
  // If memory is unavailable we skip caching silently; the block can always
  // be re-read from disk via acquire_buffer().
  std::lock_guard<std::mutex> lock(block_mutexes_[block_id]);
  if (page_table_.acquire_block(block_id) == nullptr) {
    // Block not yet loaded: try to cache it.
    char *cache_buf = nullptr;
    if (MemoryLimitPool::get_instance().try_acquire_buffer(size, cache_buf)) {
      std::memcpy(cache_buf, data, size);
      // is_dirty is set to false by set_block_acquired because the data is
      // already on disk.
      page_table_.set_block_acquired(block_id, cache_buf, size, file_offset);
    }
  } else {
    // Block is already cached; release the extra ref from acquire_block().
    page_table_.release_block(block_id);
  }
  return 0;
}

int VecBufferPool::write_meta(size_t offset, const char *data, size_t size) {
  if (!writable_) {
    LOG_ERROR("write_meta called on read-only pool: file[%s]",
              file_name_.c_str());
    return -1;
  }
#if defined(_MSC_VER)
  ssize_t written = zvec_pwrite(fd_, data, size, offset);
#else
  ssize_t written = pwrite(fd_, data, size, offset);
#endif
  if (written != static_cast<ssize_t>(size)) {
    LOG_ERROR(
        "write_meta failed: file[%s], offset[%zu], size[%zu]",
        file_name_.c_str(), offset, size);
    return -1;
  }
  return 0;
}

int VecBufferPool::flush_all() {
  int ret = 0;
  for (size_t i = 0; i < page_table_.entry_num(); ++i) {
    if (!page_table_.is_block_dirty(i)) {
      continue;
    }
    char *buf = page_table_.get_buffer_ptr(i);
    if (!buf) {
      continue;
    }
    size_t sz = page_table_.get_block_size(i);
    size_t off = page_table_.get_block_file_offset(i);
#if defined(_MSC_VER)
    ssize_t written = zvec_pwrite(fd_, buf, sz, off);
#else
    ssize_t written = pwrite(fd_, buf, sz, off);
#endif
    if (written != static_cast<ssize_t>(sz)) {
      LOG_ERROR(
          "flush_all failed: file[%s], block_id[%zu], offset[%zu], size[%zu]",
          file_name_.c_str(), i, off, sz);
      ret = -1;
      continue;
    }
    page_table_.clear_dirty(i);
  }
  return ret;
}

char *VecBufferPoolHandle::get_block(size_t offset, size_t size,
                                     size_t block_id) {
  char *buffer = pool_.acquire_buffer(block_id, offset, size, 50);
  return buffer;
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
  return pool_.get_meta(offset, length, buffer);
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

int VecBufferPoolHandle::write_block(const char *data, size_t size,
                                     size_t block_id, size_t file_offset) {
  return pool_.write_block(block_id, file_offset, data, size);
}

int VecBufferPoolHandle::write_meta(size_t offset, const char *data,
                                    size_t size) {
  return pool_.write_meta(offset, data, size);
}

int VecBufferPoolHandle::flush_all() { return pool_.flush_all(); }

}  // namespace ailego
}  // namespace zvec