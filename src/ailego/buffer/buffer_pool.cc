#include <zvec/ailego/buffer/buffer_pool.h>
#include <zvec/core/framework/index_logger.h>

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
#endif

namespace zvec {
namespace ailego {

int LRUCache::init(size_t block_size) {
  block_size_ = block_size;
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    queues_.push_back(ConcurrentQueue(block_size));
  }
  return 0;
}

bool LRUCache::evict_single_block(BlockType &item) {
  bool found = false;
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    found = queues_[i].try_dequeue(item);
    if (found) {
      break;
    }
  }
  return found;
}

bool LRUCache::add_single_block(const LPMap *lp_map, const BlockType &block,
                                int block_type) {
  bool ok = queues_[block_type].enqueue(block);
  if (!ok) {
    LOG_ERROR("enqueue failed.");
    return false;
  }
  evict_queue_insertions_.fetch_add(1, std::memory_order_relaxed);
  if (evict_queue_insertions_ % block_size_ == 0) {
    this->clear_dead_node(lp_map);
  }
  return true;
}

void LRUCache::clear_dead_node(const LPMap *lp_map) {
  for (size_t i = 0; i < CATCH_QUEUE_NUM; i++) {
    size_t clear_size = block_size_ * 2;
    if (queues_[i].size_approx() < clear_size * 4) {
      continue;
    }
    size_t clear_count = 0;
    ConcurrentQueue tmp(block_size_);
    BlockType item;
    while (queues_[i].try_dequeue(item) && (clear_count++ < clear_size)) {
      if (!lp_map->isDeadBlock(item)) {
        if (!tmp.enqueue(item)) {
          LOG_ERROR("enqueue failed.");
        }
      }
    }
    while (tmp.try_dequeue(item)) {
      if (!lp_map->isDeadBlock(item)) {
        if (!queues_[i].enqueue(item)) {
          LOG_ERROR("enqueue failed.");
        }
      }
    }
  }
}

void LPMap::init(size_t entry_num) {
  if (entries_) {
    delete[] entries_;
  }
  entry_num_ = entry_num;
  entries_ = new Entry[entry_num_];
  for (size_t i = 0; i < entry_num_; i++) {
    entries_[i].ref_count.store(std::numeric_limits<int>::min());
    entries_[i].load_count.store(0);
    entries_[i].buffer = nullptr;
  }
  cache_.init(entry_num * 4);
}

char *LPMap::acquire_block(block_id_t block_id, bool lru_mode) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  if (!lru_mode) {
    return entry.buffer;
  }
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_acquire);
    if (current_count < 0) {
      return nullptr;
    }
    if (entry.ref_count.compare_exchange_weak(current_count, current_count + 1,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      if (current_count == 0) {
        entry.load_count.fetch_add(1, std::memory_order_relaxed);
      }
      return entry.buffer;
    }
  }
}

void LPMap::release_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];

  if (entry.ref_count.fetch_sub(1, std::memory_order_release) == 1) {
    std::atomic_thread_fence(std::memory_order_acquire);
    LRUCache::BlockType block;
    block.first = block_id;
    block.second = entry.load_count.load();
    cache_.add_single_block(this, block, 0);
  }
}

char *LPMap::evict_block(block_id_t block_id) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  int expected = 0;
  if (entry.ref_count.compare_exchange_strong(
          expected, std::numeric_limits<int>::min())) {
    char *buffer = entry.buffer;
    entry.buffer = nullptr;
    return buffer;
  } else {
    return nullptr;
  }
}

char *LPMap::set_block_acquired(block_id_t block_id, char *buffer) {
  assert(block_id < entry_num_);
  Entry &entry = entries_[block_id];
  while (true) {
    int current_count = entry.ref_count.load(std::memory_order_relaxed);
    if (current_count >= 0) {
      if (entry.ref_count.compare_exchange_weak(
              current_count, current_count + 1, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return entry.buffer;
      }
    } else {
      if (entry.ref_count.compare_exchange_weak(current_count, 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        entry.buffer = buffer;
        entry.load_count.fetch_add(1, std::memory_order_relaxed);
        return entry.buffer;
      }
    }
  }
}

void LPMap::recycle(moodycamel::ConcurrentQueue<char *> &free_buffers) {
  LRUCache::BlockType block;
  do {
    bool ok = cache_.evict_single_block(block);
    if (!ok) {
      return;
    }
  } while (isDeadBlock(block));
  char *buffer = evict_block(block.first);
  if (buffer) {
    if (!free_buffers.enqueue(buffer)) {
      LOG_ERROR("recycle buffer enqueue failed.");
      ailego_free(buffer);
    }
  }
}

VecBufferPool::VecBufferPool(const std::string &filename) {
#if defined(_MSC_VER)
  fd_ = _open(filename.c_str(), O_RDONLY | _O_BINARY);
#else
  fd_ = open(filename.c_str(), O_RDONLY);
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

int VecBufferPool::init(size_t pool_capacity, size_t block_size,
                        size_t segment_count) {
  if (block_size == 0) {
    LOG_ERROR("block_size must not be 0");
    return -1;
  }
  pool_capacity_ = pool_capacity;
  size_t buffer_num = pool_capacity_ / block_size + 10;
  size_t block_num = segment_count + 10;
  lp_map_.init(block_num);
  mutex_vec_.reserve(block_num);
  for (int i = 0; i < block_num; i++) {
    mutex_vec_.emplace_back(std::make_unique<std::mutex>());
  }
  for (size_t i = 0; i < buffer_num; i++) {
    char *buffer = (char *)ailego_malloc(block_size);
    if (buffer != nullptr) {
      if (!free_buffers_.enqueue(buffer)) {
        LOG_ERROR("recycle buffer enqueue failed.");
        ailego_free(buffer);
        return -1;
      }
    } else {
      LOG_ERROR("aligned_alloc %zu(size: %zu) failed", i, block_size);
      return -1;
    }
  }
  LOG_DEBUG("Buffer pool num: %zu, entry num: %zu", buffer_num,
            lp_map_.entry_num());
  no_lru_mode_ = false;
  if (lp_map_.entry_num() <= buffer_num) {
    no_lru_mode_ = true;
  }
  return 0;
}

VecBufferPoolHandle VecBufferPool::get_handle() {
  return VecBufferPoolHandle(*this);
}

char *VecBufferPool::acquire_buffer(block_id_t block_id, size_t offset,
                                    size_t size, int retry) {
  char *buffer = lp_map_.acquire_block(block_id, !no_lru_mode());
  if (buffer) {
    return buffer;
  }
  std::lock_guard<std::mutex> lock(*mutex_vec_[block_id]);
  buffer = lp_map_.acquire_block(block_id, !no_lru_mode());
  if (buffer) {
    return buffer;
  }
  {
    bool found = free_buffers_.try_dequeue(buffer);
    if (!found && !no_lru_mode_) {
      for (int i = 0; i < retry; i++) {
        lp_map_.recycle(free_buffers_);
        found = free_buffers_.try_dequeue(buffer);
        if (found) {
          break;
        }
      }
    }
    if (!found) {
      LOG_ERROR("Buffer pool failed to get free buffer");
      return nullptr;
    }
  }

#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, size, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, size, offset);
#endif
  if (read_bytes != static_cast<ssize_t>(size)) {
    LOG_ERROR("Buffer pool failed to read file at offset: %zu", offset);
    free_buffers_.enqueue(buffer);
    return nullptr;
  }
  return lp_map_.set_block_acquired(block_id, buffer);
}

int VecBufferPool::get_meta(size_t offset, size_t length, char *buffer) {
#if defined(_MSC_VER)
  ssize_t read_bytes = zvec_pread(fd_, buffer, length, offset);
#else
  ssize_t read_bytes = pread(fd_, buffer, length, offset);
#endif
  if (read_bytes != static_cast<ssize_t>(length)) {
    LOG_ERROR("Buffer pool failed to read file at offset: %zu", offset);
    return -1;
  }
  return 0;
}

char *VecBufferPoolHandle::get_block(size_t offset, size_t size,
                                     size_t block_id) {
  char *buffer = pool_.acquire_buffer(block_id, offset, size, 5);
  return buffer;
}

int VecBufferPoolHandle::get_meta(size_t offset, size_t length, char *buffer) {
  return pool_.get_meta(offset, length, buffer);
}

void VecBufferPoolHandle::release_one(block_id_t block_id) {
  if (!pool_.no_lru_mode()) {
    pool_.lp_map_.release_block(block_id);
  }
}

void VecBufferPoolHandle::acquire_one(block_id_t block_id) {
  if (!pool_.no_lru_mode()) {
    pool_.lp_map_.acquire_block(block_id, true);
  }
}

}  // namespace ailego
}  // namespace zvec