#pragma once

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <zvec/ailego/internal/platform.h>
#include "concurrentqueue.h"

namespace zvec {
namespace ailego {

using block_id_t = size_t;
using version_t = size_t;

class LPMap;

class LRUCache {
 public:
  typedef std::pair<block_id_t, version_t> BlockType;
  typedef moodycamel::ConcurrentQueue<BlockType> ConcurrentQueue;

  int init(size_t block_size);

  bool evict_single_block(BlockType &item);

  bool add_single_block(const LPMap *lp_map, const BlockType &block,
                        int block_type);

  void clear_dead_node(const LPMap *lp_map);

 private:
  constexpr static size_t CATCH_QUEUE_NUM = 3;
  size_t block_size_{0};
  std::vector<ConcurrentQueue> queues_;
  alignas(64) std::atomic<size_t> evict_queue_insertions_{0};
};

class LPMap {
  struct Entry {
    alignas(64) std::atomic<int> ref_count;
    alignas(64) std::atomic<version_t> load_count;
    char *buffer;
  };

 public:
  LPMap() : entry_num_(0), entries_(nullptr) {}
  ~LPMap() {
    delete[] entries_;
  }

  void init(size_t entry_num);

  char *acquire_block(block_id_t block_id, bool lru_mode);

  void release_block(block_id_t block_id);

  char *evict_block(block_id_t block_id);

  char *set_block_acquired(block_id_t block_id, char *buffer);

  void recycle(moodycamel::ConcurrentQueue<char *> &free_buffers);

  size_t entry_num() const {
    return entry_num_;
  }

  inline bool isDeadBlock(LRUCache::BlockType block) const {
    Entry &entry = entries_[block.first];
    return block.second != entry.load_count.load();
  }

 private:
  size_t entry_num_{0};
  Entry *entries_{nullptr};
  LRUCache cache_;
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  VecBufferPool(const std::string &filename);
  ~VecBufferPool() {
    // Free all buffers in the free list
    char *buf = nullptr;
    while (free_buffers_.try_dequeue(buf)) {
      ailego_free(buf);
    }
    // Free any buffers still pinned in the map
    for (size_t i = 0; i < lp_map_.entry_num(); ++i) {
      char *b = lp_map_.evict_block(i);
      if (b) ailego_free(b);
    }
    close(fd_);
  }

  int init(size_t pool_capacity, size_t block_size, size_t segment_count);

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t block_id, size_t offset, size_t size,
                       int retry = 0);

  int get_meta(size_t offset, size_t length, char *buffer);

  size_t file_size() const {
    return file_size_;
  }

  bool no_lru_mode() {
    return no_lru_mode_;
  }

 private:
  int fd_;
  size_t file_size_;
  size_t pool_capacity_;
  bool no_lru_mode_;

 public:
  LPMap lp_map_;

 private:
  std::vector<std::unique_ptr<std::mutex>> mutex_vec_;
  moodycamel::ConcurrentQueue<char *> free_buffers_;
};

class VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_block(size_t offset, size_t size, size_t block_id);

  int get_meta(size_t offset, size_t length, char *buffer);

  void release_one(block_id_t block_id);

  void acquire_one(block_id_t block_id);

 private:
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec