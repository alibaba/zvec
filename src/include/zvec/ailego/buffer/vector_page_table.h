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

#include <sys/stat.h>
#include <fcntl.h>
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
#include "block_eviction_queue.h"
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

extern const size_t kVectorPageSize;

class VectorPageTable {
  struct Entry {
    std::atomic<int> ref_count;
    std::atomic<bool> in_evict_queue;
    char *buffer;
  };

 public:
  VectorPageTable() : entry_num_(0), entries_(nullptr) {
    BlockEvictionQueue::get_instance().set_valid(this);
  }
  ~VectorPageTable() {
    BlockEvictionQueue::get_instance().set_invalid(this);
    delete[] entries_;
  }

  VectorPageTable(const VectorPageTable &) = delete;
  VectorPageTable &operator=(const VectorPageTable &) = delete;
  VectorPageTable(VectorPageTable &&) = delete;
  VectorPageTable &operator=(VectorPageTable &&) = delete;

  void init(size_t entry_num);

  char *acquire_block(block_id_t block_id);

  void release_block(block_id_t block_id);

  void evict_block(block_id_t block_id);

  char *set_block_acquired(block_id_t block_id, char *buffer);

  size_t entry_num() const {
    return entry_num_;
  }

  bool is_released(block_id_t block_id) const {
    assert(block_id < entry_num_);
    return entries_[block_id].ref_count.load(std::memory_order_relaxed) <= 0;
  }

  inline bool is_dead_block(BlockEvictionQueue::BlockType block) const {
    Entry &entry = entries_[block.vector_block.first];
    return !entry.in_evict_queue.load(std::memory_order_relaxed);
  }

 private:
  size_t entry_num_{0};
  Entry *entries_{nullptr};
};

class VecBufferPoolHandle;

class VecBufferPool {
 public:
  typedef std::shared_ptr<VecBufferPool> Pointer;

  static constexpr size_t kMutexBucketCount = 64UL * 1024UL;

  VecBufferPool(const std::string &filename);
  ~VecBufferPool() {
    for (size_t i = 0; i < page_table_.entry_num(); ++i) {
      assert(page_table_.is_released(i));
      page_table_.evict_block(i);
    }
#if defined(_MSC_VER)
    _close(fd_);
#else
    close(fd_);
#endif
  }

  int init();

  VecBufferPoolHandle get_handle();

  char *acquire_buffer(block_id_t page_id, int retry = 0);

  int get_meta(size_t offset, size_t length, char *buffer);

  size_t file_size() const {
    return file_size_;
  }

 private:
  int fd_;
  size_t file_size_;
  std::string file_name_;

 public:
  VectorPageTable page_table_;

 private:
  std::unique_ptr<std::mutex[]> block_mutexes_{};
};

class VecBufferPoolHandle {
 public:
  VecBufferPoolHandle(VecBufferPool &pool) : pool_(pool) {}
  VecBufferPoolHandle(VecBufferPoolHandle &&other) : pool_(other.pool_) {}

  ~VecBufferPoolHandle() = default;

  typedef std::shared_ptr<VecBufferPoolHandle> Pointer;

  char *get_single_page(size_t file_offset, size_t len, size_t &out_page_id);

  bool read_range(size_t file_offset, size_t len, char *out);

  int get_meta(size_t offset, size_t length, char *buffer);

  void release_one(block_id_t block_id);

  void acquire_one(block_id_t block_id);

 private:
  VecBufferPool &pool_;
};

}  // namespace ailego
}  // namespace zvec