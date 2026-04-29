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
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_logger.h>
#include "concurrentqueue.h"

#if defined(_MSC_VER)
#include <io.h>
#endif

namespace zvec {
namespace ailego {

class VectorPageTable;

using block_id_t = size_t;
using version_t = size_t;

struct ParquetBufferID {
  std::string filename;
  int column;
  int row_group;
  uint64_t file_id;
  long mtime;
  ParquetBufferID() = default;
  ParquetBufferID(std::string &filename, int column, int row_group);
  const std::string to_string() const;
};

class BlockEvictionQueue {
 public:
  struct BlockType {
    std::pair<block_id_t, version_t> vector_block;
    std::pair<ParquetBufferID, version_t> parquet_buffer_block;
    VectorPageTable *page_table{nullptr};
  };
  typedef moodycamel::ConcurrentQueue<BlockType> ConcurrentQueue;

  static BlockEvictionQueue &get_instance() {
    static BlockEvictionQueue instance;
    return instance;
  }
  BlockEvictionQueue(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue &operator=(const BlockEvictionQueue &) = delete;
  BlockEvictionQueue(BlockEvictionQueue &&) = delete;
  BlockEvictionQueue &operator=(BlockEvictionQueue &&) = delete;

  int init();

  bool evict_single_block(BlockType &item);

  bool evict_block(BlockType &item);

  bool add_single_block(const BlockType &block, int queue_index);

  // void clear_dead_node();

  bool is_valid(VectorPageTable *page_table) {
    std::shared_lock<std::shared_mutex> lock(valid_page_tables_mutex_);
    return valid_page_tables_.find(page_table) != valid_page_tables_.end();
  }

  void set_valid(VectorPageTable *page_table) {
    std::unique_lock<std::shared_mutex> lock(valid_page_tables_mutex_);
    valid_page_tables_.insert(page_table);
  }

  void set_invalid(VectorPageTable *page_table) {
    std::unique_lock<std::shared_mutex> lock(valid_page_tables_mutex_);
    valid_page_tables_.erase(page_table);
  }

  // Atomically checks under the shared lock that the page table is still valid
  // AND the block version has not been superseded, preventing TOCTOU races
  // when a VectorPageTable is concurrently destroyed.
  bool is_valid_and_alive(const BlockType &item);

  void recycle();

 private:
  BlockEvictionQueue() {
    init();
  }

 private:
  constexpr static size_t CACHE_QUEUE_NUM = 3;
  size_t evict_batch_size_{0};
  std::vector<ConcurrentQueue> evict_queues_;
  std::unordered_set<VectorPageTable *> valid_page_tables_;
  std::shared_mutex valid_page_tables_mutex_;
};

class MemoryLimitPool {
 public:
  static MemoryLimitPool &get_instance() {
    static MemoryLimitPool instance;
    return instance;
  }
  MemoryLimitPool(const MemoryLimitPool &) = delete;
  MemoryLimitPool &operator=(const MemoryLimitPool &) = delete;
  MemoryLimitPool(MemoryLimitPool &&) = delete;
  MemoryLimitPool &operator=(MemoryLimitPool &&) = delete;

  int init(size_t pool_size);

  bool try_acquire_buffer(const size_t buffer_size, char *&buffer);

  void acquire_parquet(const size_t buffer_size);

  void release_buffer(char *buffer, const size_t buffer_size);

  void release_parquet(const size_t buffer_size);

  bool is_full();

 private:
  MemoryLimitPool() = default;

 private:
  size_t pool_size_{0};
  std::atomic<size_t> used_size_{0};
};

}  // namespace ailego
}  // namespace zvec