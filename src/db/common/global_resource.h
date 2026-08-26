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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <zvec/ailego/parallel/thread_pool.h>
#include <zvec/ailego/pattern/singleton.h>

namespace zvec {

class GlobalConfig;

namespace detail {
constexpr uint32_t kRocksDbMemoryPercent = 15;
}

}  // namespace zvec

namespace rocksdb {
class Cache;
class WriteBufferManager;
}  // namespace rocksdb

namespace zvec {

class GlobalResource : public ailego::Singleton<GlobalResource> {
 public:
  struct MemoryStats {
    uint64_t total_capacity{0};
    uint64_t buffer_pool_capacity{0};
    uint64_t buffer_pool_used{0};
    uint64_t rocksdb_capacity{0};
    uint64_t rocksdb_cache_used{0};
    uint64_t rocksdb_memtable_used{0};
    uint64_t rocksdb_mutable_memtable_used{0};
  };

  static uint64_t calculate_rocksdb_memory_budget(
      uint64_t total_bytes) noexcept;
  static uint64_t calculate_buffer_pool_memory_budget(
      uint64_t total_bytes) noexcept;

  int initialize();

  MemoryStats memory_stats();

  ailego::ThreadPool *query_thread_pool() {
    if (initialize() != 0) {
      return nullptr;
    }
    return query_thread_pool_.get();
  }

  ailego::ThreadPool *optimize_thread_pool() {
    if (initialize() != 0) {
      return nullptr;
    }
    return optimize_thread_pool_.get();
  }

  std::shared_ptr<rocksdb::Cache> rocksdb_block_cache() {
    if (initialize() != 0) {
      return nullptr;
    }
    return rocksdb_block_cache_;
  }

  std::shared_ptr<rocksdb::WriteBufferManager> rocksdb_write_buffer_manager() {
    if (initialize() != 0) {
      return nullptr;
    }
    return rocksdb_write_buffer_manager_;
  }

  uint64_t rocksdb_memory_capacity() {
    if (initialize() != 0) {
      return 0;
    }
    return rocksdb_memory_capacity_;
  }

 private:
  friend class GlobalConfig;

  int initialize(uint64_t memory_limit_bytes, uint32_t query_thread_count,
                 bool query_thread_binding, uint32_t optimize_thread_count,
                 bool optimize_thread_binding);
  int initialize_with_setup(uint64_t memory_limit_bytes,
                            uint32_t query_thread_count,
                            bool query_thread_binding,
                            uint32_t optimize_thread_count,
                            bool optimize_thread_binding,
                            const std::function<int()> &setup,
                            bool preserve_existing_pool = false);

  std::mutex initialization_mutex_;
  uint64_t memory_limit_bytes_{0};
  uint32_t query_thread_count_{0};
  uint32_t optimize_thread_count_{0};
  bool query_thread_binding_{false};
  bool optimize_thread_binding_{false};
  uint64_t buffer_pool_capacity_{0};
  uint64_t rocksdb_memory_capacity_{0};
  std::unique_ptr<ailego::ThreadPool> query_thread_pool_;
  std::unique_ptr<ailego::ThreadPool> optimize_thread_pool_;
  std::shared_ptr<rocksdb::Cache> rocksdb_block_cache_;
  std::shared_ptr<rocksdb::WriteBufferManager> rocksdb_write_buffer_manager_;
};

}  // namespace zvec
