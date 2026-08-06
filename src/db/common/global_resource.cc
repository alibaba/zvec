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
#include "db/common/global_resource.h"
#include <exception>
#include <mutex>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/db/config.h>

namespace zvec {

int GlobalResource::initialize() {
  const auto &config = GlobalConfig::Instance();
  auto &memory_pool = zvec::ailego::MemoryLimitPool::get_instance();
  // Standalone/core users may configure the process-wide pool before the DB
  // layer lazily creates its thread pools. In that case the first published
  // pool budget remains authoritative; a lazy accessor must not try to resize
  // a live cache merely because GlobalConfig still contains its default.
  const uint64_t effective_memory_limit = memory_pool.initialized()
                                              ? memory_pool.capacity()
                                              : config.memory_limit_bytes();
  return initialize(effective_memory_limit, config.query_thread_count(),
                    config.query_thread_binding(),
                    config.optimize_thread_count(),
                    config.optimize_thread_binding());
}

int GlobalResource::initialize(uint64_t memory_limit_bytes,
                               uint32_t query_thread_count,
                               bool query_thread_binding,
                               uint32_t optimize_thread_count,
                               bool optimize_thread_binding) {
  return initialize_with_setup(memory_limit_bytes, query_thread_count,
                               query_thread_binding, optimize_thread_count,
                               optimize_thread_binding, {});
}

int GlobalResource::initialize_with_setup(uint64_t memory_limit_bytes,
                                          uint32_t query_thread_count,
                                          bool query_thread_binding,
                                          uint32_t optimize_thread_count,
                                          bool optimize_thread_binding,
                                          const std::function<int()> &setup) {
  std::lock_guard<std::mutex> lock(initialization_mutex_);
  try {
    auto &memory_pool = zvec::ailego::MemoryLimitPool::get_instance();
    if (query_thread_pool_ && optimize_thread_pool_) {
      if (memory_limit_bytes_ == memory_limit_bytes &&
          query_thread_count_ == query_thread_count &&
          optimize_thread_count_ == optimize_thread_count &&
          query_thread_binding_ == query_thread_binding &&
          optimize_thread_binding_ == optimize_thread_binding) {
        return setup ? setup() : 0;
      }
      LOG_ERROR(
          "GlobalResource::initialize rejected configuration change after "
          "thread pools were created");
      return -1;
    }

    // An explicit GlobalConfig initialization must never publish a memory
    // limit different from an already configured lower-level pool. Reject it
    // before logger setup or thread-pool publication; lazy initialize() above
    // deliberately passes the existing capacity and therefore remains
    // compatible with standalone/core callers.
    if (setup && memory_pool.initialized() &&
        memory_pool.capacity() != memory_limit_bytes) {
      LOG_ERROR(
          "GlobalResource::initialize rejected memory limit change after "
          "MemoryLimitPool was configured: requested_capacity=%llu "
          "current_capacity=%llu",
          static_cast<unsigned long long>(memory_limit_bytes),
          static_cast<unsigned long long>(memory_pool.capacity()));
      return -1;
    }

    auto query_thread_pool = std::make_unique<ailego::ThreadPool>(
        query_thread_count, query_thread_binding);
    auto optimize_thread_pool = std::make_unique<ailego::ThreadPool>(
        optimize_thread_count, optimize_thread_binding);
    // Run side-effecting setup only after every fallible resource allocation
    // and compatibility check that can be performed without mutating global
    // state. Holding initialization_mutex_ closes the race with lazy callers
    // attempting to initialize a different resource configuration.
    if (setup && setup() != 0) {
      return -1;
    }
    if (memory_pool.init(memory_limit_bytes) != 0) {
      return -1;
    }

    memory_limit_bytes_ = memory_limit_bytes;
    query_thread_count_ = query_thread_count;
    optimize_thread_count_ = optimize_thread_count;
    query_thread_binding_ = query_thread_binding;
    optimize_thread_binding_ = optimize_thread_binding;
    this->query_thread_pool_ = std::move(query_thread_pool);
    this->optimize_thread_pool_ = std::move(optimize_thread_pool);
    return 0;
  } catch (const std::exception &e) {
    LOG_ERROR("GlobalResource::initialize failed: %s", e.what());
  } catch (...) {
    LOG_ERROR("GlobalResource::initialize failed with an unknown exception");
  }
  return -1;
}

}  // namespace zvec
