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

class GlobalResource : public ailego::Singleton<GlobalResource> {
 public:
  int initialize();

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
                            const std::function<int()> &setup);

  std::mutex initialization_mutex_;
  uint64_t memory_limit_bytes_{0};
  uint32_t query_thread_count_{0};
  uint32_t optimize_thread_count_{0};
  bool query_thread_binding_{false};
  bool optimize_thread_binding_{false};
  std::unique_ptr<ailego::ThreadPool> query_thread_pool_;
  std::unique_ptr<ailego::ThreadPool> optimize_thread_pool_;
};

}  // namespace zvec
