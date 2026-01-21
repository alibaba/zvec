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
#include <mutex>
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/db/config.h>

namespace zvec {

void GlobalResource::initialize() {
  static std::once_flag flag;
  std::call_once(flag, [this]() mutable {
    this->query_thread_pool_.reset(
        new ailego::ThreadPool(GlobalConfig::Instance().query_thread_count()));
    this->optimize_thread_pool_.reset(new ailego::ThreadPool(
        GlobalConfig::Instance().optimize_thread_count()));
    ailego::BufferManager::Instance().init(
        GlobalConfig::Instance().memory_limit_bytes(), 1);
  });
}

}  // namespace zvec