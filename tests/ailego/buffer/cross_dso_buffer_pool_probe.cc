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

#include <cstddef>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/export.h>

extern "C" {

ZVEC_HELPER_DLL_EXPORT void *zvec_test_dso_memory_limit_pool() {
  return &zvec::ailego::MemoryLimitPool::get_instance();
}

ZVEC_HELPER_DLL_EXPORT void *zvec_test_dso_block_eviction_queue() {
  return &zvec::ailego::BlockEvictionQueue::get_instance();
}

ZVEC_HELPER_DLL_EXPORT bool zvec_test_dso_charge_external(size_t bytes) {
  return zvec::ailego::MemoryLimitPool::get_instance().try_charge_external(
      bytes);
}

ZVEC_HELPER_DLL_EXPORT void zvec_test_dso_release_external(size_t bytes) {
  zvec::ailego::MemoryLimitPool::get_instance().release_external(bytes);
}

}  // extern "C"
