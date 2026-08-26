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
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>

extern "C" {
void *zvec_test_dso_memory_limit_pool();
void *zvec_test_dso_block_eviction_queue();
bool zvec_test_dso_charge_external(size_t bytes);
void zvec_test_dso_release_external(size_t bytes);
}

namespace zvec {
namespace ailego {

TEST(CrossDsoBufferPoolTest, SharesAddressesAndAccountingState) {
  auto &pool = MemoryLimitPool::get_instance();
  auto &eviction_queue = BlockEvictionQueue::get_instance();

  EXPECT_EQ(&pool, zvec_test_dso_memory_limit_pool());
  EXPECT_EQ(&eviction_queue, zvec_test_dso_block_eviction_queue());

  const size_t bytes = 4 * kVectorPageSize;
  ASSERT_EQ(0, pool.init(bytes));
  ASSERT_TRUE(zvec_test_dso_charge_external(kVectorPageSize));
  EXPECT_EQ(kVectorPageSize, pool.external_used());
  zvec_test_dso_release_external(kVectorPageSize);
  EXPECT_EQ(0u, pool.external_used());
}

}  // namespace ailego
}  // namespace zvec
