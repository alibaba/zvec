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

#include "utility/reservoir_sample_holder.h"
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/core/framework/index_features.h>
#include "../../ailego/buffer/pool_payload_budget.h"

namespace zvec::core {
TEST(ReservoirSampleHolderTest, MatchesLegacyReservoirOrderUnderPressure) {
  auto &pool = ailego::PoolPayloadBudget::get_instance();
  const size_t old_capacity = pool.capacity();
  ASSERT_EQ(pool.used(), 0u);
  ASSERT_EQ(pool.init(ailego::VecBufferPool::metadata_bytes_for_page_count(
                          4096, true) +
                      64 * 1024),
            0);
  auto restore = ailego::ScopeGuard::Make([&]() {
    EXPECT_EQ(pool.used(), 0u);
    EXPECT_EQ(pool.init(old_capacity), 0);
  });
  {
    IndexMeta meta(IndexMeta::DT_FP32, 128);
    auto input =
        std::make_shared<MultiPassIndexHolder<IndexMeta::DT_FP32>>(128);
    for (size_t i = 0; i < 2049; ++i) {
      ailego::NumericalVector<float> row(128, static_cast<float>(i));
      ASSERT_TRUE(input->emplace(i, std::move(row)));
    }
    for (size_t limit : {size_t{1}, size_t{300}, size_t{2049}, size_t{3000}}) {
      // The caller-owned reference is outside the tested pool. Activate the
      // bounded page and reader workspace budget after preparing it.
      ASSERT_EQ(0, pool.init(0));
      SampleIndexFeatures<CompactIndexFeatures> legacy(meta, limit);
      for (auto it = input->create_iterator(); it->is_valid(); it->next())
        legacy.emplace(it->data());
      ASSERT_EQ(0,
                pool.init(pool.used() +
                          ailego::VecBufferPool::metadata_bytes_for_page_count(
                              4096, true) +
                          64 * 1024 + 2 * (meta.element_size() + 8)));
      IndexHolder::Pointer result;
      ASSERT_EQ(CreateReservoirSampleHolder(input, limit, &result), 0);
      ASSERT_EQ(result->count(), legacy.count());
      size_t i = 0;
      for (auto it = result->create_iterator(); it->is_valid();
           it->next(), ++i) {
        ASSERT_EQ(
            std::memcmp(it->data(), legacy.element(i), meta.element_size()), 0);
      }
      EXPECT_EQ(i, result->count());
      auto *ordinal = dynamic_cast<OrdinalAccessHolder *>(result.get());
      ASSERT_NE(nullptr, ordinal);
      OrdinalAccessHolder::Reader::Pointer first, second;
      ASSERT_EQ(0, ordinal->create_ordinal_reader(&first));
      ASSERT_EQ(0, ordinal->create_ordinal_reader(&second));
      // Readers keep the sample alive and own independent bounded scratch.
      result.reset();
      for (size_t id = legacy.count(); id > 0; --id) {
        uint64_t key;
        const void *data = nullptr;
        ASSERT_EQ(0, first->read(id - 1, &key, &data));
        EXPECT_EQ(id - 1, key);
        const void *other = nullptr;
        ASSERT_EQ(0, second->read(0, &key, &other));
        EXPECT_EQ(
            0, std::memcmp(data, legacy.element(id - 1), meta.element_size()));
        EXPECT_EQ(0,
                  std::memcmp(other, legacy.element(0), meta.element_size()));
        second->reset();
      }
      uint64_t key;
      const void *data = nullptr;
      EXPECT_EQ(IndexError_OutOfRange,
                first->read(legacy.count(), &key, &data));
      EXPECT_EQ(nullptr, data);
      EXPECT_EQ(IndexError_InvalidArgument, first->read(0, nullptr, &data));
    }
  }
}
}  // namespace zvec::core
