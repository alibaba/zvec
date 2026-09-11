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
#include "utility/search_heap.h"
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

using TopkHeap = SearchHeap::TopkHeap;

TEST(SearchHeap, ReusesSelectedTopkAndExportsInDistanceOrder) {
  SearchHeap heap;
  auto &topk = heap.reset<TopkHeap>(3);
  topk.emplace(30, 3.0f);
  topk.emplace(10, 1.0f);
  topk.emplace(20, 2.0f);
  const void *storage = topk.container().data();

  std::vector<uint32_t> ids;
  heap.for_each_sorted(2, [&](uint32_t id, float) {
    ids.push_back(id);
    return true;
  });
  EXPECT_EQ((std::vector<uint32_t>{10, 20}), ids);

  auto &reset = heap.reset<TopkHeap>(2);
  EXPECT_EQ(&topk, &reset);
  EXPECT_EQ(storage, reset.container().data());
  EXPECT_TRUE(reset.empty());
  EXPECT_EQ(2U, reset.limit());
}

TEST(SearchHeap, ExportsLinearPoolWithoutMaterializingTopk) {
  SearchHeap heap;
  auto &pool = heap.reset<LinearPool<float>>(3, 4);
  const uint32_t input_ids[] = {30, 10, 20, 40};
  const float distances[] = {3.0f, 1.0f, 2.0f, 4.0f};
  pool.push_block(distances, input_ids, 4);

  std::vector<uint32_t> ids;
  heap.for_each_sorted(3, [&](uint32_t id, float) {
    ids.push_back(id);
    return true;
  });
  EXPECT_EQ((std::vector<uint32_t>{10, 20, 30}), ids);

  heap.dispatch([&](auto &active) {
    EXPECT_TRUE(
        (std::is_same_v<std::decay_t<decltype(active)>, LinearPool<float>>));
    EXPECT_EQ(static_cast<void *>(&pool), static_cast<void *>(&active));
  });
}

TEST(SearchHeap, SelectsPoolBackendFromCpuFeatures) {
  SearchHeap heap;
  heap.reset_pool(4, 8);
  const bool expect_block = ailego::internal::CpuFeatures::static_flags_.AVX2;

  heap.dispatch([&](auto &active) {
    using Heap = std::decay_t<decltype(active)>;
    EXPECT_EQ(expect_block, (std::is_same_v<Heap, BlockHeap>));
    EXPECT_EQ(!expect_block, (std::is_same_v<Heap, LinearPool<float>>));
    EXPECT_EQ(0, active.size());
  });

  heap.reset_pool(2, 8);
  EXPECT_EQ(0U, heap.size());
}

}  // namespace
}  // namespace zvec::core
