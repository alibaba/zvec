// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "utility/block_heap.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

using zvec::core::BlockHeap;

TEST(BlockHeap, PopExposesNextUnexpandedCandidate) {
  BlockHeap pool;
  pool.reset(4, 4);

  const float distances[] = {30.0f, 10.0f, 40.0f, 20.0f};
  const uint32_t ids[] = {30, 10, 40, 20};
  pool.push_block(distances, ids, 4);

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);
  EXPECT_EQ(20u, pool.pop_with_next(&next));
  EXPECT_EQ(30u, next);
  EXPECT_EQ(30u, pool.pop_with_next(&next));
  EXPECT_EQ(40u, next);
  EXPECT_EQ(40u, pool.pop_with_next(&next));
  EXPECT_EQ(UINT32_MAX, next);
  EXPECT_FALSE(pool.has_next());
}

TEST(BlockHeap, RewindIsVisibleThroughNextCandidate) {
  BlockHeap pool;
  pool.reset(4, 2);

  const float initial_distances[] = {10.0f, 20.0f};
  const uint32_t initial_ids[] = {10, 20};
  pool.push_block(initial_distances, initial_ids, 2);

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  const float new_distances[] = {5.0f, 30.0f};
  const uint32_t new_ids[] = {5, 30};
  pool.push_block(new_distances, new_ids, 2);

  EXPECT_EQ(5u, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);
}

TEST(BlockHeap, ExistingPopInterfaceIsPreserved) {
  BlockHeap pool;
  pool.reset(2, 2);

  const float distances[] = {2.0f, 1.0f};
  const uint32_t ids[] = {2, 1};
  pool.push_block(distances, ids, 2);

  EXPECT_EQ(1u, pool.pop());
  EXPECT_EQ(2u, pool.pop());
  EXPECT_FALSE(pool.has_next());
}

TEST(BlockHeap, MixedPopInterfacesSkipCheckedEntriesAfterRewind) {
  BlockHeap pool;
  pool.reset(5, 4);
  const float distances[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const uint32_t ids[] = {1, 2, 3, 4};
  pool.push_block(distances, ids, 4);

  EXPECT_EQ(1u, pool.pop());
  uint32_t next = UINT32_MAX;
  EXPECT_EQ(2u, pool.pop_with_next(&next));
  EXPECT_EQ(3u, next);

  const float new_distances[] = {0.5f, 2.5f};
  const uint32_t new_ids[] = {0, 5};
  pool.push_block(new_distances, new_ids, 2);
  ASSERT_EQ(5, pool.size());  // The worst candidate (4) was truncated.
  EXPECT_EQ(0u, pool.pop_with_next(&next));
  EXPECT_EQ(5u, next);  // Skip the already popped entries 1 and 2.
  EXPECT_EQ(5u, pool.pop());
  EXPECT_EQ(3u, pool.pop_with_next(nullptr));
  EXPECT_FALSE(pool.has_next());
}

TEST(BlockHeap, NewPopMatchesLegacyAcrossTruncationRewindAndReset) {
  BlockHeap legacy, rich;
  std::mt19937 rng(701);
  for (int capacity : {1, 2, 7, 8, 16, 33}) {
    SCOPED_TRACE(capacity);
    legacy.reset(capacity, 2);
    rich.reset(capacity, 2);
    std::vector<std::pair<float, uint32_t>> reference;
    std::unordered_set<uint32_t> expanded;
    uint32_t id = 0;
    const auto pop = [&]() {
      const auto expected = std::find_if(
          reference.begin(), reference.end(),
          [&](const auto &item) { return expanded.count(item.second) == 0; });
      ASSERT_NE(reference.end(), expected);
      uint32_t next = 0;
      EXPECT_EQ(expected->second, legacy.pop());
      EXPECT_EQ(expected->second, rich.pop_with_next(&next));
      expanded.insert(expected->second);
      const auto following = std::find_if(
          reference.begin(), reference.end(),
          [&](const auto &item) { return expanded.count(item.second) == 0; });
      EXPECT_EQ(following == reference.end() ? UINT32_MAX : following->second,
                next);
    };
    for (int round = 0; round < 64; ++round) {
      const int count = round % 20;  // Includes empty blocks and > reset hint.
      std::vector<float> distances(count);
      std::vector<uint32_t> ids(count);
      for (int i = 0; i < count; ++i) {
        ids[i] = id++;
        distances[i] = static_cast<float>((rng() % 1024) * 10000 + ids[i]);
        reference.emplace_back(distances[i], ids[i]);
      }
      std::sort(reference.begin(), reference.end());
      if (reference.size() > static_cast<size_t>(capacity)) {
        reference.resize(capacity);
      }
      legacy.push_block(distances.data(), ids.data(), count);
      rich.push_block(distances.data(), ids.data(), count);
      ASSERT_EQ(reference.size(), static_cast<size_t>(rich.size()));
      ASSERT_EQ(legacy.size(), rich.size());
      for (int i = 0; i < rich.size(); ++i) {
        EXPECT_EQ(reference[i].second, rich.id(i));
        EXPECT_EQ(reference[i].first, rich.dist(i));
        EXPECT_EQ(legacy.id(i), rich.id(i));
        EXPECT_EQ(legacy.dist(i), rich.dist(i));
      }
      EXPECT_EQ(legacy.has_next(), rich.has_next());
      if (rich.has_next()) pop();
    }
    while (rich.has_next()) pop();
    EXPECT_FALSE(legacy.has_next());
    rich.reset(1, 1);
    const float score = 1.0f;
    const uint32_t node = 42;
    rich.push_block(&score, &node, 1);
    EXPECT_EQ(node, rich.pop_with_next(nullptr));
    EXPECT_FALSE(rich.has_next());
  }
}
