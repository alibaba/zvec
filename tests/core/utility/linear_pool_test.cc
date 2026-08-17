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
#include "utility/linear_pool.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>
#include <gtest/gtest.h>

using zvec::core::LinearPool;

TEST(LinearPool, InsertImmediatelyExposesAndRewindsNextCandidate) {
  LinearPool<float> pool;
  pool.reset(4, 0);

  EXPECT_TRUE(pool.insert(10, 10.0f));
  EXPECT_TRUE(pool.insert(20, 20.0f));
  ASSERT_TRUE(pool.has_next());

  uint32_t next = UINT32_MAX;
  EXPECT_EQ(10, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  bool rewound = false;
  EXPECT_TRUE(pool.insert_with_rewind(5, 5.0f, &rewound));
  EXPECT_TRUE(rewound);
  ASSERT_TRUE(pool.has_next());
  EXPECT_EQ(5, pool.pop_with_next(&next));
  EXPECT_EQ(20u, next);

  rewound = true;
  EXPECT_TRUE(pool.insert_with_rewind(30, 30.0f, &rewound));
  EXPECT_FALSE(rewound);
  EXPECT_EQ(20, pool.pop_with_next(&next));
  EXPECT_EQ(30u, next);
  EXPECT_EQ(30, pool.pop_with_next(&next));
  EXPECT_EQ(UINT32_MAX, next);
}

TEST(LinearPool, RejectedCandidateDoesNotRewind) {
  LinearPool<float> pool;
  pool.reset(2, 0);
  EXPECT_TRUE(pool.insert(1, 1.0f));
  EXPECT_TRUE(pool.insert(2, 2.0f));

  bool rewound = true;
  EXPECT_FALSE(pool.insert_with_rewind(3, 3.0f, &rewound));
  EXPECT_FALSE(rewound);
  EXPECT_EQ(2, pool.size());
  EXPECT_EQ(1, pool.pop());
}

TEST(LinearPool, NewInterfacesMatchLegacyAcrossTruncationRewindAndReset) {
  LinearPool<float> legacy, rich;
  std::mt19937 rng(701);
  for (int capacity : {1, 2, 7, 8, 16, 33}) {
    SCOPED_TRACE(capacity);
    legacy.reset(capacity, 0);
    rich.reset(capacity, 0);
    std::vector<std::pair<float, uint32_t>> reference;
    std::unordered_set<uint32_t> expanded;
    for (uint32_t id = 0; id < 256; ++id) {
      const float distance = static_cast<float>((rng() % 1024) * 10000 + id);
      const int previous_cursor = legacy.cur_;
      bool rewound = true;
      const bool inserted = legacy.insert(id, distance);
      EXPECT_EQ(inserted, rich.insert_with_rewind(id, distance, &rewound));
      EXPECT_EQ(inserted && legacy.cur_ < previous_cursor, rewound);
      reference.emplace_back(distance, id);
      std::sort(reference.begin(), reference.end());
      if (reference.size() > static_cast<size_t>(capacity)) {
        reference.resize(capacity);
      }
      ASSERT_EQ(reference.size(), static_cast<size_t>(rich.size()));
      for (int i = 0; i < rich.size(); ++i) {
        EXPECT_EQ(reference[i].second, static_cast<uint32_t>(rich.id(i)));
        EXPECT_EQ(reference[i].first, rich.dist(i));
        EXPECT_EQ(legacy.id(i), rich.id(i));
        EXPECT_EQ(legacy.dist(i), rich.dist(i));
      }
      EXPECT_EQ(legacy.has_next(), rich.has_next());
      if (id % 3 == 0 && rich.has_next()) {
        const auto expected = std::find_if(
            reference.begin(), reference.end(),
            [&](const auto &item) { return expanded.count(item.second) == 0; });
        ASSERT_NE(reference.end(), expected);
        uint32_t next = 0;
        EXPECT_EQ(expected->second, static_cast<uint32_t>(legacy.pop()));
        EXPECT_EQ(expected->second,
                  static_cast<uint32_t>(rich.pop_with_next(&next)));
        expanded.insert(expected->second);
        const auto following = std::find_if(
            reference.begin(), reference.end(),
            [&](const auto &item) { return expanded.count(item.second) == 0; });
        EXPECT_EQ(following == reference.end() ? UINT32_MAX : following->second,
                  next);
      }
    }
    while (legacy.has_next()) {
      ASSERT_TRUE(rich.has_next());
      EXPECT_EQ(legacy.pop(), rich.pop_with_next(nullptr));
    }
    EXPECT_FALSE(rich.has_next());
    rich.reset(1, 0);
    EXPECT_TRUE(rich.insert_with_rewind(42, 1.0f, nullptr));
    EXPECT_EQ(42, rich.pop_with_next(nullptr));
    EXPECT_FALSE(rich.has_next());
  }
}
