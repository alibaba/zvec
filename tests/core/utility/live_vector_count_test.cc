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

#include "utility/live_vector_count.h"
#include <array>
#include <gtest/gtest.h>

namespace zvec::core {
namespace {

TEST(LiveVectorCountTest, RescansOnlyWhenKeysOrSlotBoundsChange) {
  struct Entity {
    uint64_t key_revision() const {
      return revision;
    }
    size_t slot_count() const {
      return slots;
    }
    bool contains_node(uint32_t id) const {
      ++reads;
      return values[id];
    }
    std::array<bool, 4> values{true, false, false, true};
    uint64_t revision{0};
    size_t slots{3};
    mutable size_t reads{0};
  } entity;
  LiveVectorCount count;
  EXPECT_EQ(1U, count.get(entity));
  EXPECT_EQ(1U, count.get(entity));
  EXPECT_EQ(3U, entity.reads);

  entity.values[1] = true;
  ++entity.revision;
  EXPECT_EQ(2U, count.get(entity));
  EXPECT_EQ(2U, count.get(entity));
  EXPECT_EQ(6U, entity.reads);

  entity.slots = 4;
  EXPECT_EQ(3U, count.get(entity));
  EXPECT_EQ(3U, count.get(entity));
  EXPECT_EQ(10U, entity.reads);
}

}  // namespace
}  // namespace zvec::core
