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

// Observe addresses rather than timing a non-faulting prefetch instruction.
// This replacement is local to this translation unit and includes the inline
// helpers unchanged. MSVC uses a different intrinsic backend.
#if !defined(_MSC_VER)
static void RecordPrefetch(const void *address);
#define __builtin_prefetch(address, ...) RecordPrefetch(address)
#endif
#include <zvec/ailego/internal/platform.h>
#if !defined(_MSC_VER)
#undef __builtin_prefetch
#endif

#include <limits>
#include <vector>
#include <gtest/gtest.h>

TEST(Platform, CacheLineCountRoundsUpWithoutOverflow) {
  constexpr size_t line = AILEGO_CACHE_LINE_SIZE;
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  constexpr size_t full_lines = maximum / line;
  constexpr size_t aligned_maximum = full_lines * line;
  const struct {
    size_t bytes;
    size_t lines;
  } cases[] = {
      {0, 0},
      {1, 1},
      {line - 1, 1},
      {line, 1},
      {line + 1, 2},
      {3 * line - 1, 3},
      {3 * line, 3},
      {3 * line + 1, 4},
      {aligned_maximum - 1, full_lines},
      {aligned_maximum, full_lines},
      {aligned_maximum + 1, full_lines + 1},
      {maximum - 1, full_lines + 1},
      {maximum, full_lines + 1},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.bytes);
    EXPECT_EQ(test.lines, ailego_cache_line_count(test.bytes));
  }
}

TEST(Platform, CacheLineAlignment) {
  struct alignas(AILEGO_CACHE_LINE_SIZE) Counter {
    uint64_t value;
  };
  Counter counters[2]{};
  EXPECT_EQ(AILEGO_CACHE_LINE_SIZE, sizeof(Counter));
  EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(counters) % AILEGO_CACHE_LINE_SIZE);
  EXPECT_EQ(AILEGO_CACHE_LINE_SIZE,
            reinterpret_cast<uintptr_t>(&counters[1]) -
                reinterpret_cast<uintptr_t>(&counters[0]));
}

#if !defined(_MSC_VER)
static std::vector<const void *> prefetched;

static void RecordPrefetch(const void *address) {
  prefetched.push_back(address);
}

TEST(Platform, PrefetchLinesUseTargetStride) {
  alignas(AILEGO_CACHE_LINE_SIZE) char data[3 * AILEGO_CACHE_LINE_SIZE + 1]{};
  prefetched.clear();
  ailego_prefetch_lines(nullptr, 0);
  EXPECT_TRUE(prefetched.empty());
  ailego_prefetch_lines(data + 1, 3);
  EXPECT_EQ(
      (std::vector<const void *>{data + 1, data + 1 + AILEGO_CACHE_LINE_SIZE,
                                 data + 1 + 2 * AILEGO_CACHE_LINE_SIZE}),
      prefetched);
}

#endif
