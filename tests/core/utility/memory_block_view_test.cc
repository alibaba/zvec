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

#include <cstring>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_storage.h>

using namespace zvec;
using namespace zvec::core;

// MakeBorrowedView backs the cross-page scratch-arena path in BufferStorage:
// cross-page vectors are copied into a reused thread-local arena and exposed
// as non-owning views. The block must therefore free nothing and pin nothing
// on destruction, and copies/moves must keep aliasing the same buffer without
// taking ownership. A double-free here would trap under ASan.
TEST(MemoryBlockBorrowedView, IsNonOwning) {
  auto *buf = new char[64];
  std::memset(buf, 0xAB, 64);
  {
    auto view = IndexStorage::MemoryBlock::MakeBorrowedView(buf);
    EXPECT_EQ(static_cast<const void *>(buf), view.data());

    // Copy keeps a non-owning alias to the same buffer.
    auto copy = view;
    EXPECT_EQ(static_cast<const void *>(buf), copy.data());

    // Move keeps the alias too.
    auto moved = std::move(copy);
    EXPECT_EQ(static_cast<const void *>(buf), moved.data());
  }  // every view destroyed here; the backing buffer must NOT be freed.

  // Still readable, and safe to free exactly once by the sole owner.
  EXPECT_EQ(static_cast<unsigned char>(buf[0]), 0xABu);
  delete[] buf;
}

// A borrowed view over a slice of a shared buffer (as the arena hands out)
// must alias the exact slice and never free the shared backing storage.
TEST(MemoryBlockBorrowedView, AliasesArenaSlice) {
  std::vector<char> arena(256, 0);
  arena[128] = 0x5A;
  auto view = IndexStorage::MemoryBlock::MakeBorrowedView(arena.data() + 128);
  ASSERT_EQ(static_cast<const void *>(arena.data() + 128), view.data());
  EXPECT_EQ(0x5A, *static_cast<const char *>(view.data()));
  // Destroying the view leaves the arena intact for reuse.
  view.reset();
  EXPECT_EQ(0x5A, arena[128]);
}
