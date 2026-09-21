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

// MakeBorrowedView is only for explicitly caller-managed buffers, not batch
// results whose backing storage must survive later reads. The block must free
// nothing and pin nothing on destruction, and copies/moves must keep aliasing
// the same buffer without taking ownership. A double-free here would trap
// under ASan.
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

TEST(MemoryBlockSharedView, CopiesAndMovesKeepArenaAlive) {
  auto arena = std::make_shared<std::vector<char>>(256, 0x5A);
  std::weak_ptr<std::vector<char>> lifetime = arena;
  const void *slice = arena->data() + 128;
  auto view =
      IndexStorage::MemoryBlock::MakeSharedView(arena->data() + 128, arena);
  EXPECT_EQ(IndexStorage::MemoryBlock::MBT_SHARED_SCRATCH, view.type_);
  auto copy = view;
  IndexStorage::MemoryBlock assigned;
  assigned = copy;
  auto moved = std::move(copy);
  EXPECT_EQ(nullptr, copy.data());
  IndexStorage::MemoryBlock move_assigned;
  move_assigned = std::move(assigned);
  EXPECT_EQ(nullptr, assigned.data());
  arena.reset();
  view.reset();

  ASSERT_FALSE(lifetime.expired());
  EXPECT_EQ(slice, moved.data());
  EXPECT_EQ(slice, move_assigned.data());
  EXPECT_EQ(0x5A, *static_cast<const char *>(moved.data()));
  EXPECT_EQ(0x5A, *static_cast<const char *>(move_assigned.data()));
  moved.reset();
  EXPECT_FALSE(lifetime.expired());
  move_assigned.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(MemoryBlockSharedView, ResetAndReplacementReleasePreviousOwner) {
  auto arena = std::make_shared<std::vector<char>>(64, 0x2A);
  std::weak_ptr<std::vector<char>> lifetime = arena;
  auto view = IndexStorage::MemoryBlock::MakeSharedView(arena->data(), arena);
  auto copy = view;
  arena.reset();

  char replacement = 'x';
  view.reset(&replacement);
  EXPECT_EQ(&replacement, view.data());
  EXPECT_FALSE(lifetime.expired());
  copy = IndexStorage::MemoryBlock::MakeBorrowedView(&replacement);
  EXPECT_TRUE(lifetime.expired());

  auto next_arena = std::make_shared<std::vector<char>>(64, 0x3A);
  lifetime = next_arena;
  view =
      IndexStorage::MemoryBlock::MakeSharedView(next_arena->data(), next_arena);
  next_arena.reset();
  view = copy;
  EXPECT_TRUE(lifetime.expired());
}

TEST(MemoryBlockSharedView, DestructionReleasesLastSlice) {
  std::weak_ptr<std::vector<char>> lifetime;
  {
    auto arena = std::make_shared<std::vector<char>>(64, 0x2A);
    lifetime = arena;
    auto view = IndexStorage::MemoryBlock::MakeSharedView(arena->data(), arena);
    arena.reset();
    EXPECT_FALSE(lifetime.expired());
  }
  EXPECT_TRUE(lifetime.expired());
}
