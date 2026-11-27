// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <string>
#include <utility>
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_framework.h>

namespace {
struct AllocationProbe {
  size_t context_size;
  size_t refused_filter_allocation;
  void *context{nullptr};
  void *filter{nullptr};
  size_t filter_allocations{0};
  bool refused_filter{false};
  bool destroyed_context{false};
  bool destroyed_filter{false};
};

thread_local AllocationProbe *probe = nullptr;
thread_local size_t refused_allocation_size = 0;
}  // namespace

void *operator new(std::size_t size) {
  if (refused_allocation_size && size == refused_allocation_size) {
    refused_allocation_size = 0;
    throw std::bad_alloc();
  }
  void *ptr = std::malloc(size ? size : 1);
  if (!ptr) throw std::bad_alloc();
  if (probe && !probe->context && size == probe->context_size) {
    probe->context = ptr;
  }
  return ptr;
}

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void *operator new(std::size_t size, std::align_val_t alignment) {
  // Search visitation allocates its bits and word generations through
  // DiskAnnReadAllocator. Refuse each allocation independently, after the
  // owning context has been constructed.
  if (probe && probe->context && !probe->refused_filter &&
      ++probe->filter_allocations == probe->refused_filter_allocation) {
    probe->refused_filter = true;
    throw std::bad_alloc();
  }
  void *ptr =
      ailego_aligned_malloc(size ? size : 1, static_cast<size_t>(alignment));
  if (!ptr) throw std::bad_alloc();
  if (probe && probe->context && probe->filter_allocations == 1) {
    probe->filter = ptr;
  }
  return ptr;
}

void operator delete(void *ptr, std::align_val_t) noexcept {
  if (probe && ptr && ptr == probe->filter) {
    probe->destroyed_filter = true;
  }
  ailego_aligned_free(ptr);
}

void operator delete(void *ptr, std::size_t,
                     std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}

void operator delete(void *ptr) noexcept {
  if (probe && ptr && ptr == probe->context) {
    probe->destroyed_context = true;
  }
  std::free(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept {
  ::operator delete(ptr);
}

namespace zvec {
namespace {
namespace fs = std::filesystem;

template <typename Index>
void check_context_failure(Index &index, size_t context_size) {
  for (size_t allocation : {size_t{1}, size_t{2}}) {
    SCOPED_TRACE(allocation);
    AllocationProbe attempt{context_size, allocation};
    core::IndexContext::Pointer result;
    bool threw = false;
    {
      probe = &attempt;
      const auto reset_probe =
          ailego::ScopeGuard::Make([] { probe = nullptr; });
      try {
        result = index.create_context();
      } catch (const std::bad_alloc &) {
        threw = true;
      }
    }

    ASSERT_TRUE(attempt.refused_filter);
    EXPECT_EQ(allocation, attempt.filter_allocations);
    EXPECT_EQ(nullptr, result);
    EXPECT_FALSE(threw);
    EXPECT_TRUE(attempt.destroyed_context);
    if (allocation == 2) {
      ASSERT_NE(nullptr, attempt.filter);
      EXPECT_TRUE(attempt.destroyed_filter);
    }
    auto retry = index.create_context();
    ASSERT_NE(nullptr, retry);
  }
}

class QueryContextOomTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    const auto *test = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = test->name();
    std::replace(name.begin(), name.end(), '/', '_');
    directory_ = "query_context_oom_" + name;
    fs::remove_all(directory_);
    ASSERT_TRUE(fs::create_directory(directory_));
    auto &pool = ailego::MemoryLimitPool::get_instance();
    previous_capacity_ = pool.capacity();
    ASSERT_EQ(0, pool.init(size_t{256} << 20));
  }

  void TearDown() override {
    fs::remove_all(directory_);
    auto &pool = ailego::MemoryLimitPool::get_instance();
    EXPECT_EQ(0, pool.init(previous_capacity_));
  }

  void open_read_storage(const std::string &path,
                         core::IndexStorage::Pointer *storage) {
    *storage = core::IndexFactory::CreateStorage(
        GetParam() ? "BufferReadStorage" : "FileReadStorage");
    ASSERT_NE(nullptr, *storage);
    ASSERT_EQ(0, (*storage)->init(ailego::Params{}));
    ASSERT_EQ(0, (*storage)->open(path, false));
  }

  fs::path directory_;
  size_t previous_capacity_{0};
};

}  // namespace
}  // namespace zvec
