// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <gtest/gtest.h>
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/logger/logger.h>
#include <zvec/core/framework/index_framework.h>

namespace {
struct AllocationProbe {
  size_t context_size;
  size_t filter_size;
  bool throw_from_logging;
  void *context{nullptr};
  bool refused_filter{false};
  bool destroyed_context{false};
  bool logger_threw{false};
};

thread_local AllocationProbe *probe = nullptr;
}  // namespace

void *operator new(std::size_t size) {
  if (probe && probe->context && !probe->refused_filter &&
      size == probe->filter_size) {
    probe->refused_filter = true;
    throw std::bad_alloc();
  }
  void *ptr = std::malloc(size ? size : 1);
  if (!ptr) throw std::bad_alloc();
  if (probe && !probe->context && size == probe->context_size) {
    probe->context = ptr;
  }
  return ptr;
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

class ContextFailureLogger final : public ailego::Logger {
 public:
  int init(const ailego::Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  void log(int level, const char *, int, const char *, va_list) override {
    if (level == LEVEL_ERROR && probe && probe->refused_filter &&
        probe->throw_from_logging) {
      probe->logger_threw = true;
      throw std::bad_alloc();
    }
  }
};

template <typename Index>
void CheckContextFailure(Index &index, size_t context_size, size_t filter_size) {
  auto previous_logger =
      ailego::LoggerBroker::Register(std::make_shared<ContextFailureLogger>());
  const auto restore = ailego::ScopeGuard::Make([&] {
    ailego::LoggerBroker::Register(std::move(previous_logger));
  });
  for (bool throw_from_logging : {false, true}) {
    SCOPED_TRACE(throw_from_logging);
    AllocationProbe attempt{context_size, filter_size, throw_from_logging};
    core::IndexContext::Pointer result;
    bool threw = false;
    probe = &attempt;
    try {
      result = index.create_context();
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    probe = nullptr;

    ASSERT_TRUE(attempt.refused_filter);
    ASSERT_EQ(throw_from_logging, attempt.logger_threw);
    EXPECT_EQ(nullptr, result);
    if (!throw_from_logging) {
      EXPECT_FALSE(threw);
    }
    EXPECT_TRUE(attempt.destroyed_context);
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

  void OpenReadStorage(const std::string &path,
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
