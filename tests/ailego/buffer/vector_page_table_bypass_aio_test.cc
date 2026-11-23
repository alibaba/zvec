// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <ailego/io/libaio_loader.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "pool_payload_budget.h"

namespace zvec::ailego {

struct VecBufferPoolTestPeer {
  static void use_libaio(VecBufferPool &pool) {
    pool.aio_enabled_ = true;
    pool.io_backend_type_ = IOBackendType::kLibAio;
  }
};

namespace {

struct FakeAio {
  enum class Mode {
    SUCCESS, PARTIAL, SHORT_READ, WAIT_ERROR, DESTROY_ERROR, DUPLICATE,
    REJECTED_RING
  };
  static FakeAio *active;
  Mode mode{Mode::SUCCESS};
  std::array<uint32_t, 8> header{0, 16, 0, 0, 0xa10a10a1, 0, 0, 32};
  std::array<iocb, 16> requests{};
  std::array<iocb *, 16> request_addresses{};
  size_t submitted{0};
  size_t pending{0};
  size_t submit_calls{0};
  size_t destroy_calls{0};

  static int setup(int, io_context_t *context) {
    if (active->mode == Mode::REJECTED_RING) active->header[4] = 0;
    *context = reinterpret_cast<io_context_t>(active->header.data());
    return 0;
  }
  static int destroy(io_context_t) {
    ++active->destroy_calls;
    if ((active->mode == Mode::DESTROY_ERROR ||
         active->mode == Mode::DUPLICATE ||
         active->mode == Mode::REJECTED_RING) &&
        active->destroy_calls == 1)
      return -EIO;
    active->pending = 0;
    return 0;
  }
  static int submit(io_context_t, long count, iocb **requests) {
    ++active->submit_calls;
    EXPECT_EQ(0u, active->pending);
    const size_t accepted =
        active->mode == Mode::PARTIAL
            ? std::min<size_t>(2, static_cast<size_t>(count))
            : static_cast<size_t>(count);
    for (size_t i = 0; i < accepted; ++i) {
      active->requests[i] = *requests[i];
      active->request_addresses[i] = requests[i];
    }
    active->submitted = active->pending = accepted;
    return static_cast<int>(accepted);
  }
  static int getevents(io_context_t, long, long, io_event *events, timespec *) {
    if (active->mode == Mode::WAIT_ERROR || active->mode == Mode::DESTROY_ERROR)
      return -EIO;
    // Deliberately return one completion at a time, in reverse request order.
    size_t index = --active->pending;
    if (active->mode == Mode::DUPLICATE && index == 0)
      index = active->submitted - 1;
    const auto &request = active->requests[index];
    const ssize_t bytes = ::pread(request.aio_fildes, request.u.c.buf,
                                  request.u.c.nbytes, request.u.c.offset);
    events[0] = {};
    events[0].data = request.data;
    events[0].obj = active->request_addresses[index];
    events[0].res = active->mode == Mode::SHORT_READ ? bytes - 512 : bytes;
    return 1;
  }
};

FakeAio *FakeAio::active = nullptr;

class BypassAioTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto &loader = LibAioLoader::Instance();
    if (!loader.load()) GTEST_SKIP() << "libaio not installed";
    setup_ = loader.io_setup;
    destroy_ = loader.io_destroy;
    submit_ = loader.io_submit;
    getevents_ = loader.io_getevents;
    loader.io_setup = FakeAio::setup;
    loader.io_destroy = FakeAio::destroy;
    loader.io_submit = FakeAio::submit;
    loader.io_getevents = FakeAio::getevents;
    FakeAio::active = &fake_;
    ASSERT_EQ(0, PoolPayloadBudget::get_instance().init(8UL * 1024 * 1024));
    char path[] = "bypass_aio_test_XXXXXX";
    const int fd = ::mkstemp(path);
    ASSERT_GE(fd, 0);
    path_ = path;
    std::vector<char> page(kVectorPageSize);
    for (size_t i = 0; i < 4; ++i) {
      std::fill(page.begin(), page.end(), static_cast<char>(i + 1));
      EXPECT_EQ(static_cast<ssize_t>(page.size()),
                ::write(fd, page.data(), page.size()));
    }
    EXPECT_EQ(0, ::close(fd));
  }
  void TearDown() override {
    // Every test joins its worker before restoring callbacks: the worker's
    // TLS context must use the same destroy function that created it.
    if (setup_) {
      auto &loader = LibAioLoader::Instance();
      loader.io_setup = setup_;
      loader.io_destroy = destroy_;
      loader.io_submit = submit_;
      loader.io_getevents = getevents_;
    }
    FakeAio::active = nullptr;
    if (!path_.empty()) std::remove(path_.c_str());
  }
  void check_read(VecBufferPool &pool) {
    std::vector<char> output(4 * kVectorPageSize, 'X');
    std::array<VecBufferPool::ReadRange, 4> ranges{};
    for (size_t i = 0; i < ranges.size(); ++i)
      ranges[i] = {i * kVectorPageSize, kVectorPageSize,
                   output.data() + i * kVectorPageSize};
    ASSERT_TRUE(pool.read_ranges_bypass(ranges.data(), ranges.size()));
    for (size_t i = 0; i < ranges.size(); ++i) {
      EXPECT_TRUE(std::all_of(
          output.begin() + i * kVectorPageSize,
          output.begin() + (i + 1) * kVectorPageSize,
          [i](char byte) { return byte == static_cast<char>(i + 1); }));
      EXPECT_FALSE(pool.is_page_resident(i));
    }
  }
  void run(FakeAio::Mode mode) {
    fake_.mode = mode;
    VecBufferPool pool(path_, false);
    ASSERT_EQ(0, pool.init());
    VecBufferPoolTestPeer::use_libaio(pool);
    // Create backing once before the lifetime assertion; slab headers remain
    // charged after returning their page slots to the free list.
    std::array<char *, 4> warm{};
    auto &memory = MemoryLimitPool::get_instance();
    ASSERT_EQ(warm.size(), memory.batch_acquire_buffers(
                               kVectorPageSize, warm.data(), warm.size()));
    for (char *page : warm) memory.release_buffer(page, kVectorPageSize);
    const auto before = MemoryLimitPool::get_instance().used();
    size_t after = 0;
    std::thread worker([&] {
      check_read(pool);
      after = MemoryLimitPool::get_instance().used();
      if (mode == FakeAio::Mode::DESTROY_ERROR ||
          mode == FakeAio::Mode::DUPLICATE ||
          mode == FakeAio::Mode::REJECTED_RING) {
        EXPECT_EQ(before + (mode == FakeAio::Mode::REJECTED_RING
                                ? 0 : 4 * kVectorPageSize), after);
        const auto calls = fake_.submit_calls;
        check_read(pool);
        EXPECT_EQ(calls,
                  fake_.submit_calls);  // Quarantined context not reused.
      }
    });
    worker.join();
    EXPECT_EQ(mode == FakeAio::Mode::REJECTED_RING ? 0u : 1u,
              fake_.submit_calls);
    EXPECT_EQ(0u, fake_.pending);
    EXPECT_EQ(before, MemoryLimitPool::get_instance().used());
    const bool retried_destroy = mode == FakeAio::Mode::DESTROY_ERROR ||
                                 mode == FakeAio::Mode::DUPLICATE ||
                                 mode == FakeAio::Mode::REJECTED_RING;
    EXPECT_EQ(retried_destroy ? 2u : 1u, fake_.destroy_calls);
  }

  FakeAio fake_;
  std::string path_;
  aio_setup_fn setup_{nullptr};
  aio_destroy_fn destroy_{nullptr};
  aio_submit_fn submit_{nullptr};
  aio_getevents_fn getevents_{nullptr};
};

TEST_F(BypassAioTest, DrainsPartialSubmissionBeforeFallback) {
  run(FakeAio::Mode::PARTIAL);
}
TEST_F(BypassAioTest, QuarantinesUntilDestroyWithoutReusingContext) {
  run(FakeAio::Mode::DESTROY_ERROR);
}
TEST_F(BypassAioTest, DuplicateCompletionDoesNotReleaseKernelDestinations) {
  run(FakeAio::Mode::DUPLICATE);
}
}  // namespace
}  // namespace zvec::ailego
