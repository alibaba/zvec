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

#if (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ailego/io/iouring_loader.h>
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/io/file.h>
#include <gtest/gtest.h>
#include "../buffer/pool_payload_budget.h"

namespace zvec {
namespace ailego {

class IoUringStagingTestPeer {
 public:
  IoUringStagingTestPeer(int fd, bool duplicate)
      : duplicate_(duplicate) {
    instance_ = this;
    ring.ring_fd_ = ::dup(fd);
    ring.sq_entries_ = sqes_.size();
    ring.sq_tail_ = &sq_tail_;
    ring.sq_ring_mask_ = &ring_mask_;
    ring.sq_array_ = sq_array_.data();
    ring.sqes_ = sqes_.data();
    ring.cq_head_ = &cq_head_;
    ring.cq_tail_ = &cq_tail_;
    ring.cq_ring_mask_ = &ring_mask_;
    ring.cqes_ = cqes_.data();
    ring.enter_override_ = Enter;
  }

  void release_after_simulated_cancellation() {
    ASSERT_TRUE(File::TryMemoryUnmap(address_, size_));
    MemoryLimitPool::get_instance().release_unmapped_external(size_,
                                                              projection_);
  }

 private:
  static int Enter(int, unsigned submitted, unsigned, unsigned) {
    auto &self = *instance_;
    if (submitted != self.sqes_.size()) {
      errno = EIO;
      return -1;
    }
    self.address_ = self.ring.staging_;
    self.size_ = self.ring.staging_size_;
    self.projection_ = self.ring.staging_projection_;
    for (size_t i = 0; i < submitted; ++i) {
      const auto &sqe = self.sqes_[i];
      auto *buf = reinterpret_cast<void *>(sqe.addr);
      EXPECT_NE(buf, self.destinations[i]);
      if (sqe.opcode == IORING_OP_READ) std::memset(buf, 0x51, sqe.len);
      self.cqes_[i].user_data =
          self.sqes_[self.duplicate_ ? 0 : i].user_data;
      self.cqes_[i].res = sqe.len;
    }
    self.cq_tail_ = submitted;
    return static_cast<int>(submitted);
  }

  inline static thread_local IoUringStagingTestPeer *instance_{nullptr};
  bool duplicate_;
  unsigned sq_tail_{0};
  unsigned cq_head_{0};
  unsigned cq_tail_{0};
  unsigned ring_mask_{1};
  std::array<unsigned, 2> sq_array_{0, 1};
  std::array<io_uring_sqe, 2> sqes_{};
  std::array<io_uring_cqe, 2> cqes_{};
  char *address_{nullptr};
  size_t size_{0};
  ProcessRssBudget::MaterializationStamp projection_{};

 public:
  std::array<void *, 2> destinations{};
  IoUringRing ring;
};

namespace {

constexpr size_t kBlockSize = 512;
constexpr size_t kBlockCount = 4;

class IoUringLoaderTest : public testing::Test {
 protected:
  void SetUp() override {
    old_capacity_ = pool_.capacity();
    ASSERT_EQ(0u, pool_.used());
    ASSERT_EQ(0, pool_.init(16 * PoolPayloadBudget::page_buffer_size()));
  }
  void TearDown() override {
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(0, pool_.init(old_capacity_));
  }
  PoolPayloadBudget &pool_ = PoolPayloadBudget::get_instance();
  size_t old_capacity_{0};
};

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    ::unlink(path_);
  }

  int fd() const {
    return fd_;
  }

 private:
  char path_[64] = "IoUringLoaderTest.XXXXXX";
  int fd_;
};

void *allocate_aligned(size_t size) {
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kBlockSize, size) != 0) {
    return nullptr;
  }
  std::memset(buffer, 0, size);
  return buffer;
}

TEST(IoUringStagingTest, CompletedReadsReleaseRingOwnedStaging) {
  auto &pool = PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto reset = ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(0)); });
  const size_t before = pool.external_used();
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  IoUringStagingTestPeer fake(file.fd(), /*duplicate=*/false);
  alignas(kBlockSize) std::array<char, 2 * kBlockSize> output{};
  fake.destinations = {output.data(), output.data() + kBlockSize};
  const std::array<IoUringRead, 2> reads{{
      {0, kBlockSize, fake.destinations[0]},
      {kBlockSize, kBlockSize, fake.destinations[1]},
  }};
  EXPECT_EQ(0, fake.ring.execute(file.fd(), reads.data(), reads.size()));
  EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                          [](char value) { return value == 0x51; }));
  EXPECT_EQ(before, pool.external_used());
}

TEST(IoUringStagingTest, CancellationAfterPartialReapReleasesAsyncReadBacking) {
  auto &pool = PoolPayloadBudget::get_instance();
  const size_t old_capacity = pool.capacity();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto reset =
      ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(old_capacity)); });
  const size_t before = pool.external_used();
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  alignas(kBlockSize) std::array<char, 2 * kBlockSize> output{};
  IoUringStagingTestPeer fake(file.fd(), /*duplicate=*/false);
  fake.destinations = {output.data(), output.data() + kBlockSize};
  const std::array<IoUringRead, 2> reads{{
      {0, kBlockSize, fake.destinations[0]},
      {kBlockSize, kBlockSize, fake.destinations[1]},
  }};

  ASSERT_EQ(0, fake.ring.submit_reads(file.fd(), reads.data(), reads.size()));
  EXPECT_TRUE(fake.ring.has_pending_reads());
  EXPECT_GT(pool.external_used(), before);
  EXPECT_TRUE(std::all_of(output.begin(), output.end(),
                          [](char value) { return value == 0; }));
  uint32_t completed = 2;
  ASSERT_EQ(1, fake.ring.get_completed_reads(1, &completed, 1));
  EXPECT_EQ(0u, completed);
  EXPECT_TRUE(fake.ring.has_pending_reads());
  EXPECT_TRUE(std::all_of(output.begin(), output.begin() + kBlockSize,
                          [](char value) { return value == 0x51; }));
  fake.ring.cancel_reads();
  EXPECT_FALSE(fake.ring.has_pending_reads());
  EXPECT_TRUE(std::all_of(output.begin() + kBlockSize, output.end(),
                          [](char value) { return value == 0; }));
  EXPECT_EQ(before, pool.external_used());
}

TEST(IoUringStagingTest, AmbiguousWriteCompletionKeepsStagingCharged) {
  auto &pool = PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, pool.init(4 * 1024 * 1024));
  const auto reset = ScopeGuard::Make([&] { EXPECT_EQ(0, pool.init(0)); });
  const size_t before = pool.external_used();
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  IoUringStagingTestPeer fake(file.fd(), /*duplicate=*/true);
  std::array<char, 2 * kBlockSize> input{};
  fake.destinations = {input.data(), input.data() + kBlockSize};
  const std::array<IoUringWrite, 2> writes{{
      {0, kBlockSize, fake.destinations[0]},
      {kBlockSize, kBlockSize, fake.destinations[1]},
  }};
  EXPECT_EQ(IoUringRing::kQuarantined,
            fake.ring.execute_writes(file.fd(), writes.data(), writes.size()));
  EXPECT_FALSE(fake.ring.is_valid());
  ASSERT_GT(pool.external_used(), before);
  fake.release_after_simulated_cancellation();
  EXPECT_EQ(before, pool.external_used());
}

TEST_F(IoUringLoaderTest, ReadsBatchAndZeroPadsExpectedShortRead) {
  IoUringRing ring;
  if (!ring.setup(/*entries=*/8)) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  constexpr size_t kTailSize = 123;
  std::vector<uint8_t> source(kBlockSize + kTailSize);
  std::fill(source.begin(), source.begin() + kBlockSize, 0x3c);
  std::fill(source.begin() + kBlockSize, source.end(), 0x7d);
  ASSERT_EQ(::pwrite(file.fd(), source.data(), source.size(), 0),
            static_cast<ssize_t>(source.size()));

  void *output = allocate_aligned(2 * kBlockSize);
  ASSERT_NE(output, nullptr);
  std::vector<IoUringRead> requests;
  requests.emplace_back(/*offset=*/0, kBlockSize, output);
  requests.emplace_back(kBlockSize, kBlockSize,
                        static_cast<uint8_t *>(output) + kBlockSize, kTailSize);

  for (bool asynchronous : {false, true}) {
    SCOPED_TRACE(asynchronous);
    std::memset(output, 0xac, 2 * kBlockSize);
    if (!asynchronous) {
      ASSERT_EQ(0, ring.execute(file.fd(), requests));
    } else {
      ASSERT_EQ(0, ring.submit_reads(file.fd(), requests.data(), requests.size()));
      std::array<bool, 2> seen{};
      size_t total = 0;
      while (total < requests.size()) {
        std::array<uint32_t, 2> completed{};
        const int count = ring.get_completed_reads(1, completed.data(), completed.size());
        ASSERT_GT(count, 0);
        for (int i = 0; i < count; ++i) {
          const size_t id = completed[i];
          ASSERT_LT(id, requests.size());
          ASSERT_FALSE(seen[id]);
          seen[id] = true;
          const size_t bytes = id == 0 ? kBlockSize : kTailSize;
          EXPECT_EQ(0, std::memcmp(static_cast<uint8_t *>(output) + id * kBlockSize,
                                   source.data() + id * kBlockSize, bytes));
        }
        total += count;
      }
    }
    EXPECT_EQ(0u, pool_.used());
    EXPECT_EQ(std::memcmp(output, source.data(), kBlockSize), 0);
    const auto *tail = static_cast<const uint8_t *>(output) + kBlockSize;
    EXPECT_EQ(std::memcmp(tail, source.data() + kBlockSize, kTailSize), 0);
    EXPECT_TRUE(std::all_of(tail + kTailSize, tail + kBlockSize,
                            [](uint8_t value) { return value == 0; }));
  }
  std::free(output);
}

TEST_F(IoUringLoaderTest, WritesScatteredBatch) {
  IoUringRing ring;
  if (!ring.setup(/*entries=*/8)) {
    GTEST_SKIP() << "io_uring is unavailable";
  }

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(kBlockCount * kBlockSize);
  for (size_t block = 0; block < kBlockCount; ++block) {
    std::memset(source.data() + block * kBlockSize, static_cast<int>(block + 1),
                kBlockSize);
  }

  const std::array<size_t, kBlockCount> order = {3, 0, 2, 1};
  std::vector<IoUringWrite> requests;
  for (size_t block : order) {
    requests.emplace_back(block * kBlockSize, kBlockSize,
                          source.data() + block * kBlockSize);
  }
  ASSERT_EQ(ring.execute_writes(file.fd(), requests), 0);
  EXPECT_EQ(
      BudgetedMappedBuffer::mapping_size(kBlockCount * kIoUringStagingAlign),
      pool_.external_used());

  std::vector<uint8_t> output(source.size());
  ASSERT_EQ(::pread(file.fd(), output.data(), output.size(), 0),
            static_cast<ssize_t>(output.size()));
  EXPECT_EQ(output, source);
}

}  // namespace
}  // namespace ailego
}  // namespace zvec

#endif  // (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)
