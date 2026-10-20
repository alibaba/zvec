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

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ailego/io/iouring_loader.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include "../buffer/pool_payload_budget.h"

namespace zvec {
namespace ailego {
class IoUringStagingTestPeer {
 public:
  static void *data(const IoUringRing &ring) {
    return ring.staging_;
  }
  static size_t size(const IoUringRing &ring) {
    return ring.staging_size_;
  }
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

TEST_F(IoUringLoaderTest, StagingGrowthChargesOverlapAndCanRetry) {
  const size_t kPage = PoolPayloadBudget::page_buffer_size();
  ASSERT_EQ(0, pool_.init(3 * kPage));
  IoUringRing ring;
  ASSERT_TRUE(ring.ensure_staging(kPage));
  EXPECT_EQ(kPage, pool_.external_used());
  ASSERT_TRUE(ring.ensure_staging(2 * kPage));
  EXPECT_EQ(2 * kPage, pool_.external_used());
  auto *data = IoUringStagingTestPeer::data(ring);
  std::memset(data, 0x6a, 2 * kPage);
  EXPECT_FALSE(ring.ensure_staging(3 * kPage));
  EXPECT_EQ(data, IoUringStagingTestPeer::data(ring));
  EXPECT_EQ(0x6a, static_cast<unsigned char *>(data)[kPage]);
  EXPECT_EQ(2 * kPage, pool_.used());
  ASSERT_TRUE(ring.ensure_staging(kPage));
  EXPECT_EQ(2 * kPage, pool_.used());
  ring.teardown();
  EXPECT_EQ(0u, pool_.used());
  ASSERT_TRUE(ring.ensure_staging(3 * kPage));
  EXPECT_EQ(3 * kPage, pool_.used());
}

TEST_F(IoUringLoaderTest, PrechargedWritebackStagingIsNotChargedTwice) {
  const size_t kBytes = 4 * PoolPayloadBudget::page_buffer_size();
  ASSERT_TRUE(pool_.try_charge_metadata(kBytes));
  {
    IoUringRing ring;
    ring.adopt_staging_reservation(kBytes);
    ASSERT_TRUE(ring.ensure_staging(kIoUringStagingAlign));
    auto *data = IoUringStagingTestPeer::data(ring);
    EXPECT_EQ(kBytes, IoUringStagingTestPeer::size(ring));
    ASSERT_TRUE(ring.ensure_staging(kBytes));
    EXPECT_EQ(data, IoUringStagingTestPeer::data(ring));
    EXPECT_FALSE(ring.ensure_staging(kBytes + 1));
    EXPECT_EQ(kBytes, pool_.metadata_used());
    EXPECT_EQ(0u, pool_.external_used());
    ring.teardown();
    // Reservation follows the writeback ring, even after backend fallback.
    EXPECT_EQ(kBytes, pool_.used());
  }
  EXPECT_EQ(0u, pool_.used());
}

TEST_F(IoUringLoaderTest, AbandonedStagingKeepsItsChargeAfterDestruction) {
  const size_t kBytes = PoolPayloadBudget::page_buffer_size();
  for (bool prepaid : {false, true}) {
    SCOPED_TRACE(prepaid);
    void *abandoned = nullptr;
    {
      IoUringRing ring;
      if (prepaid) {
        ASSERT_TRUE(pool_.try_charge_metadata(kBytes));
        ring.adopt_staging_reservation(kBytes);
      }
      ASSERT_TRUE(ring.ensure_staging(kBytes));
      abandoned = IoUringStagingTestPeer::data(ring);
      ring.abandon_staging();
      EXPECT_EQ(nullptr, IoUringStagingTestPeer::data(ring));
      ring.teardown();
      EXPECT_EQ(kBytes, pool_.used());
    }
    EXPECT_EQ(kBytes, pool_.used());
    EXPECT_EQ(prepaid ? kBytes : 0u, pool_.metadata_used());
    EXPECT_EQ(prepaid ? 0u : kBytes, pool_.external_used());
    // No kernel requests were submitted in this test, so test cleanup may
    // release the simulated quarantine. Production must not do this.
    File::MemoryUnmap(abandoned, kBytes);
    if (prepaid)
      pool_.release_metadata(kBytes);
    else
      pool_.release_external(kBytes);
  }
}

TEST_F(IoUringLoaderTest, FailedReadReleasesStagingWithoutDestroyingRing) {
  IoUringRing ring;
  ASSERT_TRUE(ring.ensure_staging(kIoUringStagingAlign));
  EXPECT_EQ(PoolPayloadBudget::page_buffer_size(), pool_.used());
  EXPECT_EQ(-1, ring.execute(-1, nullptr, 0));
  EXPECT_EQ(0u, pool_.used());
  EXPECT_EQ(nullptr, IoUringStagingTestPeer::data(ring));
}

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

  ASSERT_EQ(ring.execute(file.fd(), requests), 0);
  EXPECT_EQ(0u, pool_.used());
  EXPECT_EQ(0u, IoUringStagingTestPeer::size(ring));
  EXPECT_EQ(std::memcmp(output, source.data(), kBlockSize), 0);
  const auto *tail = static_cast<const uint8_t *>(output) + kBlockSize;
  EXPECT_EQ(std::memcmp(tail, source.data() + kBlockSize, kTailSize), 0);
  EXPECT_TRUE(std::all_of(tail + kTailSize, tail + kBlockSize,
                          [](uint8_t value) { return value == 0; }));
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
