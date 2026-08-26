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

#if defined(__linux) || defined(__linux__)

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <ailego/io/iouring_loader.h>
#include <gtest/gtest.h>

namespace zvec {
namespace ailego {
namespace {

constexpr size_t kBlockSize = 512;
constexpr size_t kBlockCount = 4;

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

TEST(IoUringLoaderTest, ReadsBatchAndZeroPadsExpectedShortRead) {
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
  EXPECT_EQ(std::memcmp(output, source.data(), kBlockSize), 0);
  const auto *tail = static_cast<const uint8_t *>(output) + kBlockSize;
  EXPECT_EQ(std::memcmp(tail, source.data() + kBlockSize, kTailSize), 0);
  EXPECT_TRUE(std::all_of(tail + kTailSize, tail + kBlockSize,
                          [](uint8_t value) { return value == 0; }));
  std::free(output);
}

TEST(IoUringLoaderTest, WritesScatteredBatch) {
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

  std::vector<uint8_t> output(source.size());
  ASSERT_EQ(::pread(file.fd(), output.data(), output.size(), 0),
            static_cast<ssize_t>(output.size()));
  EXPECT_EQ(output, source);
}

}  // namespace
}  // namespace ailego
}  // namespace zvec

#endif  // __linux__
