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

#include "diskann_file_reader.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec::core;

namespace {

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

  const char *path() const {
    return path_;
  }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  char path_[64] = "DiskAnnFileReaderTest.XXXXXX";
  int fd_;
};

}  // namespace

TEST(DiskAnnFileReaderTest, BatchAlignedReads) {
  constexpr size_t kPageSize = 4096;
  constexpr size_t kPageCount = 32;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> source(kPageSize * kPageCount);
  for (size_t page = 0; page < kPageCount; ++page) {
    std::memset(source.data() + page * kPageSize,
                static_cast<int>(page + 1), kPageSize);
  }

  size_t written = 0;
  while (written < source.size()) {
    ssize_t ret = ::pwrite(file.fd(), source.data() + written,
                           source.size() - written, written);
    ASSERT_GT(ret, 0);
    written += static_cast<size_t>(ret);
  }
  ASSERT_EQ(::fsync(file.fd()), 0);
  file.close();

  void *raw_buffer = nullptr;
  ASSERT_EQ(::posix_memalign(&raw_buffer, kPageSize, source.size()), 0);
  std::unique_ptr<void, decltype(&std::free)> output(raw_buffer, &std::free);

  std::vector<AlignedRead> requests;
  requests.reserve(kPageCount);
  for (size_t i = 0; i < kPageCount; ++i) {
    // Use a non-sequential page order and more than AIO_LISTIO_MAX requests
    // so the macOS backend has to submit multiple batches.
    size_t source_page = (i * 7) % kPageCount;
    requests.emplace_back(source_page * kPageSize, kPageSize,
                          static_cast<uint8_t *>(output.get()) +
                              i * kPageSize);
  }

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx{
#if defined(__APPLE__) || defined(__MACH__)
      -1
#else
      0
#endif
  };
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_EQ(reader.read(requests, ctx), 0);

  for (size_t i = 0; i < kPageCount; ++i) {
    size_t source_page = (i * 7) % kPageCount;
    const auto *page =
        static_cast<const uint8_t *>(output.get()) + i * kPageSize;
    for (size_t byte = 0; byte < kPageSize; ++byte) {
      ASSERT_EQ(page[byte], static_cast<uint8_t>(source_page + 1));
    }
  }

  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
}
