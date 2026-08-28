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
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <gtest/gtest.h>

using namespace zvec::core;

namespace {

constexpr size_t kPageSize = 4096;

class TemporaryFile {
 public:
  TemporaryFile() : fd_(::mkstemp(path_)) {}

  ~TemporaryFile() {
    close();
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

  bool write_all(const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    size_t written = 0;
    while (written < size) {
      ssize_t ret = ::pwrite(fd_, bytes + written, size - written, written);
      if (ret < 0 && errno == EINTR) {
        continue;
      }
      if (ret <= 0) {
        return false;
      }
      written += static_cast<size_t>(ret);
    }
    return ::fsync(fd_) == 0;
  }

 private:
  char path_[64] = "/tmp/zvec-diskann-reader-XXXXXX";
  int fd_;
};

using AlignedBuffer = std::unique_ptr<void, decltype(&std::free)>;

AlignedBuffer make_aligned_buffer(size_t size) {
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, kPageSize, size) != 0) {
    return AlignedBuffer(nullptr, &std::free);
  }
  std::memset(buffer, 0, size);
  return AlignedBuffer(buffer, &std::free);
}

}  // namespace

TEST(DiskAnnFileReaderTest, BatchAlignedReadsPreserveRequestOrder) {
  constexpr size_t kPageCount = 32;

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> source(kPageSize * kPageCount);
  for (size_t page = 0; page < kPageCount; ++page) {
    std::memset(source.data() + page * kPageSize, static_cast<int>(page + 1),
                kPageSize);
  }
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();

  AlignedBuffer output = make_aligned_buffer(source.size());
  ASSERT_NE(output, nullptr);

  std::vector<AlignedRead> requests;
  requests.reserve(kPageCount);
  for (size_t i = 0; i < kPageCount; ++i) {
    const size_t source_page = (i * 7) % kPageCount;
    requests.emplace_back(source_page * kPageSize, kPageSize,
                          static_cast<uint8_t *>(output.get()) + i * kPageSize);
  }

  LinuxAlignedFileReader reader;
  reader.open(file.path());

  IOContext ctx{};
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_EQ(reader.read(requests, ctx, false), 0);

  for (size_t i = 0; i < kPageCount; ++i) {
    const size_t source_page = (i * 7) % kPageCount;
    const auto *page =
        static_cast<const uint8_t *>(output.get()) + i * kPageSize;
    for (size_t byte = 0; byte < kPageSize; ++byte) {
      ASSERT_EQ(page[byte], static_cast<uint8_t>(source_page + 1));
    }
  }

  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
}

TEST(DiskAnnFileReaderTest, ShortReadReturnsError) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> source(kPageSize, 0x5a);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();

  AlignedBuffer output = make_aligned_buffer(kPageSize * 2);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{
      {0, kPageSize * 2, output.get()},
  };

  LinuxAlignedFileReader reader;
  reader.open(file.path());

  IOContext ctx{};
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  EXPECT_NE(reader.read(requests, ctx, false), 0);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
}

TEST(DiskAnnFileReaderTest, ReadBeforeOpenReturnsError) {
  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{
      {0, kPageSize, output.get()},
  };

  LinuxAlignedFileReader reader;
  IOContext ctx{};
  EXPECT_NE(reader.read(requests, ctx, false), 0);
}

TEST(DiskAnnFileReaderTest,
     OpenFromHandleSurvivesPathReplacementBeforeHandoff) {
  TemporaryFile original;
  TemporaryFile replacement;
  ASSERT_GE(original.fd(), 0);
  ASSERT_GE(replacement.fd(), 0);

  std::vector<uint8_t> original_data(kPageSize, 0x3a);
  std::vector<uint8_t> replacement_data(kPageSize, 0xc7);
  ASSERT_TRUE(original.write_all(original_data.data(), original_data.size()));
  ASSERT_TRUE(
      replacement.write_all(replacement_data.data(), replacement_data.size()));

  const int source_flags_before = ::fcntl(original.fd(), F_GETFL);
  ASSERT_GE(source_flags_before, 0);

  // The original descriptor represents FileReadStorage after it supplied
  // metadata. Replace the path first to prove the handoff follows the open
  // file object rather than resolving the path again.
  ASSERT_EQ(::rename(replacement.path(), original.path()), 0);

  LinuxAlignedFileReader original_reader;
  ASSERT_EQ(original_reader.open_from_handle(original.path(), original.fd()),
            0);
  const int source_flags_after = ::fcntl(original.fd(), F_GETFL);
  ASSERT_GE(source_flags_after, 0);
  EXPECT_EQ(source_flags_after, source_flags_before);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{{0, kPageSize, output.get()}};
  IOContext ctx{};
  ASSERT_EQ(setup_io_ctx(ctx), 0);

  ASSERT_EQ(original_reader.read(requests, ctx, false), 0);
  EXPECT_EQ(std::memcmp(output.get(), original_data.data(), kPageSize), 0);

  LinuxAlignedFileReader replacement_reader;
  replacement_reader.open(original.path());
  ASSERT_EQ(replacement_reader.read(requests, ctx, false), 0);
  EXPECT_EQ(std::memcmp(output.get(), replacement_data.data(), kPageSize), 0);

  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  original_reader.close();
  replacement_reader.close();
}

#if defined(__linux__) || defined(__linux) || defined(__APPLE__) || \
    defined(__MACH__)
TEST(DiskAnnFileReaderTest, OpenFromHandleDoesNotChangeSourceFlags) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> source(kPageSize, 0x6b);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));

  const int flags_before = ::fcntl(file.fd(), F_GETFL);
  ASSERT_GE(flags_before, 0);
#if defined(__linux__) || defined(__linux)
  ASSERT_EQ(flags_before & O_DIRECT, 0);
#endif

  LinuxAlignedFileReader reader;
  ASSERT_EQ(reader.open_from_handle(file.path(), file.fd()), 0);

  const int flags_after = ::fcntl(file.fd(), F_GETFL);
  ASSERT_GE(flags_after, 0);
  EXPECT_EQ(flags_after, flags_before);

  // An unaligned one-byte read remains valid on the caller's buffered handle.
  uint8_t byte = 0;
  ASSERT_EQ(::pread(file.fd(), &byte, 1, 0), 1);
  EXPECT_EQ(byte, 0x6b);
  reader.close();
}
#endif

#if defined(__APPLE__) || defined(__MACH__)
TEST(DiskAnnFileReaderTest, MacOSBatchUsesSynchronousPread) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);

  std::vector<uint8_t> source(kPageSize, 0x3c);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{{0, kPageSize, output.get()}};

  LinuxAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);
  EXPECT_EQ(ctx->type, zvec::ailego::IOBackendType::kPread);
  PendingBatch batch;
  ASSERT_EQ(reader.submit(batch, requests, ctx), 0);
  EXPECT_TRUE(batch.used_pread);
  EXPECT_EQ(batch.n_submitted, 1U);

  std::vector<uint32_t> completed;
  EXPECT_EQ(reader.get_completed(batch, ctx, 1, completed), 1);
  ASSERT_EQ(completed.size(), 1U);
  EXPECT_EQ(completed[0], 0U);
  EXPECT_EQ(std::memcmp(output.get(), source.data(), source.size()), 0);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
  reader.close();
}
#endif
