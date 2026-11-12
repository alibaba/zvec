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
#include <zvec/ailego/buffer/vector_page_table.h>

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

TEST(DiskAnnFileReaderTest, BufferPoolReadsSectorSlicesFromNativePages) {
  namespace ailego = zvec::ailego;
  const size_t native_page_size = ailego::kVectorPageSize;
  const size_t sector_size = DiskAnnUtil::kSectorSize;
  ASSERT_GE(native_page_size, sector_size);
  ASSERT_EQ(native_page_size % sector_size, 0U);

  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(2 * native_page_size);
  for (size_t sector = 0; sector < source.size() / sector_size; ++sector) {
    std::memset(source.data() + sector * sector_size,
                static_cast<int>(sector + 1), sector_size);
  }
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();

  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(
      memory_pool.init(native_page_size +
                       ailego::VecBufferPool::metadata_bytes_for_page_count(2)),
      0);
  auto pool =
      std::make_shared<ailego::VecBufferPool>(file.path(), /*writable=*/false);
  ASSERT_EQ(pool->init(), 0);
  ailego::block_id_t seed_page = 0;
  ASSERT_NE(pool->acquire_buffer(seed_page, 10), nullptr);

  AlignedBuffer output = make_aligned_buffer(4 * sector_size);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{
      {sector_size, sector_size, output.get()},
      {native_page_size - sector_size, 2 * sector_size,
       static_cast<uint8_t *>(output.get()) + sector_size},
      {sector_size, sector_size,
       static_cast<uint8_t *>(output.get()) + 3 * sector_size},
  };

  BufferPoolAlignedFileReader reader(pool);
  reader.open(file.path());
  IOContext unused{};
  ASSERT_EQ(reader.read(requests, unused), 0);
  EXPECT_EQ(std::memcmp(output.get(), source.data() + sector_size, sector_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<uint8_t *>(output.get()) + sector_size,
                        source.data() + native_page_size - sector_size,
                        2 * sector_size),
            0);
  EXPECT_EQ(std::memcmp(static_cast<uint8_t *>(output.get()) + 3 * sector_size,
                        source.data() + sector_size, sector_size),
            0);
  EXPECT_GE(pool->stats().bypass_bytes, sector_size);
  pool->release_pages(&seed_page, 1);
  reader.release_io_ctx(unused);
  EXPECT_EQ(destroy_io_ctx(unused), 0);
  reader.close();
}

TEST(DiskAnnFileReaderTest,
     BufferPoolCacheAndBypassFollowOriginalFileAfterPathReplacement) {
  namespace ailego = zvec::ailego;
  const size_t native_page_size = ailego::kVectorPageSize;
  const size_t sector_size = DiskAnnUtil::kSectorSize;
  TemporaryFile original;
  TemporaryFile replacement;
  ASSERT_GE(original.fd(), 0);
  ASSERT_GE(replacement.fd(), 0);
  std::vector<uint8_t> original_data(2 * native_page_size, 0x3a);
  std::vector<uint8_t> replacement_data(original_data.size(), 0xc7);
  ASSERT_TRUE(original.write_all(original_data.data(), original_data.size()));
  ASSERT_TRUE(
      replacement.write_all(replacement_data.data(), replacement_data.size()));
  original.close();
  replacement.close();

  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(
      memory_pool.init(native_page_size +
                       ailego::VecBufferPool::metadata_bytes_for_page_count(2)),
      0);
  auto pool = std::make_shared<ailego::VecBufferPool>(original.path(), false);
  ASSERT_EQ(pool->init(), 0);
  ailego::block_id_t pinned_page = 0;
  ASSERT_NE(pool->acquire_buffer(pinned_page, 10), nullptr);

  // Pin the only page the budget can hold. The cross-page request must use
  // both the old cached page and direct I/O for the uncached second page.
  ASSERT_EQ(::rename(replacement.path(), original.path()), 0);
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(reader.open_from_pool(original.path()), 0);
  AlignedBuffer output = make_aligned_buffer(2 * sector_size);
  ASSERT_NE(output, nullptr);
  const size_t offset = native_page_size - sector_size;
  std::vector<AlignedRead> requests{{offset, 2 * sector_size, output.get()}};
  const auto before = pool->stats();
  IOContext ctx{};
  EXPECT_EQ(reader.read(requests, ctx), 0);
  EXPECT_EQ(
      std::memcmp(output.get(), original_data.data() + offset, 2 * sector_size),
      0);
  const auto after = pool->stats();
  EXPECT_EQ(after.bypass_bytes - before.bypass_bytes, sector_size);
  pool->release_pages(&pinned_page, 1);
  reader.release_io_ctx(ctx);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
  pool.reset();
  EXPECT_EQ(memory_pool.used(), 0U);
  EXPECT_EQ(memory_pool.metadata_used(), 0U);
}

TEST(DiskAnnFileReaderTest, BufferPoolWithoutPageTableUsesBypassOnly) {
  namespace ailego = zvec::ailego;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(2 * ailego::kVectorPageSize, 0x5a);
  std::memset(source.data() + ailego::kVectorPageSize, 0xa5,
              ailego::kVectorPageSize);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(memory_pool.init(1), 0);
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);

  // BufferReadStorage deliberately does not initialize a page table when
  // metadata and one page cannot fit within the shared memory budget.
  ASSERT_FALSE(pool->cache_enabled());
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(reader.open_from_pool(file.path()), 0);
  AlignedBuffer output = make_aligned_buffer(2 * kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{
      {ailego::kVectorPageSize, kPageSize, output.get()},
      {0, kPageSize, static_cast<uint8_t *>(output.get()) + kPageSize}};
  IOContext ctx{};
  EXPECT_EQ(reader.read(requests, ctx), 0);
  EXPECT_EQ(std::memcmp(output.get(), source.data() + ailego::kVectorPageSize,
                        kPageSize),
            0);
  EXPECT_EQ(std::memcmp(static_cast<uint8_t *>(output.get()) + kPageSize,
                        source.data(), kPageSize),
            0);
  EXPECT_EQ(pool->stats().bypass_bytes, 2 * kPageSize);
  EXPECT_EQ(pool->stats().page_table_metadata_bytes, 0U);
  EXPECT_EQ(memory_pool.used(), 0U);
  EXPECT_EQ(memory_pool.metadata_used(), 0U);
  reader.release_io_ctx(ctx);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
}

TEST(DiskAnnFileReaderTest, BufferPoolReadBeforeOpenReturnsError) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(kPageSize, 0x5a);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  auto pool = std::make_shared<zvec::ailego::VecBufferPool>(file.path(), false);
  BufferPoolAlignedFileReader reader(pool);
  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{{0, kPageSize, output.get()}};
  IOContext ctx{};
  EXPECT_NE(reader.read(requests, ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderTest, BufferPoolRejectsMissingPool) {
  BufferPoolAlignedFileReader reader(nullptr);
  EXPECT_NE(reader.open_from_pool("missing"), 0);
  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{{0, kPageSize, output.get()}};
  IOContext ctx{};
  EXPECT_NE(reader.read(requests, ctx), 0);
  PendingBatch batch;
  EXPECT_NE(reader.submit(batch, requests, ctx), 0);
  EXPECT_EQ(batch.n_submitted, 0U);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderTest, BufferPoolRejectsUnalignedRequestsAfterOpen) {
  namespace ailego = zvec::ailego;
  const size_t sector_size = DiskAnnUtil::kSectorSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(2 * ailego::kVectorPageSize, 0x5a);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  auto &memory_pool = ailego::MemoryLimitPool::get_instance();
  ASSERT_EQ(
      memory_pool.init(16 * ailego::kVectorPageSize +
                       ailego::VecBufferPool::metadata_bytes_for_page_count(2)),
      0);
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(pool->init(), 0);
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(reader.open_from_pool(file.path()), 0);
  AlignedBuffer output = make_aligned_buffer(sector_size);
  ASSERT_NE(output, nullptr);

  IOContext ctx{};
  std::vector<AlignedRead> bad_offset{{1, sector_size, output.get()}};
  EXPECT_EQ(reader.read(bad_offset, ctx), IndexError_InvalidArgument);
  EXPECT_EQ(ctx, nullptr);
  std::vector<AlignedRead> bad_length{{0, sector_size - 1, output.get()}};
  EXPECT_EQ(reader.read(bad_length, ctx), IndexError_InvalidArgument);
  EXPECT_EQ(ctx, nullptr);

  std::vector<AlignedRead> valid{{0, sector_size, output.get()}};
  EXPECT_EQ(reader.read(valid, ctx), 0);
  EXPECT_EQ(std::memcmp(output.get(), source.data(), sector_size), 0);
  reader.release_io_ctx(ctx);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  reader.close();
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

#if defined(__APPLE__)
TEST(DiskAnnFileReaderTest, SectorTransferCachesOnlyLogicalNode) {
  TemporaryFile file;
  std::vector<uint8_t> source(8192);
  for (size_t i = 0; i < source.size(); ++i) source[i] = i % 251;
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  LinuxAlignedFileReader reader;
  reader.open(file.path());
  ASSERT_EQ(reader.configure_read_cache(64 * 1024, 1668), 0);
  ASSERT_EQ(reader.configure_read_workers(4), 0);
  std::vector<uint8_t> out(4096, 0xa5);
  AlignedRead node(123 + 1668, 1668, out.data() + 1668);
  node.enclosing_offset = 123;
  node.enclosing_len = 4096;
  node.enclosing_buf = out.data();
  std::vector<AlignedRead> reqs{node};
  IOContext ctx{};
  EXPECT_FALSE(reader.try_read_cached(node));
  ASSERT_EQ(reader.read(reqs, ctx), 0);
  EXPECT_EQ(std::memcmp(out.data(), source.data() + 123, 4096), 0);
  std::fill(out.begin(), out.end(), 0xa5);
  ASSERT_TRUE(reader.try_read_cached(node));
  EXPECT_EQ(std::memcmp(out.data() + 1668, source.data() + 1791, 1668), 0);
  EXPECT_EQ(out.front(), 0xa5);
  EXPECT_EQ(out.back(), 0xa5);
  std::fill(out.begin(), out.end(), 0xa5);
  ASSERT_EQ(reader.read(reqs, ctx), 0);
  EXPECT_EQ(std::memcmp(out.data() + 1668, source.data() + 123 + 1668, 1668),
            0);
  EXPECT_EQ(out.front(), 0xa5);
  EXPECT_EQ(out.back(), 0xa5);
  // Invalid enclosing ranges must not dispatch a read or publish cache data.
  reqs[0].enclosing_offset = 124;
  EXPECT_NE(reader.read(reqs, ctx), 0);
  reqs[0] = node;
  reqs[0].enclosing_len = 100;
  EXPECT_NE(reader.read(reqs, ctx), 0);
  reqs[0] = node;
  reqs[0].enclosing_buf = out.data() + 1;
  EXPECT_NE(reader.read(reqs, ctx), 0);
  reqs[0] = node;
  ASSERT_EQ(reader.configure_read_cache(0, 1668), 0);
  ASSERT_EQ(reader.configure_read_workers(1), 0);
  ASSERT_EQ(reader.read(reqs, ctx), 0);
  EXPECT_EQ(std::memcmp(out.data(), source.data() + 123, 4096), 0);
}

TEST(DiskAnnFileReaderTest, PackedUnalignedReadsPreserveUntouchedBytes) {
  TemporaryFile file;
  std::vector<uint8_t> source(32768);
  for (size_t i = 0; i < source.size(); ++i) source[i] = i % 251;
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  LinuxAlignedFileReader reader;
  reader.open(file.path());
  ASSERT_EQ(reader.configure_read_workers(4), 0);
  for (size_t bytes : {1668u, 8196u}) {
    ASSERT_EQ(reader.configure_read_cache(64 * 1024, bytes), 0);
    std::vector<uint8_t> output(bytes * 2 + 16, 0xa5);
    std::vector<AlignedRead> reqs{{123, bytes, output.data() + 4},
                                  {4321, bytes, output.data() + 8 + bytes}};
    IOContext ctx{};
    for (size_t repeat = 0; repeat < 3; ++repeat) {
      ASSERT_EQ(reader.read(reqs, ctx), 0);
      EXPECT_EQ(std::memcmp(output.data() + 4, source.data() + 123, bytes), 0);
      EXPECT_EQ(
          std::memcmp(output.data() + 8 + bytes, source.data() + 4321, bytes),
          0);
      EXPECT_EQ(output.front(), 0xa5);
      EXPECT_EQ(output[4 + bytes], 0xa5);
      EXPECT_EQ(output.back(), 0xa5);
    }
  }
}

TEST(DiskAnnFileReaderTest, ParallelReadsDrainErrorsAndPreserveOrder) {
  TemporaryFile file;
  std::vector<uint8_t> source(kPageSize * 16);
  for (size_t i = 0; i < source.size(); ++i) source[i] = i / kPageSize;
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  LinuxAlignedFileReader reader;
  reader.open(file.path());
  EXPECT_NE(reader.configure_read_workers(0), 0);
  EXPECT_NE(reader.configure_read_workers(33), 0);
  ASSERT_EQ(reader.configure_read_workers(4), 0);
  ASSERT_EQ(reader.configure_read_cache(64 * 1024), 0);
  auto output = make_aligned_buffer(source.size());
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests;
  for (size_t i = 0; i < 16; ++i)
    requests.emplace_back((15 - i) * kPageSize, kPageSize,
                          static_cast<uint8_t *>(output.get()) + i * kPageSize);
  IOContext ctx{};
  PendingBatch batch;
  ASSERT_EQ(reader.submit(batch, requests, ctx), 0);
  std::vector<uint32_t> completed;
  ASSERT_EQ(reader.get_completed(batch, ctx, 1, completed), 16);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(completed[i], i);
    EXPECT_EQ(static_cast<uint8_t *>(output.get())[i * kPageSize], 15 - i);
  }
  requests[0].offset = source.size();
  EXPECT_NE(reader.submit(batch, requests, ctx), 0);
  EXPECT_EQ(batch.n_submitted, 0u);
  requests[0].offset = 0;
  EXPECT_EQ(reader.submit(batch, requests, ctx), 0);
  // Aliased destinations retain serial last-write behavior.
  requests = {{0, kPageSize, output.get()},
              {kPageSize, kPageSize, output.get()}};
  EXPECT_EQ(reader.read(requests, ctx), 0);
  EXPECT_EQ(static_cast<uint8_t *>(output.get())[0], 1);
  reader.close();
  EXPECT_EQ(reader.read_workers(), 1u);
  reader.open(file.path());
  EXPECT_EQ(reader.read_workers(), 1u);
  EXPECT_EQ(reader.read_cache_bytes(), 0u);
}

TEST(DiskAnnFileReaderTest, ParallelConcurrentCallersShareBoundedWorkers) {
  TemporaryFile file;
  std::vector<uint8_t> source(kPageSize * 16);
  for (size_t i = 0; i < source.size(); ++i) source[i] = i / kPageSize;
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  LinuxAlignedFileReader reader;
  reader.open(file.path());
  ASSERT_EQ(reader.configure_read_workers(4), 0);
  ASSERT_EQ(reader.configure_read_cache(64 * 1024), 0);
  std::vector<std::thread> callers;
  for (size_t c = 0; c < 8; ++c)
    callers.emplace_back([&] {
      auto output = make_aligned_buffer(source.size());
      ASSERT_NE(output, nullptr);
      std::vector<AlignedRead> requests;
      for (size_t i = 0; i < 16; ++i)
        requests.emplace_back(
            i * kPageSize, kPageSize,
            static_cast<uint8_t *>(output.get()) + i * kPageSize);
      IOContext ctx{};
      for (size_t repeat = 0; repeat < 10; ++repeat) {
        ASSERT_EQ(reader.read(requests, ctx), 0);
        EXPECT_EQ(std::memcmp(output.get(), source.data(), source.size()), 0);
      }
    });
  for (auto &caller : callers) caller.join();
}

TEST(DiskAnnFileReaderTest, ReadCacheReopenAndSubmitPreserveIdentity) {
  TemporaryFile first, second;
  std::vector<uint8_t> source(kPageSize * 2, 17);
  ASSERT_TRUE(first.write_all(source.data(), source.size()));
  std::fill(source.begin(), source.end(), 29);
  ASSERT_TRUE(second.write_all(source.data(), source.size()));
  auto output = make_aligned_buffer(kPageSize * 2);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{
      {0, kPageSize, output.get()},
      {0, kPageSize, static_cast<uint8_t *>(output.get()) + kPageSize}};
  LinuxAlignedFileReader reader;
  reader.open(first.path());
  ASSERT_EQ(reader.configure_read_cache(64 * 1024), 0);
  EXPECT_GT(reader.read_cache_bytes(), 0u);
  EXPECT_LE(reader.read_cache_bytes(), 64u * 1024);
  IOContext ctx{};
  PendingBatch batch;
  ASSERT_EQ(reader.submit(batch, requests, ctx), 0);
  std::vector<uint32_t> completed;
  ASSERT_EQ(reader.get_completed(batch, ctx, 1, completed), 2);
  EXPECT_EQ(completed, (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(static_cast<uint8_t *>(output.get())[0], 17);
  EXPECT_EQ(static_cast<uint8_t *>(output.get())[kPageSize], 17);
  reader.open(second.path());
  EXPECT_EQ(reader.read_cache_bytes(), 0u);
  ASSERT_EQ(reader.configure_read_cache(64 * 1024), 0);
  ASSERT_EQ(reader.read(requests, ctx), 0);
  EXPECT_EQ(static_cast<uint8_t *>(output.get())[0], 29);
  requests = {{kPageSize * 2, kPageSize, output.get()}};
  EXPECT_NE(reader.submit(batch, requests, ctx), 0);
  EXPECT_EQ(batch.n_submitted, 0u);
  reader.close();
  EXPECT_EQ(reader.read_cache_bytes(), 0u);
  EXPECT_NE(reader.read(requests, ctx), 0);
}
#endif
