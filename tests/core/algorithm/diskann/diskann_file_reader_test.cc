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
#include <ailego/pattern/scope_guard.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include "../../../ailego/buffer/pool_payload_budget.h"

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

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      memory_pool.init(2 * native_page_size +
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

TEST(DiskAnnFileReaderTest, CachedSectorBatchesKeepOrderAndReleasePins) {
  namespace ailego = zvec::ailego;
  const size_t page = ailego::kVectorPageSize;
  const size_t sector = DiskAnnUtil::kSectorSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(8 * page);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>((i / sector) * 17 + i % 251);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  auto &budget = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, budget.init(64UL << 20));
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(0, pool->init());
  for (ailego::block_id_t id = 0; id < 8; ++id) {
    ASSERT_NE(nullptr, pool->acquire_buffer(id));
    pool->release_pages(&id, 1);
  }
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(0, reader.open_from_pool(file.path()));
  auto output = make_aligned_buffer(8 * page);
  ASSERT_NE(nullptr, output);
  std::vector<AlignedRead> requests{
      {3 * page + sector, sector, output.get()},
      {page - sector, 2 * sector,
       static_cast<uint8_t *>(output.get()) + 2 * page},
      {sector, sector, static_cast<uint8_t *>(output.get()) + 4 * page},
      {3 * page + sector, sector,
       static_cast<uint8_t *>(output.get()) + 6 * page}};
  IOContext ctx{};
  PendingBatch batch;
  const auto before = pool->stats();
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> completed;
  ASSERT_EQ(4, reader.get_completed(batch, ctx, 1, completed));
  for (size_t i = 0; i < requests.size(); ++i) {
    ASSERT_EQ(i, completed[i]);
    EXPECT_EQ(0, std::memcmp(requests[i].buf,
                            source.data() + requests[i].offset,
                            requests[i].len));
  }
  EXPECT_EQ(before.miss, pool->stats().miss);
  EXPECT_EQ(before.bypass_bytes, pool->stats().bypass_bytes);
  EXPECT_EQ(nullptr, ctx);
  // No page pin may escape the synchronous copy. Reclaim before checking the
  // caller's retained bytes so both pin release and output lifetime matter.
  for (size_t i = 0; i < 32; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  EXPECT_FALSE(pool->is_page_resident(0));
  EXPECT_FALSE(pool->is_page_resident(3));
  for (const auto &request : requests)
    EXPECT_EQ(0, std::memcmp(request.buf, source.data() + request.offset,
                            request.len));
  reader.close();
}

TEST(DiskAnnFileReaderTest, CachedNodesCopyOnlyTheirWindowAndReleasePins) {
  namespace ailego = zvec::ailego;
  const size_t page = ailego::kVectorPageSize;
  constexpr size_t sector = DiskAnnUtil::kSectorSize;
  constexpr size_t node_size = 1156;
  constexpr size_t nodes_per_sector = sector / node_size;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(8 * page);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i % 251);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  auto &budget = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(0, budget.init(64UL << 20));
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(0, pool->init());
  for (ailego::block_id_t id = 0; id < 8; ++id) {
    ASSERT_NE(nullptr, pool->acquire_buffer(id));
    pool->release_pages(&id, 1);
  }
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(0, reader.open_from_pool(file.path()));
  const std::array<uint32_t, 6> ids{{0, 1, 2, 4, 13, 1}};
  auto output = make_aligned_buffer(ids.size() * sector);
  ASSERT_NE(nullptr, output);
  std::memset(output.get(), 0xad, ids.size() * sector);
  std::vector<AlignedRead> requests;
  for (size_t i = 0; i < ids.size(); ++i)
    requests.emplace_back((1 + ids[i] / nodes_per_sector) * sector, sector,
                          static_cast<char *>(output.get()) + i * sector);
  IOContext ctx{};
  PendingBatch batch;
  const auto before = pool->stats();
  ASSERT_TRUE(reader.try_submit_cached_nodes(batch, requests, ctx, ids.data(),
                                              node_size, nodes_per_sector));
  std::vector<uint32_t> completed;
  ASSERT_EQ(ids.size(), reader.get_completed(batch, ctx, 1, completed));
  for (size_t i = 0; i < requests.size(); ++i) {
    ASSERT_EQ(i, completed[i]);
    const size_t start = (ids[i] % nodes_per_sector) * node_size;
    const auto *bytes = static_cast<const uint8_t *>(requests[i].buf);
    EXPECT_EQ(0, std::memcmp(bytes + start,
                            source.data() + requests[i].offset + start,
                            node_size));
    for (size_t j = 0; j < sector; ++j) {
      if (j < start || j >= start + node_size) {
        EXPECT_EQ(0xad, bytes[j]);
      }
    }
  }
  EXPECT_EQ(before.miss, pool->stats().miss);
  EXPECT_EQ(before.bypass_bytes, pool->stats().bypass_bytes);
  EXPECT_EQ(nullptr, ctx);
  for (size_t i = 0; i < 32; ++i)
    ailego::BlockEvictionQueue::get_instance().batch_recycle(64);
  for (ailego::block_id_t id = 0; id < 8; ++id)
    EXPECT_FALSE(pool->is_page_resident(id));
  for (size_t i = 0; i < requests.size(); ++i) {
    const size_t start = (ids[i] % nodes_per_sector) * node_size;
    EXPECT_EQ(0, std::memcmp(static_cast<char *>(requests[i].buf) + start,
                            source.data() + requests[i].offset + start,
                            node_size));
  }
}

TEST(DiskAnnFileReaderTest, CachedLargeNodeSpansNativePagesWithoutCopyingPadding) {
  namespace ailego = zvec::ailego;
  const size_t page = ailego::kVectorPageSize;
  constexpr size_t sector = DiskAnnUtil::kSectorSize;
  constexpr size_t node_size = 17 * 1024 + 132;
  constexpr size_t length = 5 * sector;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(8 * page);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i % 251);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(64UL << 20));
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(0, pool->init());
  for (ailego::block_id_t id = 0; id < 8; ++id) {
    ASSERT_NE(nullptr, pool->acquire_buffer(id));
    pool->release_pages(&id, 1);
  }
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(0, reader.open_from_pool(file.path()));
  auto output = make_aligned_buffer(length);
  ASSERT_NE(nullptr, output);
  std::memset(output.get(), 0xad, length);
  std::vector<AlignedRead> requests{{sector, length, output.get()}};
  const uint32_t id = 0;
  IOContext ctx{};
  PendingBatch batch;
  ASSERT_TRUE(reader.try_submit_cached_nodes(batch, requests, ctx, &id,
                                              node_size, 0));
  EXPECT_EQ(0, std::memcmp(output.get(), source.data() + sector, node_size));
  const auto *bytes = static_cast<const uint8_t *>(output.get());
  for (size_t i = node_size; i < length; ++i) EXPECT_EQ(0xad, bytes[i]);
}

TEST(DiskAnnFileReaderTest, MissingCachedNodeLeavesAllDestinationsUntouched) {
  namespace ailego = zvec::ailego;
  const size_t page = ailego::kVectorPageSize;
  constexpr size_t sector = DiskAnnUtil::kSectorSize;
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::vector<uint8_t> source(4 * page, 0x61);
  ASSERT_TRUE(file.write_all(source.data(), source.size()));
  file.close();
  ASSERT_EQ(0, ailego::PoolPayloadBudget::get_instance().init(64UL << 20));
  auto pool = std::make_shared<ailego::VecBufferPool>(file.path(), false);
  ASSERT_EQ(0, pool->init());
  const ailego::block_id_t first = 0;
  ASSERT_NE(nullptr, pool->acquire_buffer(first));
  pool->release_pages(&first, 1);
  BufferPoolAlignedFileReader reader(pool);
  ASSERT_EQ(0, reader.open_from_pool(file.path()));
  auto output = make_aligned_buffer(2 * sector);
  ASSERT_NE(nullptr, output);
  std::memset(output.get(), 0xad, 2 * sector);
  std::vector<AlignedRead> requests{
      {0, sector, output.get()},
      {2 * page, sector, static_cast<char *>(output.get()) + sector}};
  const uint32_t ids[]{0, 0};
  IOContext ctx{};
  PendingBatch batch;
  ASSERT_FALSE(reader.try_submit_cached_nodes(batch, requests, ctx, ids, 1156, 3));
  const auto *bytes = static_cast<const uint8_t *>(output.get());
  for (size_t i = 0; i < 2 * sector; ++i) ASSERT_EQ(0xad, bytes[i]);
  EXPECT_EQ(0u, batch.n_submitted);
  ASSERT_FALSE(reader.try_submit_cached_nodes(batch, requests, ctx, ids,
                                               sector + 1, 3));
  ASSERT_EQ(0, reader.submit(batch, requests, ctx));
  std::vector<uint32_t> completed;
  ASSERT_EQ(2, reader.get_completed(batch, ctx, 2, completed));
  for (const auto &request : requests)
    EXPECT_EQ(0, std::memcmp(request.buf, source.data() + request.offset,
                            request.len));
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

  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
  ASSERT_EQ(
      memory_pool.init(2 * native_page_size +
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
  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
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
  auto &memory_pool = ailego::PoolPayloadBudget::get_instance();
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
  std::vector<AlignedRead> bad_offset{{0, sector_size, output.get()}};
  bad_offset.front().offset = 1;
  EXPECT_EQ(reader.read(bad_offset, ctx), IndexError_InvalidArgument);
  EXPECT_EQ(ctx, nullptr);
  std::vector<AlignedRead> bad_length{{0, sector_size, output.get()}};
  bad_length.front().len = sector_size - 1;
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
