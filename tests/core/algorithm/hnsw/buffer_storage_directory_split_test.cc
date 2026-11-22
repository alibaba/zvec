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

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__linux__) && !defined(__ANDROID__)
#include <dlfcn.h>
#include <unistd.h>
#elif defined(__APPLE__) && TARGET_OS_OSX
#include "buffer_storage_directory_pwrite.h"
#endif
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/budgeted_allocator.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/internal/platform.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include "algorithm/hnsw/hnsw_chunk.h"
#include "utility/utility_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace {
thread_local size_t fail_array_size = 0;
thread_local size_t fail_aligned_size = 0;
thread_local bool fault_hit = false;
#if defined(__linux__) && !defined(__ANDROID__)
thread_local bool fail_next_scalar_allocation = false;
#endif

void InjectAllocationFailure(size_t bytes, size_t &target) {
  if (target != 0 && bytes == target) {
    fail_array_size = 0;
    fail_aligned_size = 0;
    fault_hit = true;
    throw std::bad_alloc();
  }
}
// Keep GCC from diagnosing the paired aligned allocation as mismatched when
// it inlines the replacement delete into a caller.
#if defined(__GNUC__)
[[gnu::noinline]]
#endif
void FreeAligned(void *ptr) noexcept {
  ailego_aligned_free(ptr);
}
}  // namespace

// This test has its own executable, so the allocation fault is isolated from
// other tests. The two overloads cover plain and budgeted directory buffers.
void *operator new[](size_t bytes) {
  InjectAllocationFailure(bytes, fail_array_size);
  return ::operator new(bytes);
}
void operator delete[](void *ptr) noexcept {
  ::operator delete(ptr);
}
void operator delete[](void *ptr, size_t) noexcept {
  ::operator delete(ptr);
}

#if defined(__linux__) && !defined(__ANDROID__)
void *operator new(size_t bytes) {
  if (fail_next_scalar_allocation) {
    fail_next_scalar_allocation = false;
    fault_hit = true;
    throw std::bad_alloc();
  }
  if (void *ptr = std::malloc(std::max(size_t{1}, bytes))) return ptr;
  throw std::bad_alloc();
}
void operator delete(void *ptr) noexcept {
  FreeAligned(ptr);
}
void operator delete(void *ptr, size_t) noexcept {
  FreeAligned(ptr);
}
#endif

void *operator new(size_t bytes, std::align_val_t alignment) {
  InjectAllocationFailure(bytes, fail_aligned_size);
  if (void *ptr = ailego_aligned_malloc(std::max(size_t{1}, bytes),
                                        static_cast<size_t>(alignment))) {
    return ptr;
  }
  throw std::bad_alloc();
}
void operator delete(void *ptr, std::align_val_t) noexcept {
  FreeAligned(ptr);
}
void operator delete(void *ptr, size_t, std::align_val_t) noexcept {
  FreeAligned(ptr);
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
namespace {
using MetaFooter = zvec::core::IndexFormat::MetaFooter;
constexpr size_t kShortFooterBytes = offsetof(MetaFooter, total_size);
struct ShortFooterWrite {
  off_t offset;
  size_t prefix;
  bool throw_after_write{false};
  ssize_t written{-1};
};
thread_local ShortFooterWrite *short_footer_write = nullptr;
class ScopedWriteFailure;
thread_local ScopedWriteFailure *write_failure = nullptr;

class ScopedWriteFailure {
 public:
  explicit ScopedWriteFailure(off_t target) : offset(target) {
    write_failure = this;
  }
  ~ScopedWriteFailure() {
    write_failure = nullptr;
  }

  const off_t offset;
  bool hit{false};
};

// Write a real prefix of the footer. Each fault is armed for one write.
ssize_t InjectPwrite(decltype(&::pwrite) write_at, int fd, const void *buffer,
                     size_t count, off_t offset) {
  if (write_failure && offset == write_failure->offset) {
    write_failure->hit = true;
    write_failure = nullptr;
    errno = EIO;
    return -1;
  }
  if (count == sizeof(MetaFooter) && short_footer_write &&
      offset == short_footer_write->offset) {
    auto *fault = short_footer_write;
    short_footer_write = nullptr;
    fault->written = write_at(fd, buffer, fault->prefix, offset);
    // A logger's allocation failure can escape from the I/O error path.
    if (fault->throw_after_write) throw std::bad_alloc();
    return fault->written;
  }
  return write_at(fd, buffer, count, offset);
}
}  // namespace

#if defined(__linux__) && !defined(__ANDROID__)
extern "C" ssize_t pwrite(int fd, const void *buffer, size_t count,
                          off_t offset) {
  static const auto write_at =
      reinterpret_cast<decltype(&::pwrite)>(::dlsym(RTLD_NEXT, "pwrite"));
  if (!write_at) std::abort();
  return InjectPwrite(write_at, fd, buffer, count, offset);
}
#endif
#endif

namespace {
using namespace zvec::core;
constexpr uint32_t kDirectoryCapacity = 256;
constexpr size_t kMetaOverhead =
    sizeof(IndexFormat::MetaHeader) + sizeof(IndexFormat::MetaFooter);
constexpr uint64_t kCheckpoint = 73;
constexpr std::array<char, 5> kOldData = {'v', 'a', 'l', 'u', 'e'};
constexpr std::array<char, 5> kNewData = {'c', 'h', 'u', 'n', 'k'};

struct Directory {
  IndexFormat::MetaFooter footer{};
  uint64_t footer_offset{};
  size_t free_bytes{};
  uint64_t last_segment_offset{};

  uint64_t segment_meta_offset() const {
    return footer_offset - footer.segments_meta_size;
  }
};

bool ReadDirectory(const std::string &path, Directory *directory,
                   uint64_t chain_offset = 0) {
  std::ifstream in(path, std::ios::binary);
  IndexFormat::MetaHeader header{};
  in.seekg(chain_offset);
  if (!in.read(reinterpret_cast<char *>(&header), sizeof(header))) return false;
  directory->footer_offset = chain_offset + header.meta_footer_offset;
  in.seekg(directory->footer_offset);
  auto &footer = directory->footer;
  if (!in.read(reinterpret_cast<char *>(&footer), sizeof(footer))) return false;
  in.seekg(directory->segment_meta_offset());
  size_t first_name = footer.segments_meta_size;
  for (uint32_t i = 0; i < footer.segment_count; ++i) {
    IndexFormat::SegmentMeta segment{};
    if (!in.read(reinterpret_cast<char *>(&segment), sizeof(segment)))
      return false;
    first_name =
        std::min(first_name, static_cast<size_t>(segment.segment_id_offset));
    directory->last_segment_offset =
        chain_offset + header.content_offset + segment.data_index;
  }
  directory->free_bytes =
      first_name - footer.segment_count * sizeof(IndexFormat::SegmentMeta);
  return true;
}

class ScopedSplitAllocationFailure {
 public:
  explicit ScopedSplitAllocationFailure(size_t page_size) {
    const zvec::ailego::BudgetedAllocator<char> allocator;
    const size_t allocator_overhead =
        std::numeric_limits<size_t>::max() - allocator.max_size();
    const size_t split_allocation =
        (kDirectoryCapacity + kMetaOverhead + page_size - 1) / page_size *
            page_size -
        kMetaOverhead;
    fault_hit = false;
    fail_array_size = split_allocation;
    fail_aligned_size = split_allocation + allocator_overhead;
  }
  ~ScopedSplitAllocationFailure() {
    fail_array_size = 0;
    fail_aligned_size = 0;
  }
};

class BufferStorageDirectorySplitTest : public testing::Test {
 protected:
  void SetUp() override {
#if defined(__APPLE__) && TARGET_OS_OSX
    zvec::test::SetDirectorySplitPwriteHook(InjectPwrite);
#endif
    page_size_ = zvec::ailego::kVectorPageSize;
    ASSERT_NE(0u, page_size_);
    ASSERT_EQ(0,
              zvec::ailego::PoolPayloadBudget::get_instance().init(64UL << 20));
    path_ = testing::UnitTest::GetInstance()->current_test_info()->name();
    path_ = "buffer_storage_directory_split_" + path_ + ".index";
    std::remove(path_.c_str());
    writer_ = IndexFactory::CreateStorage("BufferStorage");
    ASSERT_NE(nullptr, writer_);
    zvec::ailego::Params params;
    params.set(MMAPFILE_STORAGE_SEGMENT_META_CAPACITY, kDirectoryCapacity);
    ASSERT_EQ(0, writer_->init(params));
    ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{true, false}));
  }

  void TearDown() override {
    if (writer_) writer_->close();
    writer_.reset();
    std::remove(path_.c_str());
#if defined(__APPLE__) && TARGET_OS_OSX
    zvec::test::SetDirectorySplitPwriteHook(nullptr);
#endif
  }

  void FillFirstChain() {
    ASSERT_NO_FATAL_FAILURE(FillDirectory());
    ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
    ASSERT_EQ(0, writer_->flush());
  }

  void AppendOldSegment() {
    old_name_ = "old_segment";
    ASSERT_EQ(0, writer_->append(old_name_, page_size_));
    ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  }

  void FillDirectory() {
    ASSERT_EQ(0, writer_->flush());
    Directory before{};
    ASSERT_TRUE(ReadDirectory(path_, &before));
    ASSERT_EQ(0u, before.footer.next_meta_header_offset);
    ASSERT_GT(before.free_bytes, sizeof(IndexFormat::SegmentMeta) + 1);
    old_name_.assign(before.free_bytes - sizeof(IndexFormat::SegmentMeta) - 1,
                     'x');
    ASSERT_EQ(0, writer_->append(old_name_, page_size_));
  }

  void WriteOldSegment() {
    auto segment = writer_->get(old_name_);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(kOldData.size(),
              segment->write(0, kOldData.data(), kOldData.size()));
  }

  void VerifySegment(IndexStorage &storage, const std::string &name,
                     const std::array<char, 5> &expected) {
    auto segment = storage.get(name);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(expected.size(), segment->data_size());
    std::array<char, 5> data{};
    ASSERT_EQ(data.size(), segment->fetch(0, data.data(), data.size()));
    EXPECT_EQ(expected, data);
  }

  void AppendAndVerifyReopen(const std::string &name) {
    const uint64_t committed = writer_->check_point();
    const bool was_dirty = writer_->is_dirty();
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_EQ(committed, writer_->check_point());
    EXPECT_EQ(was_dirty, writer_->is_dirty());
    {
      auto segment = writer_->get(name);
      ASSERT_NE(nullptr, segment);
      ASSERT_EQ(kNewData.size(),
                segment->write(0, kNewData.data(), kNewData.size()));
    }
    ASSERT_EQ(0, writer_->flush());
    ASSERT_EQ(0, writer_->close());
    for (const char *type : {"BufferStorage", "MMapFileStorage"}) {
      SCOPED_TRACE(type);
      auto reader = IndexFactory::CreateStorage(type);
      ASSERT_NE(nullptr, reader);
      ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
      ASSERT_EQ(0, reader->open(path_, false));
      EXPECT_EQ(kCheckpoint, reader->check_point());
      ASSERT_NO_FATAL_FAILURE(VerifySegment(*reader, old_name_, kOldData));
      ASSERT_NO_FATAL_FAILURE(VerifySegment(*reader, name, kNewData));
      ASSERT_EQ(0, reader->close());
    }
  }

  void VerifyRetry(const std::string &name) {
    EXPECT_FALSE(writer_->has(name));
    EXPECT_EQ(nullptr, writer_->get(name));
    ASSERT_NO_FATAL_FAILURE(VerifySegment(*writer_, old_name_, kOldData));
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_NE(nullptr, writer_->get(name));
  }

  size_t page_size_{};
  std::string path_;
  std::string old_name_;
  IndexStorage::Pointer writer_;
};

TEST_F(BufferStorageDirectorySplitTest, InPlaceAppendReopensAfterFlush) {
  ASSERT_NO_FATAL_FAILURE(AppendOldSegment());
  writer_->refresh(kCheckpoint);
  ASSERT_NO_FATAL_FAILURE(AppendAndVerifyReopen("new_in_place"));
}

TEST_F(BufferStorageDirectorySplitTest, SplitReopensAfterFlush) {
  ASSERT_NO_FATAL_FAILURE(FillDirectory());
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  writer_->refresh(kCheckpoint);
  ASSERT_NO_FATAL_FAILURE(AppendAndVerifyReopen("new_split"));
}

TEST_F(BufferStorageDirectorySplitTest, SplitPreservesCommittedCheckpoint) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(kCheckpoint, writer_->check_point());
  ASSERT_NO_FATAL_FAILURE(AppendAndVerifyReopen("new_split"));
}

TEST_F(BufferStorageDirectorySplitTest, FailedSplitCanRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  const std::string name = "retry_same_segment";
  {
    ScopedSplitAllocationFailure fault(page_size_);
    ASSERT_EQ(IndexError_NoMemory, writer_->append(name, page_size_));
    ASSERT_TRUE(fault_hit);
  }
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

#if defined(__APPLE__) && TARGET_OS_OSX
TEST_F(BufferStorageDirectorySplitTest,
       OldDataWritebackFailureDoesNotBlockAppend) {
  ASSERT_NO_FATAL_FAILURE(AppendOldSegment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  const std::string name = "new_in_place";
  ASSERT_GE(disk.free_bytes, sizeof(IndexFormat::SegmentMeta) + name.size() + 1);
  {
    auto segment = writer_->get(old_name_);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(kNewData.size(),
              segment->write(0, kNewData.data(), kNewData.size()));
  }
  {
    ScopedWriteFailure fault(disk.last_segment_offset);
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_FALSE(fault.hit);
    EXPECT_TRUE(writer_->is_dirty());
    EXPECT_EQ(IndexError_WriteData, writer_->flush());
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_TRUE(writer_->has(name));
  ASSERT_NO_FATAL_FAILURE(VerifySegment(*writer_, old_name_, kNewData));
  ASSERT_EQ(0, writer_->flush());
  EXPECT_FALSE(writer_->is_dirty());
}
#endif

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(BufferStorageDirectorySplitTest,
       OldMetadataWritebackFailureDoesNotBlockAppend) {
  ASSERT_NO_FATAL_FAILURE(AppendOldSegment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  const std::string name = "new_in_place";
  ASSERT_GE(disk.free_bytes, sizeof(IndexFormat::SegmentMeta) + name.size() + 1);

  auto old = writer_->get(old_name_);
  ASSERT_NE(nullptr, old);
  ASSERT_EQ(kNewData.size(),
            old->write(kOldData.size(), kNewData.data(), kNewData.size()));
  {
    ScopedWriteFailure fault(disk.segment_meta_offset());
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_FALSE(fault.hit);
    EXPECT_EQ(IndexError_WriteData, writer_->flush());
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_TRUE(writer_->has(name));
  EXPECT_EQ(kOldData.size() + kNewData.size(), old->data_size());
  ASSERT_EQ(0, writer_->flush());
  EXPECT_FALSE(writer_->is_dirty());
}

TEST_F(BufferStorageDirectorySplitTest, InPlaceAppendFailureRemovesMapping) {
  ASSERT_NO_FATAL_FAILURE(AppendOldSegment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  const std::string name = "retry_in_place";
  ASSERT_GT(disk.free_bytes,
            sizeof(IndexFormat::SegmentMeta) + name.size() + 1);

  {
    ScopedWriteFailure fault(disk.segment_meta_offset() +
                             disk.footer.segment_count *
                                 sizeof(IndexFormat::SegmentMeta));
    ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
    EXPECT_TRUE(fault.hit);
  }
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

TEST_F(BufferStorageDirectorySplitTest, FailedMetadataFlushCanRetryWithSplit) {
  ASSERT_NO_FATAL_FAILURE(FillDirectory());
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  writer_->refresh(kCheckpoint + 1);
  {
    ScopedWriteFailure fault(disk.segment_meta_offset());
    ASSERT_EQ(IndexError_WriteData, writer_->flush());
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  ASSERT_NO_FATAL_FAILURE(VerifyRetry("new_split"));
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_EQ(0, writer_->flush());
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
}

TEST_F(BufferStorageDirectorySplitTest, ShortCheckpointWritePreservesCommit) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(0, writer_->append("new_chain", page_size_));
  Directory root{}, tail{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const uint64_t tail_offset = root.footer.next_meta_header_offset;
  ASSERT_NE(0u, tail_offset);
  ASSERT_TRUE(ReadDirectory(path_, &tail, tail_offset));
  writer_->refresh(kCheckpoint + 1);

  // All checkpoint bytes and their CRC reach disk despite the short write.
  ShortFooterWrite fault{static_cast<off_t>(tail.footer_offset),
                        kShortFooterBytes};
  short_footer_write = &fault;
  const int status = writer_->flush();
  short_footer_write = nullptr;
  ASSERT_EQ(static_cast<ssize_t>(kShortFooterBytes), fault.written);
  ASSERT_EQ(IndexError_WriteData, status);
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_TRUE(ReadDirectory(path_, &root));
  ASSERT_TRUE(ReadDirectory(path_, &tail, tail_offset));
  EXPECT_EQ(kCheckpoint, root.footer.check_point);
  EXPECT_EQ(kCheckpoint, tail.footer.check_point);

  ASSERT_EQ(0, writer_->flush());
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
}

TEST_F(BufferStorageDirectorySplitTest, CloseCanRetryCheckpointWriteback) {
  ASSERT_NO_FATAL_FAILURE(AppendOldSegment());
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  writer_->refresh(kCheckpoint + 1);
  {
    ScopedWriteFailure fault(disk.footer_offset);
    ASSERT_EQ(IndexError_WriteData, writer_->close());
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
}

TEST_F(BufferStorageDirectorySplitTest, ShortFooterWriteCanRetrySplit) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));

  const std::string name = "retry_after_short_write";
  ShortFooterWrite fault{static_cast<off_t>(disk.footer_offset),
                        kShortFooterBytes};
  short_footer_write = &fault;
  const int status = writer_->append(name, page_size_);
  short_footer_write = nullptr;
  ASSERT_EQ(static_cast<ssize_t>(kShortFooterBytes), fault.written);
  ASSERT_EQ(IndexError_WriteData, status);
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

TEST_F(BufferStorageDirectorySplitTest,
       ShortFooterWriteRollsBackIfWriteThrows) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));

  const std::string name = "retry_short_write_exception";
  constexpr size_t kPrefix = 64;
  ShortFooterWrite fault{static_cast<off_t>(disk.footer_offset), kPrefix, true};
  short_footer_write = &fault;
  const int status = writer_->append(name, page_size_);
  short_footer_write = nullptr;
  ASSERT_EQ(static_cast<ssize_t>(kPrefix), fault.written);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

#endif

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(BufferStorageDirectorySplitTest, FailedMapAllocationCanRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  const std::string name = "retry_map_allocation";
  fault_hit = false;
  fail_next_scalar_allocation = true;
  const int status = writer_->append(name, page_size_);
  fail_next_scalar_allocation = false;
  EXPECT_TRUE(fault_hit);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

#endif

TEST_F(BufferStorageDirectorySplitTest, HnswBrokerCanRetryChunkAllocation) {
  IndexStreamer::Stats stats;
  ChunkBroker broker(stats);
  uint32_t chunk_size = page_size_;
  ASSERT_EQ(0, broker.open(writer_, chunk_size, false));
  broker.set_max_chunks_size(16 * page_size_);
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());

  constexpr int kType = ChunkBroker::CHUNK_TYPE_NODE;
  const auto name = ChunkBroker::make_segment_id(kType, 0);
  {
    ScopedSplitAllocationFailure fault(page_size_);
    auto [status, chunk] = broker.alloc_chunk(kType, 0);
    ASSERT_EQ(IndexError_NoMemory, status);
    ASSERT_EQ(nullptr, chunk);
    ASSERT_TRUE(fault_hit);
  }
  ASSERT_EQ(0u, broker.get_chunk_cnt(kType));
  EXPECT_FALSE(writer_->has(name));
  ASSERT_NO_FATAL_FAILURE(VerifySegment(*writer_, old_name_, kOldData));

  auto [status, chunk] = broker.alloc_chunk(kType, 0);
  ASSERT_EQ(0, status);
  ASSERT_NE(nullptr, chunk);
  ASSERT_EQ(1u, broker.get_chunk_cnt(kType));
  chunk.reset();
  ASSERT_EQ(0, broker.close());
}

TEST_F(BufferStorageDirectorySplitTest, HnswBrokerReopensChunkAfterFlush) {
  IndexStreamer::Stats stats;
  ChunkBroker broker(stats);
  uint32_t chunk_size = page_size_;
  ASSERT_EQ(0, broker.open(writer_, chunk_size, false));
  broker.set_max_chunks_size(16 * page_size_);
  ASSERT_NO_FATAL_FAILURE(FillDirectory());
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());

  constexpr int kType = ChunkBroker::CHUNK_TYPE_NODE;
  auto [status, chunk] = broker.alloc_chunk(kType, 0);
  ASSERT_EQ(0, status);
  ASSERT_NE(nullptr, chunk);
  ASSERT_EQ(kOldData.size(), chunk->write(0, kOldData.data(), kOldData.size()));
  chunk.reset();
  ASSERT_EQ(0, broker.flush(kCheckpoint));
  ASSERT_EQ(0, broker.flush(0));
  EXPECT_EQ(kCheckpoint, stats.check_point());
  ASSERT_EQ(0, broker.close());
  ASSERT_EQ(0, writer_->close());

  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  ASSERT_EQ(0, broker.open(writer_, chunk_size, false));
  EXPECT_EQ(kCheckpoint, stats.check_point());
  ASSERT_EQ(1u, broker.get_chunk_cnt(kType));
  auto reopened = broker.get_chunk(kType, 0);
  ASSERT_NE(nullptr, reopened);
  std::array<char, kOldData.size()> data{};
  ASSERT_EQ(data.size(), reopened->fetch(0, data.data(), data.size()));
  EXPECT_EQ(kOldData, data);
  reopened.reset();
  ASSERT_EQ(0, broker.close());
}
}  // namespace
