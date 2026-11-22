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
#include <functional>
#include <limits>
#include <new>
#include <string>
#if defined(__linux__) && !defined(__ANDROID__)
#include <dlfcn.h>
#include <unistd.h>
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
thread_local size_t fail_scalar_size = 0;
thread_local size_t fail_aligned_size = 0;
thread_local bool fault_hit = false;

void InjectAllocationFailure(size_t bytes, size_t &target) {
  if (target != 0 && bytes == target) {
    fail_array_size = 0;
    fail_scalar_size = 0;
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
#if defined(__linux__) && !defined(__ANDROID__)
struct FooterPublicationProbe {
  off_t offset;
  std::function<void()> inspect;
  bool reject_next_allocation{false};
  bool written{false};
  bool allocation_attempted{false};
};
thread_local FooterPublicationProbe *footer_publication = nullptr;

void RejectProbedAllocation() {
  if (footer_publication && (footer_publication->written ||
                             footer_publication->reject_next_allocation)) {
    auto *probe = footer_publication;
    footer_publication = nullptr;
    probe->allocation_attempted = true;
    probe->inspect();
    throw std::bad_alloc();
  }
}

class ScopedFooterPublicationProbe {
 public:
  explicit ScopedFooterPublicationProbe(FooterPublicationProbe &probe) {
    footer_publication = &probe;
  }
  ~ScopedFooterPublicationProbe() {
    footer_publication = nullptr;
  }
};
#endif
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
  RejectProbedAllocation();
  InjectAllocationFailure(bytes, fail_scalar_size);
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
#if defined(__linux__) && !defined(__ANDROID__)
  RejectProbedAllocation();
#endif
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

#if defined(__linux__) && !defined(__ANDROID__)
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
thread_local off_t rejected_metadata_offset = -1;
thread_local size_t rejected_metadata_writes = 0;
thread_local bool reject_only_one_metadata_write = false;
}  // namespace

// Write a real prefix of the old footer, including the new chain link. This
// interposition is confined to this test executable and armed for one write.
extern "C" ssize_t pwrite(int fd, const void *buffer, size_t count,
                          off_t offset) {
  static const auto write_at =
      reinterpret_cast<decltype(&::pwrite)>(::dlsym(RTLD_NEXT, "pwrite"));
  if (!write_at) std::abort();
  if (offset == rejected_metadata_offset) {
    ++rejected_metadata_writes;
    if (reject_only_one_metadata_write) rejected_metadata_offset = -1;
    errno = EIO;
    return -1;
  }
  const bool watching_publication =
      footer_publication && offset == footer_publication->offset;
  const bool injecting_short_write =
      short_footer_write && offset == short_footer_write->offset;
  if (count == sizeof(MetaFooter) &&
      (injecting_short_write || watching_publication)) {
    uint64_t next_offset;
    std::memcpy(&next_offset,
                static_cast<const char *>(buffer) +
                    offsetof(MetaFooter, next_meta_header_offset),
                sizeof(next_offset));
    if (next_offset != 0) {
      if (injecting_short_write) {
        auto *fault = short_footer_write;
        short_footer_write = nullptr;
        fault->written = write_at(fd, buffer, fault->prefix, offset);
        // Simulate write_meta throwing after the partial write. A logger's
        // allocation failure can escape from that path.
        if (fault->throw_after_write) throw std::bad_alloc();
        return fault->written;
      }
      const ssize_t written = write_at(fd, buffer, count, offset);
      footer_publication->written = written == static_cast<ssize_t>(count);
      return written;
    }
  }
  return write_at(fd, buffer, count, offset);
}
#endif

namespace {
using namespace zvec::core;
constexpr uint32_t kDirectoryCapacity = 256;
constexpr size_t kMetaOverhead =
    sizeof(IndexFormat::MetaHeader) + sizeof(IndexFormat::MetaFooter);
constexpr std::array<char, 5> kOldData = {'v', 'a', 'l', 'u', 'e'};

struct Directory {
  IndexFormat::MetaFooter footer{};
  uint32_t footer_offset{};
  size_t free_bytes{};
};

bool ReadRootDirectory(const std::string &path, Directory *directory) {
  std::ifstream in(path, std::ios::binary);
  IndexFormat::MetaHeader root{};
  if (!in.read(reinterpret_cast<char *>(&root), sizeof(root))) return false;
  directory->footer_offset = root.meta_footer_offset;
  in.seekg(root.meta_footer_offset);
  auto &footer = directory->footer;
  if (!in.read(reinterpret_cast<char *>(&footer), sizeof(footer))) return false;
  in.seekg(root.meta_footer_offset - footer.segments_meta_size);
  size_t first_name = footer.segments_meta_size;
  for (uint32_t i = 0; i < footer.segment_count; ++i) {
    IndexFormat::SegmentMeta segment{};
    if (!in.read(reinterpret_cast<char *>(&segment), sizeof(segment)))
      return false;
    first_name =
        std::min(first_name, static_cast<size_t>(segment.segment_id_offset));
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
  }

  void FillFirstChain() {
    ASSERT_NO_FATAL_FAILURE(FillDirectory());
    ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
    ASSERT_EQ(0, writer_->flush());
  }

  void FillDirectory() {
    ASSERT_EQ(0, writer_->flush());
    Directory before{};
    ASSERT_TRUE(ReadRootDirectory(path_, &before));
    ASSERT_EQ(0u, before.footer.next_meta_header_offset);
    ASSERT_GT(before.free_bytes, sizeof(IndexFormat::SegmentMeta) + 1);
    old_name_.assign(before.free_bytes - sizeof(IndexFormat::SegmentMeta) - 1,
                     'x');
    ASSERT_EQ(0, writer_->append(old_name_, page_size_));
    Directory full{};
    ASSERT_TRUE(ReadRootDirectory(path_, &full));
    ASSERT_EQ(0u, full.footer.next_meta_header_offset);
    ASSERT_EQ(0u, full.free_bytes);
  }

  void WriteOldSegment() {
    auto segment = writer_->get(old_name_);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(kOldData.size(),
              segment->write(0, kOldData.data(), kOldData.size()));
  }

  void VerifyOldDirectory(const std::string &failed_name) {
    Directory disk{};
    ASSERT_TRUE(ReadRootDirectory(path_, &disk));
    EXPECT_EQ(0u, disk.footer.next_meta_header_offset);
    EXPECT_FALSE(writer_->has(failed_name));
    ASSERT_NO_FATAL_FAILURE(VerifyReader(failed_name, false));
  }

  void VerifyReader(const std::string &name, bool visible) {
    auto reader = IndexFactory::CreateStorage("BufferStorage");
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
    ASSERT_EQ(0, reader->open(path_, IndexStorage::OpenOptions{false, true}));
    auto old = reader->get(old_name_);
    ASSERT_NE(nullptr, old);
    ASSERT_EQ(kOldData.size(), old->data_size());
    std::array<char, kOldData.size()> data{};
    ASSERT_EQ(data.size(), old->fetch(0, data.data(), data.size()));
    EXPECT_EQ(kOldData, data);
    EXPECT_EQ(visible, reader->has(name));
    old.reset();
    EXPECT_EQ(0, reader->close());
  }

  void VerifyRetry(const std::string &name) {
    ASSERT_NO_FATAL_FAILURE(VerifyOldDirectory(name));
    ASSERT_EQ(0, writer_->append(name, page_size_));
    ASSERT_TRUE(writer_->has(name));
    ASSERT_EQ(0, writer_->close());
    ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
    ASSERT_TRUE(writer_->has(old_name_));
    ASSERT_TRUE(writer_->has(name));
  }

  size_t page_size_{};
  std::string path_;
  std::string old_name_;
  IndexStorage::Pointer writer_;
};

TEST_F(BufferStorageDirectorySplitTest,
       FailedSplitCanReopenAndRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  const std::string name = "retry_same_segment";
  {
    ScopedSplitAllocationFailure fault(page_size_);
    ASSERT_EQ(IndexError_NoMemory, writer_->append(name, page_size_));
    ASSERT_TRUE(fault_hit);
  }
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(BufferStorageDirectorySplitTest,
       WritebackAllocationFailureKeepsDirtySegment) {
  ASSERT_NO_FATAL_FAILURE(FillDirectory());
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  constexpr uint64_t kCheckpoint = 73;
  writer_->refresh(kCheckpoint);
  ASSERT_TRUE(writer_->is_dirty());

  const std::string name = "retry_dirty_split";
  fault_hit = false;
  fail_scalar_size = zvec::ailego::VecBufferPool::kWritebackBatchPages *
                     sizeof(zvec::ailego::block_id_t);
  const int status = writer_->append(name, page_size_);
  fail_scalar_size = 0;
  ASSERT_TRUE(fault_hit);
  ASSERT_EQ(IndexError_NoMemory, status);
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_FALSE(writer_->has(name));
  Directory old_directory{};
  ASSERT_TRUE(ReadRootDirectory(path_, &old_directory));
  EXPECT_EQ(0u, old_directory.footer.next_meta_header_offset);
  EXPECT_EQ(0u, old_directory.footer.check_point);

  ASSERT_EQ(0, writer_->append(name, page_size_));
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(VerifyReader(name, true));
  Directory persisted{};
  ASSERT_TRUE(ReadRootDirectory(path_, &persisted));
  EXPECT_EQ(kCheckpoint, persisted.footer.check_point);
}

TEST_F(BufferStorageDirectorySplitTest, AppendValidationPrecedesWriteback) {
  ASSERT_NO_FATAL_FAILURE(FillDirectory());
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  ASSERT_TRUE(writer_->is_dirty());
  Directory disk{};
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));
  ASSERT_GT(disk.footer_offset, disk.footer.segments_meta_size);

  const std::string name = "retry_after_writeback_failure";
  rejected_metadata_offset =
      disk.footer_offset - disk.footer.segments_meta_size;
  rejected_metadata_writes = 0;
  EXPECT_EQ(IndexError_Duplicate, writer_->append(old_name_, page_size_));
  EXPECT_EQ(IndexError_InvalidArgument, writer_->append(name, 0));
  EXPECT_EQ(0u, rejected_metadata_writes);
  EXPECT_TRUE(writer_->is_dirty());

  const int status = writer_->append(name, page_size_);
  rejected_metadata_offset = -1;
  EXPECT_EQ(1u, rejected_metadata_writes);
  ASSERT_EQ(IndexError_WriteData, status);
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_FALSE(writer_->has(name));
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));
  EXPECT_EQ(0u, disk.footer.next_meta_header_offset);

  ASSERT_EQ(0, writer_->append(name, page_size_));
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(VerifyReader(name, true));
}

TEST_F(BufferStorageDirectorySplitTest, InPlaceAppendFailureRemovesMapping) {
  old_name_ = "old_segment";
  ASSERT_EQ(0, writer_->append(old_name_, page_size_));
  ASSERT_NO_FATAL_FAILURE(WriteOldSegment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));
  const std::string name = "retry_in_place";
  ASSERT_GT(disk.free_bytes,
            sizeof(IndexFormat::SegmentMeta) + name.size() + 1);

  rejected_metadata_offset =
      disk.footer_offset - disk.footer.segments_meta_size;
  reject_only_one_metadata_write = true;
  const int status = writer_->append(name, page_size_);
  reject_only_one_metadata_write = false;
  rejected_metadata_offset = -1;
  ASSERT_EQ(IndexError_WriteData, status);
  ASSERT_NO_FATAL_FAILURE(VerifyOldDirectory(name));

  // BP owns the key in directory metadata, independent of the caller's ID.
  auto caller_name = name;
  ASSERT_EQ(0, writer_->append(caller_name, page_size_));
  std::fill(caller_name.begin(), caller_name.end(), 'x');
  EXPECT_TRUE(writer_->has(name));
  EXPECT_FALSE(writer_->has(caller_name));
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(VerifyReader(name, true));
}

TEST_F(BufferStorageDirectorySplitTest, ShortFooterWriteKeepsOldDirectory) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  Directory disk{};
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));

  const std::string name = "retry_after_short_write";
  ShortFooterWrite fault{disk.footer_offset, kShortFooterBytes};
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
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));

  const std::string name = "retry_short_write_exception";
  constexpr size_t kPrefix = 64;
  ShortFooterWrite fault{disk.footer_offset, kPrefix, true};
  short_footer_write = &fault;
  const int status = writer_->append(name, page_size_);
  short_footer_write = nullptr;
  ASSERT_EQ(static_cast<ssize_t>(kPrefix), fault.written);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
  ASSERT_NO_FATAL_FAILURE(VerifyReader(name, true));
}

TEST_F(BufferStorageDirectorySplitTest, PublishingSplitNeedsNoMoreAllocations) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  Directory disk{};
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));

  const std::string name = "published_segment";
  FooterPublicationProbe probe{disk.footer_offset,
                               [&] { VerifyReader(name, false); }};
  int status;
  {
    ScopedFooterPublicationProbe watch(probe);
    status = writer_->append(name, page_size_);
  }
  ASSERT_TRUE(probe.written);
  EXPECT_FALSE(probe.allocation_attempted);
  ASSERT_EQ(0, status);
  ASSERT_NO_FATAL_FAILURE(VerifyReader(name, true));
}

TEST_F(BufferStorageDirectorySplitTest, FailedMapAllocationKeepsOldDirectory) {
  ASSERT_NO_FATAL_FAILURE(FillFirstChain());
  Directory disk{};
  ASSERT_TRUE(ReadRootDirectory(path_, &disk));

  const std::string name = "retry_map_allocation";
  FooterPublicationProbe probe{disk.footer_offset,
                               [&] { VerifyReader(name, false); }};
  // The directory is clean and the first allocation is for the pending map.
  probe.reject_next_allocation = true;
  int status;
  {
    ScopedFooterPublicationProbe watch(probe);
    status = writer_->append(name, page_size_);
  }
  EXPECT_TRUE(probe.allocation_attempted);
  EXPECT_FALSE(probe.written);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(VerifyRetry(name));
}

#endif

TEST_F(BufferStorageDirectorySplitTest, HnswBrokerCanRetryAndReopenChunk) {
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
  ASSERT_NO_FATAL_FAILURE(VerifyOldDirectory(name));

  auto [status, chunk] = broker.alloc_chunk(kType, 0);
  ASSERT_EQ(0, status);
  ASSERT_NE(nullptr, chunk);
  ASSERT_EQ(1u, broker.get_chunk_cnt(kType));
  ASSERT_EQ(kOldData.size(), chunk->write(0, kOldData.data(), kOldData.size()));
  chunk.reset();
  ASSERT_EQ(0, broker.close());
  ASSERT_EQ(0, writer_->close());

  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  ASSERT_EQ(0, broker.open(writer_, chunk_size, false));
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
