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

#include <sys/stat.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
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
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_format.h>
#include <zvec/core/framework/index_version.h>
#include "algorithm/hnsw/hnsw_chunk.h"
#include "utility/utility_params.h"
#include "../../../ailego/buffer/pool_payload_budget.h"

namespace {
thread_local size_t fail_array_size = 0;
thread_local size_t fail_aligned_size = 0;
thread_local bool fault_hit = false;
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
thread_local bool fail_next_allocation = false;
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

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
void *operator new(size_t bytes) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
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
#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
  if (fail_next_allocation) {
    fail_next_allocation = false;
    fault_hit = true;
    throw std::bad_alloc();
  }
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

struct PausedFooterWrite {
  std::mutex mutex;
  std::condition_variable cv;
  bool reached{false};
  bool resume{false};
};
thread_local PausedFooterWrite *paused_footer_write = nullptr;

struct RefreshOnFooterWrite {
  off_t offset;
  zvec::core::IndexStorage &storage;
  uint64_t checkpoint;
  bool hit{false};
  uint64_t visible_checkpoint{0};
};
thread_local RefreshOnFooterWrite *refresh_on_footer_write = nullptr;

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

// A short write on the second data range of a directory-sharing page.
thread_local off_t short_data_offset = -1;
thread_local size_t short_data_prefix = 0;
thread_local bool short_data_hit = false;
// Throw a C++ exception from an ordinary data-page writeback.
thread_local off_t throw_data_offset = -1;
// Fail the same offset repeatedly (e.g. a link write and its rollback).
thread_local off_t repeated_fail_offset = -1;
thread_local int repeated_fail_remaining = 0;

// Pause one data write so a same-stripe writer can queue behind it.
struct PausedDataWrite {
  std::mutex mutex;
  std::condition_variable cv;
  size_t offset{0};
  bool reached{false};
  bool resume{false};
};
thread_local PausedDataWrite *paused_data_write = nullptr;

// Write a real prefix of the footer. Each fault is armed for one write.
ssize_t InjectPwrite(decltype(&::pwrite) write_at, int fd, const void *buffer,
                     size_t count, off_t offset) {
  if (refresh_on_footer_write && refresh_on_footer_write->offset == offset) {
    auto &request = *refresh_on_footer_write;
    refresh_on_footer_write = nullptr;
    request.storage.refresh(request.checkpoint);
    request.visible_checkpoint = request.storage.check_point();
    request.hit = true;
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
  if (write_failure && offset == write_failure->offset) {
    if (paused_footer_write) {
      auto &pause = *paused_footer_write;
      std::unique_lock<std::mutex> lock(pause.mutex);
      pause.reached = true;
      pause.cv.notify_all();
      pause.cv.wait(lock, [&]() { return pause.resume; });
    }
    write_failure->hit = true;
    write_failure = nullptr;
    errno = EIO;
    return -1;
  }
  if (offset == short_data_offset && count > short_data_prefix) {
    const size_t prefix = short_data_prefix;
    short_data_offset = -1;
    short_data_hit = true;
    // Persist only the prefix so the first range succeeds and the second
    // range reports a short write.
    return write_at(fd, buffer, prefix, offset);
  }
  if (offset == throw_data_offset) {
    throw_data_offset = -1;
    throw std::bad_alloc();
  }
  if (offset == repeated_fail_offset && repeated_fail_remaining > 0) {
    if (--repeated_fail_remaining == 0) repeated_fail_offset = -1;
    errno = EIO;
    return -1;
  }
  if (paused_data_write &&
      offset == static_cast<off_t>(paused_data_write->offset)) {
    auto &pause = *paused_data_write;
    std::unique_lock<std::mutex> lock(pause.mutex);
    pause.reached = true;
    pause.cv.notify_all();
    pause.cv.wait(lock, [&] { return pause.resume; });
  }
  return write_at(fd, buffer, count, offset);
}

thread_local off_t failed_read_offset = -1;
thread_local bool failed_read_hit = false;

ssize_t InjectPread(decltype(&::pread) read_at, int fd, void *buffer,
                    size_t count, off_t offset) {
  if (offset == failed_read_offset) {
    failed_read_offset = -1;
    failed_read_hit = true;
    errno = EIO;
    return -1;
  }
  return read_at(fd, buffer, count, offset);
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

extern "C" ssize_t pread(int fd, void *buffer, size_t count, off_t offset) {
  static const auto read_at =
      reinterpret_cast<decltype(&::pread)>(::dlsym(RTLD_NEXT, "pread"));
  if (!read_at) std::abort();
  return InjectPread(read_at, fd, buffer, count, offset);
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
    zvec::test::SetDirectorySplitPreadHook(InjectPread);
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
    zvec::test::SetDirectorySplitPreadHook(nullptr);
#endif
  }

  void fill_first_chain() {
    ASSERT_NO_FATAL_FAILURE(fill_directory());
    ASSERT_NO_FATAL_FAILURE(write_old_segment());
    ASSERT_EQ(0, writer_->flush());
  }

  void append_old_segment() {
    old_name_ = "old_segment";
    ASSERT_EQ(0, writer_->append(old_name_, page_size_));
    ASSERT_NO_FATAL_FAILURE(write_old_segment());
  }

  void fill_directory() {
    ASSERT_EQ(0, writer_->flush());
    Directory before{};
    ASSERT_TRUE(ReadDirectory(path_, &before));
    ASSERT_EQ(0u, before.footer.next_meta_header_offset);
    ASSERT_GT(before.free_bytes, sizeof(IndexFormat::SegmentMeta) + 1);
    old_name_.assign(before.free_bytes - sizeof(IndexFormat::SegmentMeta) - 1,
                     'x');
    ASSERT_EQ(0, writer_->append(old_name_, page_size_));
  }

  void write_old_segment() {
    auto segment = writer_->get(old_name_);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(kOldData.size(),
              segment->write(0, kOldData.data(), kOldData.size()));
  }

  void verify_segment(IndexStorage &storage, const std::string &name,
                      const std::array<char, 5> &expected) {
    auto segment = storage.get(name);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(expected.size(), segment->data_size());
    std::array<char, 5> data{};
    ASSERT_EQ(data.size(), segment->fetch(0, data.data(), data.size()));
    EXPECT_EQ(expected, data);
  }

  void write_and_verify_reopen(const std::string &name, uint64_t checkpoint) {
    {
      auto segment = writer_->get(name);
      ASSERT_NE(nullptr, segment);
      ASSERT_EQ(kNewData.size(),
                segment->write(0, kNewData.data(), kNewData.size()));
    }
    ASSERT_EQ(0, writer_->flush());
    EXPECT_EQ(checkpoint, writer_->check_point());
    ASSERT_EQ(0, writer_->close());
    ASSERT_NO_FATAL_FAILURE(verify_reopened_segments(name, checkpoint));
  }

  void verify_m_map_can_continue(const std::string &name) {
    auto mmap = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, mmap);
    ASSERT_EQ(0, mmap->init(zvec::ailego::Params{}));
    ASSERT_EQ(0, mmap->open(path_, false));
    ASSERT_EQ(0, mmap->append(name, page_size_));
    {
      auto segment = mmap->get(name);
      ASSERT_NE(nullptr, segment);
      ASSERT_EQ(kNewData.size(),
                segment->write(0, kNewData.data(), kNewData.size()));
    }
    ASSERT_EQ(0, mmap->flush());
    ASSERT_EQ(0, mmap->close());
    ASSERT_NO_FATAL_FAILURE(verify_reopened_segments(name, 0));
  }

  void verify_reopened_segments(const std::string &name, uint64_t checkpoint) {
    for (const char *type : {"BufferStorage", "MMapFileStorage"}) {
      SCOPED_TRACE(type);
      auto reader = IndexFactory::CreateStorage(type);
      ASSERT_NE(nullptr, reader);
      ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
      ASSERT_EQ(0, reader->open(path_, false));
      EXPECT_EQ(checkpoint, reader->check_point());
      ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, old_name_, kOldData));
      ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, name, kNewData));
      ASSERT_EQ(0, reader->close());
    }
  }

  void verify_retry(const std::string &name) {
    EXPECT_FALSE(writer_->has(name));
    EXPECT_EQ(nullptr, writer_->get(name));
    ASSERT_NO_FATAL_FAILURE(verify_segment(*writer_, old_name_, kOldData));
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_NE(nullptr, writer_->get(name));
    ASSERT_NO_FATAL_FAILURE(write_and_verify_reopen(name, 0));
  }

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
  void recover_failed_append_rollback(const std::string &name) {
    Directory root{};
    ASSERT_TRUE(ReadDirectory(path_, &root));
    {
      ScopedWriteFailure restore_fault(root.footer_offset);
      ShortFooterWrite fault{static_cast<off_t>(root.footer_offset),
                             sizeof(root.footer.footer_crc)};
      short_footer_write = &fault;
      const int status = writer_->append(name, page_size_);
      short_footer_write = nullptr;
      ASSERT_EQ(IndexError_WriteData, status);
      ASSERT_EQ(static_cast<ssize_t>(fault.prefix), fault.written);
      ASSERT_TRUE(restore_fault.hit);
    }
    EXPECT_FALSE(writer_->has(name));
    EXPECT_TRUE(writer_->is_dirty());
    ASSERT_EQ(0, writer_->flush());
    auto reader = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
    ASSERT_EQ(0, reader->open(path_, false));
    EXPECT_FALSE(reader->has(name));
    ASSERT_NO_FATAL_FAILURE(verify_retry(name));
  }
#endif

  size_t page_size_{};
  std::string path_;
  std::string old_name_;
  IndexStorage::Pointer writer_;
};

TEST_F(BufferStorageDirectorySplitTest, InPlaceAppendReopensAfterFlush) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  writer_->refresh(kCheckpoint);
  const std::string name = "new_in_place";
  ASSERT_EQ(0, writer_->append(name, page_size_));
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_NO_FATAL_FAILURE(write_and_verify_reopen(name, kCheckpoint));
}

TEST_F(BufferStorageDirectorySplitTest,
       SplitPublishesOldChainBeforeExplicitFlush) {
  ASSERT_NO_FATAL_FAILURE(fill_directory());
  ASSERT_NO_FATAL_FAILURE(write_old_segment());
  writer_->refresh(kCheckpoint);
  const std::string name = "new_split";
  ASSERT_EQ(0, writer_->append(name, page_size_));
  EXPECT_EQ(0u, writer_->check_point());

  auto reader = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  EXPECT_TRUE(reader->has(name));
  // The storage accessor reports the new tail's checkpoint.
  EXPECT_EQ(0u, reader->check_point());
  Directory old_chain{};
  ASSERT_TRUE(ReadDirectory(path_, &old_chain));
  EXPECT_EQ(kCheckpoint, old_chain.footer.check_point);
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, old_name_, kOldData));
  ASSERT_EQ(0, reader->close());

  ASSERT_NO_FATAL_FAILURE(write_and_verify_reopen(name, 0));
}

TEST_F(BufferStorageDirectorySplitTest, RejectsUnrepresentableSegmentNames) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  ASSERT_EQ(0, writer_->flush());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  ASSERT_GT(root.footer.segments_meta_size, sizeof(IndexFormat::SegmentMeta));
  const std::string too_long(
      root.footer.segments_meta_size - sizeof(IndexFormat::SegmentMeta), 'x');
  const std::string embedded_nul("prefix\0suffix",
                                 sizeof("prefix\0suffix") - 1);

  ASSERT_EQ(IndexError_InvalidLength, writer_->append(too_long, page_size_));
  EXPECT_EQ(IndexError_InvalidArgument,
            writer_->append(embedded_nul, page_size_));
  const std::string valid_name = "valid_segment";
  const std::string exact_fit_name(too_long.size() - 1, 'x');
  ASSERT_EQ(0, writer_->append(valid_name, page_size_));
  ASSERT_EQ(0, writer_->append(exact_fit_name, page_size_));
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_TRUE(writer_->has(valid_name));
  EXPECT_TRUE(writer_->has(exact_fit_name));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*writer_, old_name_, kOldData));
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest, FailedSplitCanRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  const std::string name = "retry_same_segment";
  {
    ScopedSplitAllocationFailure fault(page_size_);
    ASSERT_EQ(IndexError_NoMemory, writer_->append(name, page_size_));
    ASSERT_TRUE(fault_hit);
  }
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

TEST_F(BufferStorageDirectorySplitTest, LargeDirectoryCanReopen) {
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const size_t name_size =
      root.footer.segments_meta_size - sizeof(IndexFormat::SegmentMeta) - 1;
  std::string first;
  std::string last;
  for (size_t i = 0; i < 1024; ++i) {
    last = std::to_string(i);
    last.resize(name_size, 'x');
    if (i == 0) first = last;
    ASSERT_EQ(0, writer_->append(last, page_size_)) << i;
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_TRUE(writer_->has(first));
  EXPECT_TRUE(writer_->has(last));
}

#if (defined(__linux__) && !defined(__ANDROID__)) || \
    (defined(__APPLE__) && TARGET_OS_OSX)
TEST_F(BufferStorageDirectorySplitTest,
       OldMetadataWritebackFailureDoesNotBlockAppend) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  const std::string name = "new_in_place";
  ASSERT_GE(disk.free_bytes,
            sizeof(IndexFormat::SegmentMeta) + name.size() + 1);

  auto old = writer_->get(old_name_);
  ASSERT_NE(nullptr, old);
  ASSERT_EQ(kNewData.size(),
            old->write(kOldData.size(), kNewData.data(), kNewData.size()));
  writer_->refresh(kCheckpoint);
  {
    ScopedWriteFailure fault(disk.segment_meta_offset());
    ASSERT_EQ(0, writer_->append(name, page_size_));
    EXPECT_EQ(kCheckpoint, writer_->check_point());
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
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
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
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

TEST_F(BufferStorageDirectorySplitTest, MMapAppendAfterFailedInPlaceAppend) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  {
    ScopedWriteFailure fault(disk.footer_offset);
    ASSERT_EQ(IndexError_WriteData,
              writer_->append("failed_append", page_size_));
    EXPECT_TRUE(fault.hit);
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(verify_m_map_can_continue("mmap_segment"));
}

TEST_F(BufferStorageDirectorySplitTest, MMapAppendAfterFailedSplit) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  // This ID needs a new chain. The shorter mmap ID fits in the old chain.
  ASSERT_GT(disk.free_bytes,
            sizeof(IndexFormat::SegmentMeta) + sizeof("mmap_segment"));
  const std::string name(disk.free_bytes - sizeof(IndexFormat::SegmentMeta),
                         'x');
  {
    ScopedWriteFailure fault(disk.footer_offset);
    ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
    EXPECT_TRUE(fault.hit);
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(verify_m_map_can_continue("mmap_segment"));
}

TEST_F(BufferStorageDirectorySplitTest,
       OldMetadataWritebackFailurePreventsSplit) {
  ASSERT_NO_FATAL_FAILURE(fill_directory());
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
  Directory disk{};
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  ASSERT_NO_FATAL_FAILURE(write_old_segment());
  writer_->refresh(kCheckpoint + 1);
  const std::string name = "new_split";
  {
    ScopedWriteFailure fault(disk.segment_meta_offset());
    ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
  EXPECT_EQ(0u, writer_->check_point());
  EXPECT_FALSE(writer_->is_dirty());
  ASSERT_TRUE(ReadDirectory(path_, &disk));
  EXPECT_EQ(kCheckpoint + 1, disk.footer.check_point);
}

TEST_F(BufferStorageDirectorySplitTest,
       InPlaceAppendRepairsInterruptedDirectoryWriteback) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  const std::string tail_name = "tail_segment";
  const std::string name = "following_append";
  ASSERT_EQ(0, writer_->append(tail_name, page_size_));
  Directory root{}, tail{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const uint64_t tail_offset = root.footer.next_meta_header_offset;
  ASSERT_NE(0u, tail_offset);
  ASSERT_TRUE(ReadDirectory(path_, &tail, tail_offset));
  ASSERT_GE(tail.free_bytes,
            sizeof(IndexFormat::SegmentMeta) + name.size() + 1);

  auto old = writer_->get(old_name_);
  ASSERT_NE(nullptr, old);
  ASSERT_EQ(kNewData.size(),
            old->write(kOldData.size(), kNewData.data(), kNewData.size()));
  {
    ScopedWriteFailure fault(root.footer_offset);
    ASSERT_EQ(IndexError_WriteData, writer_->flush());
    ASSERT_TRUE(fault.hit);
  }
  {
    ScopedWriteFailure fault(root.footer_offset);
    EXPECT_EQ(IndexError_WriteData,
              writer_->append("blocked_append", page_size_));
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_FALSE(writer_->has("blocked_append"));
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_EQ(0, writer_->append(name, page_size_));

  // Reopen while the writer is live so its close cannot repair the directory.
  for (const char *type : {"BufferStorage", "MMapFileStorage"}) {
    SCOPED_TRACE(type);
    auto reader = IndexFactory::CreateStorage(type);
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
    const int status = reader->open(path_, false);
    EXPECT_EQ(0, status);
    if (status != 0) continue;
    EXPECT_TRUE(reader->has(tail_name));
    EXPECT_TRUE(reader->has(name));
    EXPECT_FALSE(reader->has("blocked_append"));
    auto segment = reader->get(old_name_);
    ASSERT_NE(nullptr, segment);
    EXPECT_EQ(kOldData.size() + kNewData.size(), segment->data_size());
    std::array<char, kNewData.size()> actual{};
    EXPECT_EQ(actual.size(),
              segment->fetch(kOldData.size(), actual.data(), actual.size()));
    EXPECT_EQ(kNewData, actual);
  }
}

TEST_F(BufferStorageDirectorySplitTest, FailedClosePreservesWritebackForRetry) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  const std::string tail = "tail_segment";
  ASSERT_EQ(0, writer_->append(tail, page_size_));
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  ASSERT_NE(0u, root.footer.next_meta_header_offset);
  {
    auto old = writer_->get(old_name_);
    ASSERT_NE(nullptr, old);
    ASSERT_EQ(kNewData.size(),
              old->write(kOldData.size(), kNewData.data(), kNewData.size()));
  }
  {
    ScopedWriteFailure fault(root.footer_offset);
    EXPECT_EQ(IndexError_WriteData, writer_->close());
    EXPECT_TRUE(fault.hit);
  }
  EXPECT_TRUE(writer_->is_dirty());
  EXPECT_TRUE(writer_->has(tail));
  writer_->refresh(kCheckpoint);
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  RefreshOnFooterWrite request{static_cast<off_t>(root.footer_offset), *writer_,
                               kCheckpoint + 1};
  refresh_on_footer_write = &request;
  const int status = writer_->close();
  refresh_on_footer_write = nullptr;
  ASSERT_EQ(0, status);
  ASSERT_TRUE(request.hit);
  EXPECT_EQ(kCheckpoint, request.visible_checkpoint);
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  ASSERT_TRUE(writer_->has(tail));
  auto old = writer_->get(old_name_);
  ASSERT_NE(nullptr, old);
  EXPECT_EQ(kOldData.size() + kNewData.size(), old->data_size());
}

TEST_F(BufferStorageDirectorySplitTest, ConcurrentFlushRetriesFailedWriteback) {
  using namespace std::chrono_literals;
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  PausedFooterWrite pause;
  auto first = std::async(std::launch::async, [&]() {
#if defined(__APPLE__) && TARGET_OS_OSX
    zvec::test::SetDirectorySplitPwriteHook(InjectPwrite);
#endif
    paused_footer_write = &pause;
    ScopedWriteFailure fault(root.footer_offset);
    return writer_->flush();
  });
  bool reached;
  {
    std::unique_lock<std::mutex> lock(pause.mutex);
    reached = pause.cv.wait_for(lock, 5s, [&]() { return pause.reached; });
    if (!reached) {
      pause.resume = true;
      pause.cv.notify_all();
    }
  }
  ASSERT_TRUE(reached);
  // Observe the dirty state while the first writeback is still in flight.
  EXPECT_TRUE(writer_->is_dirty());
  std::promise<void> started;
  auto second = std::async(std::launch::async, [&]() {
    started.set_value();
    return writer_->flush();
  });
  started.get_future().wait();
  {
    std::lock_guard<std::mutex> lock(pause.mutex);
    pause.resume = true;
    pause.cv.notify_all();
  }
  EXPECT_EQ(IndexError_WriteData, first.get());
  EXPECT_EQ(0, second.get());
  EXPECT_FALSE(writer_->is_dirty());
  auto reader = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, old_name_, kOldData));
}

TEST_F(BufferStorageDirectorySplitTest, RefreshDuringWritebackStaysDirty) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  writer_->refresh(kCheckpoint);
  RefreshOnFooterWrite request{static_cast<off_t>(root.footer_offset), *writer_,
                               kCheckpoint + 1};
  refresh_on_footer_write = &request;
  const int status = writer_->flush();
  refresh_on_footer_write = nullptr;
  ASSERT_EQ(0, status);
  ASSERT_TRUE(request.hit);
  EXPECT_EQ(kCheckpoint + 1, request.visible_checkpoint);
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_TRUE(ReadDirectory(path_, &root));
  EXPECT_EQ(kCheckpoint, root.footer.check_point);
  ASSERT_EQ(0, writer_->flush());
  EXPECT_FALSE(writer_->is_dirty());
  ASSERT_TRUE(ReadDirectory(path_, &root));
  EXPECT_EQ(kCheckpoint + 1, root.footer.check_point);
}

TEST_F(BufferStorageDirectorySplitTest, FlushRecoversFailedAppendRollback) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  ASSERT_EQ(0, writer_->flush());
  ASSERT_NO_FATAL_FAILURE(recover_failed_append_rollback("retry_in_place"));
}

TEST_F(BufferStorageDirectorySplitTest, FlushRecoversFailedSplitRollback) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  ASSERT_NO_FATAL_FAILURE(recover_failed_append_rollback("retry_split"));
}

TEST_F(BufferStorageDirectorySplitTest, CrossPageWriteReportsWrittenPrefix) {
  constexpr std::array<char, 8> original = {'o', 'l', 'd', '1',
                                            'o', 'l', 'd', '2'};
  constexpr std::array<char, 8> replacement = {'n', 'e', 'w', '1',
                                               'n', 'e', 'w', '2'};
  constexpr std::array<char, 8> expected = {'n', 'e', 'w', '1',
                                            'o', 'l', 'd', '2'};
  const std::string name = "cross_page";
  const size_t offset = page_size_ - 4;
  ASSERT_EQ(0, writer_->append(name, 2 * page_size_));
  {
    auto segment = writer_->get(name);
    ASSERT_NE(nullptr, segment);
    ASSERT_EQ(original.size(),
              segment->write(offset, original.data(), original.size()));
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  auto segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  std::array<char, 8> actual{};
  ASSERT_EQ(4u, segment->fetch(offset, actual.data(), 4));

  failed_read_hit = false;
  failed_read_offset = static_cast<off_t>(segment->data_offset() + page_size_);
  const size_t written =
      segment->write(offset, replacement.data(), replacement.size());
  failed_read_offset = -1;
  ASSERT_TRUE(failed_read_hit);
  EXPECT_EQ(4u, written);
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_EQ(0, writer_->flush());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  auto reopened = reader->get(name);
  ASSERT_NE(nullptr, reopened);
  ASSERT_EQ(actual.size(),
            reopened->fetch(offset, actual.data(), actual.size()));
  EXPECT_EQ(expected, actual);
}

TEST_F(BufferStorageDirectorySplitTest, ShortCheckpointWriteCanRetryFlush) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  ASSERT_EQ(0, writer_->append("new_chain", page_size_));
  writer_->refresh(kCheckpoint);
  ASSERT_EQ(0, writer_->flush());
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
  EXPECT_TRUE(writer_->is_dirty());

  ASSERT_EQ(0, writer_->flush());
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_EQ(kCheckpoint + 1, writer_->check_point());
}

TEST_F(BufferStorageDirectorySplitTest, ShortFooterWriteCanRetrySplit) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
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
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

TEST_F(BufferStorageDirectorySplitTest,
       ShortFooterWriteRollsBackIfWriteThrows) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
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
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

// Build an index whose directory uses a 4 KiB format page on purpose. Hosts
// with a larger page size then share cache pages between the directory and
// segment data.
void CreateFormatPageLayoutIndex(const std::string &path,
                                 uint32_t format_page) {
  const uint32_t content_offset = format_page;
  const uint32_t meta_footer_offset =
      content_offset - sizeof(IndexFormat::MetaFooter);
  const uint32_t segments_meta_size =
      meta_footer_offset - sizeof(IndexFormat::MetaHeader);
  IndexFormat::MetaHeader header{};
  IndexFormat::SetupMetaHeader(&header, meta_footer_offset, content_offset);

  const std::string details = IndexVersion::Details();
  const std::string version_name = INDEX_VERSION_SEGMENT_NAME;
  // Match the writer's layout: meta records grow from the front, segment
  // names grow down from the end, and the gap is free directory space.
  std::vector<char> table(segments_meta_size, 0);
  auto *version = reinterpret_cast<IndexFormat::SegmentMeta *>(table.data());
  version->segment_id_offset =
      segments_meta_size - static_cast<uint32_t>(version_name.size() + 1);
  version->data_index = 0;
  version->data_size = details.size();
  version->padding_size = format_page - details.size();
  version->data_crc =
      zvec::ailego::Crc32c::Hash(details.data(), details.size(), 0);
  std::memcpy(table.data() + version->segment_id_offset, version_name.c_str(),
              version_name.size() + 1);

  IndexFormat::MetaFooter footer{};
  IndexFormat::SetupMetaFooter(&footer);
  footer.segment_count = 1;
  footer.segments_meta_size = segments_meta_size;
  footer.content_size = format_page;
  footer.total_size = content_offset + format_page;
  footer.segments_meta_crc =
      zvec::ailego::Crc32c::Hash(table.data(), segments_meta_size, 0);
  IndexFormat::UpdateMetaFooter(&footer, 0);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out);
  out.write(reinterpret_cast<const char *>(&header), sizeof(header));
  out.write(reinterpret_cast<const char *>(table.data()), segments_meta_size);
  out.write(reinterpret_cast<const char *>(&footer), sizeof(footer));
  out.seekp(content_offset);
  out.write(details.data(), details.size());
  out.seekp(footer.total_size - 1);
  out.put('\0');
  ASSERT_TRUE(out);
}

TEST_F(BufferStorageDirectorySplitTest, SplitOnLegacyFormatPageKeepsDirectory) {
  if (page_size_ <= 4096) {
    GTEST_SKIP() << "requires a host page larger than the 4 KiB layout";
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(CreateFormatPageLayoutIndex(path_, 4096));
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));

  ASSERT_EQ(0, writer_->append("filler", page_size_));
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  std::string padding_name(
      root.free_bytes - sizeof(IndexFormat::SegmentMeta) - 1, 'x');
  ASSERT_EQ(0, writer_->append(padding_name, page_size_));
  // This append splits the directory onto a fresh chain.
  const std::string name = "new_segment";
  ASSERT_EQ(0, writer_->append(name, page_size_));
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const uint64_t tail_offset = root.footer.next_meta_header_offset;
  ASSERT_NE(0u, tail_offset);

  auto segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  ASSERT_EQ(kNewData.size(),
            segment->write(0, kNewData.data(), kNewData.size()));
  // The first segment write must not zero the new chain's header, and the
  // data writeback must leave every directory byte untouched.
  ASSERT_EQ(0, writer_->vec_buffer_pool()->flush_all());
  ASSERT_EQ(0, writer_->close());

  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, name, kNewData));
  ASSERT_EQ(0, reader->close());
  Directory tail{};
  EXPECT_TRUE(ReadDirectory(path_, &tail, tail_offset));
}

TEST_F(BufferStorageDirectorySplitTest,
       InPlaceAppendAfterDataWritebackKeepsNewSegment) {
  if (page_size_ <= 4096) {
    GTEST_SKIP() << "requires a host page larger than the 4 KiB layout";
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(CreateFormatPageLayoutIndex(path_, 4096));
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));

  const std::string old_name = "old_segment";
  ASSERT_EQ(0, writer_->append(old_name, page_size_));
  auto old = writer_->get(old_name);
  ASSERT_NE(nullptr, old);
  // Cache the directory page and dirty the old segment's data bytes.
  ASSERT_EQ(kOldData.size(), old->write(0, kOldData.data(), kOldData.size()));

  const std::string name = "new_in_place";
  ASSERT_EQ(0, writer_->append(name, page_size_));
  // A pure data writeback must not flush the stale cached directory back.
  ASSERT_EQ(0, writer_->vec_buffer_pool()->flush_all());

  // Before close, the on-disk table is exactly what append published. The
  // new segment must still be visible after the pure data writeback.
  {
    auto reader = IndexFactory::CreateStorage("BufferStorage");
    ASSERT_NE(nullptr, reader);
    ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
    ASSERT_EQ(0, reader->open(path_, false));
    EXPECT_TRUE(reader->has(name));
    ASSERT_EQ(0, reader->close());
  }
  // close() publishes the data sizes as well; verify the payload then.
  ASSERT_EQ(0, writer_->close());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  EXPECT_TRUE(reader->has(name));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, old_name, kOldData));
  ASSERT_EQ(0, reader->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       SharedPageShortSecondRangeKeepsDirtyForRetry) {
  if (page_size_ <= 4096) {
    GTEST_SKIP() << "requires a host page larger than the 4 KiB layout";
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(CreateFormatPageLayoutIndex(path_, 4096));
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));

  ASSERT_EQ(0, writer_->append("filler", page_size_));
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  std::string padding_name(
      root.free_bytes - sizeof(IndexFormat::SegmentMeta) - 1, 'x');
  ASSERT_EQ(0, writer_->append(padding_name, page_size_));
  const std::string name = "new_segment";
  ASSERT_EQ(0, writer_->append(name, page_size_));
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const uint64_t tail_offset = root.footer.next_meta_header_offset;
  ASSERT_NE(0u, tail_offset);
  const size_t data_offset = tail_offset + 4096;

  auto filler = writer_->get("filler");
  ASSERT_NE(nullptr, filler);
  // Page 0 shares the root directory; filler data on a directory-free page.
  ASSERT_EQ(kOldData.size(),
            filler->write(0, kOldData.data(), kOldData.size()));
  ASSERT_EQ(kNewData.size(),
            filler->write(8192, kNewData.data(), kNewData.size()));
  auto segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  ASSERT_EQ(kNewData.size(),
            segment->write(0, kNewData.data(), kNewData.size()));

  // The second data range of the split chain's shared page is short-written.
  short_data_hit = false;
  short_data_prefix = 4096;
  short_data_offset = static_cast<off_t>(data_offset);
  EXPECT_NE(0, writer_->vec_buffer_pool()->flush_all());
  short_data_offset = -1;
  EXPECT_TRUE(short_data_hit);
  EXPECT_TRUE(writer_->is_dirty());

  // The retry persists both ranges; every directory and payload is intact.
  ASSERT_EQ(0, writer_->vec_buffer_pool()->flush_all());
  ASSERT_EQ(0, writer_->close());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, name, kNewData));
  auto reopened = reader->get("filler");
  ASSERT_NE(nullptr, reopened);
  std::array<char, 5> data{};
  ASSERT_EQ(data.size(), reopened->fetch(8192, data.data(), data.size()));
  EXPECT_EQ(kNewData, data);
  ASSERT_EQ(0, reader->close());
  Directory tail{};
  EXPECT_TRUE(ReadDirectory(path_, &tail, tail_offset));
}

TEST_F(BufferStorageDirectorySplitTest,
       DataWritebackExceptionReleasesPagePins) {
  const std::string name_a = "seg_a";
  const std::string name_b = "seg_b";
  ASSERT_EQ(0, writer_->append(name_a, page_size_));
  ASSERT_EQ(0, writer_->append(name_b, page_size_));
  auto seg_a = writer_->get(name_a);
  auto seg_b = writer_->get(name_b);
  ASSERT_NE(nullptr, seg_a);
  ASSERT_NE(nullptr, seg_b);
  ASSERT_EQ(kOldData.size(), seg_a->write(0, kOldData.data(), kOldData.size()));
  ASSERT_EQ(kNewData.size(), seg_b->write(0, kNewData.data(), kNewData.size()));
  auto &pool = *writer_->vec_buffer_pool();
  const size_t page_a = seg_a->data_offset() / page_size_;
  const size_t page_b = seg_b->data_offset() / page_size_;

  // An exception escaping the ordinary run's pwrite must still release every
  // pin the run acquired; otherwise close() trips the residency assert.
  throw_data_offset = static_cast<off_t>(seg_a->data_offset());
  EXPECT_THROW(pool.flush_all(), std::bad_alloc);
  EXPECT_TRUE(pool.page_table_.is_released(page_a)) << "page pin leaked";
  EXPECT_TRUE(pool.page_table_.is_released(page_b)) << "page pin leaked";

  // The dirty pages survive and the retry persists them.
  ASSERT_EQ(0, pool.flush_all());
  ASSERT_EQ(0, writer_->close());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, name_a, kOldData));
  ASSERT_NO_FATAL_FAILURE(verify_segment(*reader, name_b, kNewData));
  ASSERT_EQ(0, reader->close());
}

TEST_F(BufferStorageDirectorySplitTest, PartialWriteKeepsRecordCursor) {
  const std::string name = "record";
  ASSERT_EQ(0, writer_->append(name, 2 * page_size_));
  auto segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  std::vector<char> prefix(page_size_ - 2, 'a');
  ASSERT_EQ(prefix.size(), segment->write(0, prefix.data(), prefix.size()));
  // Reopen writably so the second page is loaded by a real pread instead of
  // the new-page zero-fill path (which would skip the injected read fault).
  ASSERT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  const size_t cursor = segment->data_size();

  // The next record crosses into a page whose load fails. Only its prefix
  // reaches the cache; the record cursor must not advance.
  std::vector<char> record(16, 'b');
  failed_read_hit = false;
  failed_read_offset = static_cast<off_t>(segment->data_offset() + page_size_);
  const size_t written = segment->write(cursor, record.data(), record.size());
  failed_read_offset = -1;
  ASSERT_TRUE(failed_read_hit);
  EXPECT_LT(written, record.size());
  EXPECT_EQ(cursor, segment->data_size());

  // The retry starts at the unchanged cursor and publishes the whole record.
  ASSERT_EQ(record.size(),
            segment->write(cursor, record.data(), record.size()));
  EXPECT_EQ(cursor + record.size(), segment->data_size());
  std::vector<char> actual(record.size(), '\0');
  ASSERT_EQ(record.size(),
            segment->fetch(cursor, actual.data(), actual.size()));
  EXPECT_EQ(record, actual);
}

TEST_F(BufferStorageDirectorySplitTest, SharedTailPageWritebackKeepsFileSize) {
  if (page_size_ <= 4096) {
    GTEST_SKIP() << "requires a host page larger than the 4 KiB layout";
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(CreateFormatPageLayoutIndex(path_, 4096));
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  auto version = writer_->get(INDEX_VERSION_SEGMENT_NAME);
  ASSERT_NE(nullptr, version);
  ASSERT_EQ(kNewData.size(),
            version->write(0, kNewData.data(), kNewData.size()));
  ASSERT_EQ(0, writer_->vec_buffer_pool()->flush_all());
  struct stat info {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &info));
  EXPECT_EQ(8192, info.st_size);
  EXPECT_EQ(8192u, writer_->vec_buffer_pool()->file_size());
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       OrdinaryTailPageWritebackKeepsFileSize) {
  if (page_size_ <= 4096) {
    GTEST_SKIP() << "requires a host page larger than the 4 KiB layout";
  }
  ASSERT_EQ(0, writer_->close());
  ASSERT_NO_FATAL_FAILURE(CreateFormatPageLayoutIndex(path_, 4096));
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  ASSERT_EQ(0, writer_->append("filler", page_size_));
  ASSERT_EQ(0, writer_->close());
  // Reopening truncates the unused aligned tail to the format's real end.
  ASSERT_EQ(0, writer_->open(path_, IndexStorage::OpenOptions{false, false}));
  auto filler = writer_->get("filler");
  ASSERT_NE(nullptr, filler);
  ASSERT_EQ(kOldData.size(),
            filler->write(0, kOldData.data(), kOldData.size()));
  ASSERT_EQ(0, writer_->vec_buffer_pool()->flush_all());
  struct stat info {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &info));
  EXPECT_EQ(24576, info.st_size);
  EXPECT_EQ(24576u, writer_->vec_buffer_pool()->file_size());
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       FooterShortWriteRestoresOldCheckpointAfterRetry) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  writer_->refresh(kCheckpoint);
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  // The first write persists only a prefix (which still covers the new
  // checkpoint). The first restore fails; the retry succeeds.
  ShortFooterWrite fault{static_cast<off_t>(root.footer_offset),
                         kShortFooterBytes};
  short_footer_write = &fault;
  ScopedWriteFailure restore_fault(root.footer_offset);
  const int status = writer_->flush();
  short_footer_write = nullptr;
  ASSERT_EQ(IndexError_WriteData, status);
  ASSERT_TRUE(restore_fault.hit);

  // The restored old footer keeps the previous checkpoint readable.
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  EXPECT_EQ(0u, reader->check_point());
  ASSERT_EQ(0, reader->close());

  writer_->refresh(kCheckpoint + 1);
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(kCheckpoint + 1, writer_->check_point());
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       FooterShortWriteAndFailedRestoreMarksIncomplete) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  writer_->refresh(kCheckpoint);
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  ShortFooterWrite fault{static_cast<off_t>(root.footer_offset),
                         kShortFooterBytes};
  short_footer_write = &fault;
  // Both restore attempts fail: the outcome is reported as incomplete and a
  // later writeback reconciles the footer instead of hiding the split state.
  repeated_fail_offset = static_cast<off_t>(root.footer_offset);
  repeated_fail_remaining = 2;
  const int status = writer_->flush();
  short_footer_write = nullptr;
  repeated_fail_offset = -1;
  ASSERT_EQ(IndexError_WriteData, status);
  EXPECT_TRUE(writer_->is_dirty());
  ASSERT_EQ(0, writer_->flush());
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  ASSERT_EQ(0, writer_->close());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  EXPECT_EQ(kCheckpoint, reader->check_point());
  ASSERT_EQ(0, reader->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       FailedLinkAndRollbackReclaimsGrownTail) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  struct stat before {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &before));
  const auto original_eof = before.st_size;
  const std::string name = "new_split";
  // Both the link write and the rollback restore fail.
  repeated_fail_offset = static_cast<off_t>(root.footer_offset);
  repeated_fail_remaining = 2;
  ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
  repeated_fail_offset = -1;
  EXPECT_TRUE(writer_->is_dirty());
  // The repaired link lets a successful flush reclaim the grown tail.
  ASSERT_EQ(0, writer_->flush());
  struct stat after {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &after));
  EXPECT_EQ(original_eof, after.st_size);
  // The same segment can be appended again afterwards.
  ASSERT_EQ(0, writer_->append(name, page_size_));
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest, RollbackBookkeepingDoesNotAllocate) {
  // Grow the rollback reserve once on a fallible path, then force an
  // allocation failure: the rollback's push_back must not allocate.
  ASSERT_EQ(0, writer_->append("warm", page_size_));
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const std::string name = "new_split";
  const zvec::ailego::BudgetedAllocator<size_t> allocator;
  const size_t max_slots = std::numeric_limits<size_t>::max() / sizeof(size_t);
  const size_t allocator_overhead = (max_slots - allocator.max_size()) * sizeof(size_t);
  fault_hit = false;
  fail_aligned_size = sizeof(size_t) + allocator_overhead;
  repeated_fail_offset = static_cast<off_t>(root.footer_offset);
  repeated_fail_remaining = 2;
  ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
  repeated_fail_offset = -1;
  fail_aligned_size = 0;
  EXPECT_FALSE(fault_hit);
  ASSERT_EQ(0, writer_->flush());
  ASSERT_EQ(0, writer_->close());
}

TEST_F(BufferStorageDirectorySplitTest, WritebackRunReleasesLatchBetweenRuns) {
  auto &pool = *writer_->vec_buffer_pool();
  const size_t page_a = 1;
  const size_t page_b = page_a + zvec::ailego::VecBufferPool::kMutexBucketCount;
  ASSERT_TRUE(pool.extend_file((page_b + 1) * page_size_));
  // Two non-contiguous dirty pages map to the same stripe.
  ASSERT_EQ(0, pool.write_range(page_a * page_size_, 5, "aaaaa"));
  ASSERT_EQ(0, pool.write_range(page_b * page_size_, 5, "bbbbb"));

  PausedDataWrite pause;
  pause.offset = page_a * page_size_;
  paused_data_write = &pause;
  std::atomic<bool> flush_done{false};
  std::thread flusher([&] {
#if defined(__APPLE__) && TARGET_OS_OSX
    zvec::test::SetDirectorySplitPwriteHook(InjectPwrite);
#endif
    paused_data_write = &pause;
    EXPECT_EQ(0, pool.flush_all());
    flush_done.store(true, std::memory_order_release);
  });
  {
    std::unique_lock<std::mutex> lock(pause.mutex);
    ASSERT_TRUE(pause.cv.wait_for(lock, std::chrono::seconds(5),
                                  [&] { return pause.reached; }));
  }
  // A writer on the same stripe now queues for the exclusive latch.
  std::atomic<bool> write_done{false};
  std::thread writer([&] {
    EXPECT_EQ(0, pool.write_range(page_b * page_size_, 5, "ccccc"));
    write_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  {
    std::lock_guard<std::mutex> lock(pause.mutex);
    pause.resume = true;
    pause.cv.notify_all();
  }
  paused_data_write = nullptr;
  // The first run must release its latch before the next run re-acquires
  // the same stripe, otherwise flusher and writer deadlock each other.
  flusher.join();
  writer.join();
  EXPECT_TRUE(flush_done.load(std::memory_order_acquire));
  EXPECT_TRUE(write_done.load(std::memory_order_acquire));
}

TEST_F(BufferStorageDirectorySplitTest,
       FailedLinkRollbackThenSuccessfulRetryKeepsCommittedTail) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  const std::string name = "new_split";
  // Only the link write fails; the rollback restore succeeds.
  repeated_fail_offset = static_cast<off_t>(root.footer_offset);
  repeated_fail_remaining = 1;
  ASSERT_EQ(IndexError_WriteData, writer_->append(name, page_size_));
  repeated_fail_offset = -1;
  // A committed retry makes that record obsolete: a later flush must not
  // truncate the freshly committed chain.
  ASSERT_EQ(0, writer_->append(name, page_size_));
  struct stat committed {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &committed));
  // Do NOT write segment data: the new chain's page stays unloaded, so the
  // stale truncation record can actually discard it if left uncleared.
  writer_->refresh(0);
  ASSERT_EQ(0, writer_->flush());
  struct stat after {};
  ASSERT_EQ(0, ::stat(path_.c_str(), &after));
  EXPECT_EQ(committed.st_size, after.st_size);
  ASSERT_EQ(0, writer_->close());
  auto reader = IndexFactory::CreateStorage("BufferStorage");
  ASSERT_NE(nullptr, reader);
  ASSERT_EQ(0, reader->init(zvec::ailego::Params{}));
  ASSERT_EQ(0, reader->open(path_, false));
  EXPECT_TRUE(reader->has(name));
  ASSERT_EQ(0, reader->close());
}

TEST_F(BufferStorageDirectorySplitTest,
       AppendAllocationFailureLeavesNoPendingSegment) {
  const std::string name = "pending_segment";
  // Fail allocation of the rollback reserve before publishing the segment.
  fault_hit = false;
  fail_next_allocation = true;
  EXPECT_EQ(IndexError_NoMemory, writer_->append(name, page_size_));
  fail_next_allocation = false;
  EXPECT_TRUE(fault_hit);
  EXPECT_FALSE(writer_->has(name));
  EXPECT_EQ(nullptr, writer_->get(name));
  // The same name can be appended again (no Duplicate, no dangling meta).
  ASSERT_EQ(0, writer_->append(name, page_size_));
  EXPECT_TRUE(writer_->has(name));
  auto segment = writer_->get(name);
  ASSERT_NE(nullptr, segment);
  ASSERT_EQ(kNewData.size(),
            segment->write(0, kNewData.data(), kNewData.size()));
  ASSERT_EQ(0, writer_->close());
}

#endif

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(BufferStorageDirectorySplitTest,
       FailedMapAllocationCanRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  const std::string name = "retry_map_allocation";
  fault_hit = false;
  fail_next_allocation = true;
  const int status = writer_->append(name, page_size_);
  fail_next_allocation = false;
  EXPECT_TRUE(fault_hit);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

TEST_F(BufferStorageDirectorySplitTest, HnswBrokerCanRetryChunkAllocation) {
  IndexStreamer::Stats stats;
  ChunkBroker broker(stats);
  uint32_t chunk_size = page_size_;
  ASSERT_EQ(0, broker.open(writer_, chunk_size, false));
  broker.set_max_chunks_size(16 * page_size_);
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());

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
  ASSERT_NO_FATAL_FAILURE(verify_segment(*writer_, old_name_, kOldData));

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
  ASSERT_NO_FATAL_FAILURE(fill_directory());
  ASSERT_NO_FATAL_FAILURE(write_old_segment());

  constexpr int kType = ChunkBroker::CHUNK_TYPE_NODE;
  auto [status, chunk] = broker.alloc_chunk(kType, 0);
  ASSERT_EQ(0, status);
  ASSERT_NE(nullptr, chunk);
  ASSERT_EQ(kOldData.size(), chunk->write(0, kOldData.data(), kOldData.size()));
  chunk.reset();
  ASSERT_EQ(0, writer_->flush());
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

#endif

}  // namespace
