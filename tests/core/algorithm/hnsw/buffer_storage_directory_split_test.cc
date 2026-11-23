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
#include <atomic>
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
#include <thread>
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

TEST_F(BufferStorageDirectorySplitTest,
       SegmentCapacitySurvivesConcurrentResize) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  auto first = writer_->get(old_name_);
  auto second = writer_->get(old_name_);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  ASSERT_NE(first, second);
  std::atomic<bool> stop{false};
  std::atomic<bool> started{false};
  std::thread resize([&]() {
    started.store(true, std::memory_order_release);
    for (size_t i = 0; !stop.load(std::memory_order_relaxed); ++i) {
      first->resize(i % 2 == 0 ? 10 : 1000);
    }
  });
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  size_t observed_capacity = page_size_;
  for (size_t i = 0; i < 50000; ++i) {
    second->resize(i % 2 == 0 ? 20 : 2000);
    observed_capacity = writer_->get(old_name_)->capacity();
    if (observed_capacity != page_size_) break;
  }
  stop.store(true, std::memory_order_relaxed);
  resize.join();
  EXPECT_EQ(page_size_, observed_capacity);
  EXPECT_EQ(page_size_, first->data_size() + first->padding_size());
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
  EXPECT_EQ(0, writer_->close());
  ASSERT_EQ(0, writer_->open(path_, false));
  EXPECT_EQ(kCheckpoint, writer_->check_point());
  ASSERT_TRUE(writer_->has(tail));
  auto old = writer_->get(old_name_);
  ASSERT_NE(nullptr, old);
  EXPECT_EQ(kOldData.size() + kNewData.size(), old->data_size());
}

TEST_F(BufferStorageDirectorySplitTest, ConcurrentFlushWaitsForWriteback) {
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
  EXPECT_TRUE(writer_->is_dirty());
  std::promise<void> started;
  auto second = std::async(std::launch::async, [&]() {
    started.set_value();
    return writer_->flush();
  });
  started.get_future().wait();
  const auto second_status = second.wait_for(100ms);
  {
    std::lock_guard<std::mutex> lock(pause.mutex);
    pause.resume = true;
    pause.cv.notify_all();
  }
  EXPECT_EQ(std::future_status::timeout, second_status);
  EXPECT_EQ(IndexError_WriteData, first.get());
  EXPECT_EQ(0, second.get());
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

TEST_F(BufferStorageDirectorySplitTest,
       ClosePersistsRequestsAcceptedBeforeClose) {
  ASSERT_NO_FATAL_FAILURE(append_old_segment());
  Directory root{};
  ASSERT_TRUE(ReadDirectory(path_, &root));
  writer_->refresh(kCheckpoint);
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

#endif

#if defined(__linux__) && !defined(__ANDROID__)
TEST_F(BufferStorageDirectorySplitTest,
       FailedMapAllocationCanRetrySameSegment) {
  ASSERT_NO_FATAL_FAILURE(fill_first_chain());
  const std::string name = "retry_map_allocation";
  fault_hit = false;
  fail_next_scalar_allocation = true;
  const int status = writer_->append(name, page_size_);
  fail_next_scalar_allocation = false;
  EXPECT_TRUE(fault_hit);
  ASSERT_EQ(IndexError_NoMemory, status);
  ASSERT_NO_FATAL_FAILURE(verify_retry(name));
}

#endif

}  // namespace
