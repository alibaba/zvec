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

#include <malloc.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/io/file.h>
#include <zvec/ailego/utility/file_helper.h>
#include "diskann_file_reader.h"

namespace zvec {
namespace core {

class WindowsAlignedFileReaderTestPeer {
 public:
  static HANDLE stable_file_handle(const WindowsAlignedFileReader &reader) {
    return reader.stable_file_handle_;
  }
};

}  // namespace core
}  // namespace zvec

using namespace zvec::core;

namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kPageCount = 256;

uint8_t page_value(size_t page, uint8_t bias = 0) {
  return static_cast<uint8_t>((page * 37 + 11 + bias) & 0xff);
}

class TemporaryFile {
 public:
  explicit TemporaryFile(bool unicode_path = false) {
    wchar_t temp_dir[MAX_PATH]{};
    DWORD length = ::GetTempPathW(MAX_PATH, temp_dir);
    if (length == 0 || length >= static_cast<DWORD>(MAX_PATH) ||
        ::GetTempFileNameW(temp_dir, L"zvr", 0, wide_path_) == 0) {
      wide_path_[0] = L'\0';
      return;
    }

    if (unicode_path) {
      std::wstring unicode_wide_path = wide_path_;
      unicode_wide_path += L"_磁盘索引_テスト";
      if (unicode_wide_path.size() >= static_cast<size_t>(MAX_PATH) ||
          !::DeleteFileW(wide_path_)) {
        return;
      }
      std::copy(unicode_wide_path.begin(), unicode_wide_path.end(), wide_path_);
      wide_path_[unicode_wide_path.size()] = L'\0';
    }
    path_ = zvec::ailego::FileHelper::WideToUtf8(wide_path_);
  }

  ~TemporaryFile() {
    if (wide_path_[0] != L'\0') {
      ::DeleteFileW(wide_path_);
    }
  }

  bool valid() const {
    return wide_path_[0] != L'\0' && !path_.empty();
  }

  const char *path() const {
    return path_.c_str();
  }

  const wchar_t *wide_path() const {
    return wide_path_;
  }

  bool write_pages(uint8_t bias = 0) const {
    if (!valid()) {
      return false;
    }

    HANDLE file = ::CreateFileW(wide_path_, GENERIC_WRITE, 0, nullptr,
                                CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      return false;
    }

    std::vector<uint8_t> contents(kPageSize * kPageCount);
    for (size_t page = 0; page < kPageCount; ++page) {
      std::memset(contents.data() + page * kPageSize, page_value(page, bias),
                  kPageSize);
    }

    DWORD written = 0;
    BOOL result =
        ::WriteFile(file, contents.data(), static_cast<DWORD>(contents.size()),
                    &written, nullptr);
    bool success = result && written == static_cast<DWORD>(contents.size()) &&
                   ::FlushFileBuffers(file);
    ::CloseHandle(file);
    return success;
  }

 private:
  wchar_t wide_path_[MAX_PATH]{};
  std::string path_;
};

struct AlignedFree {
  void operator()(uint8_t *buffer) const {
    ::_aligned_free(buffer);
  }
};

using AlignedBuffer = std::unique_ptr<uint8_t, AlignedFree>;

AlignedBuffer make_aligned_buffer(size_t size) {
  auto *buffer = static_cast<uint8_t *>(::_aligned_malloc(size, kPageSize));
  if (buffer != nullptr) {
    std::memset(buffer, 0, size);
  }
  return AlignedBuffer(buffer);
}

bool verify_page(const uint8_t *buffer, size_t page, uint8_t bias = 0) {
  return std::all_of(buffer, buffer + kPageSize, [page, bias](uint8_t value) {
    return value == page_value(page, bias);
  });
}

DWORD replace_open_file_atomically(const wchar_t *replacement_path,
                                    const wchar_t *target_path) {
  // diskann_file_reader.h targets Windows Vista, whose SDK view predates the
  // extended rename declarations. These values and this layout are the Win10
  // FILE_RENAME_INFO_EX ABI used by FileRenameInfoEx.
  constexpr auto kFileRenameInfoEx =
      static_cast<FILE_INFO_BY_HANDLE_CLASS>(22);
  constexpr DWORD kReplaceIfExists = 0x00000001;
  constexpr DWORD kPosixSemantics = 0x00000002;
  struct ExtendedFileRenameInfo {
    DWORD flags;
    HANDLE root_directory;
    DWORD file_name_length;
    WCHAR file_name[MAX_PATH];
  };
  static_assert(offsetof(ExtendedFileRenameInfo, root_directory) ==
                offsetof(FILE_RENAME_INFO, RootDirectory));
  static_assert(offsetof(ExtendedFileRenameInfo, file_name_length) ==
                offsetof(FILE_RENAME_INFO, FileNameLength));
  static_assert(offsetof(ExtendedFileRenameInfo, file_name) ==
                offsetof(FILE_RENAME_INFO, FileName));

  HANDLE replacement_handle = ::CreateFileW(
      replacement_path, DELETE | SYNCHRONIZE,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (replacement_handle == INVALID_HANDLE_VALUE) {
    return ::GetLastError();
  }

  const size_t target_path_length = std::wcslen(target_path);
  if (target_path_length >= MAX_PATH) {
    ::CloseHandle(replacement_handle);
    return ERROR_FILENAME_EXCED_RANGE;
  }

  const size_t target_path_bytes =
      target_path_length * sizeof(target_path[0]);
  ExtendedFileRenameInfo rename_info{};
  rename_info.flags = kReplaceIfExists | kPosixSemantics;
  rename_info.root_directory = nullptr;
  rename_info.file_name_length = static_cast<DWORD>(target_path_bytes);
  std::memcpy(rename_info.file_name, target_path, target_path_bytes);

  // POSIX rename semantics keep existing handles bound to the old file object
  // while new opens of the target path observe the replacement.
  const BOOL renamed = ::SetFileInformationByHandle(
      replacement_handle, kFileRenameInfoEx, &rename_info,
      static_cast<DWORD>(sizeof(rename_info)));
  const DWORD error = renamed ? ERROR_SUCCESS : ::GetLastError();
  ::CloseHandle(replacement_handle);
  return error;
}

DWORD issue_misaligned_read(HANDLE handle) {
  LARGE_INTEGER offset{};
  if (!::SetFilePointerEx(handle, offset, nullptr, FILE_BEGIN)) {
    return ::GetLastError();
  }
  uint8_t byte = 0;
  DWORD bytes_read = 0;
  ::SetLastError(ERROR_SUCCESS);
  if (::ReadFile(handle, &byte, 1, &bytes_read, nullptr)) {
    return ERROR_SUCCESS;
  }
  return ::GetLastError();
}

class ScopedCurrentDirectory {
 public:
  ScopedCurrentDirectory() {
    const DWORD capacity = ::GetCurrentDirectoryW(0, nullptr);
    if (capacity == 0) {
      return;
    }
    original_.resize(capacity, L'\0');
    const DWORD length =
        ::GetCurrentDirectoryW(capacity, original_.data());
    if (length == 0 || length >= capacity) {
      original_.clear();
      return;
    }
    original_.resize(length);
  }

  ~ScopedCurrentDirectory() {
    restore();
  }

  bool valid() const {
    return !original_.empty();
  }

  bool change_to(const std::wstring &directory) {
    return ::SetCurrentDirectoryW(directory.c_str()) != FALSE;
  }

  bool restore() {
    if (original_.empty()) {
      return false;
    }
    return ::SetCurrentDirectoryW(original_.c_str()) != FALSE;
  }

 private:
  std::wstring original_;
};

}  // namespace

TEST(DiskAnnFileReaderWindowsTest, OpenKeepsStableHandleUnbuffered) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());

  HANDLE stable_handle =
      WindowsAlignedFileReaderTestPeer::stable_file_handle(reader);
  ASSERT_NE(stable_handle, INVALID_HANDLE_VALUE);
  EXPECT_EQ(issue_misaligned_read(stable_handle), ERROR_INVALID_PARAMETER);
}

TEST(DiskAnnFileReaderWindowsTest, OpenSupportsUtf8Path) {
  TemporaryFile file(/*unicode_path=*/true);
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  // Exercise the UTF-8 to UTF-16 conversion used by open() before CreateFileW.
  WindowsAlignedFileReader reader;
  reader.open(file.path());

  HANDLE stable_handle =
      WindowsAlignedFileReaderTestPeer::stable_file_handle(reader);
  ASSERT_NE(stable_handle, INVALID_HANDLE_VALUE);

  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> request{{0, kPageSize, output.get()}};
  EXPECT_EQ(reader.read(request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest,
     RelativePathSurvivesWorkingDirectoryChange) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  const std::wstring full_path(file.wide_path());
  const size_t separator = full_path.find_last_of(L"\\/");
  ASSERT_NE(separator, std::wstring::npos);
  const std::wstring directory = full_path.substr(0, separator);
  const std::wstring filename = full_path.substr(separator + 1);

  ScopedCurrentDirectory current_directory;
  ASSERT_TRUE(current_directory.valid());
  ASSERT_TRUE(current_directory.change_to(directory));

  WindowsAlignedFileReader reader;
  reader.open(zvec::ailego::FileHelper::WideToUtf8(filename));

  // prepare_io_ctx() opens the actual IOCP handle.  Moving away from the
  // directory used by open() must not change which file it resolves.
  ASSERT_TRUE(current_directory.restore());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> request{{0, kPageSize, output.get()}};
  EXPECT_EQ(reader.read(request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest, ConcurrentContextsKeepCompletionsIsolated) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kReadsPerThread = 64;

  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());

  std::atomic<size_t> ready{0};
  std::atomic<bool> start{false};
  std::atomic<size_t> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreadCount);

  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    workers.emplace_back([&, thread_index]() {
      bool success = true;
      IOContext ctx = nullptr;
      if (setup_io_ctx(ctx) != 0 || ctx == nullptr) {
        success = false;
      }

      AlignedBuffer output = make_aligned_buffer(kPageSize * kReadsPerThread);
      if (output == nullptr) {
        success = false;
      }

      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      if (success) {
        std::vector<AlignedRead> requests;
        std::vector<size_t> source_pages;
        requests.reserve(kReadsPerThread);
        source_pages.reserve(kReadsPerThread);
        for (size_t i = 0; i < kReadsPerThread; ++i) {
          const size_t source_page = (thread_index * 13 + i * 7) % kPageCount;
          source_pages.push_back(source_page);
          requests.emplace_back(source_page * kPageSize, kPageSize,
                                output.get() + i * kPageSize);
        }

        PendingBatch batch;
        if (reader.submit(batch, requests, ctx) != 0 || batch.n_reaped != 0 ||
            ctx->outstanding_count != batch.n_submitted) {
          success = false;
        }

        std::vector<uint8_t> seen(kReadsPerThread, 0);
        while (success && batch.n_reaped < batch.n_submitted) {
          std::vector<uint32_t> completed;
          int count = reader.get_completed(batch, ctx, 1, completed);
          if (count <= 0 || static_cast<size_t>(count) != completed.size()) {
            success = false;
            break;
          }
          for (uint32_t index : completed) {
            if (index >= seen.size() || seen[index] != 0 ||
                !verify_page(output.get() + index * kPageSize,
                             source_pages[index])) {
              success = false;
              break;
            }
            seen[index] = 1;
          }
        }

        if (success && !std::all_of(seen.begin(), seen.end(),
                                    [](uint8_t value) { return value == 1; })) {
          success = false;
        }
      }

      if (destroy_io_ctx(ctx) != 0 || ctx != nullptr) {
        success = false;
      }
      if (!success) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (std::thread &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(failures.load(), 0U);
}

TEST(DiskAnnFileReaderWindowsTest, ContextCanMoveBetweenRunnableThreads) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer first_output = make_aligned_buffer(kPageSize);
  AlignedBuffer second_output = make_aligned_buffer(kPageSize);
  ASSERT_NE(first_output, nullptr);
  ASSERT_NE(second_output, nullptr);

  std::atomic<bool> first_completed{false};
  std::atomic<bool> release_first{false};
  std::atomic<bool> second_completed{false};
  int first_result = IndexError_Runtime;
  int second_result = IndexError_Runtime;

  std::thread first([&]() {
    std::vector<AlignedRead> requests{{0, kPageSize, first_output.get()}};
    first_result = reader.read(requests, ctx, false);
    first_completed.store(true, std::memory_order_release);

    // Stay runnable after dequeuing from the IOCP. With a port concurrency of
    // one this thread occupies the only slot even though its batch is done.
    while (!release_first.load(std::memory_order_acquire)) {
      YieldProcessor();
    }
  });

  while (!first_completed.load(std::memory_order_acquire)) {
    ::Sleep(1);
  }

  std::thread second([&]() {
    std::vector<AlignedRead> requests{
        {kPageSize, kPageSize, second_output.get()}};
    second_result = reader.read(requests, ctx, false);
    second_completed.store(true, std::memory_order_release);
  });

  constexpr ULONGLONG kHandoffTimeoutMs = 5000;
  const ULONGLONG deadline = ::GetTickCount64() + kHandoffTimeoutMs;
  while (!second_completed.load(std::memory_order_acquire) &&
         ::GetTickCount64() < deadline) {
    ::Sleep(1);
  }
  const bool completed_while_first_runnable =
      second_completed.load(std::memory_order_acquire);

  // Always let the first thread exit before asserting. The old concurrency=1
  // behavior then releases its IOCP association, allowing the second thread
  // to finish instead of leaving a permanently hung regression test.
  release_first.store(true, std::memory_order_release);
  first.join();
  second.join();

  EXPECT_TRUE(completed_while_first_runnable);
  EXPECT_EQ(first_result, 0);
  EXPECT_EQ(second_result, 0);
  EXPECT_TRUE(verify_page(first_output.get(), 0));
  EXPECT_TRUE(verify_page(second_output.get(), 1));
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest, DestroyContextDrainsOutstandingBatch) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize * MAX_IO_DEPTH);
  ASSERT_NE(output, nullptr);

  std::vector<AlignedRead> requests;
  requests.reserve(MAX_IO_DEPTH);
  for (size_t i = 0; i < MAX_IO_DEPTH; ++i) {
    // A zero-byte ReadFile normally completes synchronously. Because the
    // reader does not opt into FILE_SKIP_COMPLETION_PORT_ON_SUCCESS, teardown
    // must still dequeue its completion packet along with pending reads.
    const size_t length = i == 0 ? 0 : kPageSize;
    requests.emplace_back(i * kPageSize, length,
                          output.get() + i * kPageSize);
  }

  PendingBatch batch;
  ASSERT_EQ(reader.submit(batch, requests, ctx), 0);
  ASSERT_EQ(ctx->outstanding_count, batch.n_submitted);

  // This must cancel and reap the batch before it destroys the OVERLAPPED
  // slots or lets the caller release output.
  ASSERT_EQ(destroy_io_ctx(ctx), 0);
  ASSERT_EQ(ctx, nullptr);

  IOContext replacement_ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(replacement_ctx), 0);
  ASSERT_NE(replacement_ctx, nullptr);
  std::vector<AlignedRead> one_read{{0, kPageSize, output.get()}};
  ASSERT_EQ(reader.read(one_read, replacement_ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));
  EXPECT_EQ(destroy_io_ctx(replacement_ctx), 0);
  EXPECT_EQ(replacement_ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest, RejectsMisalignedUnbufferedRead) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize * 2);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> requests{{0, kPageSize, output.get() + 1}};

  PendingBatch batch;
  EXPECT_EQ(reader.submit(batch, requests, ctx), IndexError_InvalidArgument);
  EXPECT_EQ(ctx->outstanding_count, 0U);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest,
     ReleaseCompletedContextDropsFileHandleAndCanReadAgain) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> request{{0, kPageSize, output.get()}};
  ASSERT_EQ(reader.read(request, ctx, false), 0);
  EXPECT_NE(ctx->file_handle, INVALID_HANDLE_VALUE);
  EXPECT_NE(ctx->completion_port, nullptr);
  reader.release_io_ctx(ctx);
  EXPECT_EQ(ctx->file_handle, INVALID_HANDLE_VALUE);
  EXPECT_EQ(ctx->completion_port, nullptr);
  EXPECT_EQ(ctx->outstanding_count, 0U);

  // Search contexts may outlive the reader because the high-level context pool
  // retains them. Releasing at the complete operation boundary must drop the
  // private context handle. The reader's stable handle still owns the old
  // contents and can lazily prepare this same context again after deletion.
  ASSERT_TRUE(::DeleteFileW(file.wide_path()));
  EXPECT_EQ(reader.read(request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));
  reader.release_io_ctx(ctx);
  EXPECT_EQ(ctx->file_handle, INVALID_HANDLE_VALUE);
  EXPECT_EQ(ctx->completion_port, nullptr);
  EXPECT_EQ(ctx->outstanding_count, 0U);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest,
     OpenFromHandleSurvivesPathReplacementBeforeHandoff) {
  constexpr uint8_t kReplacementBias = 97;

  TemporaryFile original;
  TemporaryFile replacement;
  ASSERT_TRUE(original.valid());
  ASSERT_TRUE(replacement.valid());
  ASSERT_TRUE(original.write_pages());
  ASSERT_TRUE(replacement.write_pages(kReplacementBias));

  // This buffered handle represents FileReadStorage, which has already
  // supplied metadata from the original file.
  zvec::ailego::File source;
  ASSERT_TRUE(source.open(original.path(), true, false));

  ASSERT_EQ(replace_open_file_atomically(replacement.wide_path(),
                                         original.wide_path()),
            ERROR_SUCCESS);

  WindowsAlignedFileReader original_reader;
  ASSERT_EQ(original_reader.open_from_handle(original.path(),
                                             source.native_handle()),
            0);
  HANDLE stable_handle =
      WindowsAlignedFileReaderTestPeer::stable_file_handle(original_reader);
  ASSERT_NE(stable_handle, INVALID_HANDLE_VALUE);
  EXPECT_EQ(issue_misaligned_read(stable_handle), ERROR_INVALID_PARAMETER);
  source.close();

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  std::vector<AlignedRead> request{{0, kPageSize, output.get()}};
  ASSERT_EQ(original_reader.read(request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));

  WindowsAlignedFileReader replacement_reader;
  replacement_reader.open(original.path());
  ASSERT_EQ(replacement_reader.read(request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0, kReplacementBias));

  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest,
     ReusedContextTracksFileObjectAfterAtomicReplacement) {
  constexpr uint8_t kReplacementBias = 83;

  TemporaryFile original;
  TemporaryFile replacement;
  ASSERT_TRUE(original.valid());
  ASSERT_TRUE(replacement.valid());
  ASSERT_TRUE(original.write_pages());
  ASSERT_TRUE(replacement.write_pages(kReplacementBias));

  WindowsAlignedFileReader original_reader;
  original_reader.open(original.path());
  IOContext shared_ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(shared_ctx), 0);
  ASSERT_NE(shared_ctx, nullptr);

  // The context has not performed I/O yet. Replacing the path must not make
  // its first lazy read observe bytes from a different file object.
  ASSERT_EQ(replace_open_file_atomically(replacement.wide_path(),
                                         original.wide_path()),
            ERROR_SUCCESS);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> request{{0, kPageSize, output.get()}};
  ASSERT_EQ(original_reader.read(request, shared_ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));

  WindowsAlignedFileReader replacement_reader;
  replacement_reader.open(original.path());
  // Reuse the context that is currently bound to original_reader. The path is
  // unchanged, so the readers' file identities must force an IOCP rebind.
  ASSERT_EQ(replacement_reader.read(request, shared_ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0, kReplacementBias));

  // Switching back must likewise restore the original file object rather than
  // reuse the replacement reader's handle solely because the paths match.
  ASSERT_EQ(original_reader.read(request, shared_ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));

  EXPECT_EQ(destroy_io_ctx(shared_ctx), 0);
  EXPECT_EQ(shared_ctx, nullptr);
}

TEST(DiskAnnFileReaderWindowsTest, ShortReadResetsContextForNextBatch) {
  TemporaryFile file;
  ASSERT_TRUE(file.valid());
  ASSERT_TRUE(file.write_pages());

  WindowsAlignedFileReader reader;
  reader.open(file.path());
  IOContext ctx = nullptr;
  ASSERT_EQ(setup_io_ctx(ctx), 0);
  ASSERT_NE(ctx, nullptr);

  AlignedBuffer output = make_aligned_buffer(kPageSize);
  ASSERT_NE(output, nullptr);
  std::vector<AlignedRead> short_request{
      {kPageCount * kPageSize, kPageSize, output.get()}};
  PendingBatch short_batch;
  int result = reader.submit(short_batch, short_request, ctx);
  if (result == 0) {
    std::vector<uint32_t> completed;
    result = reader.get_completed(short_batch, ctx, 1, completed);
  }
  EXPECT_NE(result, 0);

  std::vector<AlignedRead> valid_request{{0, kPageSize, output.get()}};
  EXPECT_EQ(reader.read(valid_request, ctx, false), 0);
  EXPECT_TRUE(verify_page(output.get(), 0));
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
}
