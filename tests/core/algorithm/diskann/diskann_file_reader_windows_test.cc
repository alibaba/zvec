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
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/utility/file_helper.h>
#include "diskann_file_reader.h"

using namespace zvec::core;

namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kPageCount = 256;

uint8_t page_value(size_t page) {
  return static_cast<uint8_t>((page * 37 + 11) & 0xff);
}

class TemporaryFile {
 public:
  TemporaryFile() {
    wchar_t temp_dir[MAX_PATH]{};
    DWORD length = ::GetTempPathW(MAX_PATH, temp_dir);
    if (length == 0 || length >= static_cast<DWORD>(MAX_PATH) ||
        ::GetTempFileNameW(temp_dir, L"zvr", 0, wide_path_) == 0) {
      wide_path_[0] = L'\0';
      return;
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

  bool write_pages() const {
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
      std::memset(contents.data() + page * kPageSize, page_value(page),
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

bool verify_page(const uint8_t *buffer, size_t page) {
  return std::all_of(buffer, buffer + kPageSize, [page](uint8_t value) {
    return value == page_value(page);
  });
}

}  // namespace

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
    requests.emplace_back(i * kPageSize, kPageSize,
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

TEST(DiskAnnFileReaderWindowsTest, OpenContextPreventsIndexReplacement) {
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

  EXPECT_FALSE(::DeleteFileW(file.wide_path()));
  EXPECT_EQ(::GetLastError(), ERROR_SHARING_VIOLATION);
  EXPECT_EQ(destroy_io_ctx(ctx), 0);
  EXPECT_EQ(ctx, nullptr);
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
