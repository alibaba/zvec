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
#pragma once

#define MAX_IO_DEPTH 128

#include <fcntl.h>

#if (defined(__linux) || defined(__linux__))
#include <ailego/io/iouring_loader.h>  // raw-syscall io_uring wrapper
#include <ailego/io/libaio_loader.h>   // dlopen-based libaio wrapper
#elif defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <Windows.h>
#endif

#include <atomic>
#include <string>
#include <utility>
#include <vector>
#include <zvec/core/framework/index_context.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))

// IoBackend holds the per-thread I/O context for whichever async backend
// was successfully initialised at setup time.  The priority is:
//   1. io_uring  (raw kernel syscalls — zero dependency)
//   2. libaio    (dlopen — soft dependency)
//   3. pread     (always available — synchronous fallback)
//
// IOContext is a *pointer* to IoBackend, which preserves the existing
// sentinel conventions: nullptr means uninitialised and (IOContext)-1 is
// the invalid-handle sentinel returned by get_ctx() for unregistered
// threads.
struct IoBackend {
  enum Backend : uint8_t {
    NONE = 0,      // synchronous pread
    IO_URING = 1,  // io_uring via raw syscalls
    LIBAIO = 2,    // libaio via dlopen
  };

  Backend backend{NONE};
  IoUringRing ring{};
  io_context_t aio_ctx{nullptr};
};

typedef IoBackend *IOContext;

#elif defined(_WIN32) || defined(_WIN64)
struct IOContext {
  std::vector<OVERLAPPED> reqs;
  HANDLE file_handle{INVALID_HANDLE_VALUE};
  HANDLE completion_port{NULL};
  std::wstring file_path;

  IOContext() = default;
  IOContext(const IOContext &) = delete;
  IOContext &operator=(const IOContext &) = delete;

  IOContext(IOContext &&other) noexcept
      : reqs(std::move(other.reqs)),
        file_handle(other.file_handle),
        completion_port(other.completion_port),
        file_path(std::move(other.file_path)) {
    other.file_handle = INVALID_HANDLE_VALUE;
    other.completion_port = NULL;
  }

  IOContext &operator=(IOContext &&other) noexcept {
    if (this != &other) {
      close_handles();
      reqs = std::move(other.reqs);
      file_handle = other.file_handle;
      completion_port = other.completion_port;
      file_path = std::move(other.file_path);
      other.file_handle = INVALID_HANDLE_VALUE;
      other.completion_port = NULL;
    }
    return *this;
  }

  ~IOContext() {
    close_handles();
  }

  void close_handles() {
    if (file_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(file_handle);
      file_handle = INVALID_HANDLE_VALUE;
    }
    if (completion_port != NULL) {
      CloseHandle(completion_port);
      completion_port = NULL;
    }
    file_path.clear();
  }
};
#else
typedef uint32_t IOContext;
#endif

int setup_io_ctx(IOContext &ctx);
int destroy_io_ctx(IOContext &ctx);

// Log the current DiskAnn I/O backend status (async vs. synchronous pread).
// Probes the backend on first call.  No-op on non-Linux platforms.
void log_diskann_io_backend();

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;

  AlignedRead() : offset(0), len(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
    ailego_assert(static_cast<size_t>(offset) % 512 == 0);
    ailego_assert(static_cast<size_t>(len) % 512 == 0);
    ailego_assert(reinterpret_cast<size_t>(buf) % 512 == 0);
  }
};

class AlignedFileReader {
 protected:
  std::map<std::thread::id, IOContext> ctx_map;
  std::mutex ctx_mut;

 public:
  virtual IOContext &get_ctx() = 0;

  virtual ~AlignedFileReader() {}

  virtual void register_thread() = 0;
  virtual void deregister_thread() = 0;
  virtual void deregister_all_threads() = 0;

  virtual void open(const std::string &fname) = 0;
  virtual void close() = 0;

  virtual int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                   bool async = false) = 0;
};

#if (defined(__linux) || defined(__linux__))
class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  int file_desc;

  IOContext bad_ctx = (IOContext)-1;

 public:
  LinuxAlignedFileReader();
  LinuxAlignedFileReader(int file_desc);
  ~LinuxAlignedFileReader();

 public:
  IOContext &get_ctx();

  void register_thread();
  void deregister_thread();
  void deregister_all_threads();
  void open(const std::string &fname);
  void close();

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false);
};
#elif defined(_WIN32) || defined(_WIN64)
class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  std::wstring file_path_;

  int prepare_io_ctx(IOContext &ctx);

 public:
  WindowsAlignedFileReader();
  ~WindowsAlignedFileReader();

 public:
  IOContext &get_ctx();

  void register_thread();
  void deregister_thread();
  void deregister_all_threads();
  void open(const std::string &fname);
  void close();

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false);
};
#endif

// Platform-agnostic reader typedef
#if (defined(__linux) || defined(__linux__))
typedef LinuxAlignedFileReader PlatformAlignedFileReader;
#elif defined(_WIN32) || defined(_WIN64)
typedef WindowsAlignedFileReader PlatformAlignedFileReader;
#endif

}  // namespace core
}  // namespace zvec
