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
#include <array>

#if (defined(__linux) || defined(__linux__))
#include <ailego/io/iouring_loader.h>  // raw-syscall io_uring wrapper (IoUringRing)
#include <ailego/io/libaio_loader.h>  // dlopen-based libaio wrapper
#elif defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <Windows.h>

// Do not leak Win32's function-like aliases into headers included below.
// They otherwise rewrite qualified names such as FileHelper::DeleteFile().
#ifdef DeleteFile
#undef DeleteFile
#endif
#ifdef RemoveDirectory
#undef RemoveDirectory
#endif
#endif

#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#endif
#include <string>
#include <vector>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/core/framework/index_context.h>
#include "diskann_util.h"

namespace zvec {
namespace core {

// IoBackend holds the selected backend for each thread. On Linux it also owns
// the resources required by the asynchronous backends. The priority is:
//   1. io_uring  (raw kernel syscalls — zero dependency)
//   2. libaio    (dlopen — soft dependency)
//   3. pread     (always available — synchronous fallback)
//
// macOS uses a real context with type kPread so that the active backend can be
// inspected and reported consistently instead of using an opaque placeholder.
// Windows stores a private file handle, completion port, and stable OVERLAPPED
// request slots in each I/O context. Keeping completion ports private prevents
// one context from consuming another context's completions.
// IOContext is a pointer to IoBackend; nullptr means uninitialised.
struct IoBackend {
  ailego::IOBackendType type{ailego::IOBackendType::kPread};

#if (defined(__linux) || defined(__linux__))
  IoUringRing ring{};
  io_context_t aio_ctx{nullptr};
#elif defined(_WIN32) || defined(_WIN64)
  std::array<OVERLAPPED, MAX_IO_DEPTH> reqs{};
  std::array<uint8_t, MAX_IO_DEPTH> active_requests{};
  HANDLE file_handle{INVALID_HANDLE_VALUE};
  HANDLE completion_port{nullptr};
  std::wstring file_path;
  uint64_t file_identity{0};
  uint32_t outstanding_count{0};
  uint64_t generation{0};
#endif
};

typedef IoBackend *IOContext;

int setup_io_ctx(IOContext &ctx);
int destroy_io_ctx(IOContext &ctx);

// Log the current DiskAnn I/O backend (io_uring, libaio, or pread). Probes the
// backend on first call. No-op outside Linux and macOS.
void log_diskann_io_backend();

struct AlignedRead {
  uint64_t offset;
  uint64_t len;
  void *buf;

  AlignedRead() : offset(0), len(0), buf(nullptr) {}

  AlignedRead(uint64_t offset, uint64_t len, void *buf)
      : offset(offset), len(len), buf(buf) {
#if defined(__linux__) || defined(__linux)
    // O_DIRECT requires 512-byte alignment on Linux.
    ailego_assert(static_cast<size_t>(offset) % 512 == 0);
    ailego_assert(static_cast<size_t>(len) % 512 == 0);
    ailego_assert(reinterpret_cast<size_t>(buf) % 512 == 0);
#endif
  }
};

struct PendingBatch {
#if (defined(__linux) || defined(__linux__))
  std::vector<struct iocb> cbs;
  std::vector<struct iocb *> cb_ptrs;
#elif defined(_WIN32) || defined(_WIN64)
  std::vector<uint64_t> expected_lengths;
  std::vector<uint8_t> completed;
  uint64_t generation{0};
#endif
  uint32_t n_submitted{0};
  uint32_t n_reaped{0};
  bool used_pread{false};
};

class AlignedFileReader {
 public:
  virtual ~AlignedFileReader() {}

  virtual void open(const std::string &fname) = 0;
  virtual void close() = 0;

  virtual int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
                   bool async = false) = 0;

  virtual int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
                     IOContext &ctx) = 0;

  virtual int get_completed(PendingBatch &batch, IOContext &ctx,
                            int min_completed,
                            std::vector<uint32_t> &completed_indices) = 0;

  // Release any lazy per-context file resources at an operation boundary.
  // POSIX backends keep their process-local queue resources for reuse; Windows
  // overrides this to close private file and completion-port handles.
  virtual void release_io_ctx(IOContext &ctx) = 0;
};

// POSIX reader implementation. Linux selects io_uring, libaio, or pread;
// macOS ARM64 uses synchronous pread.
#if !defined(_WIN32) && !defined(_WIN64)
class LinuxAlignedFileReader : public AlignedFileReader {
 private:
  int file_desc;

 public:
  LinuxAlignedFileReader();
  LinuxAlignedFileReader(int file_desc);
  ~LinuxAlignedFileReader() override;

 public:
  void open(const std::string &fname) override;
  // Duplicate an already-open descriptor so metadata and graph reads stay on
  // the same file object even if fname is atomically replaced.
  int open_from_handle(const std::string &fname, int source_fd);
  void close() override;

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) override;

  int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
             IOContext &ctx) override;

  int get_completed(PendingBatch &batch, IOContext &ctx, int min_completed,
                    std::vector<uint32_t> &completed_indices) override;
  void release_io_ctx(IOContext & /*ctx*/) override {}
};
#else
class WindowsAlignedFileReader : public AlignedFileReader {
 private:
  friend class WindowsAlignedFileReaderTestPeer;

  std::wstring file_path_;
  HANDLE stable_file_handle_{INVALID_HANDLE_VALUE};
  uint64_t file_identity_{0};

  int prepare_io_ctx(IOContext &ctx);
  void reset_io_ctx(IOContext &ctx);

 public:
  ~WindowsAlignedFileReader() override;

  void open(const std::string &fname) override;
  // Capture the same file object as an already-open buffered handle. This is
  // used while loading an index so metadata and later graph reads cannot come
  // from different files if fname is atomically replaced between the two.
  int open_from_handle(const std::string &fname, HANDLE source_handle);
  void close() override;

  int read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
           bool async = false) override;
  int submit(PendingBatch &batch, std::vector<AlignedRead> &read_reqs,
             IOContext &ctx) override;
  int get_completed(PendingBatch &batch, IOContext &ctx, int min_completed,
                    std::vector<uint32_t> &completed_indices) override;
  void release_io_ctx(IOContext &ctx) override;
};
#endif

#if defined(_WIN32) || defined(_WIN64)
using PlatformAlignedFileReader = WindowsAlignedFileReader;
#else
using PlatformAlignedFileReader = LinuxAlignedFileReader;
#endif

}  // namespace core
}  // namespace zvec
