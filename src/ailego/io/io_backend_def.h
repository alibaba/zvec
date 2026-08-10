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

// Abstract I/O backend selector — internal header.
//
// Wraps the low-level backends (io_uring via raw syscalls, LibAioLoader for
// libaio) and provides a uniform way to initialize, query, and report the
// active I/O backend.  The actual I/O operations are still performed by the
// underlying backends; this class is responsible only for backend
// initialization and reporting.
//
// When no async backend is available, the caller should fall back to
// synchronous pread().

#pragma once

#include <ailego/io/libaio_loader.h>
#include <zvec/ailego/io/io_backend.h>

#if defined(__linux) || defined(__linux__)
#include <unistd.h>                 // ::syscall(), ::close() — POSIX only
#include <cstring>                  // std::memset
#include <ailego/io/iouring_def.h>  // io_uring_params, __NR_io_uring_setup
#endif

namespace zvec {
namespace ailego {

// Returns a human-readable name for the given backend type.
inline const char *IOBackendTypeName(IOBackendType type) {
  switch (type) {
    case IOBackendType::kIoUring:
      return "io_uring";
    case IOBackendType::kLibAio:
      return "libaio";
    case IOBackendType::kPread:
      return "pread";
  }
  return "unknown";
}

// Returns a human-readable description for the given backend type.
// When the backend is kPread, includes installation guidance for libaio.
inline const char *IOBackendDescription(IOBackendType type) {
  switch (type) {
    case IOBackendType::kIoUring:
      return "io_uring async I/O backend (raw kernel syscalls, zero "
             "dependency).";
    case IOBackendType::kLibAio:
      return "libaio async I/O backend loaded at runtime via dlopen().";
    case IOBackendType::kPread:
      return "No async I/O backend available. Install libaio (e.g. "
             "'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) "
             "and retry. DiskAnn will fall back to synchronous pread() \u2014 "
             "performance will be degraded.";
  }
  return "Unknown I/O backend.";
}

// Singleton that probes and caches the I/O backend on first use.
//
// available() probes backends by priority (io_uring > libaio > pread)
// exactly once and caches the result — including the pread-only outcome,
// so systems without async I/O don't re-probe on every call.
// Use type() / name() to query the cached backend without probing.
class IOBackend {
 public:
  static IOBackend &Instance() {
    static IOBackend instance;
    return instance;
  }

  // Returns the active backend, probing on the first call
  // (io_uring > libaio > pread).  Idempotent — later calls return the
  // cached result immediately, even when the outcome is kPread.
  IOBackendType available() {
    if (probed_) {
      return type_;
    }
#if defined(__linux) || defined(__linux__)
    if (io_uring_supported()) {
      type_ = IOBackendType::kIoUring;
    } else if (LibAioLoader::Instance().load() &&
               LibAioLoader::Instance().is_available()) {
      type_ = IOBackendType::kLibAio;
    } else {
      type_ = IOBackendType::kPread;
    }
#else
    type_ = IOBackendType::kPread;
#endif
    probed_ = true;
    return type_;
  }

  bool is_pread() {
    return available() == IOBackendType::kPread;
  }

  bool is_libaio() {
    return available() == IOBackendType::kLibAio;
  }

  bool is_io_uring() {
    return available() == IOBackendType::kIoUring;
  }

  // Returns the cached backend type without triggering the probe.
  IOBackendType type() const {
    return type_;
  }

  // Human-readable name for the selected backend.
  const char *name() const {
    return IOBackendTypeName(type_);
  }

  // Human-readable description for the selected backend.
  const char *description() const {
    return IOBackendDescription(type_);
  }

 private:
  IOBackend() = default;

#if defined(__linux) || defined(__linux__)
  // Probe io_uring availability with a minimal ring setup using only raw
  // syscalls — no dependency on liburing.  A successful setup alone is NOT
  // sufficient: io_uring_setup() exists since Linux 5.1, but the read path
  // uses IORING_OP_READ, which was only added in 5.6.  We therefore also
  // require IORING_FEAT_RW_CUR_POS in params.features — a feature flag
  // introduced in the same 5.6 release — so kernels 5.1–5.5 fall back to
  // libaio/pread instead of failing every read with -EINVAL.
  static bool io_uring_supported() {
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));
    int fd = static_cast<int>(::syscall(__NR_io_uring_setup, 1, &params));
    if (fd < 0) {
      return false;
    }
    ::close(fd);
    return (params.features & IORING_FEAT_RW_CUR_POS) != 0;
  }
#endif

  // kPread doubles as the pre-probe default; probed_ marks whether the
  // one-shot probe has run so that a pread-only outcome is cached too.
  // (IOBackendType values are C ABI — no kNone sentinel is added there.)
  IOBackendType type_{IOBackendType::kPread};
  bool probed_{false};
};

}  // namespace ailego
}  // namespace zvec
