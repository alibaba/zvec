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

// Abstract I/O backend selector.
//
// Wraps the low-level loaders (LibAioLoader for libaio) and provides a uniform
// way to initialize, query, and report the active I/O backend.  The actual I/O
// operations are still performed by the underlying loaders; this class is
// responsible only for backend initialization and reporting.
//
// When no async backend is available, the caller should fall back to
// synchronous pread().
//
// Usage:
//   auto& backend = ailego::IOBackend::Instance();
//   if (!backend.is_pread()) { ... }
//   LOG_INFO("I/O backend: %s", backend.name());

#pragma once

#include <cstring>
#include <ailego/io/iouring_def.h>
#include <ailego/io/libaio_loader.h>
#include <zvec/ailego/io/io_backend.h>

#if defined(__linux) || defined(__linux__)
#include <unistd.h>
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
    case IOBackendType::kWindowsOverlapped:
      return "windows_overlapped";
  }
  return "unknown";
}

// Returns a platform-specific human-readable backend description.
inline const char *IOBackendDescription(IOBackendType type) {
  switch (type) {
    case IOBackendType::kIoUring:
      return "io_uring async I/O backend accessed through the Linux kernel ABI.";
    case IOBackendType::kLibAio:
      return "libaio async I/O backend loaded at runtime via dlopen().";
    case IOBackendType::kPread:
#if defined(__linux) || defined(__linux__)
      return "No async I/O backend available. Install libaio (e.g. "
             "'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) "
             "and retry. DiskAnn will fall back to synchronous pread() \u2014 "
             "performance will be degraded.";
#else
      return "Synchronous pread I/O backend.";
#endif
    case IOBackendType::kWindowsOverlapped:
      return "Windows overlapped I/O backend with per-request events.";
  }
  return "Unknown I/O backend.";
}

// Singleton that loads and queries an I/O backend on demand.
//
// available() (no arg) tries the best backend with priority
// (io_uring > libaio > pread)
// and returns the loaded backend type.
// available(IOBackendType) tries a specific backend.
// Use type() / name() to query the loaded backend without triggering a load.
class IOBackend {
 public:
  static IOBackend &Instance() {
    static IOBackend instance;
    return instance;
  }

  // Select Windows overlapped I/O, or the best available Linux backend
  // (io_uring > libaio > pread).
  // Returns the loaded backend type.
  // Idempotent — if already loaded, returns immediately.
  IOBackendType available() {
#if defined(_WIN32) || defined(_WIN64)
    type_ = IOBackendType::kWindowsOverlapped;
    return type_;
#else
    if (type_ != IOBackendType::kPread) {
      return type_;
    }
    IOBackendType type = available(IOBackendType::kIoUring);
    if (type == IOBackendType::kPread) {
      type = available(IOBackendType::kLibAio);
    }
    return type;
#endif
  }

  // Try to load the requested backend. On Linux, failed requests fall back to
  // kPread. Windows always reports its native overlapped-I/O backend.
  // Idempotent — if the same backend is already loaded, returns immediately.
  IOBackendType available(IOBackendType requested) {
#if defined(_WIN32) || defined(_WIN64)
    (void)requested;
    type_ = IOBackendType::kWindowsOverlapped;
    return type_;
#else
    if (type_ == requested && type_ != IOBackendType::kPread) {
      return type_;
    }
#if defined(__linux) || defined(__linux__)
    if (requested == IOBackendType::kIoUring) {
      struct io_uring_params params;
      std::memset(&params, 0, sizeof(params));
      int fd = static_cast<int>(::syscall(__NR_io_uring_setup, 1, &params));
      if (fd >= 0) {
        ::close(fd);
        type_ = IOBackendType::kIoUring;
        return type_;
      }
    }
    if (requested == IOBackendType::kLibAio) {
      if (LibAioLoader::Instance().load() &&
          LibAioLoader::Instance().is_available()) {
        type_ = IOBackendType::kLibAio;
        return type_;
      }
    }
#endif
    type_ = IOBackendType::kPread;
    return type_;
#endif
  }

  bool is_pread() {
    return available() == IOBackendType::kPread;
  }

  bool is_libaio() {
    return available() == IOBackendType::kLibAio;
  }

  bool is_iouring() {
    return available() == IOBackendType::kIoUring;
  }

  bool is_windows_overlapped() {
    return available() == IOBackendType::kWindowsOverlapped;
  }

  // Returns the loaded backend type.
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

#if defined(_WIN32) || defined(_WIN64)
  IOBackendType type_{IOBackendType::kWindowsOverlapped};
#else
  IOBackendType type_{IOBackendType::kPread};
#endif
};

}  // namespace ailego
}  // namespace zvec
