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

#include <cerrno>
#include <zvec/ailego/io/file.h>
#include <zvec/core/framework/index_error.h>
#include "ailego/io/file_lock.h"

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace zvec::core {

// Serialize backing-file growth, directory publication and tail recovery
// across storage backends, including independent handles to the same file.
class IndexFileWriteLock {
 public:
  explicit IndexFileWriteLock(ailego::File::NativeHandle handle)
      : handle_(handle) {}

  ~IndexFileWriteLock() {
    if (!locked_) return;
#if defined(_WIN32) || defined(_WIN64)
    UnlockFileEx(handle_, 0, 1, 0, &range_);
#else
    ailego::FileLock::Unlock(handle_);
#endif
  }

  int try_lock() {
#if defined(_WIN32) || defined(_WIN64)
    // Windows enforces byte-range locks on ordinary I/O. Lock a byte beyond
    // the index data so other handles can still read the file.
    range_.OffsetHigh = 0x40000000;
    locked_ =
        LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                   0, 1, 0, &range_) != 0;
    if (!locked_ && GetLastError() == ERROR_LOCK_VIOLATION) {
      return IndexError_Busy;
    }
#else
    do {
      locked_ = ailego::FileLock::TryLock(handle_);
    } while (!locked_ && errno == EINTR);
    if (!locked_ && (errno == EWOULDBLOCK || errno == EAGAIN)) {
      return IndexError_Busy;
    }
#endif
    return locked_ ? IndexError_Success : IndexError_IO;
  }

  IndexFileWriteLock(const IndexFileWriteLock &) = delete;
  IndexFileWriteLock &operator=(const IndexFileWriteLock &) = delete;

 private:
  const ailego::File::NativeHandle handle_;
#if defined(_WIN32) || defined(_WIN64)
  OVERLAPPED range_{};
#endif
  bool locked_{false};
};

}  // namespace zvec::core
