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

// Raw-syscall wrapper for Linux io_uring: queue lifecycle via
// io_uring_setup/io_uring_enter + mmap, with zero dependency on liburing.
//
// setup() returns false when the kernel lacks the io_uring features we
// need (pre-5.6, missing IORING_OP_READ, or disabled), letting callers
// fall back to libaio or pread.
//
// Not thread-safe: each I/O thread must own its IoUringRing instance.

#pragma once

#if defined(__linux) || defined(__linux__)

#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <ailego/io/iouring_def.h>

namespace zvec {
namespace core {

// AlignedRead lives in diskann_file_reader.h; a forward declaration
// suffices since execute() takes it by reference.
struct AlignedRead;

// Max SQEs submitted per io_uring_enter() call.
static constexpr uint32_t kIoUringMaxBatch = 128;

// Staging slot alignment — keeps O_DIRECT reads legal.
static constexpr size_t kIoUringStagingAlign = 4096;

class IoUringRing {
 public:
  IoUringRing() = default;
  ~IoUringRing() {
    teardown();
  }

  IoUringRing(const IoUringRing &) = delete;
  IoUringRing &operator=(const IoUringRing &) = delete;

  // Create an io_uring with `entries` queue slots.  Returns false if the
  // kernel lacks io_uring or setup failed.  In iouring_loader.cc.
  bool setup(uint32_t entries);

  // munmap all regions, close the ring fd, free staging.  Closing the fd
  // only *starts* asynchronous cancellation of in-flight requests, so call
  // this only when the ring is quiesced — or after abandon_staging().
  // In iouring_loader.cc.
  void teardown();

  // Grow the ring-owned staging pool to at least `bytes`.  Only safe while
  // the ring is quiesced (the old pool is freed here).  In iouring_loader.cc.
  bool ensure_staging(size_t bytes);

  // Deliberately leak the staging pool when in-flight requests cannot be
  // drained: the kernel may keep writing into it after the fd is closed.
  // Caller buffers are never handed to the kernel and remain safe.
  void abandon_staging() {
    staging_ = nullptr;
    staging_size_ = 0;
  }

  bool is_valid() const {
    return ring_fd_ >= 0;
  }

  // Execute a batch of aligned reads via io_uring.  Returns 0 on success,
  // -1 on failure — the caller may always fall back to pread, since the
  // kernel only writes into the staging pool.  In diskann_file_reader.cc
  // (AlignedRead is defined there).
  int execute(int fd, std::vector<AlignedRead> &read_reqs);

 private:
  int ring_fd_{-1};

  // mmap'd region bases (needed for munmap).
  void *sq_ring_ptr_{nullptr};
  struct io_uring_sqe *sqes_ptr_{nullptr};
  void *cq_ring_ptr_{nullptr};

  // SQ ring field pointers (into sq_ring_ptr_).
  unsigned *sq_head_{nullptr};
  unsigned *sq_tail_{nullptr};
  unsigned *sq_ring_mask_{nullptr};
  unsigned *sq_ring_entries_{nullptr};
  unsigned *sq_flags_{nullptr};
  unsigned *sq_dropped_{nullptr};
  unsigned *sq_array_{nullptr};

  // CQ ring field pointers (into cq_ring_ptr_).
  unsigned *cq_head_{nullptr};
  unsigned *cq_tail_{nullptr};
  unsigned *cq_ring_mask_{nullptr};
  unsigned *cq_ring_entries_{nullptr};
  unsigned *cq_overflow_{nullptr};
  struct io_uring_cqe *cqes_{nullptr};

  struct io_uring_sqe *sqes_{nullptr};

  // Ring-owned staging pool: the kernel reads into it and execute() copies
  // verified completions out, decoupling caller buffer lifetime from async
  // io_uring teardown (see abandon_staging()).
  char *staging_{nullptr};
  size_t staging_size_{0};

  unsigned sq_entries_{0};
  unsigned cq_entries_{0};
};

}  // namespace core
}  // namespace zvec

#endif  // __linux__
