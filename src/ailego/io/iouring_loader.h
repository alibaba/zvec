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

// Raw-syscall wrapper for Linux io_uring.
//
// This class implements the io_uring submission/completion queue lifecycle
// using *only* kernel syscalls (io_uring_setup, io_uring_enter) and mmap.
// There is zero dependency on liburing or liburing-dev: no build-time header,
// no runtime .so, and no dlopen.  This mirrors the project's existing
// zero-dependency philosophy established by the libaio dlopen approach, but
// goes one step further — io_uring is a pure kernel ABI accessed via syscall.
//
// Runtime detection is automatic: if the kernel does not support io_uring
// (pre-5.1 or io_uring disabled), io_uring_setup() returns -ENOSYS and
// setup() returns false, allowing callers to fall back to libaio or pread.
//
// The ring is **not** thread-safe.  Each thread that performs I/O must have
// its own IoUringRing instance (managed through IoBackend / IOContext).

#pragma once

#if defined(__linux) || defined(__linux__)

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <ailego/io/iouring_def.h>
#include <zvec/core/framework/index_logger.h>

namespace zvec {
namespace core {

// Forward declaration — AlignedRead is defined in diskann_file_reader.h
// after this header is included.  The forward declaration is sufficient
// because IoUringRing::execute takes it by reference.
struct AlignedRead;

// Maximum number of SQEs we submit in a single io_uring_enter() call.
static constexpr uint32_t kIoUringMaxBatch = 128;

class IoUringRing {
 public:
  IoUringRing() = default;
  ~IoUringRing() {
    teardown();
  }

  IoUringRing(const IoUringRing &) = delete;
  IoUringRing &operator=(const IoUringRing &) = delete;

  // Create an io_uring with the given number of entries.
  // Returns true on success, false if the kernel does not support io_uring
  // or setup failed for any reason.
  bool setup(uint32_t entries) {
    struct io_uring_params params;
    std::memset(&params, 0, sizeof(params));

    // io_uring_setup is a raw syscall — returns fd (>=0) or -1 with errno.
    ring_fd_ = static_cast<int>(
        syscall(__NR_io_uring_setup, static_cast<int>(entries), &params));
    if (ring_fd_ < 0) {
      // ENOSYS = kernel doesn't support io_uring.
      // EPERM  = io_uring disabled via sysctl.
      // EINVAL = invalid parameters.
      if (errno != ENOSYS) {
        LOG_WARN("io_uring_setup failed; errno=%d, %s", errno,
                 ::strerror(errno));
      }
      return false;
    }

    sq_entries_ = params.sq_entries;
    cq_entries_ = params.cq_entries;

    // --- mmap the three shared regions ---

    // 1. SQ ring (includes head, tail, mask, entries, flags, dropped, array).
    size_t sq_ring_sz = static_cast<size_t>(params.sq_off.array) +
                        sq_entries_ * sizeof(uint32_t);
    sq_ring_ptr_ = ::mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ring_fd_, IORING_OFF_SQ_RING);
    if (sq_ring_ptr_ == MAP_FAILED) {
      LOG_ERROR("mmap SQ ring failed: %s", ::strerror(errno));
      sq_ring_ptr_ = nullptr;
      teardown();
      return false;
    }

    // 2. SQE array.
    size_t sqes_sz = sq_entries_ * sizeof(struct io_uring_sqe);
    sqes_ptr_ = reinterpret_cast<struct io_uring_sqe *>(
        ::mmap(nullptr, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd_,
               IORING_OFF_SQES));
    if (sqes_ptr_ == MAP_FAILED) {
      LOG_ERROR("mmap SQEs failed: %s", ::strerror(errno));
      sqes_ptr_ = nullptr;
      teardown();
      return false;
    }

    // 3. CQ ring (includes head, tail, mask, entries, overflow, cqes[]).
    size_t cq_ring_sz = static_cast<size_t>(params.cq_off.cqes) +
                        cq_entries_ * sizeof(struct io_uring_cqe);
    cq_ring_ptr_ = ::mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ring_fd_, IORING_OFF_CQ_RING);
    if (cq_ring_ptr_ == MAP_FAILED) {
      LOG_ERROR("mmap CQ ring failed: %s", ::strerror(errno));
      cq_ring_ptr_ = nullptr;
      teardown();
      return false;
    }

    // --- Set up typed pointers into the mmap'd regions ---

    // SQ ring fields.
    sq_head_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_ptr_) +
                                            params.sq_off.head);
    sq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_ptr_) +
                                            params.sq_off.tail);
    sq_ring_mask_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(sq_ring_ptr_) + params.sq_off.ring_mask);
    sq_ring_entries_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(sq_ring_ptr_) + params.sq_off.ring_entries);
    sq_flags_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_ptr_) +
                                             params.sq_off.flags);
    sq_dropped_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(sq_ring_ptr_) + params.sq_off.dropped);
    sq_array_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_ptr_) +
                                             params.sq_off.array);

    // CQ ring fields.
    cq_head_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ring_ptr_) +
                                            params.cq_off.head);
    cq_tail_ = reinterpret_cast<unsigned *>(static_cast<char *>(cq_ring_ptr_) +
                                            params.cq_off.tail);
    cq_ring_mask_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(cq_ring_ptr_) + params.cq_off.ring_mask);
    cq_ring_entries_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(cq_ring_ptr_) + params.cq_off.ring_entries);
    cq_overflow_ = reinterpret_cast<unsigned *>(
        static_cast<char *>(cq_ring_ptr_) + params.cq_off.overflow);
    cqes_ = reinterpret_cast<struct io_uring_cqe *>(
        static_cast<char *>(cq_ring_ptr_) + params.cq_off.cqes);

    // SQE array.
    sqes_ = sqes_ptr_;

    // Initialize the SQ array to identity mapping so that logical index ==
    // physical SQE index.  This is the simplest and most common configuration.
    for (uint32_t i = 0; i < sq_entries_; i++) {
      sq_array_[i] = i;
    }

    return true;
  }

  // Tear down the ring: munmap all regions and close the ring fd.
  void teardown() {
    if (sq_ring_ptr_ && sq_ring_ptr_ != MAP_FAILED) {
      // We don't track the exact mmap size; munmap with a large enough size
      // is safe because the kernel only unmaps what was actually mapped.
      // However, to be correct we use the page-aligned size.
      size_t sz = static_cast<size_t>(sq_entries_) * sizeof(uint32_t) + 4096;
      ::munmap(sq_ring_ptr_, sz);
    }
    if (sqes_ptr_ && sqes_ptr_ != MAP_FAILED) {
      size_t sz =
          static_cast<size_t>(sq_entries_) * sizeof(struct io_uring_sqe);
      ::munmap(sqes_ptr_, sz);
    }
    if (cq_ring_ptr_ && cq_ring_ptr_ != MAP_FAILED) {
      size_t sz =
          static_cast<size_t>(cq_entries_) * sizeof(struct io_uring_cqe) + 4096;
      ::munmap(cq_ring_ptr_, sz);
    }

    sq_ring_ptr_ = nullptr;
    sqes_ptr_ = nullptr;
    cq_ring_ptr_ = nullptr;
    sqes_ = nullptr;
    cqes_ = nullptr;
    sq_head_ = sq_tail_ = sq_ring_mask_ = sq_ring_entries_ = nullptr;
    sq_flags_ = sq_dropped_ = sq_array_ = nullptr;
    cq_head_ = cq_tail_ = cq_ring_mask_ = cq_ring_entries_ = nullptr;
    cq_overflow_ = nullptr;

    if (ring_fd_ >= 0) {
      ::close(ring_fd_);
      ring_fd_ = -1;
    }

    sq_entries_ = 0;
    cq_entries_ = 0;
  }

  bool is_valid() const {
    return ring_fd_ >= 0;
  }

  // Execute a batch of aligned read requests via io_uring.
  //
  // Implemented in diskann_file_reader.cc to avoid a circular dependency:
  // AlignedRead is defined in diskann_file_reader.h *after* this header is
  // included, so the method body cannot be inline here.
  //
  // On success returns 0.  On failure returns -1; the caller should fall
  // back to pread.
  int execute(int fd, std::vector<AlignedRead> &read_reqs);

 private:
  int ring_fd_{-1};

  // mmap'd region base pointers (needed for munmap).
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

  // SQE array pointer.
  struct io_uring_sqe *sqes_{nullptr};

  // Ring capacities.
  unsigned sq_entries_{0};
  unsigned cq_entries_{0};
};

}  // namespace core
}  // namespace zvec

#endif  // __linux__
