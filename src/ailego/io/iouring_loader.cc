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

#if defined(__linux) || defined(__linux__)

#include <sys/syscall.h>  // syscall(), __NR_io_uring_setup
#include <unistd.h>       // close()
#include <cerrno>
#include <cstring>
#include <ailego/io/iouring_loader.h>
#include <zvec/ailego/logger/logger.h>

namespace zvec {
namespace core {

bool IoUringRing::setup(uint32_t entries) {
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
      LOG_WARN("io_uring_setup failed; errno=%d, %s", errno, ::strerror(errno));
    }
    return false;
  }

  sq_entries_ = params.sq_entries;
  cq_entries_ = params.cq_entries;

  // The read path uses IORING_OP_READ (Linux 5.6+).  On older kernels
  // (5.1–5.5) io_uring_setup() succeeds but every read would fail with
  // -EINVAL, so reject the ring here and let callers fall back.
  if ((params.features & IORING_FEAT_RW_CUR_POS) == 0) {
    LOG_WARN(
        "io_uring lacks IORING_OP_READ support (kernel < 5.6); falling "
        "back");
    teardown();
    return false;
  }

  // --- mmap the three shared regions ---

  // 1. SQ ring (includes head, tail, mask, entries, flags, dropped, array).
  size_t sq_ring_sz =
      static_cast<size_t>(params.sq_off.array) + sq_entries_ * sizeof(uint32_t);
  sq_ring_ptr_ = ::mmap(nullptr, sq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED,
                        ring_fd_, IORING_OFF_SQ_RING);
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
  cq_ring_ptr_ = ::mmap(nullptr, cq_ring_sz, PROT_READ | PROT_WRITE, MAP_SHARED,
                        ring_fd_, IORING_OFF_CQ_RING);
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
  sq_dropped_ = reinterpret_cast<unsigned *>(static_cast<char *>(sq_ring_ptr_) +
                                             params.sq_off.dropped);
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

void IoUringRing::teardown() {
  if (sq_ring_ptr_ && sq_ring_ptr_ != MAP_FAILED) {
    // We don't track the exact mmap size; munmap with a large enough size
    // is safe because the kernel only unmaps what was actually mapped.
    // However, to be correct we use the page-aligned size.
    size_t sz = static_cast<size_t>(sq_entries_) * sizeof(uint32_t) + 4096;
    ::munmap(sq_ring_ptr_, sz);
  }
  if (sqes_ptr_ && sqes_ptr_ != MAP_FAILED) {
    size_t sz = static_cast<size_t>(sq_entries_) * sizeof(struct io_uring_sqe);
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

  ::free(staging_);
  staging_ = nullptr;
  staging_size_ = 0;

  sq_entries_ = 0;
  cq_entries_ = 0;
}

bool IoUringRing::ensure_staging(size_t bytes) {
  if (staging_size_ >= bytes) {
    return true;
  }
  void *ptr = nullptr;
  int err = ::posix_memalign(&ptr, kIoUringStagingAlign, bytes);
  if (err != 0) {
    LOG_WARN("io_uring staging allocation failed; size=%zu, %s", bytes,
             ::strerror(err));
    return false;
  }
  ::free(staging_);
  staging_ = static_cast<char *>(ptr);
  staging_size_ = bytes;
  return true;
}

}  // namespace core
}  // namespace zvec

#endif  // __linux__
