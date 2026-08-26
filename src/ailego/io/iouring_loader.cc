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
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>
#include <ailego/io/iouring_loader.h>
#include <zvec/ailego/logger/logger.h>

namespace zvec {
namespace ailego {

// Retry budget for draining in-flight requests when the kernel keeps
// returning EAGAIN/EBUSY (100 us sleep per retry, about one second total).
static constexpr size_t kIoUringDrainRetries = 10000;

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
  sq_ring_size_ =
      static_cast<size_t>(params.sq_off.array) + sq_entries_ * sizeof(uint32_t);
  sq_ring_ptr_ = ::mmap(nullptr, sq_ring_size_, PROT_READ | PROT_WRITE,
                        MAP_SHARED, ring_fd_, IORING_OFF_SQ_RING);
  if (sq_ring_ptr_ == MAP_FAILED) {
    LOG_ERROR("mmap SQ ring failed: %s", ::strerror(errno));
    sq_ring_ptr_ = nullptr;
    teardown();
    return false;
  }

  // 2. SQE array.
  sqes_size_ = sq_entries_ * sizeof(struct io_uring_sqe);
  sqes_ptr_ = reinterpret_cast<struct io_uring_sqe *>(
      ::mmap(nullptr, sqes_size_, PROT_READ | PROT_WRITE, MAP_SHARED, ring_fd_,
             IORING_OFF_SQES));
  if (sqes_ptr_ == MAP_FAILED) {
    LOG_ERROR("mmap SQEs failed: %s", ::strerror(errno));
    sqes_ptr_ = nullptr;
    teardown();
    return false;
  }

  // 3. CQ ring (includes head, tail, mask, entries, overflow, cqes[]).
  cq_ring_size_ = static_cast<size_t>(params.cq_off.cqes) +
                  cq_entries_ * sizeof(struct io_uring_cqe);
  cq_ring_ptr_ = ::mmap(nullptr, cq_ring_size_, PROT_READ | PROT_WRITE,
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
    ::munmap(sq_ring_ptr_, sq_ring_size_);
  }
  if (sqes_ptr_ && sqes_ptr_ != MAP_FAILED) {
    ::munmap(sqes_ptr_, sqes_size_);
  }
  if (cq_ring_ptr_ && cq_ring_ptr_ != MAP_FAILED) {
    ::munmap(cq_ring_ptr_, cq_ring_size_);
  }

  sq_ring_ptr_ = nullptr;
  sqes_ptr_ = nullptr;
  cq_ring_ptr_ = nullptr;
  sq_ring_size_ = 0;
  sqes_size_ = 0;
  cq_ring_size_ = 0;
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

int IoUringRing::execute(int fd, const IoUringRead *read_reqs, size_t count) {
  return execute_impl(fd, read_reqs, nullptr, count);
}

int IoUringRing::execute_writes(int fd, const IoUringWrite *write_reqs,
                                size_t count) {
  return execute_impl(fd, nullptr, write_reqs, count);
}

int IoUringRing::execute_impl(int fd, const IoUringRead *read_reqs,
                              const IoUringWrite *write_reqs, size_t count) {
  const bool is_write = write_reqs != nullptr;
  if (!is_valid() ||
      (count != 0 && ((read_reqs == nullptr) == (write_reqs == nullptr)))) {
    return -1;
  }
  if (count == 0) {
    return 0;
  }

  const size_t batch_size = std::min<size_t>(sq_entries_, kIoUringMaxBatch);
  if (batch_size == 0) {
    return -1;
  }

  for (size_t batch_start = 0; batch_start < count; batch_start += batch_size) {
    const size_t n_ops = std::min(batch_size, count - batch_start);
    std::array<size_t, kIoUringMaxBatch> slot_offsets{};
    size_t staging_bytes = 0;
    for (size_t j = 0; j < n_ops; ++j) {
      const size_t req_idx = batch_start + j;
      const uint64_t offset =
          is_write ? write_reqs[req_idx].offset : read_reqs[req_idx].offset;
      const uint64_t len =
          is_write ? write_reqs[req_idx].len : read_reqs[req_idx].len;
      const uint64_t expected_len =
          is_write ? len
                   : (read_reqs[req_idx].expected_len == 0
                          ? len
                          : read_reqs[req_idx].expected_len);
      const void *buf =
          is_write ? write_reqs[req_idx].buf : read_reqs[req_idx].buf;
      if (buf == nullptr || len == 0 || expected_len > len ||
          len > std::numeric_limits<uint32_t>::max() || offset % 512 != 0 ||
          len % 512 != 0 ||
          (!is_write && reinterpret_cast<uintptr_t>(buf) % 512 != 0)) {
        return -1;
      }
      const size_t aligned_len =
          (static_cast<size_t>(len) + kIoUringStagingAlign - 1) &
          ~(kIoUringStagingAlign - 1);
      if (aligned_len < len ||
          staging_bytes > std::numeric_limits<size_t>::max() - aligned_len) {
        return -1;
      }
      slot_offsets[j] = staging_bytes;
      staging_bytes += aligned_len;
    }
    if (!ensure_staging(staging_bytes)) {
      return -1;
    }

    const unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
    const unsigned mask = *sq_ring_mask_;
    for (size_t j = 0; j < n_ops; ++j) {
      const unsigned idx = (tail + static_cast<unsigned>(j)) & mask;
      const unsigned sqe_idx = sq_array_[idx];
      struct io_uring_sqe *sqe = &sqes_[sqe_idx];
      const size_t req_idx = batch_start + j;
      if (is_write) {
        const IoUringWrite &req = write_reqs[req_idx];
        std::memcpy(staging_ + slot_offsets[j], req.buf, req.len);
        io_uring_prep_write(sqe, fd, staging_ + slot_offsets[j],
                            static_cast<uint32_t>(req.len), req.offset);
      } else {
        const IoUringRead &req = read_reqs[req_idx];
        io_uring_prep_read(sqe, fd, staging_ + slot_offsets[j],
                           static_cast<uint32_t>(req.len), req.offset);
      }
      sqe->user_data = req_idx;
    }

    __sync_synchronize();
    __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(n_ops),
                     __ATOMIC_RELEASE);

    size_t submitted = 0;
    size_t completed = 0;
    bool all_ok = true;
    auto reap_available = [&]() {
      unsigned chead = *cq_head_;
      const unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      const unsigned cq_mask = *cq_ring_mask_;
      while (chead != ctail) {
        struct io_uring_cqe *cqe = &cqes_[chead & cq_mask];
        const size_t req_idx = static_cast<size_t>(cqe->user_data);
        if (req_idx < batch_start || req_idx >= batch_start + n_ops) {
          LOG_WARN("io_uring completion referenced unknown request: %zu",
                   req_idx);
          all_ok = false;
        } else {
          const uint64_t offset =
              is_write ? write_reqs[req_idx].offset : read_reqs[req_idx].offset;
          const uint64_t len =
              is_write ? write_reqs[req_idx].len : read_reqs[req_idx].len;
          const uint64_t expected_len =
              is_write ? len
                       : (read_reqs[req_idx].expected_len == 0
                              ? len
                              : read_reqs[req_idx].expected_len);
          const char *operation = is_write ? "write" : "read";
          if (cqe->res < 0) {
            LOG_WARN("io_uring %s failed: req=%zu, res=%d, offset=%lu",
                     operation, req_idx, cqe->res,
                     static_cast<unsigned long>(offset));
            all_ok = false;
          } else if (static_cast<uint64_t>(cqe->res) != expected_len) {
            LOG_WARN("io_uring short %s: req=%zu, got=%d, expected=%lu",
                     operation, req_idx, cqe->res,
                     static_cast<unsigned long>(expected_len));
            all_ok = false;
          } else if (!is_write) {
            const IoUringRead &req = read_reqs[req_idx];
            const size_t slot = req_idx - batch_start;
            std::memcpy(req.buf, staging_ + slot_offsets[slot], expected_len);
            if (expected_len < len) {
              std::memset(static_cast<char *>(req.buf) + expected_len, 0,
                          len - expected_len);
            }
          }
        }
        ++chead;
        ++completed;
      }
      __atomic_store_n(cq_head_, chead, __ATOMIC_RELEASE);
    };

    while (completed < n_ops) {
      reap_available();
      if (completed >= n_ops) {
        break;
      }

      const unsigned to_submit = static_cast<unsigned>(n_ops - submitted);
      const int ret = static_cast<int>(syscall(
          __NR_io_uring_enter, ring_fd_, to_submit, 1u, IORING_ENTER_GETEVENTS,
          static_cast<void *>(nullptr), static_cast<size_t>(0)));
      if (ret >= 0) {
        submitted += static_cast<size_t>(ret);
        continue;
      }
      if (errno == EINTR ||
          ((errno == EAGAIN || errno == EBUSY) && completed < submitted)) {
        continue;
      }

      LOG_WARN(
          "io_uring_enter failed; errno=%d, %s, submitted=%zu/%zu, "
          "completed=%zu. draining before falling back to p%s",
          errno, ::strerror(errno), submitted, n_ops, completed,
          is_write ? "write" : "read");
      __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(submitted),
                       __ATOMIC_RELEASE);

      size_t drain_retries = 0;
      while (completed < submitted) {
        reap_available();
        if (completed >= submitted) {
          break;
        }
        const int wait_ret = static_cast<int>(syscall(
            __NR_io_uring_enter, ring_fd_, 0u, 1u, IORING_ENTER_GETEVENTS,
            static_cast<void *>(nullptr), static_cast<size_t>(0)));
        if (wait_ret >= 0 || errno == EINTR) {
          continue;
        }
        if ((errno == EAGAIN || errno == EBUSY) &&
            drain_retries++ < kIoUringDrainRetries) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }
        LOG_ERROR(
            "io_uring drain failed; errno=%d, %s. leaking the staging pool "
            "and disabling io_uring for this context",
            errno, ::strerror(errno));
        abandon_staging();
        teardown();
        return -1;
      }
      return -1;
    }

    if (!all_ok) {
      return -1;
    }
  }
  return 0;
}

}  // namespace ailego
}  // namespace zvec

#endif  // __linux__
