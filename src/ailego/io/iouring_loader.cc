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

#if (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)

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
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/budgeted_buffer.h>
#include <zvec/ailego/buffer/budgeted_mapped_buffer.h>
#include <zvec/ailego/logger/logger.h>

namespace zvec {
namespace ailego {

// Retry budget for draining in-flight requests when the kernel keeps
// returning EAGAIN/EBUSY (100 us sleep per retry, about one second total).
static constexpr size_t kIoUringDrainRetries = 10000;

struct IoUringRing::AsyncReads {
  MemoryReservation memory;
  struct Slot {
    IoUringRead request;
    size_t offset;
    bool seen{false};
  };
  BudgetedVector<Slot> slots;
  size_t submitted{0};
  size_t completed{0};
};

IoUringRing::IoUringRing() = default;

int IoUringRing::enter(unsigned submit, unsigned minimum, unsigned flags) {
  if (enter_override_) return enter_override_(ring_fd_, submit, minimum, flags);
  return static_cast<int>(syscall(__NR_io_uring_enter, ring_fd_, submit,
                                  minimum, flags, nullptr, size_t{0}));
}

bool IoUringRing::has_pending_reads() const {
  return async_reads_ != nullptr;
}

int IoUringRing::submit_reads(int fd, const IoUringRead *reads, size_t count) {
  if (async_reads_ || !is_valid() || !reads || count == 0 ||
      count > std::min<size_t>(sq_entries_, kIoUringMaxBatch))
    return -1;
  try {
    MemoryReservation memory;
    memory.add(sizeof(AsyncReads));
    auto state = std::make_unique<AsyncReads>();
    state->memory.swap(memory);
    state->slots.reserve(count);
    size_t bytes = 0;
    for (size_t i = 0; i < count; ++i) {
      const auto &read = reads[i];
      if (!read.buf || !read.len || read.offset % 512 || read.len % 512 ||
          reinterpret_cast<uintptr_t>(read.buf) % 512 ||
          read.len > std::numeric_limits<uint32_t>::max() ||
          read.expected_len > read.len)
        return -1;
      const size_t size =
          (read.len + kIoUringStagingAlign - 1) & ~(kIoUringStagingAlign - 1);
      if (bytes > SIZE_MAX - size) return -1;
      state->slots.push_back({read, bytes, false});
      bytes += size;
    }
    if (!ensure_staging(bytes)) return -1;
    const unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
    const unsigned mask = *sq_ring_mask_;
    for (size_t i = 0; i < count; ++i) {
      auto *sqe = &sqes_[sq_array_[(tail + i) & mask]];
      io_uring_prep_read(sqe, fd, staging_ + state->slots[i].offset,
                         static_cast<uint32_t>(reads[i].len), reads[i].offset);
      sqe->user_data = i;
    }
    async_reads_ = std::move(state);
    __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(count),
                     __ATOMIC_RELEASE);
    size_t retries = 0;
    while (async_reads_->submitted < count) {
      const unsigned remaining = count - async_reads_->submitted;
      const int ret = enter(remaining, 0, 0);
      if (ret > 0 && static_cast<unsigned>(ret) <= remaining) {
        async_reads_->submitted += ret;
        retries = 0;
        continue;
      }
      if ((ret == 0 || errno == EINTR || errno == EAGAIN || errno == EBUSY) &&
          retries++ < 8)
        continue;
      // Remove the unsubmitted suffix before retiring accepted requests.
      __atomic_store_n(sq_tail_, tail + async_reads_->submitted,
                       __ATOMIC_RELEASE);
      if (ret > 0) {
        abandon_staging();
        async_reads_.reset();
        teardown();
      } else {
        cancel_reads();
      }
      return -1;
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return -1;
  }
}

int IoUringRing::reap_reads(size_t minimum, uint32_t *indices, size_t capacity,
                            bool deliver) {
  if (!async_reads_ || (capacity && !indices)) return -1;
  auto &state = *async_reads_;
  minimum = std::min({minimum, capacity, state.submitted - state.completed});
  size_t count = 0;
  for (;;) {
    unsigned head = __atomic_load_n(cq_head_, __ATOMIC_ACQUIRE);
    const unsigned tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
    while (head != tail && count < capacity) {
      const auto event = cqes_[head & *cq_ring_mask_];
      const size_t id = event.user_data;
      if (id >= state.submitted || state.slots[id].seen) {
        abandon_staging();
        async_reads_.reset();
        teardown();
        return -1;
      }
      auto &slot = state.slots[id];
      slot.seen = true;
      ++state.completed;
      ++head;
      __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
      const auto &request = slot.request;
      const size_t expected =
          request.expected_len ? request.expected_len : request.len;
      if (deliver &&
          (event.res < 0 || static_cast<size_t>(event.res) != expected)) {
        cancel_reads();
        return -1;
      }
      if (deliver) {
        std::memcpy(request.buf, staging_ + slot.offset, expected);
        if (expected < request.len)
          std::memset(static_cast<char *>(request.buf) + expected, 0,
                      request.len - expected);
      }
      indices[count++] = id;
    }
    if (state.completed == state.submitted) {
      async_reads_.reset();
      release_staging();
      return static_cast<int>(count);
    }
    if (count >= minimum) return static_cast<int>(count);
    const int ret = enter(0, 1, IORING_ENTER_GETEVENTS);
    if (ret >= 0 || errno == EINTR) continue;
    return -1;
  }
}

int IoUringRing::get_completed_reads(size_t minimum, uint32_t *indices,
                                     size_t capacity) {
  const int ret = reap_reads(minimum, indices, capacity, true);
  if (ret < 0) cancel_reads();
  return ret;
}

void IoUringRing::cancel_reads() {
  if (!async_reads_) return;
  // Never touch caller destinations during cancellation. A caller may already
  // have unwound its result buffers before releasing this context.
  std::array<uint32_t, kIoUringMaxBatch> discarded{};
  size_t retries = 0;
  while (async_reads_) {
    if (async_reads_->completed == async_reads_->submitted) {
      async_reads_.reset();
      release_staging();
      break;
    }
    if (reap_reads(1, discarded.data(), discarded.size(), false) >= 0) continue;
    if (!async_reads_) break;
    if ((errno == EAGAIN || errno == EBUSY) &&
        retries++ < kIoUringDrainRetries) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    abandon_staging();
    async_reads_.reset();
    teardown();
  }
}

IoUringRing::~IoUringRing() {
  teardown();
  if (prepaid_staging_bytes_ && !staging_)
    MemoryLimitPool::get_instance().release_metadata(prepaid_staging_bytes_);
}

void IoUringRing::adopt_staging_reservation(size_t bytes) {
  assert(staging_ == nullptr && staging_charge_ == 0 &&
         prepaid_staging_bytes_ == 0);
  prepaid_staging_bytes_ = bytes;
}

bool IoUringRing::release_staging() {
  if (!staging_) return true;
  if (!File::TryMemoryUnmap(staging_, staging_size_)) {
    LOG_ERROR("io_uring staging unmap failed; size=%zu", staging_size_);
    return false;
  }
  auto &pool = MemoryLimitPool::get_instance();
  if (staging_charge_) {
    pool.release_unmapped_external(staging_charge_, staging_projection_);
  } else if (prepaid_staging_bytes_) {
    pool.release_materialization(staging_size_, staging_projection_);
  } else {
    pool.release_unmapped_process_backing(staging_size_, staging_projection_);
  }
  staging_ = nullptr;
  staging_size_ = 0;
  staging_charge_ = 0;
  staging_projection_ = {};
  return true;
}

bool IoUringRing::setup(uint32_t entries) {
  return setup_with_status(entries) == SetupResult::kReady;
}

IoUringRing::SetupResult IoUringRing::setup_with_status(uint32_t entries) {
  if (is_valid()) return SetupResult::kReady;
  auto &pool = MemoryLimitPool::get_instance();
  if (pool.under_process_rss_pressure()) return SetupResult::kNoMemory;
  struct io_uring_params params;
  std::memset(&params, 0, sizeof(params));

  // io_uring_setup is a raw syscall — returns fd (>=0) or -1 with errno.
  ring_fd_ = static_cast<int>(
      syscall(__NR_io_uring_setup, static_cast<int>(entries), &params));
  if (ring_fd_ < 0) {
    // ENOSYS = kernel doesn't support io_uring.
    // EPERM  = io_uring disabled via sysctl.
    // EINVAL = invalid parameters.
    const int error = errno;
    if (error != ENOSYS) {
      LOG_WARN("io_uring_setup failed; errno=%d, %s", error, ::strerror(error));
    }
    return error == ENOMEM ? SetupResult::kNoMemory : SetupResult::kUnavailable;
  }

  // The read path uses IORING_OP_READ (Linux 5.6+).  On older kernels
  // (5.1–5.5) io_uring_setup() succeeds but every read would fail with
  // -EINVAL, so reject the ring here and let callers fall back.
  if ((params.features & IORING_FEAT_RW_CUR_POS) == 0) {
    LOG_WARN(
        "io_uring lacks IORING_OP_READ support (kernel < 5.6); falling "
        "back");
    teardown();
    return SetupResult::kUnavailable;
  }

  bool ready = false;
  const auto rollback = ScopeGuard::Make([&] {
    if (!ready) teardown();
  });
  const auto result = map_control_regions(params);
  ready = result == SetupResult::kReady;
  return result;
}

IoUringRing::SetupResult IoUringRing::map_control_regions(
    const io_uring_params &params) {
  sq_entries_ = params.sq_entries;
  cq_entries_ = params.cq_entries;
  sq_ring_size_ =
      static_cast<size_t>(params.sq_off.array) + sq_entries_ * sizeof(uint32_t);
  sqes_size_ = sq_entries_ * sizeof(struct io_uring_sqe);
  cq_ring_size_ = static_cast<size_t>(params.cq_off.cqes) +
                  cq_entries_ * sizeof(struct io_uring_cqe);

  auto &pool = MemoryLimitPool::get_instance();
  // A thread-local ring can outlive the storage that first used it. Charge
  // all mapped views to the process for their lifetime, independently of a
  // standalone cache's capacity or reinitialization. No eviction is needed
  // here, since the caller may already hold storage locks.
  const size_t sq_charge = BudgetedMappedBuffer::mapping_size(sq_ring_size_);
  const size_t sqes_charge = BudgetedMappedBuffer::mapping_size(sqes_size_);
  const size_t cq_charge = BudgetedMappedBuffer::mapping_size(cq_ring_size_);
  if (!sq_charge || !sqes_charge || !cq_charge ||
      sqes_charge > std::numeric_limits<size_t>::max() - sq_charge ||
      cq_charge > std::numeric_limits<size_t>::max() - sq_charge - sqes_charge)
    return SetupResult::kNoMemory;
  size_t pending = sq_charge + sqes_charge + cq_charge;
  if (!pool.try_reserve_process_backing(pending)) return SetupResult::kNoMemory;
  const auto cancel = ScopeGuard::Make([&] {
    if (pending) pool.finish_process_backing(pending, false);
  });

  auto map_failure = SetupResult::kUnavailable;
  const auto map_region = [&](size_t size, off_t offset, size_t charge,
                              ProcessRssBudget::MaterializationStamp &stamp) {
    void *region = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                          ring_fd_, offset);
    if (region == MAP_FAILED) {
      const int error = errno;
      if (error == ENOMEM) map_failure = SetupResult::kNoMemory;
      LOG_ERROR("io_uring control mapping failed: %s", ::strerror(error));
      return static_cast<void *>(nullptr);
    }
    stamp = pool.finish_process_backing(charge, true);
    pending -= charge;
    return region;
  };
  sq_ring_ptr_ = map_region(sq_ring_size_, IORING_OFF_SQ_RING, sq_charge,
                            sq_ring_projection_);
  if (!sq_ring_ptr_) return map_failure;
  sqes_ptr_ = static_cast<struct io_uring_sqe *>(
      map_region(sqes_size_, IORING_OFF_SQES, sqes_charge, sqes_projection_));
  if (!sqes_ptr_) return map_failure;
  cq_ring_ptr_ = map_region(cq_ring_size_, IORING_OFF_CQ_RING, cq_charge,
                            cq_ring_projection_);
  if (!cq_ring_ptr_) return map_failure;

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

  return SetupResult::kReady;
}

void IoUringRing::teardown() {
  cancel_reads();
  const auto release_region = [](void *region, size_t size,
                                 ProcessRssBudget::MaterializationStamp stamp) {
    if (!region) return;
    if (::munmap(region, size) == 0) {
      MemoryLimitPool::get_instance().release_unmapped_process_backing(
          BudgetedMappedBuffer::mapping_size(size), stamp);
    }
  };
  release_region(sq_ring_ptr_, sq_ring_size_, sq_ring_projection_);
  release_region(sqes_ptr_, sqes_size_, sqes_projection_);
  release_region(cq_ring_ptr_, cq_ring_size_, cq_ring_projection_);

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

  release_staging();

  sq_entries_ = 0;
  cq_entries_ = 0;
}

bool IoUringRing::ensure_staging(size_t bytes) {
  if (async_reads_) return false;
  const size_t mapped_bytes = BudgetedMappedBuffer::mapping_size(bytes);
  if (bytes != 0 && mapped_bytes == 0) return false;
  bytes = mapped_bytes;
  auto &pool = MemoryLimitPool::get_instance();
  if (prepaid_staging_bytes_ && pool.under_process_rss_pressure()) {
    // The synchronous path can use its existing cached page. Drop the
    // optional mapping before another writeback batch starts.
    release_staging();
    return false;
  }
  if (staging_size_ >= bytes) {
    return true;
  }
  // Writeback has already reserved its maximum batch. Allocate that capacity
  // once, avoiding both a second charge and overlapping growth allocations.
  if (prepaid_staging_bytes_) {
    if (bytes > prepaid_staging_bytes_) return false;
    bytes = prepaid_staging_bytes_;
  }
  // The prepaid workspace may be first used long after the pool opened.
  // Respect current RSS pressure before mapping it. Writeback can fall back
  // to pwrite using the existing cached page if staging is denied.
  const size_t charge = prepaid_staging_bytes_ || !pool.capacity() ? 0 : bytes;
  // Keep the old allocation charged until the replacement actually exists.
  using Pending = MemoryLimitPool::PendingAllocation;
  Pending pending(pool, Pending::Kind::kLockedExternal);
  if (!pending.reserve(charge)) return false;
  if (prepaid_staging_bytes_ && !pool.try_reserve_materialization(bytes))
    return false;
  const bool process_backing = !prepaid_staging_bytes_ && !charge;
  if (process_backing && !pool.try_reserve_process_backing(bytes)) return false;
  // Anonymous mappings are page-aligned, satisfying the I/O alignment. They
  // bypass malloc arenas, so release_staging() really returns physical pages.
  void *ptr = File::MemoryMap(bytes, 0);
  const auto prepaid_projection =
      prepaid_staging_bytes_
          ? pool.finish_materialization(bytes, ptr != nullptr)
          : ProcessRssBudget::MaterializationStamp{};
  const auto process_projection =
      process_backing ? pool.finish_process_backing(bytes, ptr != nullptr)
                      : ProcessRssBudget::MaterializationStamp{};
  if (!ptr) {
    LOG_WARN("io_uring staging mapping failed; size=%zu", bytes);
    return false;
  }
  if (!release_staging()) {
    if (!File::TryMemoryUnmap(ptr, bytes)) {
      LOG_ERROR("io_uring replacement staging unmap failed; size=%zu", bytes);
      pending.commit_mapping();
    } else if (prepaid_staging_bytes_) {
      pool.release_materialization(bytes, prepaid_projection);
    } else if (process_backing) {
      pool.release_unmapped_process_backing(bytes, process_projection);
    }
    return false;
  }
  staging_projection_ = prepaid_staging_bytes_ ? prepaid_projection
                        : process_backing      ? process_projection
                                               : pending.commit_mapping();
  staging_ = static_cast<char *>(ptr);
  staging_size_ = bytes;
  staging_charge_ = charge;
  return true;
}

int IoUringRing::execute(int fd, const IoUringRead *read_reqs, size_t count) {
  if (async_reads_) return -1;
  // Reuse across batches of this blocking call, not across a TLS ring's
  // lifetime. execute_impl drains or quarantines all outstanding I/O first.
  auto release = ScopeGuard::Make([&] { release_staging(); });
  return execute_impl(fd, read_reqs, nullptr, count) == 0 ? 0 : -1;
}

int IoUringRing::execute_writes(int fd, const IoUringWrite *write_reqs,
                                size_t count) {
  return execute_impl(fd, nullptr, write_reqs, count);
}

int IoUringRing::execute_impl(int fd, const IoUringRead *read_reqs,
                              const IoUringWrite *write_reqs, size_t count) {
  const bool is_write = write_reqs != nullptr;
  if (async_reads_ || !is_valid() ||
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

  const auto quarantine = [&] {
    // close() starts asynchronous kernel cancellation. Keep staging and its
    // reservation alive, and tell a writer that a synchronous retry is unsafe.
    abandon_staging();
    teardown();
    return kQuarantined;
  };

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

    try {
      size_t submitted = 0;
      size_t completed = 0;
      bool all_ok = true;
      std::array<bool, kIoUringMaxBatch> seen{};
      auto reap_available = [&]() {
        unsigned chead = *cq_head_;
        const unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        const unsigned cq_mask = *cq_ring_mask_;
        while (chead != ctail) {
          struct io_uring_cqe *cqe = &cqes_[chead & cq_mask];
          const size_t req_idx = static_cast<size_t>(cqe->user_data);
          if (req_idx < batch_start || req_idx >= batch_start + n_ops ||
              seen[req_idx - batch_start]) {
            LOG_ERROR("io_uring completion did not identify a new request: %zu",
                      req_idx);
            return false;
          } else {
            seen[req_idx - batch_start] = true;
            const uint64_t offset = is_write ? write_reqs[req_idx].offset
                                             : read_reqs[req_idx].offset;
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
        return true;
      };

      while (completed < n_ops) {
        if (!reap_available()) return quarantine();
        if (completed >= n_ops) {
          break;
        }

        const unsigned to_submit = static_cast<unsigned>(n_ops - submitted);
        const int ret = enter(to_submit, 1u, IORING_ENTER_GETEVENTS);
        if (ret >= 0) {
          if (static_cast<unsigned>(ret) > to_submit) return quarantine();
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
          if (!reap_available()) return quarantine();
          if (completed >= submitted) {
            break;
          }
          const int wait_ret = enter(0u, 1u, IORING_ENTER_GETEVENTS);
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
          return quarantine();
        }
        return -1;
      }

      if (!all_ok) {
        return -1;
      }
    } catch (...) {
      // After publication even a userspace exception cannot establish that
      // every kernel reference has gone away.
      return quarantine();
    }
  }
  return 0;
}

}  // namespace ailego
}  // namespace zvec

#endif  // __linux__
