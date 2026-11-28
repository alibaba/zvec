// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <mutex>
#include <zvec/ailego/buffer/memory_reservation.h>
#include <zvec/ailego/parallel/worker_memory.h>

namespace zvec::ailego {
namespace {
class WorkerMemoryBudget {
 public:
  static WorkerMemoryBudget &instance() {
    // Pools may finish during static teardown. Keep accounting alive until
    // process exit, just like the permanently retained worker backing.
    static auto *budget = new WorkerMemoryBudget;
    return *budget;
  }

  void acquire(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto &pool = MemoryLimitPool::get_instance();
    if (bytes > std::numeric_limits<size_t>::max() - active_)
      throw std::bad_alloc();
    const size_t target = active_ + bytes;
    if (target > reservation_.bytes()) {
      reservation_.add(target - reservation_.bytes(), bytes);
    } else if (!pool.try_materialize(bytes)) {
      throw std::bad_alloc();
    }
    active_ = target;
  }
  void started(size_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    started_ += bytes;
    assert(started_ <= active_);
    retained_ = std::max(retained_, started_);
  }
  void release(size_t bytes, size_t started) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(active_ >= bytes && started_ >= started);
    active_ -= bytes;
    started_ -= started;
    reservation_.retain_at_most(std::max(active_, retained_));
  }

 private:
  std::mutex mutex_;
  MemoryReservation reservation_;
  size_t active_{0};
  size_t started_{0};
  size_t retained_{0};
};
}  // namespace

WorkerMemoryLease::WorkerMemoryLease(size_t workers) {
  if (!workers || !MemoryLimitPool::get_instance().process_rss_limit()) return;
  worker_bytes_ = DefaultNativeWorkerStackBytes();
  if (!worker_bytes_) return;
  if (workers > std::numeric_limits<size_t>::max() / worker_bytes_)
    throw std::bad_alloc();
  const size_t bytes = workers * worker_bytes_;
  WorkerMemoryBudget::instance().acquire(bytes);
  bytes_ = bytes;
}

void WorkerMemoryLease::worker_started() noexcept {
  if (!bytes_) return;
  assert(worker_bytes_ <= bytes_ - started_bytes_);
  WorkerMemoryBudget::instance().started(worker_bytes_);
  started_bytes_ += worker_bytes_;
}

WorkerMemoryLease::~WorkerMemoryLease() {
  if (bytes_) WorkerMemoryBudget::instance().release(bytes_, started_bytes_);
}
}  // namespace zvec::ailego
