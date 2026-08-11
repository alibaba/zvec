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

#include "diskann_file_reader.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>

#define MAX_EVENTS 1024

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

// Retry budget for draining in-flight io_uring requests when the kernel
// keeps returning EAGAIN/EBUSY (100 us sleep per retry, ~1 s total).
static constexpr size_t kIoUringDrainRetries = 10000;

// Ensures the I/O backend selection is logged exactly once per process,
// regardless of which entry point (setup_io_ctx or register_thread)
// triggers it first.
static std::once_flag g_io_backend_log_once;
#endif

void log_diskann_io_backend() {
#if (defined(__linux) || defined(__linux__))
  auto &backend = ailego::IOBackend::Instance();
  if (backend.is_pread()) {
    LOG_WARN(
        "DiskAnn: no async I/O backend available. Install libaio (e.g. "
        "'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) and "
        "retry. DiskAnn will fall back to synchronous pread() — performance "
        "will be degraded.");
  } else {
    LOG_INFO("DiskAnn: I/O backend '%s' loaded — async I/O enabled.",
             backend.name());
  }
#endif
}

int setup_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  std::call_once(g_io_backend_log_once, log_diskann_io_backend);
  if (ailego::IOBackend::Instance().is_pread()) {
    // No async backend available — leave ctx null so callers fall back to
    // synchronous pread().
    return 0;
  }

  ctx = new IoBackend();
  ailego::IOBackendType selected = ailego::IOBackend::Instance().available();

  // Priority 1: io_uring (raw kernel syscalls — zero dependency).
  if (selected == ailego::IOBackendType::kIoUring &&
      ctx->ring.setup(MAX_EVENTS)) {
    ctx->backend = IoBackend::IO_URING;
    return 0;
  }

  // Priority 2: libaio (dlopen — soft dependency).
  if (selected != ailego::IOBackendType::kPread &&
      LibAioLoader::Instance().load() &&
      LibAioLoader::Instance().is_available()) {
    int ret = LibAioLoader::Instance().io_setup(MAX_EVENTS, &ctx->aio_ctx);
    if (ret == 0) {
      ctx->backend = IoBackend::LIBAIO;
      return 0;
    }
    LOG_WARN("io_setup failed; returned: %d, %s. falling back to pread", ret,
             ::strerror(-ret));
  }

  // Priority 3: synchronous pread (always available).
  ctx->backend = IoBackend::NONE;
  return 0;
#else
  return 0;
#endif
}

int destroy_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  if (ctx == nullptr) {
    return 0;
  }

  if (ctx->backend == IoBackend::IO_URING) {
    ctx->ring.teardown();
  } else if (ctx->backend == IoBackend::LIBAIO &&
             LibAioLoader::Instance().is_available()) {
    LibAioLoader::Instance().io_destroy(ctx->aio_ctx);
  }
  // IoUringRing destructor also calls teardown() — idempotent and safe.

  delete ctx;
  ctx = nullptr;
  return 0;
#else
  return 0;
#endif
}

static int execute_io_pread(int fd, std::vector<AlignedRead> &read_reqs) {
  for (auto &req : read_reqs) {
    ssize_t bytes_read = ::pread(fd, req.buf, req.len, req.offset);
    if (bytes_read < 0) {
      LOG_ERROR("pread failed; errno=%d, %s, offset=%lu, len=%lu", errno,
                ::strerror(errno), (unsigned long)req.offset,
                (unsigned long)req.len);
      return IndexError_Runtime;
    }
    if ((size_t)bytes_read != req.len) {
      LOG_ERROR("pread short read; got=%zd, expected=%lu", bytes_read,
                (unsigned long)req.len);
      return IndexError_Runtime;
    }
  }
  return 0;
}

#if (defined(__linux) || defined(__linux__))
// io_getevents() should only fail permanently for an invalid context or
// invalid arguments. If that happens after submission, io_destroy() is the
// only safe way to quiesce the context before synchronous I/O touches the same
// destination buffers. Recreate the context so later reads can still use AIO.
static bool reset_aio_context(io_context_t &ctx) {
  auto &loader = LibAioLoader::Instance();
  int ret;
  do {
    ret = loader.io_destroy(ctx);
  } while (ret == -EINTR);

  if (ret != 0) {
    LOG_ERROR("io_destroy failed while draining AIO; returned: %d, %s", ret,
              ::strerror(-ret));
    return false;
  }

  ctx = nullptr;
  io_context_t replacement = nullptr;
  ret = loader.io_setup(MAX_EVENTS, &replacement);
  if (ret != 0) {
    LOG_ERROR(
        "io_setup failed while recreating an AIO context; returned: %d, %s. "
        "this context will use pread",
        ret, ::strerror(-ret));
    return true;
  }
  ctx = replacement;
  return true;
}

int execute_io_libaio(io_context_t &ctx, int fd,
                      std::vector<AlignedRead> &read_reqs, uint64_t n_retries) {
  uint64_t iters = DiskAnnUtil::div_round_up(read_reqs.size(), MAX_EVENTS);

  for (uint64_t iter = 0; iter < iters; iter++) {
    uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS),
                              (uint64_t)MAX_EVENTS);

    std::vector<iocb_t *> cbs(n_ops, nullptr);
    std::vector<io_event_t> evts(n_ops);
    std::vector<struct iocb> cb(n_ops);
    for (uint64_t j = 0; j < n_ops; j++) {
      io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf,
                    read_reqs[j + iter * MAX_EVENTS].len,
                    read_reqs[j + iter * MAX_EVENTS].offset);
    }

    for (uint64_t i = 0; i < n_ops; i++) {
      cbs[i] = cb.data() + i;
    }

    size_t n_tries = 0;
    size_t submitted = 0;
    bool submission_ok = true;

    // Phase 1: accumulate partial submissions. A positive return value means
    // that exactly that prefix is now in flight and must never be submitted
    // again.
    while (submitted < n_ops) {
      size_t remaining = n_ops - submitted;
      int ret = LibAioLoader::Instance().io_submit(ctx, (int64_t)remaining,
                                                   cbs.data() + submitted);
      if (ret > 0 && static_cast<size_t>(ret) <= remaining) {
        submitted += static_cast<size_t>(ret);
        n_tries = 0;
        continue;
      }
      if ((ret == -EAGAIN || ret == -EINTR) && n_tries < n_retries) {
        n_tries++;
        continue;
      }
      LOG_WARN(
          "io_submit stopped after %zu/%lu requests; returned: %d. "
          "falling back to pread after draining submitted AIO",
          submitted, (unsigned long)n_ops, ret);
      submission_ok = false;
      break;
    }

    // Phase 2: accumulate completions for every request that was actually
    // submitted. Partial completion is normal and must not trigger fallback:
    // the remaining requests can still write into the caller's buffers.
    size_t completed = 0;
    while (completed < submitted) {
      size_t remaining = submitted - completed;
      int ret = LibAioLoader::Instance().io_getevents(
          ctx, (int64_t)remaining, (int64_t)remaining, evts.data() + completed,
          nullptr);
      if (ret > 0 && static_cast<size_t>(ret) <= remaining) {
        completed += static_cast<size_t>(ret);
        continue;
      }
      if (ret == -EINTR) {
        // Once requests are in flight, EINTR cannot safely turn into pread
        // regardless of the caller's submission retry budget.
        continue;
      }

      LOG_ERROR(
          "io_getevents failed after %zu/%zu completions; returned: %d, %s. "
          "resetting the AIO context before falling back to pread",
          completed, submitted, ret,
          ret < 0 ? ::strerror(-ret) : "invalid completion count");
      if (!reset_aio_context(ctx)) {
        // Do not run pread unless io_destroy confirmed that no request can
        // still write into these buffers.
        return IndexError_Runtime;
      }
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 3: verify every harvested event. Completion order is unspecified,
    // so use io_event::obj instead of assuming it matches request order.
    bool all_ok = true;
    std::vector<bool> seen(submitted, false);
    for (size_t i = 0; i < completed; i++) {
      auto cb_it = std::find(cbs.begin(), cbs.begin() + submitted, evts[i].obj);
      if (cb_it == cbs.begin() + submitted) {
        LOG_WARN("aio completion %zu referenced an unknown request", i);
        all_ok = false;
        continue;
      }

      size_t request_index = static_cast<size_t>(cb_it - cbs.begin());
      const AlignedRead &req = read_reqs[request_index + iter * MAX_EVENTS];
      int64_t result = static_cast<int64_t>(evts[i].res);
      int64_t result2 = static_cast<int64_t>(evts[i].res2);
      if (seen[request_index] || result != static_cast<int64_t>(req.len) ||
          result2 != 0) {
        LOG_WARN(
            "aio request %zu failed: res=%ld, res2=%ld, expected=%lu, "
            "offset=%lu",
            request_index, (long)result, (long)result2, (unsigned long)req.len,
            (unsigned long)req.offset);
        all_ok = false;
      }
      seen[request_index] = true;
    }

    if (!submission_ok || !all_ok) {
      // All submitted requests have been harvested at this point. It is now
      // safe for synchronous reads to reuse their destination buffers.
      return execute_io_pread(fd, read_reqs);
    }
  }

  return 0;
}
#endif

int execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs,
               uint64_t n_retries = 0) {
#if (defined(__linux) || defined(__linux__))
  // Guard against null or sentinel contexts.
  if (ctx == nullptr || ctx == (IOContext)-1) {
    return execute_io_pread(fd, read_reqs);
  }

  // Dispatch based on the active backend.
  if (ctx->backend == IoBackend::IO_URING) {
    int ret = ctx->ring.execute(fd, read_reqs);
    if (ret == 0) {
      return 0;
    }
    // The kernel only ever writes into the ring-owned staging pool, never
    // into the caller's buffers, so a pread fallback can never race with
    // requests that are still in flight.
    LOG_WARN("io_uring execute failed; falling back to pread");
    return execute_io_pread(fd, read_reqs);
  }

  if (ctx->backend == IoBackend::LIBAIO) {
    return execute_io_libaio(ctx->aio_ctx, fd, read_reqs, n_retries);
  }

  // NONE backend — synchronous pread.
  return execute_io_pread(fd, read_reqs);
#else
  return execute_io_pread(fd, read_reqs);
#endif
}

// ---------------------------------------------------------------------------
// IoUringRing::execute — defined here (not in iouring_loader.h) because it
// accesses AlignedRead members, and AlignedRead is defined in
// diskann_file_reader.h after iouring_loader.h is included.
// ---------------------------------------------------------------------------
#if (defined(__linux) || defined(__linux__))
int IoUringRing::execute(int fd, std::vector<AlignedRead> &read_reqs) {
  if (!is_valid()) {
    return -1;
  }
  if (read_reqs.empty()) {
    return 0;
  }

  // Process in batches limited by the SQ ring size.
  uint32_t batch_size =
      std::min(sq_entries_, static_cast<uint32_t>(kIoUringMaxBatch));
  uint64_t iters = DiskAnnUtil::div_round_up(read_reqs.size(), batch_size);

  for (uint64_t iter = 0; iter < iters; iter++) {
    uint64_t n_ops =
        std::min(static_cast<uint64_t>(read_reqs.size()) - iter * batch_size,
                 static_cast<uint64_t>(batch_size));

    // --- Phase 1: Fill SQEs ---
    //
    // Reads land in the ring-owned staging pool, never in the caller's
    // buffers.  io_uring teardown is asynchronous — closing the ring fd
    // only initiates cancellation — so the kernel may still write into
    // request buffers after execute() has returned an error.  Staging
    // memory can simply be leaked in that case (abandon_staging()), while
    // the caller's buffers stay safe to reuse or free.  The copy-out below
    // costs one sector-scale memcpy per read, negligible next to the I/O.
    std::vector<size_t> slot_off(n_ops);
    size_t staging_bytes = 0;
    for (uint64_t j = 0; j < n_ops; j++) {
      slot_off[j] = staging_bytes;
      size_t len = read_reqs[j + iter * batch_size].len;
      // Round every slot up so each staging pointer stays O_DIRECT-legal.
      staging_bytes +=
          (len + kIoUringStagingAlign - 1) & ~(kIoUringStagingAlign - 1);
    }
    // Safe: the previous batch is fully drained before we get here, so no
    // in-flight request can reference the old pool being freed on growth.
    if (!ensure_staging(staging_bytes)) {
      return -1;  // nothing submitted; pread fallback is safe
    }

    unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
    unsigned mask = *sq_ring_mask_;

    for (uint64_t j = 0; j < n_ops; j++) {
      unsigned idx = (tail + static_cast<unsigned>(j)) & mask;
      unsigned sqe_idx = sq_array_[idx];
      struct io_uring_sqe *sqe = &sqes_[sqe_idx];

      uint64_t req_idx = j + iter * batch_size;
      io_uring_prep_read(sqe, fd, staging_ + slot_off[j],
                         static_cast<uint32_t>(read_reqs[req_idx].len),
                         read_reqs[req_idx].offset);
      // Store the request index so we can verify the completion.
      sqe->user_data = req_idx;
    }

    // Memory barrier: ensure SQE contents are visible before tail update.
    __sync_synchronize();
    __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(n_ops),
                     __ATOMIC_RELEASE);

    // --- Phase 2: Submit and reap completions ---
    //
    // io_uring_enter() returns the number of SQEs consumed, not the number
    // of CQEs available.  A partial submission returns before the wait
    // phase, and a signal can interrupt the wait while preserving a
    // positive submission count, so IORING_ENTER_GETEVENTS guarantees
    // min_complete completions only when the call finishes normally.
    // Completions must therefore be counted against cq_tail instead of
    // assuming n_ops CQEs are ready.
    uint64_t submitted = 0;
    uint64_t completed = 0;
    bool all_ok = true;

    // Consume every CQE the kernel has published so far and verify it.
    // Completion order is unspecified, so use cqe->user_data to find the
    // request instead of assuming submission order.
    auto reap_available = [&]() {
      unsigned chead = *cq_head_;  // single consumer — plain load is enough
      unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
      unsigned cq_mask = *cq_ring_mask_;
      if (chead == ctail) {
        return;
      }
      while (chead != ctail) {
        struct io_uring_cqe *cqe = &cqes_[chead & cq_mask];
        uint64_t req_idx = cqe->user_data;

        if (req_idx < iter * batch_size ||
            req_idx >= iter * batch_size + n_ops) {
          LOG_WARN("io_uring completion referenced unknown request: %lu",
                   (unsigned long)req_idx);
          all_ok = false;
        } else if (cqe->res < 0) {
          LOG_WARN("io_uring read failed: req=%lu, res=%d, offset=%lu",
                   (unsigned long)req_idx, cqe->res,
                   (unsigned long)read_reqs[req_idx].offset);
          all_ok = false;
        } else if (static_cast<uint64_t>(cqe->res) != read_reqs[req_idx].len) {
          LOG_WARN("io_uring short read: req=%lu, got=%d, expected=%lu",
                   (unsigned long)req_idx, cqe->res,
                   (unsigned long)read_reqs[req_idx].len);
          all_ok = false;
        } else {
          // Verified completion — copy from staging into the caller's
          // buffer.  This is the only place caller memory is written.
          std::memcpy(read_reqs[req_idx].buf,
                      staging_ + slot_off[req_idx - iter * batch_size],
                      read_reqs[req_idx].len);
        }
        chead++;
        completed++;
      }
      // Release: CQE reads must complete before the kernel may reuse slots.
      __atomic_store_n(cq_head_, chead, __ATOMIC_RELEASE);
    };

    while (completed < n_ops) {
      reap_available();
      if (completed >= n_ops) {
        break;
      }

      unsigned to_submit = static_cast<unsigned>(n_ops - submitted);
      int ret = static_cast<int>(syscall(
          __NR_io_uring_enter, ring_fd_, to_submit, 1u, IORING_ENTER_GETEVENTS,
          static_cast<void *>(nullptr), static_cast<size_t>(0)));
      if (ret >= 0) {
        submitted += static_cast<uint64_t>(ret);
        continue;
      }
      if (errno == EINTR) {
        // Interrupted during submit or wait; the SQEs already consumed are
        // tracked in `submitted`, so simply retry.
        continue;
      }
      if ((errno == EAGAIN || errno == EBUSY) && completed < submitted) {
        // Kernel resources are exhausted, but in-flight requests will free
        // them as they complete; keep reaping and retrying.
        continue;
      }

      // Unrecoverable failure (or EAGAIN with nothing in flight).
      LOG_WARN(
          "io_uring_enter failed; errno=%d, %s, submitted=%lu/%lu, "
          "completed=%lu. draining before falling back to pread",
          errno, ::strerror(errno), (unsigned long)submitted,
          (unsigned long)n_ops, (unsigned long)completed);

      // Un-publish the SQEs the kernel never consumed so a later batch
      // cannot submit them against stale buffers.
      __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(submitted),
                       __ATOMIC_RELEASE);

      // Drain every in-flight request before the staging pool may be
      // freed or reused by a later batch.  CQEs are posted to the shared
      // ring by the kernel on its own, so completions can still be reaped
      // here even when io_uring_enter() keeps failing.
      size_t drain_retries = 0;
      while (completed < submitted) {
        reap_available();
        if (completed >= submitted) {
          break;
        }
        int wret = static_cast<int>(syscall(
            __NR_io_uring_enter, ring_fd_, 0u, 1u, IORING_ENTER_GETEVENTS,
            static_cast<void *>(nullptr), static_cast<size_t>(0)));
        if (wret >= 0 || errno == EINTR) {
          continue;
        }
        if ((errno == EAGAIN || errno == EBUSY) &&
            drain_retries++ < kIoUringDrainRetries) {
          // Give in-flight requests time to complete; entering the kernel
          // via the sleep also lets pending completion task-work run.
          std::this_thread::sleep_for(std::chrono::microseconds(100));
          continue;
        }
        // The ring cannot be drained.  Leak the staging pool — the kernel
        // may keep writing into it through the asynchronous teardown — and
        // disable io_uring for this context.  The caller's buffers were
        // never exposed to the kernel, so the pread fallback stays safe.
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
      // Every request completed and the staging pool is quiesced, but at
      // least one read failed or was short — let the caller retry with
      // pread.
      return -1;
    }
  }

  return 0;
}
#endif  // __linux__

LinuxAlignedFileReader::LinuxAlignedFileReader(int file_desc) {
  this->file_desc = file_desc;
}

LinuxAlignedFileReader::LinuxAlignedFileReader() {
  this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader() {
  deregister_all_threads();
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

IOContext &LinuxAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  auto it = ctx_map.find(std::this_thread::get_id());
  if (it == ctx_map.end()) {
    LOG_ERROR("bad thread access; returning invalid IOContext");
    return this->bad_ctx;
  } else {
    return it->second;
  }
}

void LinuxAlignedFileReader::register_thread() {
#if (defined(__linux) || defined(__linux__))
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(thread_id) != ctx_map.end()) {
    LOG_ERROR("multiple calls to register_thread from the same thread");
    return;
  }

  IOContext ctx = nullptr;
  int ret = setup_io_ctx(ctx);
  if (ret != 0) {
    LOG_ERROR("setup_io_ctx failed; returned: %d", ret);
    lk.unlock();
    return;
  }
  if (ctx != nullptr) {
    LOG_INFO("allocating ctx: %lu", (uint64_t)ctx);
  }
  ctx_map[thread_id] = ctx;
  lk.unlock();
#endif
}

void LinuxAlignedFileReader::deregister_thread() {
#if (defined(__linux) || defined(__linux__))
  auto thread_id = std::this_thread::get_id();
  IOContext ctx;

  {
    std::lock_guard<std::mutex> lk(ctx_mut);
    auto it = ctx_map.find(thread_id);
    if (it == ctx_map.end()) {
      LOG_ERROR("deregister_thread: thread not registered");
      return;
    }
    ctx = it->second;
    ctx_map.erase(it);
  }

  // Teardown is a syscall; keep it outside the lock to avoid blocking others.
  destroy_io_ctx(ctx);
  LOG_INFO("returned ctx from thread");
#endif
}

void LinuxAlignedFileReader::deregister_all_threads() {
#if (defined(__linux) || defined(__linux__))
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    destroy_io_ctx(x->second);
  }
  ctx_map.clear();
#endif
}

void LinuxAlignedFileReader::open(const std::string &fname) {
  int flags = O_RDONLY;

#if defined(__linux__) || defined(__linux)
  flags |= O_DIRECT | O_LARGEFILE;
#endif

  this->file_desc = ::open(fname.c_str(), flags);

#if defined(__linux__) || defined(__linux)
  // O_DIRECT may not be supported on all filesystems (e.g. tmpfs, overlay).
  // Fall back to regular buffered I/O when it fails.
  if (this->file_desc == -1) {
    LOG_WARN(
        "open with O_DIRECT failed for %s (errno=%d: %s), "
        "falling back to buffered I/O",
        fname.c_str(), errno, ::strerror(errno));
    this->file_desc = ::open(fname.c_str(), O_RDONLY | O_LARGEFILE);
  }
#endif

  if (this->file_desc == -1) {
    LOG_ERROR("Failed to open file: %s (errno=%d: %s)", fname.c_str(), errno,
              ::strerror(errno));
  }

  LOG_INFO("Opened file : %s", fname.c_str());
}

void LinuxAlignedFileReader::close() {
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

int LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                 IOContext &ctx, bool async) {
  if (async == true) {
    LOG_WARN("Async currently not supported");
  }

  if (this->file_desc == -1) {
    LOG_ERROR("Attempt to read from invalid file descriptor");
    return IndexError_Runtime;
  }

  int ret = execute_io(ctx, this->file_desc, read_reqs);

  return ret;
}

#if (defined(__linux) || defined(__linux__))
int LinuxAlignedFileReader::submit(PendingBatch &batch,
                                   std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx) {
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
  batch.cbs.clear();
  batch.cb_ptrs.clear();

  if (this->file_desc == -1) {
    LOG_ERROR("submit: invalid file descriptor");
    return IndexError_Runtime;
  }

  if (read_reqs.empty()) {
    return 0;
  }

  // If this context has no async I/O backend (null/sentinel context or
  // explicit pread backend), use synchronous pread.
  if (ctx == nullptr || ctx == (IOContext)-1 ||
      ctx->backend == IoBackend::NONE) {
    int pread_ret = execute_io_pread(this->file_desc, read_reqs);
    if (pread_ret != 0) {
      return pread_ret;
    }
    batch.used_pread = true;
    batch.n_submitted = (uint32_t)read_reqs.size();
    return 0;
  }

  // io_uring only offers a synchronous batched execute(): the reads are
  // already copied into the caller's buffers when it returns, so report the
  // batch as complete the same way the pread path does.
  if (ctx->backend == IoBackend::IO_URING) {
    int ring_ret = ctx->ring.execute(this->file_desc, read_reqs);
    if (ring_ret != 0) {
      // The kernel only ever writes into the ring-owned staging pool, so a
      // pread fallback cannot race with requests still in flight.
      LOG_WARN("submit: io_uring execute failed; falling back to pread");
      int pread_ret = execute_io_pread(this->file_desc, read_reqs);
      if (pread_ret != 0) {
        return pread_ret;
      }
    }
    batch.used_pread = true;
    batch.n_submitted = (uint32_t)read_reqs.size();
    return 0;
  }

  uint32_t n_ops = (uint32_t)read_reqs.size();
  batch.cbs.resize(n_ops);
  batch.cb_ptrs.resize(n_ops);

  for (uint32_t j = 0; j < n_ops; j++) {
    io_prep_pread(&batch.cbs[j], this->file_desc, read_reqs[j].buf,
                  read_reqs[j].len, read_reqs[j].offset);
    batch.cbs[j].data = (void *)(uintptr_t)j;
    batch.cb_ptrs[j] = &batch.cbs[j];
  }

  int ret = LibAioLoader::Instance().io_submit(ctx->aio_ctx, (int64_t)n_ops,
                                               batch.cb_ptrs.data());
  if (ret == (int)n_ops) {
    batch.n_submitted = n_ops;
    return 0;
  }

  // Partial submission: a positive return value means exactly that prefix is
  // now in flight and must never be submitted again. Keep submitting the
  // remainder; -EAGAIN/-EINTR are transient and worth a bounded retry.
  constexpr size_t kMaxSubmitRetries = 8;
  uint32_t submitted = (ret > 0 && ret < (int)n_ops) ? (uint32_t)ret : 0;
  size_t n_tries = 0;
  bool submission_ok = (submitted > 0) || ret == -EAGAIN || ret == -EINTR;
  while (submission_ok && submitted < n_ops) {
    uint32_t remaining = n_ops - submitted;
    ret = LibAioLoader::Instance().io_submit(ctx->aio_ctx, (int64_t)remaining,
                                             batch.cb_ptrs.data() + submitted);
    if (ret > 0 && (uint32_t)ret <= remaining) {
      submitted += (uint32_t)ret;
      n_tries = 0;
      continue;
    }
    if ((ret == -EAGAIN || ret == -EINTR) && n_tries < kMaxSubmitRetries) {
      n_tries++;
      continue;
    }
    submission_ok = false;
  }

  if (submission_ok) {
    batch.n_submitted = n_ops;
    return 0;
  }

  LOG_WARN(
      "submit: io_submit stopped after %u/%u requests; returned: %d. "
      "falling back to pread after draining submitted AIO",
      submitted, n_ops, ret);

  // Drain every request already in flight before any synchronous read can
  // reuse its destination buffer, and before batch.cbs may be reused; the
  // kernel keeps writing through those iocbs until their events are reaped.
  std::vector<io_event_t> evts(submitted);
  uint32_t drained = 0;
  while (drained < submitted) {
    uint32_t remaining = submitted - drained;
    ret = LibAioLoader::Instance().io_getevents(
        ctx->aio_ctx, (int64_t)remaining, (int64_t)remaining,
        evts.data() + drained, nullptr);
    if (ret > 0 && (uint32_t)ret <= remaining) {
      drained += (uint32_t)ret;
      continue;
    }
    if (ret == -EINTR) {
      continue;
    }
    LOG_ERROR(
        "submit: io_getevents failed while draining %u in-flight requests; "
        "returned: %d. resetting the AIO context before falling back to pread",
        submitted, ret);
    if (!reset_aio_context(ctx->aio_ctx)) {
      // Do not run pread unless io_destroy confirmed that no request can
      // still write into these buffers.
      return IndexError_Runtime;
    }
    break;
  }

  int pread_ret = execute_io_pread(this->file_desc, read_reqs);
  if (pread_ret != 0) {
    return pread_ret;
  }
  batch.used_pread = true;
  batch.n_submitted = n_ops;
  return 0;
}

// Quiesce any requests of the batch still in flight before reporting an
// error, so the kernel cannot keep writing into the caller's buffers or
// leave stale completion events for the next batch on this context.
static void quiesce_batch(PendingBatch &batch, IOContext &ctx) {
  // Only the libaio path leaves requests in flight: pread and io_uring
  // batches are complete before submit() returns (used_pread == true).
  if (batch.n_reaped < batch.n_submitted && !batch.used_pread) {
    if (reset_aio_context(ctx->aio_ctx)) {
      batch.n_reaped = batch.n_submitted;
    }
  }
}

int LinuxAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext &ctx, int min_completed,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();

  if (batch.n_reaped >= batch.n_submitted) {
    return 0;
  }

  if (batch.used_pread) {
    for (uint32_t i = batch.n_reaped; i < batch.n_submitted; i++) {
      completed_indices.push_back(i);
    }
    batch.n_reaped = batch.n_submitted;
    return (int)completed_indices.size();
  }

  uint32_t n_remaining = batch.n_submitted - batch.n_reaped;
  int min_req = std::min((int)n_remaining, min_completed);
  if (min_req < 1) min_req = 1;

  std::vector<io_event_t> evts(n_remaining);
  int ret;
  do {
    // Once requests are in flight, EINTR must be retried: returning here
    // would leave them unquiesced, free to overwrite the caller's buffers
    // or leak completion events into the next batch.
    ret = LibAioLoader::Instance().io_getevents(ctx->aio_ctx, (int64_t)min_req,
                                                (int64_t)n_remaining,
                                                evts.data(), nullptr);
  } while (ret == -EINTR);
  if (ret < 0) {
    LOG_ERROR("get_completed: io_getevents failed, ret=%d, %s", ret,
              ::strerror(-ret));
    quiesce_batch(batch, ctx);
    return IndexError_Runtime;
  }

  for (int i = 0; i < ret; i++) {
    uint32_t idx = (uint32_t)(uintptr_t)evts[i].data;
    if (idx >= batch.n_submitted) {
      LOG_ERROR("get_completed: completion referenced an unknown request %u",
                idx);
      batch.n_reaped += (uint32_t)ret;
      quiesce_batch(batch, ctx);
      return IndexError_Runtime;
    }
    int64_t res = (int64_t)evts[i].res;
    int64_t res2 = (int64_t)evts[i].res2;
    int64_t expected = (int64_t)batch.cbs[idx].u.c.nbytes;
    if (res != expected || res2 != 0) {
      // The async read failed, so the destination buffer content is
      // undefined. Degrade to a synchronous pread for this request before
      // handing the buffer to the caller.
      LOG_WARN(
          "get_completed: read %u failed: res=%ld, res2=%ld, expected=%ld; "
          "retrying with pread",
          idx, (long)res, (long)res2, (long)expected);
      ssize_t bytes_read =
          ::pread(this->file_desc, batch.cbs[idx].u.c.buf,
                  batch.cbs[idx].u.c.nbytes, batch.cbs[idx].u.c.offset);
      if (bytes_read != (ssize_t)expected) {
        LOG_ERROR(
            "get_completed: pread retry for read %u failed; got=%zd, "
            "expected=%ld, errno=%d, %s",
            idx, bytes_read, (long)expected, errno, ::strerror(errno));
        batch.n_reaped += (uint32_t)ret;
        quiesce_batch(batch, ctx);
        return IndexError_Runtime;
      }
    }
    completed_indices.push_back(idx);
  }

  batch.n_reaped += (uint32_t)ret;
  return ret;
}
#endif


}  // namespace core
}  // namespace zvec
