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
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>
#if defined(_WIN32) || defined(_WIN64)
#include <zvec/ailego/utility/file_helper.h>
#endif
#if defined(__APPLE__) || defined(__MACH__)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define MAX_EVENTS 1024

namespace zvec {
namespace core {

// Ensures the I/O backend selection is logged exactly once per process,
// regardless of which DiskAnn entry point triggers it first.
static std::once_flag g_io_backend_log_once;

static void log_diskann_io_backend(ailego::IOBackendType type) {
#if (defined(__linux) || defined(__linux__) || defined(__APPLE__) || \
     defined(__MACH__) || defined(_WIN32) || defined(_WIN64))
  std::call_once(g_io_backend_log_once, [type]() {
#if (defined(__linux) || defined(__linux__))
    if (type == ailego::IOBackendType::kPread) {
      LOG_WARN(
          "DiskAnn: no async I/O backend available: io_uring is unavailable "
          "and libaio could not be loaded. Enable io_uring or install libaio "
          "(e.g. 'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) "
          "and retry. DiskAnn will use synchronous pread(); performance may "
          "be degraded.");
    } else {
      LOG_INFO("DiskAnn: I/O backend '%s' loaded — async I/O enabled.",
               ailego::IOBackendTypeName(type));
    }
#elif defined(_WIN32) || defined(_WIN64)
    LOG_INFO("DiskAnn: I/O backend '%s' — asynchronous I/O enabled.",
             ailego::IOBackendTypeName(type));
#else
    LOG_INFO("DiskAnn: I/O backend '%s' — synchronous I/O enabled.",
             ailego::IOBackendTypeName(type));
#endif
  });
#else
  (void)type;
#endif
}

#if (defined(__linux) || defined(__linux__))
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

// Retry budget for draining in-flight io_uring requests when the kernel
// keeps returning EAGAIN/EBUSY (100 us sleep per retry, ~1 s total).
static constexpr size_t kIoUringDrainRetries = 10000;
#endif

void log_diskann_io_backend() {
  log_diskann_io_backend(ailego::IOBackend::Instance().available());
}

#if defined(_WIN32) || defined(_WIN64)
// The reader's stable handle owns the selected file object across lazy I/O
// batches. Allow its path to be deleted or atomically replaced, but keep
// in-place writes blocked while reads are active.
static constexpr DWORD kDiskAnnFileShareMode =
    FILE_SHARE_READ | FILE_SHARE_DELETE;
// Every handle kept beside an IOCP handle must bypass the system cache. A
// buffered handle to the same file object substantially degrades 4 KiB random
// reads even when the actual ReadFile calls use a separate unbuffered handle.
static constexpr DWORD kDiskAnnStableHandleFlags = FILE_FLAG_NO_BUFFERING;

// Threads stay associated with an IOCP after dequeuing a completion. An
// IOContext can move between serialized callers, so a concurrency limit of one
// can permanently starve a later caller while the previous thread remains
// runnable. Callers still serialize each context, and outstanding_count
// rejects overlapping batches; keep the private port effectively unthrottled.
static constexpr DWORD kDiskAnnIoCompletionConcurrency =
    static_cast<DWORD>(MAXLONG);

// An IOContext may be reused with different reader instances. Assign every
// successfully opened file object a process-wide identity so two readers for
// the same path cannot accidentally share an IOCP handle to stale contents.
static std::atomic<uint64_t> g_next_windows_file_identity{1};

static uint64_t next_windows_file_identity() {
  uint64_t identity =
      g_next_windows_file_identity.load(std::memory_order_relaxed);
  while (identity != 0) {
    const uint64_t next =
        identity == std::numeric_limits<uint64_t>::max() ? 0 : identity + 1;
    if (g_next_windows_file_identity.compare_exchange_weak(
            identity, next, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return identity;
    }
  }
  return 0;
}

// Cancel and reap every request that may still reference caller-owned buffers.
// Closing the file or completion port before the cancellation packets have
// arrived would allow the kernel to keep writing into buffers already returned
// to the caller.
static void close_windows_io_handles(IOContext ctx) {
  if (ctx == nullptr) {
    return;
  }

  size_t active_remaining = static_cast<size_t>(
      std::count_if(ctx->active_requests.begin(), ctx->active_requests.end(),
                    [](uint8_t active) { return active != 0; }));
  // The branches below are internal-state failures, not ordinary I/O errors.
  // Returning after one of them could free an OVERLAPPED or destination
  // buffer that the kernel still owns, so fail closed instead of risking UAF.
  if (active_remaining != ctx->outstanding_count) {
    LOG_FATAL("DiskAnn Windows I/O context lost track of outstanding requests");
    std::abort();
  }
  if (active_remaining != 0) {
    if (ctx->file_handle == INVALID_HANDLE_VALUE ||
        ctx->completion_port == nullptr) {
      LOG_FATAL(
          "Cannot drain DiskAnn overlapped requests without valid handles");
      std::abort();
    }
    if (!::CancelIoEx(ctx->file_handle, nullptr)) {
      DWORD error = ::GetLastError();
      if (error != ERROR_NOT_FOUND) {
        LOG_WARN("CancelIoEx failed while draining DiskAnn I/O (error=%lu)",
                 error);
      }
    }

    // A file associated with an IOCP does not publish the final OVERLAPPED
    // status until its completion packet is dequeued. Polling
    // GetOverlappedResult() here can therefore return ERROR_IO_INCOMPLETE
    // forever. The completion port is private to this context, so drain it
    // until every active slot has produced its terminal packet.
    while (active_remaining != 0) {
      OVERLAPPED_ENTRY entries[MAX_IO_DEPTH]{};
      ULONG removed = 0;
      const ULONG max_entries =
          static_cast<ULONG>(std::min<size_t>(active_remaining, MAX_IO_DEPTH));
      if (!::GetQueuedCompletionStatusEx(ctx->completion_port, entries,
                                         max_entries, &removed, INFINITE,
                                         FALSE) ||
          removed == 0) {
        LOG_FATAL(
            "GetQueuedCompletionStatusEx failed while draining DiskAnn I/O "
            "(error=%lu)",
            ::GetLastError());
        std::abort();
      }

      for (ULONG i = 0; i < removed; ++i) {
        if (entries[i].lpCompletionKey != reinterpret_cast<ULONG_PTR>(ctx)) {
          LOG_FATAL(
              "DiskAnn teardown received a completion for another context");
          std::abort();
        }

        const uintptr_t address =
            reinterpret_cast<uintptr_t>(entries[i].lpOverlapped);
        const uintptr_t begin = reinterpret_cast<uintptr_t>(ctx->reqs.data());
        const uintptr_t span = ctx->reqs.size() * sizeof(OVERLAPPED);
        if (address < begin || address >= begin + span ||
            (address - begin) % sizeof(OVERLAPPED) != 0) {
          LOG_FATAL("DiskAnn teardown received an unknown OVERLAPPED request");
          std::abort();
        }

        const size_t index = (address - begin) / sizeof(OVERLAPPED);
        if (ctx->active_requests[index] == 0) {
          LOG_FATAL("DiskAnn teardown received a duplicate completion");
          std::abort();
        }
        ctx->active_requests[index] = 0;
        --active_remaining;
        if (ctx->outstanding_count != 0) {
          --ctx->outstanding_count;
        }
      }
    }
    ctx->outstanding_count = 0;
  }

  if (ctx->file_handle != INVALID_HANDLE_VALUE) {
    ::CloseHandle(ctx->file_handle);
    ctx->file_handle = INVALID_HANDLE_VALUE;
  }

  if (ctx->completion_port != nullptr) {
    ::CloseHandle(ctx->completion_port);
    ctx->completion_port = nullptr;
  }
  ctx->file_path.clear();
  ctx->file_identity = 0;
  ctx->active_requests.fill(0);
  ctx->outstanding_count = 0;
}
#endif

int setup_io_ctx(IOContext &ctx) {
  auto selected = ailego::IOBackend::Instance().available();
  ctx = new (std::nothrow) IoBackend();
  if (ctx == nullptr) {
    LOG_ERROR("Failed to allocate DiskAnn I/O context");
    return IndexError_NoMemory;
  }
  ctx->type = selected;

#if defined(_WIN32) || defined(_WIN64)
  log_diskann_io_backend(ctx->type);
  return 0;
#elif defined(__linux) || defined(__linux__)
  if (selected == ailego::IOBackendType::kPread) {
    log_diskann_io_backend(ctx->type);
    return 0;
  }

  // Priority 1: io_uring (raw kernel syscalls — zero dependency).
  if (selected == ailego::IOBackendType::kIoUring &&
      ctx->ring.setup(MAX_EVENTS)) {
    log_diskann_io_backend(ctx->type);
    return 0;
  }

  // Priority 2: libaio (dlopen — soft dependency).
  if (selected != ailego::IOBackendType::kPread &&
      LibAioLoader::Instance().load() &&
      LibAioLoader::Instance().is_available()) {
    int ret = LibAioLoader::Instance().io_setup(MAX_EVENTS, &ctx->aio_ctx);
    if (ret == 0) {
      ctx->type = ailego::IOBackendType::kLibAio;
      log_diskann_io_backend(ctx->type);
      return 0;
    }
    LOG_WARN("io_setup failed; returned: %d, %s. falling back to pread", ret,
             ::strerror(-ret));
  }

  // Priority 3: synchronous pread (always available).
  ctx->type = ailego::IOBackendType::kPread;
#endif
  log_diskann_io_backend(ctx->type);
  return 0;
}

int destroy_io_ctx(IOContext &ctx) {
  if (ctx == nullptr) {
    return 0;
  }

#if defined(_WIN32) || defined(_WIN64)
  close_windows_io_handles(ctx);
#elif defined(__linux) || defined(__linux__)
  if (ctx->type == ailego::IOBackendType::kIoUring) {
    ctx->ring.teardown();
  } else if (ctx->type == ailego::IOBackendType::kLibAio &&
             LibAioLoader::Instance().is_available()) {
    LibAioLoader::Instance().io_destroy(ctx->aio_ctx);
  }
  // IoUringRing destructor also calls teardown() — idempotent and safe.
#endif

  delete ctx;
  ctx = nullptr;
  return 0;
}

#if !defined(_WIN32) && !defined(_WIN64)
static int execute_one_pread(int fd, const AlignedRead &req) {
  auto *buf = static_cast<uint8_t *>(req.buf);
  uint64_t offset = req.offset;
  uint64_t remaining = req.len;

  while (remaining > 0) {
    ssize_t bytes_read =
        ::pread(fd, buf, static_cast<size_t>(remaining), offset);
    if (bytes_read > 0) {
      buf += bytes_read;
      offset += static_cast<uint64_t>(bytes_read);
      remaining -= static_cast<uint64_t>(bytes_read);
      continue;
    }
    if (bytes_read == 0) {
      LOG_ERROR("pread returned EOF; offset=%llu, remaining=%llu",
                (unsigned long long)offset, (unsigned long long)remaining);
      return IndexError_Runtime;
    }
    if (errno == EINTR) {
      continue;
    }

    LOG_ERROR("pread failed; errno=%d, %s, offset=%llu, len=%llu", errno,
              ::strerror(errno), (unsigned long long)offset,
              (unsigned long long)remaining);
    return IndexError_Runtime;
  }

  return 0;
}

static int execute_io_pread(int fd, std::vector<AlignedRead> &read_reqs) {
  for (const auto &req : read_reqs) {
    int ret = execute_one_pread(fd, req);
    if (ret != 0) {
      return ret;
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
  // A missing asynchronous context falls back to synchronous pread.
  if (ctx == nullptr) {
    return execute_io_pread(fd, read_reqs);
  }
  // Dispatch based on the active backend.
  if (ctx->type == ailego::IOBackendType::kIoUring) {
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

  if (ctx->type == ailego::IOBackendType::kLibAio) {
    return execute_io_libaio(ctx->aio_ctx, fd, read_reqs, n_retries);
  }

  // NONE backend — synchronous pread.
  return execute_io_pread(fd, read_reqs);
#else
  (void)ctx;
  (void)n_retries;
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
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

static int duplicate_file_descriptor(int source_fd) {
#if defined(F_DUPFD_CLOEXEC)
  return ::fcntl(source_fd, F_DUPFD_CLOEXEC, 0);
#else
  int duplicate_fd = ::dup(source_fd);
  if (duplicate_fd >= 0 && ::fcntl(duplicate_fd, F_SETFD, FD_CLOEXEC) == -1) {
    const int saved_errno = errno;
    ::close(duplicate_fd);
    errno = saved_errno;
    return -1;
  }
  return duplicate_fd;
#endif
}

#if defined(__linux__) || defined(__linux)
static int reopen_file_descriptor_with_direct_io(int source_fd) {
  // dup()/F_DUPFD_CLOEXEC shares one open-file description with source_fd, so
  // changing O_DIRECT through F_SETFL would also change the caller's buffered
  // FileReadStorage handle. Reopening the procfs descriptor gives DiskAnn an
  // independent open-file description while still referring to the exact
  // inode captured during metadata loading (including an unlinked/replaced
  // file). Some restricted environments do not mount procfs; callers fall
  // back to a buffered duplicate in that case.
  char fd_path[64];
  const int path_length =
      std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", source_fd);
  if (path_length <= 0 || static_cast<size_t>(path_length) >= sizeof(fd_path)) {
    errno = EINVAL;
    return -1;
  }

  int flags = O_RDONLY | O_DIRECT | O_LARGEFILE;
#if defined(O_CLOEXEC)
  flags |= O_CLOEXEC;
#endif
  return ::open(fd_path, flags);
}
#endif

#if defined(__APPLE__) || defined(__MACH__)
static int reopen_macos_file_descriptor(const std::string &fname,
                                        int source_fd) {
  int flags = O_RDONLY;
#if defined(O_CLOEXEC)
  flags |= O_CLOEXEC;
#endif
  int reopened_fd = ::open(fname.c_str(), flags);
  if (reopened_fd < 0) {
    return -1;
  }

#if !defined(O_CLOEXEC)
  if (::fcntl(reopened_fd, F_SETFD, FD_CLOEXEC) == -1) {
    const int saved_errno = errno;
    ::close(reopened_fd);
    errno = saved_errno;
    return -1;
  }
#endif

  struct stat source_stat {};
  struct stat reopened_stat {};
  if (::fstat(source_fd, &source_stat) == -1 ||
      ::fstat(reopened_fd, &reopened_stat) == -1) {
    const int saved_errno = errno;
    ::close(reopened_fd);
    errno = saved_errno;
    return -1;
  }
  if (source_stat.st_dev != reopened_stat.st_dev ||
      source_stat.st_ino != reopened_stat.st_ino) {
    ::close(reopened_fd);
    errno = ESTALE;
    return -1;
  }
  return reopened_fd;
}

static void configure_macos_reader(int file_desc, const std::string &fname) {
  // macOS has no O_DIRECT. F_NOCACHE is its closest per-file equivalent: it
  // asks the kernel to minimize caching for I/O through this descriptor. This
  // is advisory rather than a guarantee that every read reaches the device.
  // Disable read-ahead as well because DiskAnn performs random reads.
  //
  // Do not mmap the entire index and call msync(MS_INVALIDATE) here. That does
  // not provide a reliable global cache eviction guarantee and makes open time
  // and virtual-address usage scale with the size of the index.
  if (::fcntl(file_desc, F_NOCACHE, 1) == -1) {
    LOG_WARN(
        "fcntl(F_NOCACHE) failed for %s (errno=%d: %s); reads will use "
        "the page cache",
        fname.c_str(), errno, ::strerror(errno));
  } else {
    LOG_INFO("DiskAnn macOS: F_NOCACHE enabled for %s", fname.c_str());
  }

  if (::fcntl(file_desc, F_RDAHEAD, 0) == -1) {
    LOG_WARN("fcntl(F_RDAHEAD, 0) failed for %s (errno=%d: %s)", fname.c_str(),
             errno, ::strerror(errno));
  }
}
#endif

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

#if defined(__APPLE__) || defined(__MACH__)
  if (this->file_desc != -1) {
    configure_macos_reader(this->file_desc, fname);
  }
#endif

  LOG_INFO("Opened file : %s", fname.c_str());
}

int LinuxAlignedFileReader::open_from_handle(const std::string &fname,
                                             int source_fd) {
  close();
  if (source_fd < 0) {
    LOG_ERROR("Cannot capture DiskAnn file from an invalid descriptor");
    return IndexError_InvalidArgument;
  }

  int duplicate_fd = -1;
  bool has_independent_file_description = false;
#if defined(__linux__) || defined(__linux)
  duplicate_fd = reopen_file_descriptor_with_direct_io(source_fd);
  if (duplicate_fd < 0) {
    const int direct_errno = errno;
    duplicate_fd = duplicate_file_descriptor(source_fd);
    if (duplicate_fd >= 0) {
      LOG_WARN(
          "opening an independent O_DIRECT descriptor failed for %s "
          "(errno=%d: %s); falling back to a buffered duplicate",
          fname.c_str(), direct_errno, ::strerror(direct_errno));
    }
  } else {
    has_independent_file_description = true;
  }
#elif defined(__APPLE__) || defined(__MACH__)
  duplicate_fd = reopen_macos_file_descriptor(fname, source_fd);
  if (duplicate_fd < 0) {
    const int reopen_errno = errno;
    duplicate_fd = duplicate_file_descriptor(source_fd);
    if (duplicate_fd >= 0) {
      LOG_WARN(
          "opening an independent macOS descriptor failed for %s "
          "(errno=%d: %s); falling back to a buffered duplicate",
          fname.c_str(), reopen_errno, ::strerror(reopen_errno));
    }
  } else {
    has_independent_file_description = true;
  }
#else
  duplicate_fd = duplicate_file_descriptor(source_fd);
#endif
  if (duplicate_fd < 0) {
    LOG_ERROR(
        "Failed to duplicate DiskAnn file descriptor for %s "
        "(errno=%d: %s)",
        fname.c_str(), errno, ::strerror(errno));
    return IndexError_OpenFile;
  }

#if defined(__APPLE__) || defined(__MACH__)
  if (has_independent_file_description) {
    configure_macos_reader(duplicate_fd, fname);
  }
#else
  (void)has_independent_file_description;
#endif

  file_desc = duplicate_fd;
  LOG_INFO("Captured open DiskAnn file object: %s", fname.c_str());
  return 0;
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

  // If this context has no async I/O backend (null context or explicit pread
  // backend), use synchronous pread.
  if (ctx == nullptr || ctx->type == ailego::IOBackendType::kPread) {
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
  if (ctx->type == ailego::IOBackendType::kIoUring) {
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
      AlignedRead retry_read(static_cast<uint64_t>(batch.cbs[idx].u.c.offset),
                             static_cast<uint64_t>(batch.cbs[idx].u.c.nbytes),
                             batch.cbs[idx].u.c.buf);
      if (execute_one_pread(this->file_desc, retry_read) != 0) {
        LOG_ERROR("get_completed: pread retry for read %u failed", idx);
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
#else
int LinuxAlignedFileReader::submit(PendingBatch &batch,
                                   std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx) {
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;

  int ret = read(read_reqs, ctx);
  if (ret != 0) {
    return ret;
  }

  // The portable fallback completes reads synchronously.
  batch.used_pread = true;
  batch.n_submitted = static_cast<uint32_t>(read_reqs.size());
  return 0;
}

int LinuxAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext & /*ctx*/, int /*min_completed*/,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();

  for (uint32_t i = batch.n_reaped; i < batch.n_submitted; ++i) {
    completed_indices.push_back(i);
  }
  batch.n_reaped = batch.n_submitted;
  return static_cast<int>(completed_indices.size());
}
#endif

#else  // Windows

// Windows uses one file handle and one I/O completion port per IOContext, so a
// context can only dequeue completion packets for requests submitted through
// that context. PendingBatch keeps the expected lengths and completion bitmap
// alive until every request has been harvested.
WindowsAlignedFileReader::~WindowsAlignedFileReader() {
  close();
}

static bool resolve_windows_file_path(const std::string &fname,
                                      std::wstring &absolute_path) {
  const std::wstring wide_fname = ailego::FileHelper::Utf8ToWide(fname);
  if (wide_fname.empty()) {
    LOG_ERROR("Failed to convert DiskAnn file path from UTF-8: %s",
              fname.c_str());
    return false;
  }

  const DWORD path_capacity =
      ::GetFullPathNameW(wide_fname.c_str(), 0, nullptr, nullptr);
  if (path_capacity == 0) {
    LOG_ERROR("Failed to resolve absolute DiskAnn file path: %s (error=%lu)",
              fname.c_str(), ::GetLastError());
    return false;
  }
  absolute_path.assign(path_capacity, L'\0');
  const DWORD path_length = ::GetFullPathNameW(
      wide_fname.c_str(), path_capacity, absolute_path.data(), nullptr);
  if (path_length == 0 || path_length >= path_capacity) {
    LOG_ERROR("Failed to resolve absolute DiskAnn file path: %s (error=%lu)",
              fname.c_str(), ::GetLastError());
    absolute_path.clear();
    return false;
  }
  absolute_path.resize(path_length);
  return true;
}

void WindowsAlignedFileReader::open(const std::string &fname) {
  close();
  std::wstring absolute_path;
  if (!resolve_windows_file_path(fname, absolute_path)) {
    return;
  }

  HANDLE stable_file_handle = ::CreateFileW(
      absolute_path.c_str(), GENERIC_READ, kDiskAnnFileShareMode, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_READONLY | kDiskAnnStableHandleFlags,
      nullptr);
  if (stable_file_handle == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Failed to open file: %s (error=%lu)", fname.c_str(),
              ::GetLastError());
    return;
  }
  const uint64_t file_identity = next_windows_file_identity();
  if (file_identity == 0) {
    ::CloseHandle(stable_file_handle);
    LOG_ERROR("Exhausted DiskAnn Windows file identities");
    return;
  }
  stable_file_handle_ = stable_file_handle;
  file_path_ = std::move(absolute_path);
  file_identity_ = file_identity;
  LOG_INFO("Opened file: %s", fname.c_str());
}

int WindowsAlignedFileReader::open_from_handle(const std::string &fname,
                                               HANDLE source_handle) {
  close();
  if (source_handle == INVALID_HANDLE_VALUE || source_handle == nullptr) {
    LOG_ERROR("Cannot capture DiskAnn file from an invalid source handle");
    return IndexError_InvalidArgument;
  }

  std::wstring absolute_path;
  if (!resolve_windows_file_path(fname, absolute_path)) {
    return IndexError_InvalidArgument;
  }

  // ReOpenFile refers to the same underlying file object even if fname has
  // already been renamed or replaced. Keep this stable handle unbuffered too:
  // an ordinary buffered handle beside the private unbuffered IOCP handles can
  // severely reduce random-read throughput.
  HANDLE stable_file_handle =
      ::ReOpenFile(source_handle, GENERIC_READ, kDiskAnnFileShareMode,
                   kDiskAnnStableHandleFlags);
  if (stable_file_handle == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Failed to capture DiskAnn file object (error=%lu)",
              ::GetLastError());
    return IndexError_Runtime;
  }
  const uint64_t file_identity = next_windows_file_identity();
  if (file_identity == 0) {
    ::CloseHandle(stable_file_handle);
    LOG_ERROR("Exhausted DiskAnn Windows file identities");
    return IndexError_Runtime;
  }

  stable_file_handle_ = stable_file_handle;
  file_path_ = std::move(absolute_path);
  file_identity_ = file_identity;
  LOG_INFO("Captured open DiskAnn file object: %s", fname.c_str());
  return 0;
}

void WindowsAlignedFileReader::close() {
  if (stable_file_handle_ != INVALID_HANDLE_VALUE) {
    ::CloseHandle(stable_file_handle_);
    stable_file_handle_ = INVALID_HANDLE_VALUE;
  }
  file_path_.clear();
  file_identity_ = 0;
}

int WindowsAlignedFileReader::prepare_io_ctx(IOContext &ctx) {
  if (ctx == nullptr ||
      ctx->type != ailego::IOBackendType::kWindowsOverlapped) {
    LOG_ERROR("Attempt to prepare an invalid Windows I/O context");
    return IndexError_Runtime;
  }
  if (file_path_.empty() || stable_file_handle_ == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Attempt to read before opening a DiskAnn file");
    return IndexError_Runtime;
  }
  if (ctx->file_handle != INVALID_HANDLE_VALUE &&
      ctx->completion_port != nullptr && ctx->file_path == file_path_ &&
      ctx->file_identity == file_identity_) {
    return 0;
  }
  if (ctx->outstanding_count != 0) {
    LOG_ERROR("Cannot replace a Windows I/O context with requests in flight");
    return IndexError_Runtime;
  }

  close_windows_io_handles(ctx);
  // Derive each private IOCP handle from the file object captured by open().
  // Reopening the path here could bind a lazy context to a replacement index
  // while the indexer still holds metadata for the original one.
  ctx->file_handle =
      ::ReOpenFile(stable_file_handle_, GENERIC_READ, kDiskAnnFileShareMode,
                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
  if (ctx->file_handle == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Failed to reopen DiskAnn file object for IOCP (error=%lu)",
              ::GetLastError());
    return IndexError_Runtime;
  }

  ctx->completion_port = ::CreateIoCompletionPort(
      ctx->file_handle, nullptr, reinterpret_cast<ULONG_PTR>(ctx),
      kDiskAnnIoCompletionConcurrency);
  if (ctx->completion_port == nullptr) {
    LOG_ERROR("CreateIoCompletionPort failed (error=%lu)", ::GetLastError());
    close_windows_io_handles(ctx);
    return IndexError_Runtime;
  }
  if (!::SetFileCompletionNotificationModes(ctx->file_handle,
                                            FILE_SKIP_SET_EVENT_ON_HANDLE)) {
    LOG_WARN("SetFileCompletionNotificationModes failed (error=%lu)",
             ::GetLastError());
  }
  try {
    ctx->file_path = file_path_;
    ctx->file_identity = file_identity_;
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to store the Windows DiskAnn file path");
    close_windows_io_handles(ctx);
    return IndexError_NoMemory;
  }
  return 0;
}

void WindowsAlignedFileReader::reset_io_ctx(IOContext &ctx) {
  close_windows_io_handles(ctx);
}

void WindowsAlignedFileReader::release_io_ctx(IOContext &ctx) {
  close_windows_io_handles(ctx);
}

static int validate_windows_read_requests(
    const std::vector<AlignedRead> &read_reqs) {
  constexpr uint64_t kSectorLen = 4096;
  for (size_t i = 0; i < read_reqs.size(); ++i) {
    const AlignedRead &req = read_reqs[i];
    if (req.buf == nullptr ||
        reinterpret_cast<uintptr_t>(req.buf) % kSectorLen != 0 ||
        req.offset % kSectorLen != 0 || req.len % kSectorLen != 0) {
      LOG_ERROR(
          "Invalid unbuffered read request %zu: buffer=%p, offset=%llu, "
          "len=%llu; all values must be aligned to %llu bytes",
          i, req.buf, static_cast<unsigned long long>(req.offset),
          static_cast<unsigned long long>(req.len),
          static_cast<unsigned long long>(kSectorLen));
      return IndexError_InvalidArgument;
    }
    if (req.len > (std::numeric_limits<DWORD>::max)()) {
      LOG_ERROR("Windows read request %zu is too large: %llu bytes", i,
                static_cast<unsigned long long>(req.len));
      return IndexError_InvalidArgument;
    }
  }
  return 0;
}

int WindowsAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx, bool async) {
  if (async) {
    LOG_WARN(
        "read() waits for completion; use submit()/get_completed() for "
        "asynchronous Windows I/O");
  }
  int ret = validate_windows_read_requests(read_reqs);
  if (ret != 0) {
    return ret;
  }

  std::vector<uint32_t> completed;
  try {
    completed.reserve(MAX_IO_DEPTH);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  for (size_t start = 0; start < read_reqs.size(); start += MAX_IO_DEPTH) {
    const size_t count =
        std::min<size_t>(read_reqs.size() - start, MAX_IO_DEPTH);
    std::vector<AlignedRead> requests(read_reqs.begin() + start,
                                      read_reqs.begin() + start + count);
    PendingBatch batch;
    ret = submit(batch, requests, ctx);
    if (ret != 0) {
      return ret;
    }

    while (batch.n_reaped < batch.n_submitted) {
      ret = get_completed(batch, ctx, 1, completed);
      if (ret < 0) {
        return ret;
      }
    }
  }
  return 0;
}

int WindowsAlignedFileReader::submit(PendingBatch &batch,
                                     std::vector<AlignedRead> &read_reqs,
                                     IOContext &ctx) {
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
  batch.expected_lengths.clear();
  batch.completed.clear();
  batch.generation = 0;

  if (read_reqs.empty()) {
    return 0;
  }
  if (read_reqs.size() > MAX_IO_DEPTH) {
    LOG_ERROR("Windows IOCP batch has %zu requests; maximum is %u",
              read_reqs.size(), static_cast<unsigned>(MAX_IO_DEPTH));
    return IndexError_InvalidArgument;
  }
  int ret = validate_windows_read_requests(read_reqs);
  if (ret != 0) {
    return ret;
  }
  ret = prepare_io_ctx(ctx);
  if (ret != 0) {
    return ret;
  }
  if (ctx->outstanding_count != 0) {
    LOG_ERROR("Windows I/O context already has an active batch");
    return IndexError_Runtime;
  }

  ++ctx->generation;
  if (ctx->generation == 0) {
    ++ctx->generation;
  }
  batch.generation = ctx->generation;
  try {
    batch.expected_lengths.reserve(read_reqs.size());
    batch.completed.assign(read_reqs.size(), 0);
  } catch (const std::bad_alloc &) {
    LOG_ERROR("Failed to allocate Windows IOCP batch metadata");
    batch.expected_lengths.clear();
    batch.completed.clear();
    batch.generation = 0;
    return IndexError_NoMemory;
  }

  uint32_t issued_count = 0;
  ctx->active_requests.fill(0);
  for (size_t i = 0; i < read_reqs.size(); ++i) {
    ctx->reqs[i] = OVERLAPPED{};

    const AlignedRead &req = read_reqs[i];
    OVERLAPPED &request = ctx->reqs[i];
    request.Offset = static_cast<DWORD>(req.offset & 0xffffffffULL);
    request.OffsetHigh = static_cast<DWORD>(req.offset >> 32);

    BOOL queued = ::ReadFile(ctx->file_handle, req.buf,
                             static_cast<DWORD>(req.len), nullptr, &request);
    if (!queued && ::GetLastError() != ERROR_IO_PENDING) {
      LOG_ERROR("Error queuing IOCP read %zu (error=%lu)", i, ::GetLastError());
      ctx->outstanding_count = issued_count;
      reset_io_ctx(ctx);
      batch.expected_lengths.clear();
      batch.completed.clear();
      batch.generation = 0;
      return IndexError_Runtime;
    }

    batch.expected_lengths.push_back(req.len);
    ctx->active_requests[i] = 1;
    ++issued_count;
    ctx->outstanding_count = issued_count;
  }

  batch.n_submitted = issued_count;
  return 0;
}

int WindowsAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext &ctx, int min_completed,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();
  if (batch.n_reaped >= batch.n_submitted) {
    return 0;
  }
  if (ctx == nullptr || ctx->completion_port == nullptr ||
      batch.generation == 0 || batch.generation != ctx->generation ||
      batch.expected_lengths.size() != batch.n_submitted ||
      batch.completed.size() != batch.n_submitted ||
      ctx->outstanding_count != batch.n_submitted - batch.n_reaped) {
    LOG_ERROR("Invalid or stale Windows IOCP batch");
    reset_io_ctx(ctx);
    batch.n_reaped = batch.n_submitted;
    return IndexError_Runtime;
  }

  if (completed_indices.capacity() < ctx->outstanding_count) {
    try {
      completed_indices.reserve(ctx->outstanding_count);
    } catch (const std::bad_alloc &) {
      LOG_ERROR("Failed to allocate Windows IOCP completion metadata");
      reset_io_ctx(ctx);
      batch.n_reaped = batch.n_submitted;
      return IndexError_NoMemory;
    }
  }

  const uint32_t remaining = batch.n_submitted - batch.n_reaped;
  const uint32_t target = std::min<uint32_t>(
      remaining, static_cast<uint32_t>(std::max(min_completed, 1)));

  while (completed_indices.size() < target) {
    OVERLAPPED_ENTRY entries[MAX_IO_DEPTH]{};
    ULONG removed = 0;
    const ULONG max_entries = static_cast<ULONG>(std::min<uint32_t>(
        ctx->outstanding_count, static_cast<uint32_t>(MAX_IO_DEPTH)));
    BOOL dequeued = ::GetQueuedCompletionStatusEx(
        ctx->completion_port, entries, max_entries, &removed, INFINITE, FALSE);
    if (!dequeued || removed == 0) {
      LOG_ERROR("GetQueuedCompletionStatusEx failed (error=%lu)",
                ::GetLastError());
      reset_io_ctx(ctx);
      batch.n_reaped = batch.n_submitted;
      completed_indices.clear();
      return IndexError_Runtime;
    }

    bool completion_error = false;
    for (ULONG i = 0; i < removed; ++i) {
      if (entries[i].lpCompletionKey != reinterpret_cast<ULONG_PTR>(ctx)) {
        LOG_ERROR("IOCP returned a completion for a different context");
        completion_error = true;
        continue;
      }

      const uintptr_t address =
          reinterpret_cast<uintptr_t>(entries[i].lpOverlapped);
      const uintptr_t begin = reinterpret_cast<uintptr_t>(ctx->reqs.data());
      const uintptr_t span =
          static_cast<uintptr_t>(batch.n_submitted) * sizeof(OVERLAPPED);
      if (address < begin || address >= begin + span ||
          (address - begin) % sizeof(OVERLAPPED) != 0) {
        LOG_ERROR("IOCP returned an unknown OVERLAPPED request");
        completion_error = true;
        continue;
      }

      const uint32_t index =
          static_cast<uint32_t>((address - begin) / sizeof(OVERLAPPED));
      if (batch.completed[index] != 0) {
        LOG_ERROR("IOCP returned duplicate completion for request %u", index);
        completion_error = true;
        continue;
      }

      if (ctx->active_requests[index] == 0 || ctx->outstanding_count == 0) {
        LOG_ERROR("IOCP returned an inactive or excess completion");
        completion_error = true;
        continue;
      }

      DWORD bytes_transferred = 0;
      bool terminal = true;
      if (!::GetOverlappedResult(ctx->file_handle, &ctx->reqs[index],
                                 &bytes_transferred, FALSE)) {
        DWORD error = ::GetLastError();
        terminal = error != ERROR_IO_INCOMPLETE;
        LOG_ERROR("IOCP read %u failed (error=%lu)", index, error);
        completion_error = true;
      } else if (static_cast<uint64_t>(bytes_transferred) !=
                 batch.expected_lengths[index]) {
        LOG_ERROR(
            "IOCP read %u completed with %lu bytes, expected %llu", index,
            static_cast<unsigned long>(bytes_transferred),
            static_cast<unsigned long long>(batch.expected_lengths[index]));
        completion_error = true;
      }

      if (terminal) {
        ctx->active_requests[index] = 0;
        --ctx->outstanding_count;
        batch.completed[index] = 1;
        ++batch.n_reaped;
      }
      if (!completion_error) {
        completed_indices.push_back(index);
      }
    }

    if (completion_error) {
      reset_io_ctx(ctx);
      batch.n_reaped = batch.n_submitted;
      completed_indices.clear();
      return IndexError_Runtime;
    }
  }

  return static_cast<int>(completed_indices.size());
}

#endif  // Windows/POSIX reader implementation

}  // namespace core
}  // namespace zvec
