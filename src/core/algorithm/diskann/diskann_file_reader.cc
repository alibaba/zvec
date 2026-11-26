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
#include <ailego/pattern/scope_guard.h>
#include <zvec/ailego/buffer/block_eviction_queue.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>
#if defined(_WIN32) || defined(_WIN64)
#include <io.h>
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
#if defined(__linux__) && !defined(__ANDROID__)
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

#if defined(__linux__) && !defined(__ANDROID__)
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

// A cleanup diagnostic must not interrupt retirement or escape a destructor.
static void log_aio_cleanup_failure(int level, const char *message,
                                    int status) noexcept {
  try {
    ailego::LoggerBroker::Log(level, __FILE__, __LINE__, "%s: %d", message,
                              status);
  } catch (...) {
  }
}
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

// Diagnostics must not interrupt completion accounting or teardown.
template <typename Log>
static void log_windows_io_failure(Log log) noexcept {
  try {
    log();
  } catch (...) {
  }
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
        log_windows_io_failure([&] {
          LOG_WARN("CancelIoEx failed while draining DiskAnn I/O (error=%lu)",
                   error);
        });
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
  ctx->iocp_batch = nullptr;
  ctx->active_requests.fill(0);
  ctx->outstanding_count = 0;
}
#endif

int setup_io_ctx(IOContext &ctx) {
  const auto selected = ailego::IOBackend::Instance().available();
  DiskAnnReadReservation memory;
  try {
    ctx = memory.create<IoBackend>();
  } catch (const std::bad_alloc &) {
    ctx = nullptr;
    LOG_ERROR("Failed to allocate DiskAnn I/O context");
    return IndexError_NoMemory;
  }
  ctx->memory = std::move(memory);
  ctx->type = selected;

#if defined(_WIN32) || defined(_WIN64)
  log_diskann_io_backend(ctx->type);
  return 0;
#elif defined(__linux__) && !defined(__ANDROID__)
  if (selected == ailego::IOBackendType::kPread) {
    log_diskann_io_backend(ctx->type);
    return 0;
  }

  // Priority 1: io_uring (raw kernel syscalls — zero dependency).
  if (selected == ailego::IOBackendType::kIoUring) {
    const auto setup = ctx->ring.setup_with_status(MAX_EVENTS);
    if (setup != ailego::IoUringRing::SetupResult::kUnavailable) {
      if (setup == ailego::IoUringRing::SetupResult::kNoMemory)
        ctx->type = ailego::IOBackendType::kPread;
      log_diskann_io_backend(ctx->type);
      return 0;
    }
  }

  // Priority 2: libaio (dlopen — soft dependency).
  if (selected != ailego::IOBackendType::kPread &&
      LibAioLoader::Instance().load()) {
    const int ret = LibAioLoader::Instance().setup_budgeted(
        MAX_EVENTS, ctx->aio_ctx, ctx->aio_memory);
    if (ret == 0 || ctx->aio_ctx) {
      ctx->type = ailego::IOBackendType::kLibAio;
      ctx->aio_quarantined = ret != 0;
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

  int status = 0;
#if defined(_WIN32) || defined(_WIN64)
  close_windows_io_handles(ctx);
#elif defined(__linux__) && !defined(__ANDROID__)
  if (ctx->type == ailego::IOBackendType::kIoUring) {
    ctx->ring.teardown();
  } else if (ctx->type == ailego::IOBackendType::kLibAio && ctx->aio_ctx) {
    const int ret = LibAioLoader::Instance().destroy_budgeted(ctx->aio_ctx,
                                                              ctx->aio_memory);
    if (ret != 0) {
      // The kernel may still own these destinations. Preserve their mapping
      // and its charge even if the context's final owner must go away.
      (void)ctx->aio_staging.release();
      log_aio_cleanup_failure(ailego::Logger::LEVEL_ERROR,
                              "io_destroy failed for DiskANN", ret);
      status = IndexError_Runtime;
    }
  }
  // IoUringRing destructor also calls teardown() — idempotent and safe.
#endif

  auto memory = std::move(ctx->memory);
  delete ctx;
  ctx = nullptr;
  return status;
}

#if !defined(_WIN32) && !defined(_WIN64)
static int execute_one_pread(int fd, const AlignedRead &req) {
  auto *buf = static_cast<uint8_t *>(req.buf);
  uint64_t offset = req.offset;
  uint64_t remaining = req.len;
#if defined(__linux__) && !defined(__ANDROID__)
  const auto expected = req.expected_len;
  if (expected > req.len) return IndexError_InvalidArgument;
#endif

  while (remaining > 0) {
    ssize_t bytes_read =
        ::pread(fd, buf, static_cast<size_t>(remaining), offset);
    if (bytes_read > 0) {
#if defined(__linux__) && !defined(__ANDROID__)
      if (expected && expected < req.len) {
        if (static_cast<uint64_t>(bytes_read) != expected)
          return IndexError_ReadData;
        std::memset(buf + expected, 0, req.len - expected);
        return 0;
      }
#endif
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

#if defined(__linux__) && !defined(__ANDROID__)
// AIO never receives a caller-owned address. If cancelling a batch fails,
// retain its staging in the context and fall back to pread for later reads.
static void finish_libaio_reads(IoBackend &ctx) {
  ctx.aio_active = false;
  ctx.aio_batch = nullptr;
  if (!ctx.aio_quarantined) ctx.aio_staging.reset();
}

static int retire_aio_context(IoBackend &ctx) {
  const int ret =
      LibAioLoader::Instance().destroy_budgeted(ctx.aio_ctx, ctx.aio_memory);
  ctx.aio_quarantined = ret != 0;
  finish_libaio_reads(ctx);
  if (ret != 0) {
    log_aio_cleanup_failure(ailego::Logger::LEVEL_ERROR,
                            "io_destroy failed while draining DiskANN AIO",
                            ret);
  }
  return ret;
}

static bool reset_aio_context(IoBackend &ctx) {
  if (retire_aio_context(ctx) != 0) return false;
  const int ret = LibAioLoader::Instance().setup_budgeted(
      MAX_EVENTS, ctx.aio_ctx, ctx.aio_memory);
  if (ret != 0) {
    ctx.aio_quarantined = ctx.aio_ctx != nullptr;
    log_aio_cleanup_failure(ailego::Logger::LEVEL_WARN,
                            "Cannot recreate DiskANN AIO context; using pread",
                            ret);
  }
  return true;
}

static bool prepare_libaio_staging(IoBackend &ctx, const AlignedRead *reads,
                                   size_t count) try {
  using Buffer = ailego::BudgetedMappedBuffer;
  auto &pool = ailego::MemoryLimitPool::get_instance();
  if (pool.process_rss_limit() &&
      (!pool.capacity() || pool.under_process_rss_pressure())) {
    return false;
  }
  size_t bytes = 0;
  for (size_t i = 0; i < count; ++i) {
    if (!reads[i].buf || !reads[i].len ||
        reads[i].len > (std::numeric_limits<size_t>::max)())
      return false;
    const size_t slot = Buffer::mapping_size(static_cast<size_t>(reads[i].len));
    if (!slot || slot > (std::numeric_limits<size_t>::max)() - bytes)
      return false;
    bytes += slot;
  }
  if (!ctx.aio_staging) ctx.aio_staging = std::make_unique<Buffer>();
  return ctx.aio_staging->size() >= bytes || ctx.aio_staging->try_resize(bytes);
} catch (const std::bad_alloc &) {
  return false;
}

static void prepare_libaio_request(IoBackend &ctx, struct iocb *cb, int fd,
                                   const AlignedRead &read, size_t &offset) {
  io_prep_pread(cb, fd, ctx.aio_staging->data() + offset, read.len,
                read.offset);
  offset += ailego::BudgetedMappedBuffer::mapping_size(read.len);
}

int execute_io_libaio(IoBackend &ctx, int fd,
                      std::vector<AlignedRead> &read_reqs, uint64_t n_retries) {
  if (ctx.aio_active) return IndexError_InvalidArgument;
  if (!ctx.aio_ctx || ctx.aio_quarantined)
    return execute_io_pread(fd, read_reqs);
  const size_t batch_size = std::min<size_t>(read_reqs.size(), MAX_EVENTS);
  ailego::BudgetedVector<iocb_t *> cbs;
  ailego::BudgetedVector<io_event_t> evts;
  ailego::BudgetedVector<struct iocb> cb;
  ailego::BudgetedVector<uint8_t> seen;
  const auto finish = ailego::ScopeGuard::Make([&] {
    if (ctx.aio_active)
      retire_aio_context(ctx);
    else
      finish_libaio_reads(ctx);
  });
  try {
    cbs.resize(batch_size);
    evts.resize(batch_size);
    cb.resize(batch_size);
    seen.resize(batch_size);
  } catch (const std::bad_alloc &) {
    return execute_io_pread(fd, read_reqs);
  }

  const uint64_t iters =
      DiskAnnUtil::div_round_up(read_reqs.size(), MAX_EVENTS);
  for (uint64_t iter = 0; iter < iters; iter++) {
    const size_t n_ops =
        std::min<size_t>(read_reqs.size() - iter * MAX_EVENTS, MAX_EVENTS);
    const auto *batch = read_reqs.data() + iter * MAX_EVENTS;
    if (!prepare_libaio_staging(ctx, batch, n_ops)) {
      finish_libaio_reads(ctx);
      return execute_io_pread(fd, read_reqs);
    }
    size_t offset = 0;
    for (size_t i = 0; i < n_ops; ++i) {
      cbs[i] = &cb[i];
      prepare_libaio_request(ctx, cbs[i], fd, batch[i], offset);
    }
    ctx.aio_active = true;

    size_t n_tries = 0;
    size_t submitted = 0;
    bool submission_ok = true;

    // Phase 1: accumulate partial submissions. A positive return value means
    // that exactly that prefix is now in flight and must never be submitted
    // again.
    while (submitted < n_ops) {
      size_t remaining = n_ops - submitted;
      int ret = LibAioLoader::Instance().io_submit(
          ctx.aio_ctx, (int64_t)remaining, cbs.data() + submitted);
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
    // the remaining requests can still write into the staging buffer.
    size_t completed = 0;
    while (completed < submitted) {
      size_t remaining = submitted - completed;
      int ret = LibAioLoader::Instance().io_getevents(
          ctx.aio_ctx, (int64_t)remaining, (int64_t)remaining,
          evts.data() + completed, nullptr);
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
        // The context quarantines the staging buffer until it can be drained.
        return IndexError_Runtime;
      }
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 3: verify every harvested event. Completion order is unspecified,
    // so use io_event::obj instead of assuming it matches request order.
    bool all_ok = true;
    std::fill_n(seen.begin(), submitted, 0);
    for (size_t i = 0; i < completed; i++) {
      auto cb_it = std::find(cbs.begin(), cbs.begin() + submitted, evts[i].obj);
      if (cb_it == cbs.begin() + submitted || seen[cb_it - cbs.begin()]) {
        LOG_ERROR("aio completion %zu did not identify a new request", i);
        if (!reset_aio_context(ctx)) return IndexError_Runtime;
        return execute_io_pread(fd, read_reqs);
      }

      size_t request_index = static_cast<size_t>(cb_it - cbs.begin());
      const AlignedRead &req = read_reqs[request_index + iter * MAX_EVENTS];
      int64_t result = static_cast<int64_t>(evts[i].res);
      int64_t result2 = static_cast<int64_t>(evts[i].res2);
      if (result != static_cast<int64_t>(req.len) || result2 != 0) {
        LOG_WARN(
            "aio request %zu failed: res=%ld, res2=%ld, expected=%lu, "
            "offset=%lu",
            request_index, (long)result, (long)result2, (unsigned long)req.len,
            (unsigned long)req.offset);
        all_ok = false;
      }
      seen[request_index] = true;
    }

    ctx.aio_active = false;
    if (!submission_ok || !all_ok) {
      finish_libaio_reads(ctx);
      return execute_io_pread(fd, read_reqs);
    }
    for (size_t i = 0; i < n_ops; ++i) {
      std::memcpy(batch[i].buf, cb[i].u.c.buf, batch[i].len);
    }
  }

  return 0;
}
#endif

int execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs,
               uint64_t n_retries = 0) {
#if defined(__linux__) && !defined(__ANDROID__)
  // A missing asynchronous context falls back to synchronous pread.
  if (ctx == nullptr) {
    return execute_io_pread(fd, read_reqs);
  }
  // Dispatch based on the active backend.
  if (ctx->type == ailego::IOBackendType::kIoUring) {
    if (ctx->ring.has_pending_reads()) return IndexError_InvalidArgument;
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
    return execute_io_libaio(*ctx, fd, read_reqs, n_retries);
  }

  // NONE backend — synchronous pread.
  return execute_io_pread(fd, read_reqs);
#else
  (void)ctx;
  (void)n_retries;
  return execute_io_pread(fd, read_reqs);
#endif
}

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

#if defined(__linux__) && !defined(__ANDROID__)
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

#if defined(__linux__) && !defined(__ANDROID__)
  flags |= O_DIRECT | O_LARGEFILE;
#endif

  this->file_desc = ::open(fname.c_str(), flags);

#if defined(__linux__) && !defined(__ANDROID__)
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
#if defined(__linux__) && !defined(__ANDROID__)
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

void LinuxAlignedFileReader::release_io_ctx(IOContext &ctx) {
#if defined(__linux__) && !defined(__ANDROID__)
  if (ctx && ctx->type == ailego::IOBackendType::kIoUring) {
    ctx->ring.cancel_reads();
    ctx->ring_batch = nullptr;
  }
  if (ctx && ctx->aio_active) reset_aio_context(*ctx);
#else
  (void)ctx;
#endif
}

#endif  // POSIX reader declarations

BufferPoolAlignedFileReader::BufferPoolAlignedFileReader(
    std::shared_ptr<ailego::VecBufferPool> pool)
    : pool_(std::move(pool)) {}

BufferPoolAlignedFileReader::~BufferPoolAlignedFileReader() = default;

void BufferPoolAlignedFileReader::open(const std::string &fname) {
  (void)open_from_pool(fname);
}

int BufferPoolAlignedFileReader::open_from_pool(const std::string &fname) {
  opened_ = false;
  bypass_reader_.close();
  if (!pool_) {
    return IndexError_InvalidArgument;
  }
#if defined(_WIN32) || defined(_WIN64)
  const auto handle =
      reinterpret_cast<HANDLE>(_get_osfhandle(pool_->file_descriptor()));
  const int ret = bypass_reader_.open_from_handle(fname, handle);
#else
  const int ret =
      bypass_reader_.open_from_handle(fname, pool_->file_descriptor());
#endif
  opened_ = ret == 0;
  return ret;
}

void BufferPoolAlignedFileReader::close() {
  opened_ = false;
  bypass_reader_.close();
  pool_.reset();
}

// A large graph file can still have a small random-query working set. Use
// initially spare cache headroom instead of requiring three disk observations
// before retaining its pages. After any eviction, or under shared pressure,
// preserve the existing frequency admission policy that protects reused pages.
static bool admit_graph_page(ailego::VecBufferPool &pool,
                             ailego::block_id_t page_id) {
  if (!pool.writable() && !pool.has_evicted() &&
      !ailego::MemoryLimitPool::get_instance().under_cache_pressure()) {
    return true;
  }
  return pool.should_admit_page(page_id);
}

int BufferPoolAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                      IOContext &ctx, bool /*async*/) {
  if (!pool_ || !opened_) {
    LOG_ERROR("BufferPoolAlignedFileReader: backing file is not open");
    return IndexError_Runtime;
  }
  if (read_reqs.empty()) {
    return 0;
  }

  try {
    struct UniquePage {
      ailego::block_id_t page_id;
      char *cached_page{nullptr};
      bool bypass_candidate{false};
    };
    struct PageOccurrence {
      size_t unique_index;
      char *destination;
      size_t offset_in_page;
      size_t length;
      size_t canonical_index;
    };

    size_t total_pages = 0;
    for (const AlignedRead &req : read_reqs) {
      if (req.buf == nullptr || req.len == 0 ||
          req.offset > std::numeric_limits<size_t>::max() ||
          req.len > std::numeric_limits<size_t>::max()) {
        return IndexError_InvalidArgument;
      }
      const size_t offset = static_cast<size_t>(req.offset);
      const size_t length = static_cast<size_t>(req.len);
      if (ailego::kVectorPageSize < DiskAnnUtil::kSectorSize ||
          ailego::kVectorPageSize % DiskAnnUtil::kSectorSize != 0 ||
          offset % DiskAnnUtil::kSectorSize != 0 ||
          length % DiskAnnUtil::kSectorSize != 0 ||
          offset > pool_->file_size() || length > pool_->file_size() - offset) {
        return IndexError_InvalidArgument;
      }
      const size_t first_page = offset / ailego::kVectorPageSize;
      const size_t last_page = (offset + length - 1) / ailego::kVectorPageSize;
      const size_t pages = last_page - first_page + 1;
      if (pages > std::numeric_limits<size_t>::max() - total_pages) {
        return IndexError_InvalidLength;
      }
      total_pages += pages;
    }

    // A pool may own the file without a page table when even metadata plus
    // one page exceeds its budget. Keep batched direct I/O in this case and
    // never enter page-cache operations that require initialized entries.
    if (!pool_->cache_enabled()) {
      if (ctx == nullptr) {
        const int status = setup_io_ctx(ctx);
        if (status != 0) return status;
      }
      const int ret = bypass_reader_.read(read_reqs, ctx);
      if (ret == 0) {
        for (const AlignedRead &req : read_reqs) {
          pool_->record_bypass_read(static_cast<size_t>(req.len));
        }
      }
      return ret;
    }

#if defined(__APPLE__)
    // Darwin's native pages can contain several DiskANN sectors. submit()
    // routes those sub-page requests here, including wholly resident batches.
    // Borrow only during this call: search keeps its existing output lifetime
    // and can reclaim graph pages before admitting PQ or result workspace.
    // The 4 KiB Linux/Windows paths are intentionally unchanged.
    if (ailego::kVectorPageSize > DiskAnnUtil::kSectorSize &&
        !pool_->writable() && total_pages <= MAX_IO_DEPTH) {
      std::array<ailego::block_id_t, MAX_IO_DEPTH> ids;
      std::array<char *, MAX_IO_DEPTH> pages;
      size_t count = 0;
      auto release = ailego::ScopeGuard::Make(
          [&] { pool_->release_pages(ids.data(), count); });
      bool resident = true;
      for (const auto &req : read_reqs) {
        const size_t first = req.offset / ailego::kVectorPageSize;
        const size_t last =
            (req.offset + req.len - 1) / ailego::kVectorPageSize;
        for (size_t id = first; id <= last; ++id) {
          size_t index = 0;
          while (index < count && ids[index] != id) ++index;
          if (index != count) continue;
          char *page = pool_->try_acquire_buffer(id);
          if (!page) {
            resident = false;
            break;
          }
          ids[count] = id;
          pages[count++] = page;
        }
        if (!resident) break;
      }
      if (resident) {
        for (const auto &req : read_reqs) {
          size_t offset = req.offset;
          size_t left = req.len;
          auto *destination = static_cast<char *>(req.buf);
          while (left) {
            const auto id = offset / ailego::kVectorPageSize;
            const size_t within = offset % ailego::kVectorPageSize;
            const size_t bytes =
                std::min(left, ailego::kVectorPageSize - within);
            const auto found = std::find(ids.begin(), ids.begin() + count, id);
            std::memcpy(destination, pages[found - ids.begin()] + within,
                        bytes);
            offset += bytes;
            destination += bytes;
            left -= bytes;
          }
        }
        return 0;
      }
    }
#endif

    // Reserve required read workspace against RSS before acquiring cache pins.
    DiskAnnReadReservation split_memory;
    std::vector<UniquePage> unique_pages;
    std::vector<PageOccurrence> occurrences;
    constexpr size_t kSplitBytes = sizeof(UniquePage) + sizeof(PageOccurrence);
    if (total_pages > (std::numeric_limits<size_t>::max)() / kSplitBytes)
      return IndexError_NoMemory;
    split_memory.allocate(total_pages * kSplitBytes, [&] {
      unique_pages.reserve(total_pages);
      occurrences.reserve(total_pages);
      return unique_pages.capacity() * sizeof(UniquePage) +
             occurrences.capacity() * sizeof(PageOccurrence);
    });
    size_t unique_slices = 0;
    for (const AlignedRead &req : read_reqs) {
      size_t source_offset = static_cast<size_t>(req.offset);
      size_t remaining = static_cast<size_t>(req.len);
      char *destination = static_cast<char *>(req.buf);
      while (remaining != 0) {
        const auto page_id = static_cast<ailego::block_id_t>(
            source_offset / ailego::kVectorPageSize);
        const size_t offset_in_page = source_offset % ailego::kVectorPageSize;
        const size_t copy_length =
            std::min(remaining, ailego::kVectorPageSize - offset_in_page);
        size_t unique_index = 0;
        while (unique_index < unique_pages.size() &&
               unique_pages[unique_index].page_id != page_id) {
          ++unique_index;
        }
        if (unique_index == unique_pages.size()) {
          unique_pages.push_back(UniquePage{page_id, nullptr, false});
        }
        size_t canonical_index = occurrences.size();
        for (size_t i = 0; i < occurrences.size(); ++i) {
          const PageOccurrence &prior = occurrences[i];
          if (prior.unique_index == unique_index &&
              prior.offset_in_page == offset_in_page &&
              prior.length == copy_length) {
            canonical_index = prior.canonical_index;
            break;
          }
        }
        if (canonical_index == occurrences.size()) ++unique_slices;
        occurrences.push_back(PageOccurrence{unique_index, destination,
                                             offset_in_page, copy_length,
                                             canonical_index});
        source_offset += copy_length;
        destination += copy_length;
        remaining -= copy_length;
      }
    }

    DiskAnnReadReservation lookup_memory;
    std::vector<ailego::block_id_t> admitted_ids;
    std::vector<size_t> admitted_indices;
    std::vector<char *> admitted_pages;
    std::vector<AlignedRead> bypass_requests;
    const size_t page_count = unique_pages.size();
    constexpr size_t kLookupBytes =
        sizeof(ailego::block_id_t) + sizeof(size_t) + sizeof(char *);
    if (total_pages > (std::numeric_limits<size_t>::max)() /
                          (kLookupBytes + sizeof(AlignedRead)))
      return IndexError_NoMemory;
    // Distinct slices of one page can become separate bypass requests.
    const size_t lookup_bytes =
        page_count * kLookupBytes + unique_slices * sizeof(AlignedRead);
    lookup_memory.allocate(lookup_bytes, [&] {
      admitted_ids.reserve(page_count);
      admitted_indices.reserve(page_count);
      admitted_pages.resize(page_count);
      bypass_requests.reserve(unique_slices);
      return admitted_ids.capacity() * sizeof(ailego::block_id_t) +
             admitted_indices.capacity() * sizeof(size_t) +
             admitted_pages.capacity() * sizeof(char *) +
             bypass_requests.capacity() * sizeof(AlignedRead);
    });

    auto release_cached_pages = [&]() {
      for (UniquePage &page : unique_pages) {
        if (page.cached_page != nullptr) {
          pool_->release_pages(&page.page_id, 1);
          page.cached_page = nullptr;
        }
      }
    };
    struct CachedPageGuard {
      decltype(release_cached_pages) &release;
      ~CachedPageGuard() {
        release();
      }
    } cached_page_guard{release_cached_pages};

    for (size_t i = 0; i < unique_pages.size(); ++i) {
      UniquePage &page = unique_pages[i];
      page.cached_page = pool_->try_acquire_buffer(page.page_id);
      if (page.cached_page != nullptr) {
        continue;
      }
      if (admit_graph_page(*pool_, page.page_id)) {
        admitted_ids.push_back(page.page_id);
        admitted_indices.push_back(i);
      } else {
        page.bypass_candidate = true;
      }
    }

    if (!admitted_ids.empty()) {
      if (pool_->acquire_pages(admitted_ids.data(), admitted_ids.size(),
                               admitted_pages.data())) {
        for (size_t i = 0; i < admitted_ids.size(); ++i) {
          unique_pages[admitted_indices[i]].cached_page = admitted_pages[i];
        }
      } else {
        for (const size_t index : admitted_indices) {
          unique_pages[index].bypass_candidate = true;
        }
      }
    }

    size_t bypass_rechecks = 0;
    size_t bypass_cache_joins = 0;
    admitted_ids.clear();
    admitted_indices.clear();
    for (size_t i = 0; i < unique_pages.size(); ++i) {
      UniquePage &page = unique_pages[i];
      if (!page.bypass_candidate) {
        continue;
      }
      ++bypass_rechecks;
      page.cached_page = pool_->try_acquire_buffer(page.page_id);
      if (page.cached_page != nullptr) {
        page.bypass_candidate = false;
        ++bypass_cache_joins;
      } else if (pool_->should_join_cache_path(page.page_id)) {
        admitted_ids.push_back(page.page_id);
        admitted_indices.push_back(i);
      }
    }
    if (!admitted_ids.empty() &&
        pool_->acquire_pages(admitted_ids.data(), admitted_ids.size(),
                             admitted_pages.data())) {
      for (size_t i = 0; i < admitted_ids.size(); ++i) {
        UniquePage &page = unique_pages[admitted_indices[i]];
        page.cached_page = admitted_pages[i];
        page.bypass_candidate = false;
      }
      bypass_cache_joins += admitted_ids.size();
    }
    pool_->record_bypass_recheck(bypass_rechecks, bypass_cache_joins);

    uint64_t run_offset = 0;
    uint64_t run_length = 0;
    char *run_destination = nullptr;
    size_t bypass_bytes = 0;
    auto flush_bypass_run = [&]() {
      if (run_length != 0) {
        bypass_requests.emplace_back(run_offset, run_length, run_destination);
        run_length = 0;
      }
    };
    for (size_t i = 0; i < occurrences.size(); ++i) {
      const PageOccurrence &occurrence = occurrences[i];
      const UniquePage &page = unique_pages[occurrence.unique_index];
      if (!page.bypass_candidate || occurrence.canonical_index != i) {
        continue;
      }
      const uint64_t slice_offset =
          static_cast<uint64_t>(page.page_id) * ailego::kVectorPageSize +
          occurrence.offset_in_page;
      if (run_length != 0 && run_offset + run_length == slice_offset &&
          run_destination + run_length == occurrence.destination) {
        run_length += occurrence.length;
      } else {
        flush_bypass_run();
        run_offset = slice_offset;
        run_length = occurrence.length;
        run_destination = occurrence.destination;
      }
      bypass_bytes += occurrence.length;
    }
    flush_bypass_run();

    if (!bypass_requests.empty()) {
      if (ctx == nullptr) {
        const int status = setup_io_ctx(ctx);
        if (status != 0) return status;
      }
      const int read_ret = bypass_reader_.read(bypass_requests, ctx);
      if (read_ret != 0) {
        return read_ret;
      }
      pool_->record_bypass_read(bypass_bytes, bypass_requests.size());
    }

    for (size_t i = 0; i < occurrences.size(); ++i) {
      const PageOccurrence &occurrence = occurrences[i];
      const UniquePage &page = unique_pages[occurrence.unique_index];
      if (page.cached_page != nullptr) {
        std::memcpy(occurrence.destination,
                    page.cached_page + occurrence.offset_in_page,
                    occurrence.length);
      } else if (occurrence.canonical_index != i) {
        std::memcpy(occurrence.destination,
                    occurrences[occurrence.canonical_index].destination,
                    occurrence.length);
      }
    }
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
}

// Per-operation state owns cache-load claims, not the caller's output buffers.
// Kernel destinations belong to the native reader's charged I/O context.
struct BufferReadBatch {
  ailego::MemoryReservation memory;
  std::shared_ptr<ailego::VecBufferPool> pool;
  const BufferPoolAlignedFileReader *owner{nullptr};
  PendingBatch native;
  struct Page {
    ailego::block_id_t id;
    char *destination;
    char *claimed{nullptr};
    bool done{false};
  };
  struct Piece {
    size_t page, request;
    char *destination;
  };
  ailego::BudgetedVector<Page> pages;
  ailego::BudgetedVector<Piece> pieces;
  ailego::BudgetedVector<size_t> remaining, order, ends;
  ailego::BudgetedVector<uint32_t> ready;
  ailego::MemoryReservation io_memory;
  std::vector<AlignedRead> reads;
  std::vector<uint32_t> completed;

  ~BufferReadBatch() {
    for (auto &page : pages)
      if (page.claimed) pool->cancel_read_page(page.id, page.claimed);
  }
  void finish(size_t index) {
    auto &page = pages[index];
    if (page.claimed) {
      pool->complete_read_page(page.id, std::exchange(page.claimed, nullptr),
                               page.destination);
    }
    page.done = true;
    for (const auto &piece : pieces) {
      if (piece.page != index) continue;
      if (piece.destination != page.destination)
        std::memcpy(piece.destination, page.destination,
                    ailego::kVectorPageSize);
      if (--remaining[piece.request] == 0) ready.push_back(piece.request);
    }
  }
};

bool BufferPoolAlignedFileReader::try_submit_cached_nodes(
    PendingBatch &batch, const std::vector<AlignedRead> &requests,
    IOContext &ctx, const uint32_t *node_ids, size_t node_size,
    size_t nodes_per_sector) {
  if (!pool_ || !opened_ || pool_->writable() || !pool_->cache_enabled() ||
      requests.empty() || requests.size() > MAX_IO_DEPTH || !node_ids ||
      !node_size || batch.has_async_requests())
    return false;
#if defined(__linux__) && !defined(__ANDROID__)
  if (ctx && (ctx->aio_active || ctx->ring.has_pending_reads())) return false;
#elif defined(_WIN32) || defined(_WIN64)
  if (ctx && ctx->outstanding_count) return false;
#else
  (void)ctx;
#endif
  constexpr size_t sector = DiskAnnUtil::kSectorSize;
  if (nodes_per_sector && nodes_per_sector > sector / node_size) return false;
  const size_t page_size = ailego::kVectorPageSize;
  // The search context supplies disjoint, ascending sector-buffer slots. Other
  // destination layouts retain submit()'s existing overlap and I/O handling.
  uintptr_t previous_end = 0;
  for (size_t i = 0; i < requests.size(); ++i) {
    const auto &request = requests[i];
    const auto destination = reinterpret_cast<uintptr_t>(request.buf);
    const size_t within =
        nodes_per_sector ? (node_ids[i] % nodes_per_sector) * node_size : 0;
    if (!destination || destination % sector || !request.len ||
        request.offset % sector || request.len % sector ||
        request.offset > pool_->file_size() ||
        request.len > pool_->file_size() - request.offset ||
        request.len > UINTPTR_MAX - destination ||
        (i && destination < previous_end) || within > request.len ||
        node_size > request.len - within)
      return false;
    previous_end = destination + request.len;
  }
  std::array<ailego::block_id_t, MAX_IO_DEPTH> ids;
  std::array<char *, MAX_IO_DEPTH> pages;
  size_t count = 0;
  auto release = ailego::ScopeGuard::Make(
      [&] { pool_->release_pages(ids.data(), count); });
  for (size_t i = 0; i < requests.size(); ++i) {
    const size_t within =
        nodes_per_sector ? (node_ids[i] % nodes_per_sector) * node_size : 0;
    const uint64_t first = (requests[i].offset + within) / page_size;
    const uint64_t last =
        (requests[i].offset + within + node_size - 1) / page_size;
    for (auto id = first; id <= last; ++id) {
      size_t index = 0;
      while (index < count && ids[index] != id) ++index;
      if (index != count) continue;
      if (count == MAX_IO_DEPTH) return false;
      char *page = pool_->try_acquire_buffer(id);
      if (!page) return false;
      ids[count] = id;
      pages[count++] = page;
    }
  }
  for (size_t i = 0; i < requests.size(); ++i) {
    const size_t within =
        nodes_per_sector ? (node_ids[i] % nodes_per_sector) * node_size : 0;
    uint64_t source = requests[i].offset + within;
    auto *destination = static_cast<char *>(requests[i].buf) + within;
    size_t remaining = node_size;
    while (remaining) {
      const auto id = source / page_size;
      const size_t offset = source % page_size;
      const size_t bytes = std::min(remaining, page_size - offset);
      const auto found = std::find(ids.begin(), ids.begin() + count, id);
      std::memcpy(destination, pages[found - ids.begin()] + offset, bytes);
      source += bytes;
      destination += bytes;
      remaining -= bytes;
    }
  }
  // No graph pin may span the PQ/candidate allocations between submission and
  // completion. The copied nodes have the same lifetime as the old sectors.
  batch.buffer.reset();
  batch.n_submitted = requests.size();
  batch.n_reaped = 0;
  batch.used_pread = true;
  return true;
}

int BufferPoolAlignedFileReader::submit(PendingBatch &batch,
                                        std::vector<AlignedRead> &requests,
                                        IOContext &ctx) {
  if (batch.has_async_requests()) return IndexError_InvalidArgument;
#if defined(__linux__) && !defined(__ANDROID__)
  if (ctx && (ctx->aio_active || ctx->ring.has_pending_reads()))
    return IndexError_InvalidArgument;
#elif defined(_WIN32) || defined(_WIN64)
  if (ctx && ctx->outstanding_count) return IndexError_InvalidArgument;
#endif
  batch.buffer.reset();
  batch.n_submitted = batch.n_reaped = 0;
  batch.used_pread = false;
  if (!pool_ || !opened_) return IndexError_Runtime;
  if (requests.empty()) return 0;
  if (requests.size() > std::numeric_limits<uint32_t>::max())
    return IndexError_InvalidArgument;
  // Mutable and sub-page reads retain the synchronized storage path. DiskANN
  // frontier reads normally consist of complete native pages.
  const size_t page_size = ailego::kVectorPageSize;
#if defined(_WIN32) || defined(_WIN64)
  constexpr size_t alignment = 4096;
#else
  constexpr size_t alignment = 512;
#endif
  size_t occurrences = 0;
  bool full_pages = !pool_->writable();
  for (const auto &request : requests) {
    if (!request.buf || !request.len || request.offset > pool_->file_size() ||
        request.len > pool_->file_size() - request.offset)
      return IndexError_InvalidArgument;
    if (request.offset % page_size || request.len % page_size ||
        reinterpret_cast<uintptr_t>(request.buf) % alignment)
      full_pages = false;
    if (request.len / page_size >
        MAX_IO_DEPTH - std::min<size_t>(MAX_IO_DEPTH, occurrences))
      full_pages = false;
    occurrences = std::min<size_t>(
        MAX_IO_DEPTH + 1,
        occurrences +
            std::min<uint64_t>(request.len / page_size, MAX_IO_DEPTH + 1));
  }
  for (size_t i = 0; i < requests.size() && full_pages; ++i) {
    const auto begin = reinterpret_cast<uintptr_t>(requests[i].buf);
    if (requests[i].len > UINTPTR_MAX - begin)
      return IndexError_InvalidArgument;
    for (size_t j = 0; j < i; ++j) {
      const auto other = reinterpret_cast<uintptr_t>(requests[j].buf);
      if (begin < other + requests[j].len && other < begin + requests[i].len)
        full_pages = false;
    }
  }
  const auto synchronous = [&] {
    const int ret = read(requests, ctx);
    if (!ret) {
      batch.used_pread = true;
      batch.n_submitted = requests.size();
    }
    return ret;
  };
  if (!pool_->cache_enabled() || !full_pages || occurrences > MAX_IO_DEPTH)
    return synchronous();
  if (pool_->cache_enabled()) {
    // The all-resident path needs neither a heap plan nor I/O staging. Pin
    // the bounded set before copying, including duplicate page requests.
    std::array<ailego::block_id_t, MAX_IO_DEPTH> ids;
    std::array<char *, MAX_IO_DEPTH> pages;
    size_t count = 0;
    bool resident = true;
    for (const auto &request : requests) {
      for (size_t offset = 0; offset < request.len && resident;
           offset += page_size) {
        const auto id = (request.offset + offset) / page_size;
        size_t index = 0;
        while (index < count && ids[index] != id) ++index;
        if (index != count) continue;
        char *page = pool_->try_acquire_buffer(id);
        if (!page)
          resident = false;
        else {
          ids[count] = id;
          pages[count++] = page;
        }
      }
      if (!resident) break;
    }
    if (resident) {
      for (const auto &request : requests)
        for (size_t offset = 0; offset < request.len; offset += page_size) {
          const auto id = (request.offset + offset) / page_size;
          const auto found = std::find(ids.begin(), ids.begin() + count, id);
          std::memcpy(static_cast<char *>(request.buf) + offset,
                      pages[found - ids.begin()], page_size);
        }
    }
    pool_->release_pages(ids.data(), count);
    if (resident) {
      batch.used_pread = true;
      batch.n_submitted = requests.size();
      return 0;
    }
  }
#if defined(__linux__) && !defined(__ANDROID__)
  if ((ctx && ctx->type == ailego::IOBackendType::kPread) ||
      (!ctx && pool_->cache_enabled() &&
       pool_->io_backend_type() == ailego::IOBackendType::kPread))
    return synchronous();
#elif !defined(_WIN32) && !defined(_WIN64)
  // In particular, preserve Darwin's native batched cache loader instead of
  // replacing it with the portable file reader's scalar pread fallback.
  return synchronous();
#endif
  std::shared_ptr<BufferReadBatch> plan;
  try {
    ailego::MemoryReservation memory;
    memory.add(sizeof(BufferReadBatch) + 64);
    plan = std::make_shared<BufferReadBatch>();
    plan->memory.swap(memory);
    plan->pool = pool_;
    plan->owner = this;
    plan->pages.reserve(occurrences);
    plan->pieces.reserve(occurrences);
    plan->order.reserve(occurrences);
    plan->ends.reserve(occurrences);
    plan->ready.reserve(requests.size());
    plan->remaining.resize(requests.size());
    ailego::ReserveVectorCapacity(plan->reads, plan->io_memory, occurrences);
    ailego::ReserveVectorCapacity(plan->completed, plan->io_memory,
                                  occurrences);
    for (size_t r = 0; r < requests.size(); ++r) {
      const auto &request = requests[r];
      plan->remaining[r] = request.len / page_size;
      for (size_t offset = 0; offset < request.len; offset += page_size) {
        const auto id = (request.offset + offset) / page_size;
        auto *destination = static_cast<char *>(request.buf) + offset;
        size_t index = 0;
        while (index < plan->pages.size() && plan->pages[index].id != id)
          ++index;
        if (index == plan->pages.size())
          plan->pages.push_back({id, destination, nullptr, false});
        plan->pieces.push_back({index, r, destination});
      }
    }
    for (size_t i = 0; i < plan->pages.size(); ++i) {
      auto &page = plan->pages[i];
      if (pool_->cache_enabled()) {
        if (char *cached = pool_->try_acquire_buffer(page.id)) {
          std::memcpy(page.destination, cached, page_size);
          pool_->release_pages(&page.id, 1);
          plan->finish(i);
          continue;
        }
        if (admit_graph_page(*pool_, page.id)) {
          const auto claim = pool_->claim_read_page(page.id, page.claimed);
          if (claim == ailego::VecBufferPool::ReadPageClaim::kJoin) {
            // Join the existing page load; do not issue duplicate disk I/O.
            plan.reset();
            return synchronous();
          }
        }
      }
      plan->order.push_back(i);
    }
    // Preserve result/comparison order while presenting scattered cold pages
    // in file order, coalescing adjacent source and destination ranges.
    std::sort(plan->order.begin(), plan->order.end(), [&](size_t a, size_t b) {
      return plan->pages[a].id < plan->pages[b].id;
    });
    for (size_t n = 0; n < plan->order.size(); ++n) {
      const auto &page = plan->pages[plan->order[n]];
      if (!plan->reads.empty() &&
          plan->reads.back().offset + plan->reads.back().len ==
              page.id * page_size &&
          static_cast<char *>(plan->reads.back().buf) +
                  plan->reads.back().len ==
              page.destination) {
        plan->reads.back().len += page_size;
#if defined(__linux__) && !defined(__ANDROID__)
        plan->reads.back().expected_len += page_size;
#endif
        plan->ends.back() = n + 1;
      } else {
        plan->reads.emplace_back(page.id * page_size, page_size,
                                 page.destination);
        plan->ends.push_back(n + 1);
      }
    }
  } catch (const std::bad_alloc &) {
    plan.reset();
    return synchronous();
  }
  if (!plan->reads.empty()) {
    if (!ctx && setup_io_ctx(ctx) != 0) {
      plan.reset();
      return synchronous();
    }
    // Another query may have started loading a bypassed page while this
    // plan was being prepared. Drop our claims before joining its load.
    size_t rechecked = 0;
    for (const size_t index : plan->order) {
      const auto &page = plan->pages[index];
      if (page.claimed) continue;
      ++rechecked;
      if (pool_->should_join_cache_path(page.id)) {
        pool_->record_bypass_recheck(rechecked, 0);
        plan.reset();
        return synchronous();
      }
    }
    pool_->record_bypass_recheck(rechecked, 0);
    const int ret = bypass_reader_.submit(plan->native, plan->reads, ctx);
    if (ret) return ret;
  } else {
    plan->native.used_pread = true;
  }
  batch.used_pread = plan->native.used_pread;
  batch.n_submitted = requests.size();
  batch.buffer = std::move(plan);
  return 0;
}

int BufferPoolAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext &ctx, int minimum,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();
  auto plan = batch.buffer;
  if (!plan) {
    if (batch.has_async_requests()) return IndexError_InvalidArgument;
    try {
      completed_indices.reserve(batch.n_submitted - batch.n_reaped);
      for (uint32_t i = batch.n_reaped; i < batch.n_submitted; ++i)
        completed_indices.push_back(i);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
    batch.n_reaped = batch.n_submitted;
    return completed_indices.size();
  }
  if (plan->owner != this) return IndexError_InvalidArgument;
  try {
    completed_indices.reserve(batch.n_submitted - batch.n_reaped);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
  const size_t required = std::min<size_t>(std::max(0, minimum),
                                           batch.n_submitted - batch.n_reaped);
  while (plan->native.n_reaped < plan->native.n_submitted &&
         (plan->ready.size() < required ||
          (minimum <= 0 && plan->ready.empty()))) {
    const int count = bypass_reader_.get_completed(
        plan->native, ctx, minimum > 0 ? 1 : 0, plan->completed);
    if (count < 0) {
      if (count != IndexError_NoMemory && count != IndexError_InvalidArgument) {
        bypass_reader_.release_io_ctx(ctx);
        batch.n_reaped = batch.n_submitted;
        batch.used_pread = true;
        batch.buffer.reset();
      }
      return count;
    }
    for (const uint32_t index : plan->completed) {
      const size_t begin = index ? plan->ends[index - 1] : 0;
      size_t bypass_bytes = 0;
      for (size_t i = begin; i < plan->ends[index]; ++i) {
        const auto &page = plan->pages[plan->order[i]];
        if (!page.claimed) bypass_bytes += ailego::kVectorPageSize;
        plan->finish(plan->order[i]);
      }
      if (bypass_bytes) plan->pool->record_bypass_read(bypass_bytes);
    }
    if (minimum <= 0 || count == 0) break;
  }
  completed_indices.assign(plan->ready.begin(), plan->ready.end());
  plan->ready.clear();
  batch.n_reaped += completed_indices.size();
  if (batch.n_reaped == batch.n_submitted) batch.buffer.reset();
  return completed_indices.size();
}

void BufferPoolAlignedFileReader::release_io_ctx(IOContext &ctx) {
  bypass_reader_.release_io_ctx(ctx);
}

#if !defined(_WIN32) && !defined(_WIN64)
#if defined(__linux__) && !defined(__ANDROID__)
int LinuxAlignedFileReader::submit(PendingBatch &batch,
                                   std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx) {
  if ((ctx && (ctx->aio_active || ctx->ring.has_pending_reads())) ||
      batch.has_async_requests())
    return IndexError_InvalidArgument;
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
  batch.cbs.clear();
  batch.cb_ptrs.clear();
  batch.events.clear();
  batch.destinations.clear();

  if (this->file_desc == -1) {
    LOG_ERROR("submit: invalid file descriptor");
    return IndexError_Runtime;
  }

  if (read_reqs.empty()) {
    return 0;
  }

  if (read_reqs.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    return IndexError_InvalidArgument;
  const auto n_ops = static_cast<uint32_t>(read_reqs.size());
  const auto read_synchronously = [&] {
    const int ret = execute_io_pread(this->file_desc, read_reqs);
    if (ret == 0) {
      batch.used_pread = true;
      batch.n_submitted = n_ops;
    }
    return ret;
  };
  if (ctx == nullptr || ctx->type == ailego::IOBackendType::kPread)
    return read_synchronously();

  if (ctx->type == ailego::IOBackendType::kIoUring) {
    if (n_ops > ailego::kIoUringMaxBatch) return read_synchronously();
    try {
      batch.ring_reads.assign(read_reqs.begin(), read_reqs.end());
    } catch (const std::bad_alloc &) {
      return read_synchronously();
    }
    const int ret =
        ctx->ring.submit_reads(file_desc, read_reqs.data(), read_reqs.size());
    if (ret != 0) {
      batch.ring_reads.clear();
      LOG_WARN("submit: io_uring submission failed; falling back to pread");
      return read_synchronously();
    }
    ctx->ring_batch = &batch;
    batch.n_submitted = n_ops;
    return 0;
  }

  if (!ctx->aio_ctx || ctx->aio_quarantined || n_ops > MAX_EVENTS)
    return read_synchronously();
  try {
    batch.cbs.resize(n_ops);
    batch.cb_ptrs.resize(n_ops);
    batch.events.resize(n_ops);
    batch.destinations.resize(n_ops);
  } catch (const std::bad_alloc &) {
    // Nothing has been submitted. Free any partial scratch before falling
    // back to reading directly into the already admitted destinations.
    decltype(batch.cbs)().swap(batch.cbs);
    decltype(batch.cb_ptrs)().swap(batch.cb_ptrs);
    decltype(batch.events)().swap(batch.events);
    decltype(batch.destinations)().swap(batch.destinations);
    return read_synchronously();
  }

  if (!prepare_libaio_staging(*ctx, read_reqs.data(), n_ops)) {
    finish_libaio_reads(*ctx);
    return read_synchronously();
  }
  size_t offset = 0;
  for (uint32_t j = 0; j < n_ops; ++j) {
    prepare_libaio_request(*ctx, &batch.cbs[j], file_desc, read_reqs[j],
                           offset);
    batch.cbs[j].data = (void *)(uintptr_t)j;
    batch.cb_ptrs[j] = &batch.cbs[j];
    batch.destinations[j] = read_reqs[j].buf;
  }
  ctx->aio_active = true;
  ctx->aio_batch = &batch;
  const auto abort_submission = ailego::ScopeGuard::Make([&] {
    if (ctx->aio_active && !batch.has_async_requests())
      retire_aio_context(*ctx);
  });

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

  // Drain every request already in flight before releasing staging. The
  // completion array was admitted before submission so memory pressure
  // cannot interrupt this drain.
  uint32_t drained = 0;
  while (drained < submitted) {
    uint32_t remaining = submitted - drained;
    ret = LibAioLoader::Instance().io_getevents(
        ctx->aio_ctx, (int64_t)remaining, (int64_t)remaining,
        batch.events.data() + drained, nullptr);
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
    if (!reset_aio_context(*ctx)) {
      // The staging buffer stays charged while the kernel can still use it.
      return IndexError_Runtime;
    }
    break;
  }

  finish_libaio_reads(*ctx);
  return read_synchronously();
}

int LinuxAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext &ctx, int min_completed,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();

  if (batch.n_reaped >= batch.n_submitted) {
    return 0;
  }

  if (batch.used_pread) {
    try {
      completed_indices.reserve(batch.n_submitted - batch.n_reaped);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
    for (uint32_t i = batch.n_reaped; i < batch.n_submitted; i++) {
      completed_indices.push_back(i);
    }
    batch.n_reaped = batch.n_submitted;
    return (int)completed_indices.size();
  }

  if (ctx && ctx->type == ailego::IOBackendType::kIoUring) {
    if (ctx->ring_batch != &batch) return IndexError_InvalidArgument;
    try {
      completed_indices.reserve(batch.n_submitted - batch.n_reaped);
    } catch (const std::bad_alloc &) {
      return IndexError_NoMemory;
    }
    std::array<uint32_t, ailego::kIoUringMaxBatch> indices{};
    const int count = ctx->ring.get_completed_reads(
        std::max(0, min_completed), indices.data(), indices.size());
    if (count < 0) {
      // The ring has drained or quarantined its staging. Retry only requests
      // that were not delivered by an earlier call to get_completed().
      ctx->ring_batch = nullptr;
      batch.n_reaped = batch.n_submitted;
      for (uint32_t index = 0; index < batch.ring_reads.size(); ++index) {
        const auto &request = batch.ring_reads[index];
        if (!request.buf) continue;
        const int status = execute_one_pread(file_desc, request);
        if (status != 0) {
          completed_indices.clear();
          batch.ring_reads.clear();
          return status;
        }
        completed_indices.push_back(index);
      }
      batch.ring_reads.clear();
      return static_cast<int>(completed_indices.size());
    }
    completed_indices.assign(indices.begin(), indices.begin() + count);
    for (const auto index : completed_indices)
      batch.ring_reads[index].buf = nullptr;
    batch.n_reaped += count;
    if (batch.n_reaped == batch.n_submitted) {
      ctx->ring_batch = nullptr;
      batch.ring_reads.clear();
    }
    return count;
  }

  if (!ctx || !ctx->aio_active || ctx->aio_batch != &batch)
    return IndexError_InvalidArgument;
  uint32_t n_remaining = batch.n_submitted - batch.n_reaped;
  int min_req = std::min((int)n_remaining, min_completed);
  if (min_req < 1) min_req = 1;

  bool keep_pending = false;
  const auto abort_completion = ailego::ScopeGuard::Make([&] {
    if (!keep_pending && ctx->aio_active) {
      retire_aio_context(*ctx);
      batch.n_reaped = batch.n_submitted;
      completed_indices.clear();
    }
  });

  try {
    completed_indices.reserve(n_remaining);
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }

  const auto read_undelivered = [&]() -> int {
    batch.n_reaped = batch.n_submitted;
    if (!reset_aio_context(*ctx)) {
      completed_indices.clear();
      return IndexError_Runtime;
    }
    for (uint32_t i = 0; i < batch.n_submitted; ++i) {
      if (!batch.destinations[i]) continue;
      const auto &read = batch.cbs[i].u.c;
      const int status = execute_one_pread(
          file_desc,
          AlignedRead(read.offset, read.nbytes, batch.destinations[i]));
      if (status != 0) {
        completed_indices.clear();
        return status;
      }
      batch.destinations[i] = nullptr;
      completed_indices.push_back(i);
    }
    return static_cast<int>(completed_indices.size());
  };

  int ret;
  do {
    // Retry interrupted waits while the kernel still owns the staging.
    ret = LibAioLoader::Instance().io_getevents(ctx->aio_ctx, (int64_t)min_req,
                                                (int64_t)n_remaining,
                                                batch.events.data(), nullptr);
  } while (ret == -EINTR);
  if (ret < 0 || static_cast<uint32_t>(ret) > n_remaining) {
    LOG_ERROR("get_completed: unexpected io_getevents result: %d", ret);
    return read_undelivered();
  }

  for (int i = 0; i < ret; i++) {
    const auto event_index = reinterpret_cast<uintptr_t>(batch.events[i].data);
    if (event_index >= batch.n_submitted || !batch.destinations[event_index] ||
        batch.events[i].obj != &batch.cbs[event_index]) {
      LOG_ERROR("get_completed: completion did not identify a new request");
      return read_undelivered();
    }
    const auto idx = static_cast<uint32_t>(event_index);
    int64_t res = (int64_t)batch.events[i].res;
    int64_t res2 = (int64_t)batch.events[i].res2;
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
                             batch.destinations[idx]);
      if (execute_one_pread(this->file_desc, retry_read) != 0) {
        LOG_ERROR("get_completed: pread retry for read %u failed", idx);
        return IndexError_Runtime;
      }
    } else {
      std::memcpy(batch.destinations[idx], batch.cbs[idx].u.c.buf,
                  static_cast<size_t>(expected));
    }
    batch.destinations[idx] = nullptr;
    completed_indices.push_back(idx);
  }

  batch.n_reaped += static_cast<uint32_t>(ret);
  if (batch.n_reaped == batch.n_submitted) finish_libaio_reads(*ctx);
  keep_pending = true;
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
                                   IOContext &ctx, bool async) try {
  if (async) {
    LOG_WARN(
        "read() waits for completion; use submit()/get_completed() for "
        "asynchronous Windows I/O");
  }
  int ret = validate_windows_read_requests(read_reqs);
  if (ret != 0) {
    return ret;
  }

  const size_t batch_size = std::min<size_t>(read_reqs.size(), MAX_IO_DEPTH);
  DiskAnnReadReservation completion_memory;
  DiskAnnReadReservation request_memory;
  std::vector<uint32_t> completed;
  std::vector<AlignedRead> requests;
  completion_memory.reserve(completed, batch_size);
  request_memory.reserve(requests, batch_size);

  for (size_t start = 0; start < read_reqs.size(); start += MAX_IO_DEPTH) {
    const size_t count =
        std::min<size_t>(read_reqs.size() - start, MAX_IO_DEPTH);
    requests.assign(read_reqs.begin() + start,
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
} catch (const std::bad_alloc &) {
  return IndexError_NoMemory;
}

int WindowsAlignedFileReader::submit(PendingBatch &batch,
                                     std::vector<AlignedRead> &read_reqs,
                                     IOContext &ctx) {
  // A caller may still need this batch to drain its original context.
  if (batch.has_async_requests()) return IndexError_InvalidArgument;
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
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
  std::fill_n(batch.completed.begin(), read_reqs.size(), 0);

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
    const DWORD error = queued ? ERROR_SUCCESS : ::GetLastError();
    if (!queued && error != ERROR_IO_PENDING) {
      reset_io_ctx(ctx);
      batch.generation = 0;
      log_windows_io_failure([&] {
        LOG_ERROR("Error queuing IOCP read %zu (error=%lu)", i, error);
      });
      return IndexError_Runtime;
    }

    batch.expected_lengths[i] = req.len;
    ctx->active_requests[i] = 1;
    ++issued_count;
    ctx->outstanding_count = issued_count;
  }

  batch.n_submitted = issued_count;
  ctx->iocp_batch = &batch;
  return 0;
}

int WindowsAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext &ctx, int min_completed,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();
  if (batch.n_reaped >= batch.n_submitted) {
    return 0;
  }
  if (!ctx || ctx->iocp_batch != &batch) return IndexError_InvalidArgument;
  if (ctx->completion_port == nullptr || batch.generation == 0 ||
      batch.generation != ctx->generation || batch.n_submitted > MAX_IO_DEPTH ||
      ctx->outstanding_count != batch.n_submitted - batch.n_reaped) {
    log_windows_io_failure(
        [&] { LOG_ERROR("Invalid or stale Windows IOCP batch"); });
    reset_io_ctx(ctx);
    batch.n_reaped = batch.n_submitted;
    return IndexError_Runtime;
  }

  if (completed_indices.capacity() < ctx->outstanding_count) {
    try {
      completed_indices.reserve(ctx->outstanding_count);
    } catch (const std::bad_alloc &) {
      log_windows_io_failure([&] {
        LOG_ERROR("Failed to allocate Windows IOCP completion metadata");
      });
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
      const DWORD error = ::GetLastError();
      reset_io_ctx(ctx);
      batch.n_reaped = batch.n_submitted;
      completed_indices.clear();
      log_windows_io_failure([&] {
        LOG_ERROR("GetQueuedCompletionStatusEx failed (error=%lu)", error);
      });
      return IndexError_Runtime;
    }

    bool completion_error = false;
    for (ULONG i = 0; i < removed; ++i) {
      if (entries[i].lpCompletionKey != reinterpret_cast<ULONG_PTR>(ctx)) {
        log_windows_io_failure([&] {
          LOG_ERROR("IOCP returned a completion for a different context");
        });
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
        log_windows_io_failure(
            [&] { LOG_ERROR("IOCP returned an unknown OVERLAPPED request"); });
        completion_error = true;
        continue;
      }

      const uint32_t index =
          static_cast<uint32_t>((address - begin) / sizeof(OVERLAPPED));
      if (batch.completed[index] != 0) {
        log_windows_io_failure([&] {
          LOG_ERROR("IOCP returned duplicate completion for request %u", index);
        });
        completion_error = true;
        continue;
      }

      if (ctx->active_requests[index] == 0 || ctx->outstanding_count == 0) {
        log_windows_io_failure([&] {
          LOG_ERROR("IOCP returned an inactive or excess completion");
        });
        completion_error = true;
        continue;
      }

      DWORD bytes_transferred = 0;
      const bool read_ok = ::GetOverlappedResult(
          ctx->file_handle, &ctx->reqs[index], &bytes_transferred, FALSE);
      const DWORD error = read_ok ? ERROR_SUCCESS : ::GetLastError();
      if (read_ok || error != ERROR_IO_INCOMPLETE) {
        ctx->active_requests[index] = 0;
        --ctx->outstanding_count;
        batch.completed[index] = 1;
        ++batch.n_reaped;
      }

      if (!read_ok) {
        completion_error = true;
        log_windows_io_failure([&] {
          LOG_ERROR("IOCP read %u failed (error=%lu)", index, error);
        });
      } else if (static_cast<uint64_t>(bytes_transferred) !=
                 batch.expected_lengths[index]) {
        completion_error = true;
        log_windows_io_failure([&] {
          LOG_ERROR(
              "IOCP read %u completed with %lu bytes, expected %llu", index,
              static_cast<unsigned long>(bytes_transferred),
              static_cast<unsigned long long>(batch.expected_lengths[index]));
        });
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

  if (ctx->outstanding_count == 0) ctx->iocp_batch = nullptr;
  return static_cast<int>(completed_indices.size());
}

#endif  // Windows/POSIX reader implementation

}  // namespace core
}  // namespace zvec
