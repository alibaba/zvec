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
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/core/framework/index_logger.h>

#define MAX_EVENTS 1024

namespace zvec {
namespace core {

#if (defined(__linux) || defined(__linux__))
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

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

#if (defined(__linux) || defined(__linux__))
int setup_io_ctx(IOContext &ctx) {
  std::call_once(g_io_backend_log_once, log_diskann_io_backend);
  auto &backend = ailego::IOBackend::Instance();
  ailego::IOBackendType selected = backend.available();

  ctx = new IoBackend();

  // Priority 1: io_uring (raw kernel syscalls — zero dependency).
  if (selected == ailego::IOBackendType::kIoUring &&
      ctx->ring.setup(MAX_EVENTS)) {
    ctx->backend = IoBackend::IO_URING;
    return 0;
  }
  if (selected == ailego::IOBackendType::kIoUring) {
    selected = backend.available(ailego::IOBackendType::kLibAio);
  }

  // Priority 2: libaio (dlopen — soft dependency).
  if (selected == ailego::IOBackendType::kLibAio &&
      LibAioLoader::Instance().load()) {
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
}

int destroy_io_ctx(IOContext &ctx) {
  if (ctx == nullptr || ctx == (IOContext)-1) {
    return 0;
  }

  if (ctx->backend == IoBackend::IO_URING) {
    ctx->ring.teardown();
  } else if (ctx->backend == IoBackend::LIBAIO &&
             LibAioLoader::Instance().is_available()) {
    int ret;
    do {
      ret = LibAioLoader::Instance().io_destroy(ctx->aio_ctx);
    } while (ret == -EINTR);
    if (ret != 0) {
      return ret;
    }
    ctx->aio_ctx = nullptr;
  }
  // IoUringRing destructor also calls teardown() — idempotent and safe.

  delete ctx;
  ctx = nullptr;
  return 0;
}
#elif defined(_WIN32) || defined(_WIN64)
int setup_io_ctx(IOContext &ctx) {
  ctx.reqs.resize(MAX_IO_DEPTH);
  memset(ctx.reqs.data(), 0, ctx.reqs.size() * sizeof(OVERLAPPED));
  return 0;
}

int destroy_io_ctx(IOContext &ctx) {
  ctx.close_handles();
  ctx.reqs.clear();
  return 0;
}
#endif

#if (defined(__linux) || defined(__linux__))
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

// io_getevents() should only fail permanently for an invalid context or
// invalid arguments. If that happens after submission, io_destroy() is the
// only safe way to quiesce the context before synchronous I/O touches the same
// destination buffers. Recreate the context so later reads can still use AIO.
static bool reset_aio_context(IOContext &ctx) {
  auto &loader = LibAioLoader::Instance();
  int ret;
  do {
    ret = loader.io_destroy(ctx->aio_ctx);
  } while (ret == -EINTR);

  if (ret != 0) {
    LOG_ERROR("io_destroy failed while draining AIO; returned: %d, %s", ret,
              ::strerror(-ret));
    return false;
  }

  ctx->aio_ctx = nullptr;
  io_context_t replacement = nullptr;
  ret = loader.io_setup(MAX_EVENTS, &replacement);
  if (ret != 0) {
    LOG_ERROR(
        "io_setup failed while recreating an AIO context; returned: %d, %s. "
        "this context will use pread",
        ret, ::strerror(-ret));
    ctx->backend = IoBackend::NONE;
    return true;
  }
  ctx->aio_ctx = replacement;
  return true;
}

int execute_io_libaio(IOContext &ctx, int fd,
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
      int ret = LibAioLoader::Instance().io_submit(
          ctx->aio_ctx, (int64_t)remaining, cbs.data() + submitted);
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
          ctx->aio_ctx, (int64_t)remaining, (int64_t)remaining,
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

int execute_io(IOContext &ctx, int fd, std::vector<AlignedRead> &read_reqs,
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
    // io_uring failed — fall back to pread.
    LOG_WARN("io_uring execute failed; falling back to pread");
    return execute_io_pread(fd, read_reqs);
  }

  if (ctx->backend == IoBackend::LIBAIO) {
    return execute_io_libaio(ctx, fd, read_reqs, n_retries);
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

    unsigned tail = __atomic_load_n(sq_tail_, __ATOMIC_ACQUIRE);
    unsigned mask = *sq_ring_mask_;

    for (uint64_t j = 0; j < n_ops; j++) {
      unsigned idx = (tail + static_cast<unsigned>(j)) & mask;
      unsigned sqe_idx = sq_array_[idx];
      struct io_uring_sqe *sqe = &sqes_[sqe_idx];

      uint64_t req_idx = j + iter * batch_size;
      io_uring_prep_read(sqe, fd, read_reqs[req_idx].buf,
                         static_cast<uint32_t>(read_reqs[req_idx].len),
                         read_reqs[req_idx].offset);
      // Store the request index so we can verify the completion.
      sqe->user_data = req_idx;
    }

    // Memory barrier: ensure SQE contents are visible before tail update.
    __sync_synchronize();
    __atomic_store_n(sq_tail_, tail + static_cast<unsigned>(n_ops),
                     __ATOMIC_RELEASE);

    // --- Phase 2: Submit and wait for completions ---

    int ret = static_cast<int>(
        syscall(__NR_io_uring_enter, ring_fd_, static_cast<unsigned>(n_ops),
                static_cast<unsigned>(n_ops), IORING_ENTER_GETEVENTS,
                static_cast<void *>(nullptr), static_cast<size_t>(0)));
    if (ret < 0) {
      LOG_WARN(
          "io_uring_enter failed; errno=%d, %s, n_ops=%lu. "
          "falling back to pread",
          errno, ::strerror(errno), (unsigned long)n_ops);
      return -1;
    }

    // --- Phase 3: Process CQEs ---

    unsigned head = __atomic_load_n(cq_head_, __ATOMIC_ACQUIRE);
    unsigned cq_mask = *cq_ring_mask_;
    bool all_ok = true;
    uint64_t processed = 0;

    for (unsigned i = head; processed < n_ops; i = (i + 1), processed++) {
      struct io_uring_cqe *cqe = &cqes_[i & cq_mask];
      uint64_t req_idx = cqe->user_data;

      if (cqe->res < 0) {
        LOG_WARN("io_uring read failed: req=%lu, res=%d, offset=%lu",
                 (unsigned long)req_idx, cqe->res,
                 (unsigned long)read_reqs[req_idx].offset);
        all_ok = false;
      } else if (static_cast<uint64_t>(cqe->res) != read_reqs[req_idx].len) {
        LOG_WARN("io_uring short read: req=%lu, got=%d, expected=%lu",
                 (unsigned long)req_idx, cqe->res,
                 (unsigned long)read_reqs[req_idx].len);
        all_ok = false;
      }
    }

    // Advance the CQ head to consume the completions.
    __sync_synchronize();
    __atomic_store_n(cq_head_, head + static_cast<unsigned>(n_ops),
                     __ATOMIC_RELEASE);

    if (!all_ok) {
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
  if (ret == 0 && ctx != nullptr) {
    ctx_map[thread_id] = ctx;
  }
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

  // teardown is a syscall; keep it outside the lock to avoid blocking others
  destroy_io_ctx(ctx);
  LOG_INFO("returned ctx from thread");
#endif
}

void LinuxAlignedFileReader::deregister_all_threads() {
#if (defined(__linux) || defined(__linux__))
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    IOContext ctx = x->second;
    destroy_io_ctx(ctx);
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

#endif  // (defined(__linux) || defined(__linux__))

#if defined(_WIN32) || defined(_WIN64)

// ============================================================================
// Windows implementation using unbuffered overlapped I/O. Each IOContext owns
// a separate file handle and completion port, so concurrent search contexts
// cannot consume one another's completion packets.
// ============================================================================

WindowsAlignedFileReader::WindowsAlignedFileReader() {}

WindowsAlignedFileReader::~WindowsAlignedFileReader() {
  deregister_all_threads();
  close();
}

IOContext &WindowsAlignedFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  auto thread_id = std::this_thread::get_id();
  auto it = ctx_map.find(thread_id);
  if (it == ctx_map.end()) {
    LOG_ERROR("unable to find IOContext for thread_id");
    static IOContext empty_ctx;
    return empty_ctx;
  }
  return it->second;
}

void WindowsAlignedFileReader::register_thread() {
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(thread_id) != ctx_map.end()) {
    LOG_ERROR("multiple calls to register_thread from the same thread");
    return;
  }

  IOContext ctx;
  if (setup_io_ctx(ctx) != 0) {
    return;
  }
  ctx_map.emplace(thread_id, std::move(ctx));
}

void WindowsAlignedFileReader::deregister_thread() {
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  auto it = ctx_map.find(thread_id);
  if (it == ctx_map.end()) {
    LOG_ERROR("deregister_thread: thread not registered");
    return;
  }
  destroy_io_ctx(it->second);
  ctx_map.erase(it);
}

void WindowsAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto &kv : ctx_map) {
    destroy_io_ctx(kv.second);
  }
  ctx_map.clear();
}

void WindowsAlignedFileReader::open(const std::string &fname) {
  const std::wstring wide_fname = ailego::FileHelper::Utf8ToWide(fname);
  if (wide_fname.empty()) {
    LOG_ERROR("Failed to convert DiskAnn file path from UTF-8: %s",
              fname.c_str());
    return;
  }

  close();

  HANDLE probe_handle =
      CreateFileW(wide_fname.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                      FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                  NULL);
  if (probe_handle == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    LOG_ERROR("Failed to open file: %s (error=%lu)", fname.c_str(), error);
    return;
  }
  CloseHandle(probe_handle);

  file_path_ = wide_fname;
  LOG_INFO("Opened file: %s", fname.c_str());
}

void WindowsAlignedFileReader::close() {
  file_path_.clear();
}

int WindowsAlignedFileReader::prepare_io_ctx(IOContext &ctx) {
  if (ctx.file_handle != INVALID_HANDLE_VALUE && ctx.completion_port != NULL &&
      ctx.file_path == file_path_) {
    return 0;
  }

  ctx.close_handles();
  ctx.file_handle =
      CreateFileW(file_path_.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                      FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                  NULL);
  if (ctx.file_handle == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Failed to open DiskAnn file for IOContext (error=%lu)",
              GetLastError());
    return IndexError_Runtime;
  }

  ctx.completion_port = CreateIoCompletionPort(ctx.file_handle, NULL, 0, 1);
  if (ctx.completion_port == NULL) {
    DWORD error = GetLastError();
    LOG_ERROR("CreateIoCompletionPort failed (error=%lu)", error);
    ctx.close_handles();
    return IndexError_Runtime;
  }

  ctx.file_path = file_path_;
  return 0;
}

int WindowsAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx, bool async) {
  if (async == true) {
    LOG_WARN("Async currently not supported");
  }

  if (file_path_.empty()) {
    LOG_ERROR("Attempt to read from invalid file handle");
    return IndexError_Runtime;
  }

  if (prepare_io_ctx(ctx) != 0) {
    return IndexError_Runtime;
  }

  // Ensure the context has enough stable OVERLAPPED slots. ReadFile stores the
  // pointer until completion, so this vector must not reallocate mid-batch.
  if (ctx.reqs.size() < MAX_IO_DEPTH) {
    ctx.reqs.resize(MAX_IO_DEPTH);
  }

  if (read_reqs.empty()) {
    return 0;
  }

  static const DWORD kSectorLen = 4096;
  size_t n_reqs = read_reqs.size();
  uint64_t n_batches = DiskAnnUtil::div_round_up(n_reqs, MAX_IO_DEPTH);

  for (uint64_t batch = 0; batch < n_batches; batch++) {
    uint64_t batch_start = MAX_IO_DEPTH * batch;
    uint64_t batch_size =
        std::min((uint64_t)(n_reqs - batch_start), (uint64_t)MAX_IO_DEPTH);

    // Only reset slots used by this batch. The common search path submits a
    // small beam, so clearing all MAX_IO_DEPTH slots adds avoidable hot-path
    // work.
    memset(ctx.reqs.data(), 0, batch_size * sizeof(OVERLAPPED));

    // Issue ReadFile calls for this batch.
    bool batch_error = false;
    uint64_t issued_count = 0;
    for (uint64_t j = 0; j < batch_size; j++) {
      AlignedRead &req = read_reqs[batch_start + j];
      OVERLAPPED &os = ctx.reqs[j];

      uint64_t offset = req.offset;
      uint64_t nbytes = req.len;
      char *read_buf = (char *)req.buf;

      // Alignment assertions (must be sector-aligned for unbuffered I/O).
      assert((size_t)read_buf % kSectorLen == 0);
      assert(offset % kSectorLen == 0);
      assert(nbytes % kSectorLen == 0);

      // Fill in the OVERLAPPED struct with the file offset.
      os.Offset = (DWORD)(offset & 0xffffffff);
      os.OffsetHigh = (DWORD)(offset >> 32);

      if (nbytes > (std::numeric_limits<DWORD>::max)()) {
        LOG_ERROR("Read request is too large: %llu bytes",
                  (unsigned long long)nbytes);
        batch_error = true;
        break;
      }

      BOOL ret = ReadFile(ctx.file_handle, read_buf, (DWORD)nbytes, NULL, &os);
      if (ret == FALSE) {
        DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) {
          LOG_ERROR("Error queuing IO -- error=%lu", error);
          batch_error = true;
          break;
        }
      }
      issued_count++;
    }

    // Drain every successfully issued request, including after a submission
    // error. The caller owns the buffers and may free them as soon as read()
    // returns, so no operation may remain outstanding.
    uint64_t completed_count = 0;
    while (completed_count < issued_count) {
      OVERLAPPED_ENTRY entries[MAX_IO_DEPTH];
      ULONG removed = 0;
      const ULONG max_entries = static_cast<ULONG>(
          std::min<uint64_t>(issued_count - completed_count, MAX_IO_DEPTH));
      BOOL dequeued = GetQueuedCompletionStatusEx(
          ctx.completion_port, entries, max_entries, &removed, INFINITE, FALSE);
      if (dequeued == FALSE) {
        LOG_ERROR("GetQueuedCompletionStatusEx failed (error=%lu)",
                  GetLastError());
        batch_error = true;

        // This should only happen if the completion port itself fails. Cancel
        // and poll every request to completion so the caller's buffers remain
        // safe. The handle is private to this IOContext.
        CancelIoEx(ctx.file_handle, NULL);
        for (uint64_t j = 0; j < issued_count; j++) {
          while (true) {
            DWORD ignored = 0;
            if (GetOverlappedResult(ctx.file_handle, &ctx.reqs[j], &ignored,
                                    FALSE)) {
              break;
            }
            if (GetLastError() != ERROR_IO_INCOMPLETE) {
              break;
            }
            SwitchToThread();
          }
        }
        ctx.close_handles();
        completed_count = issued_count;
        break;
      }

      completed_count += removed;
      for (ULONG i = 0; i < removed; i++) {
        OVERLAPPED *os = entries[i].lpOverlapped;
        OVERLAPPED *begin = ctx.reqs.data();
        OVERLAPPED *end = begin + batch_size;
        if (os < begin || os >= end) {
          LOG_ERROR("Completion port returned an unknown OVERLAPPED request");
          batch_error = true;
          continue;
        }

        const uint64_t slot = static_cast<uint64_t>(os - begin);
        DWORD bytes_transferred = 0;
        BOOL ok =
            GetOverlappedResult(ctx.file_handle, os, &bytes_transferred, FALSE);
        if (ok == FALSE) {
          LOG_ERROR("Overlapped read %lu failed (error=%lu)",
                    (unsigned long)slot, GetLastError());
          batch_error = true;
          continue;
        }

        const uint64_t expected = read_reqs[batch_start + slot].len;
        if (static_cast<uint64_t>(bytes_transferred) != expected) {
          LOG_ERROR(
              "Overlapped read %lu completed with %lu bytes, expected %lu",
              (unsigned long)slot, (unsigned long)bytes_transferred,
              (unsigned long)expected);
          batch_error = true;
        }
      }
    }

    if (batch_error) {
      return IndexError_Runtime;
    }
  }

  return 0;
}

#endif  // defined(_WIN32) || defined(_WIN64)

}  // namespace core
}  // namespace zvec
