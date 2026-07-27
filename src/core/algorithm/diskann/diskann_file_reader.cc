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
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
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
    LibAioLoader::Instance().io_destroy(ctx->aio_ctx);
  }
  // IoUringRing destructor also calls teardown() — idempotent and safe.

  delete ctx;
  ctx = nullptr;
  return 0;
}
#elif defined(_WIN32) || defined(_WIN64)
int setup_io_ctx(IOContext &ctx) {
  for (uint64_t i = 0; i < MAX_IO_DEPTH; i++) {
    OVERLAPPED os;
    memset(&os, 0, sizeof(OVERLAPPED));
    HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
    if (ev == NULL) {
      LOG_ERROR("CreateEventA failed (error=%lu)", GetLastError());
      destroy_io_ctx(ctx);
      return IndexError_Runtime;
    }
    os.hEvent = ev;
    ctx.reqs.push_back(os);
    ctx.events.push_back(ev);
  }
  return 0;
}

int destroy_io_ctx(IOContext &ctx) {
  for (auto &ev : ctx.events) {
    if (ev != NULL) {
      CloseHandle(ev);
    }
  }
  ctx.events.clear();
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

static int execute_io_libaio(io_context_t ctx, int fd,
                             std::vector<AlignedRead> &read_reqs,
                             uint64_t n_retries = 0) {
#if (defined(__linux) || defined(__linux__))
  if (ailego::IOBackend::Instance().is_pread()) {
    return execute_io_pread(fd, read_reqs);
  }
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
    // Phase 1: io_submit with retry.
    while (true) {
      int ret =
          LibAioLoader::Instance().io_submit(ctx, (int64_t)n_ops, cbs.data());
      if (ret == (int)n_ops) {
        break;
      }
      if ((ret == -EAGAIN || ret == -EINTR) && n_tries < n_retries) {
        n_tries++;
        continue;
      }
      LOG_WARN(
          "io_submit failed; returned: %d, expected=%lu. falling back to "
          "pread",
          ret, n_ops);
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 2: io_getevents with retry (never re-submits).
    n_tries = 0;
    while (true) {
      int ret = LibAioLoader::Instance().io_getevents(
          ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
      if (ret == (int)n_ops) {
        break;
      }
      if (ret == -EINTR && n_tries < n_retries) {
        n_tries++;
        continue;
      }
      LOG_WARN(
          "io_getevents failed; returned: %d, expected=%lu, errno=%d, %s, "
          "falling back to pread",
          ret, n_ops, errno, ::strerror(-ret));
      return execute_io_pread(fd, read_reqs);
    }

    // Phase 3: verify each completed read (res must equal requested length).
    bool all_ok = true;
    for (uint64_t i = 0; i < n_ops; i++) {
      int64_t expected_len = read_reqs[i + iter * MAX_EVENTS].len;
      if ((int64_t)evts[i].res != expected_len) {
        LOG_WARN("aio request %zu failed: res=%ld, expected=%ld, offset=%zu",
                 (size_t)i, (long)evts[i].res, (long)expected_len,
                 (size_t)read_reqs[i + iter * MAX_EVENTS].offset);
        all_ok = false;
      }
    }
    if (!all_ok) {
      return execute_io_pread(fd, read_reqs);
    }
  }

  return 0;
#else
  return execute_io_pread(fd, read_reqs);
#endif
}

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
    // io_uring failed — fall back to pread.
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
// Windows implementation using overlapped I/O with per-request events.
// Based on the DiskANN Windows aligned file reader.
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
  ctx_map[thread_id] = std::move(ctx);
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

  file_handle_ =
      CreateFileW(wide_fname.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING |
                      FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                  NULL);

  if (file_handle_ == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    LOG_ERROR("Failed to open file: %s (error=%lu)", fname.c_str(), error);
    return;
  }

  LOG_INFO("Opened file: %s", fname.c_str());
}

void WindowsAlignedFileReader::close() {
  if (file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
  }
}

int WindowsAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                   IOContext &ctx, bool async) {
  if (async == true) {
    LOG_WARN("Async currently not supported");
  }

  if (file_handle_ == INVALID_HANDLE_VALUE) {
    LOG_ERROR("Attempt to read from invalid file handle");
    return IndexError_Runtime;
  }

  // Ensure the context has enough OVERLAPPED slots.
  if (ctx.reqs.size() < MAX_IO_DEPTH) {
    for (size_t i = ctx.reqs.size(); i < MAX_IO_DEPTH; i++) {
      OVERLAPPED os;
      memset(&os, 0, sizeof(OVERLAPPED));
      HANDLE ev = CreateEventA(NULL, TRUE, FALSE, NULL);
      if (ev == NULL) {
        LOG_ERROR("CreateEventA failed (error=%lu)", GetLastError());
        return IndexError_Runtime;
      }
      os.hEvent = ev;
      ctx.reqs.push_back(os);
      ctx.events.push_back(ev);
    }
  }

  static const DWORD kSectorLen = 4096;
  size_t n_reqs = read_reqs.size();
  uint64_t n_batches = DiskAnnUtil::div_round_up(n_reqs, MAX_IO_DEPTH);

  for (uint64_t batch = 0; batch < n_batches; batch++) {
    // Reset OVERLAPPED objects for this batch (preserve hEvent).
    for (size_t i = 0; i < ctx.reqs.size(); i++) {
      OVERLAPPED &os = ctx.reqs[i];
      HANDLE ev = os.hEvent;
      memset(&os, 0, sizeof(OVERLAPPED));
      os.hEvent = ev;
      ResetEvent(ev);
    }

    uint64_t batch_start = MAX_IO_DEPTH * batch;
    uint64_t batch_size =
        std::min((uint64_t)(n_reqs - batch_start), (uint64_t)MAX_IO_DEPTH);

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

      BOOL ret = ReadFile(file_handle_, read_buf, (DWORD)nbytes, NULL, &os);
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

    // Wait for all issued reads to complete.
    // Use GetOverlappedResult per-read instead of GetQueuedCompletionStatus
    // on the shared IOCP. The shared IOCP causes completion-packet stealing
    // when multiple threads call read() concurrently, leading to data
    // corruption and segfaults.
    //
    // We must wait for ALL reads that were successfully issued, even if an
    // error occurred mid-batch. Otherwise the caller may free the read
    // buffers while the OS still has pending writes, causing a use-after-free.
    for (uint64_t j = 0; j < issued_count; j++) {
      OVERLAPPED &os = ctx.reqs[j];
      DWORD bytes_transferred = 0;

      BOOL ok =
          GetOverlappedResult(file_handle_, &os, &bytes_transferred, TRUE);
      if (ok == FALSE) {
        DWORD error = GetLastError();
        LOG_ERROR("GetOverlappedResult failed for read %lu, error=%lu",
                  (unsigned long)j, error);
        batch_error = true;
      } else {
        const uint64_t expected =
            read_reqs[batch_start + j].len;
        if (static_cast<uint64_t>(bytes_transferred) != expected) {
          LOG_ERROR(
              "Overlapped read %lu completed with %lu bytes, expected %lu",
              (unsigned long)j, (unsigned long)bytes_transferred,
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
