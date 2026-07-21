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
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <ailego/io/io_backend_def.h>
#include <zvec/core/framework/index_logger.h>
#if defined(__APPLE__) && TARGET_OS_OSX
#include <aio.h>
#endif

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
#elif defined(__APPLE__) && TARGET_OS_OSX
static std::once_flag g_io_backend_log_once;
#endif

int setup_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  auto &backend = ailego::IOBackend::Instance();
  std::call_once(g_io_backend_log_once, [&backend] {
    if (backend.available() != ailego::IOBackendType::kPread) {
      LOG_INFO("DiskAnn I/O backend: %s (async I/O enabled)", backend.name());
    } else {
      LOG_WARN(
          "DiskAnn I/O backend: synchronous pread (no async I/O available)");
    }
  });
  if (backend.available() == ailego::IOBackendType::kPread) {
    return 0;
  }
  int ret = LibAioLoader::Instance().io_setup(MAX_EVENTS, &ctx);
  return ret;
#elif defined(__APPLE__) && TARGET_OS_OSX
  std::call_once(g_io_backend_log_once, [] {
    LOG_INFO("DiskAnn I/O backend: macOS POSIX AIO with aio_suspend");
  });
  // POSIX AIO does not require a persistent per-context kernel object on
  // macOS. Keep a placeholder value for the cross-platform IOContext API.
  ctx = 0;
  return 0;
#else
  return 0;
#endif
}

int destroy_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  if (ailego::IOBackend::Instance().available() ==
      ailego::IOBackendType::kPread) {
    return 0;
  }
  int ret = LibAioLoader::Instance().io_destroy(ctx);
  return ret;
#elif defined(__APPLE__) && TARGET_OS_OSX
  ctx = 0;
  return 0;
#else
  return 0;
#endif
}

static int execute_io_pread(int fd, std::vector<AlignedRead> &read_reqs,
                            ailego::IOBackendType *used_backend = nullptr) {
  if (used_backend != nullptr) {
    *used_backend = ailego::IOBackendType::kPread;
  }
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

#if defined(__APPLE__) && TARGET_OS_OSX
// POSIX AIO on Darwin accepts a maximum of AIO_LISTIO_MAX requests in the
// interfaces used to wait for a batch. Keeping the same bound here also avoids
// exhausting the process-wide AIO request limit when several search threads
// run concurrently.
static constexpr size_t kMacAioBatchSize = AIO_LISTIO_MAX;

static bool validate_aio_result(int aio_err, int64_t result,
                                const AlignedRead &req) {
  if (aio_err == 0 && result == static_cast<int64_t>(req.len)) {
    return true;
  }

  LOG_WARN(
      "macOS aio request failed; aio_error=%d, %s, result=%ld, "
      "expected=%lu, offset=%lu",
      aio_err, aio_err == 0 ? "success" : ::strerror(aio_err), (long)result,
      (unsigned long)req.len, (unsigned long)req.offset);
  return false;
}

// Wait for and reap every submitted request. Requests and their destination
// buffers must remain alive until aio_error() no longer reports EINPROGRESS,
// and each completed request must be reaped exactly once with aio_return().
static bool drain_aio_requests(std::vector<struct aiocb> &cbs,
                               const std::vector<AlignedRead> &read_reqs,
                               size_t req_begin, size_t submitted_count) {
  size_t completed_count = 0;
  bool all_ok = true;
  std::vector<uint8_t> completed(submitted_count, 0);

  while (completed_count < submitted_count) {
    bool made_progress = false;
    std::vector<const struct aiocb *> pending;
    pending.reserve(submitted_count - completed_count);

    for (size_t i = 0; i < submitted_count; ++i) {
      if (completed[i] != 0) {
        continue;
      }

      int aio_err = ::aio_error(&cbs[i]);
      if (aio_err == EINPROGRESS) {
        pending.push_back(&cbs[i]);
        continue;
      }

      ssize_t result = ::aio_return(&cbs[i]);
      all_ok = validate_aio_result(aio_err, result, read_reqs[req_begin + i]) &&
               all_ok;
      completed[i] = 1;
      ++completed_count;
      made_progress = true;
    }

    if (completed_count == submitted_count || made_progress) {
      continue;
    }

    int ret;
    do {
      ret = ::aio_suspend(pending.data(), static_cast<int>(pending.size()),
                          nullptr);
    } while (ret == -1 && errno == EINTR);
    if (ret == -1) {
      LOG_ERROR("aio_suspend failed while draining requests; errno=%d, %s",
                errno, ::strerror(errno));
      struct timespec pause = {0, 1000000};  // 1 ms
      ::nanosleep(&pause, nullptr);
      all_ok = false;
    }
  }

  return all_ok;
}

// Submit a batch through POSIX aio_read() and wait for completion with the
// portable aio_suspend() API supported by all target macOS SDK versions.
static int execute_io_aio_suspend(int fd, std::vector<AlignedRead> &read_reqs,
                                  uint64_t n_retries,
                                  ailego::IOBackendType *used_backend) {
  for (size_t req_begin = 0; req_begin < read_reqs.size();
       req_begin += kMacAioBatchSize) {
    size_t n_ops = std::min(kMacAioBatchSize, read_reqs.size() - req_begin);
    std::vector<struct aiocb> cbs(n_ops);
    size_t submitted_count = 0;
    bool submission_ok = true;

    for (size_t i = 0; i < n_ops; ++i) {
      std::memset(&cbs[i], 0, sizeof(cbs[i]));
      cbs[i].aio_fildes = fd;
      cbs[i].aio_offset = static_cast<off_t>(read_reqs[req_begin + i].offset);
      cbs[i].aio_buf = read_reqs[req_begin + i].buf;
      cbs[i].aio_nbytes = read_reqs[req_begin + i].len;
      cbs[i].aio_reqprio = 0;
      cbs[i].aio_lio_opcode = LIO_READ;
      cbs[i].aio_sigevent.sigev_notify = SIGEV_NONE;

      uint64_t tries = 0;
      while (::aio_read(&cbs[i]) == -1) {
        int submit_errno = errno;
        if ((submit_errno == EINTR || submit_errno == EAGAIN) &&
            tries < n_retries) {
          ++tries;
          continue;
        }
        LOG_WARN(
            "aio_read submission failed; errno=%d, %s, offset=%lu, "
            "len=%lu; falling back to pread after draining submitted I/O",
            submit_errno, ::strerror(submit_errno),
            (unsigned long)read_reqs[req_begin + i].offset,
            (unsigned long)read_reqs[req_begin + i].len);
        submission_ok = false;
        break;
      }
      if (!submission_ok) {
        break;
      }
      ++submitted_count;
    }

    bool all_ok =
        drain_aio_requests(cbs, read_reqs, req_begin, submitted_count);

    if (!submission_ok || !all_ok) {
      return execute_io_pread(fd, read_reqs, used_backend);
    }
  }

  if (used_backend != nullptr) {
    *used_backend = ailego::IOBackendType::kPosixAio;
  }
  return 0;
}
#endif  // __APPLE__

int execute_io(IOContext &ctx, int fd, std::vector<AlignedRead> &read_reqs,
               uint64_t n_retries = 0,
               ailego::IOBackendType *used_backend = nullptr) {
#if (defined(__linux) || defined(__linux__))
  if (ailego::IOBackend::Instance().available() ==
      ailego::IOBackendType::kPread) {
    return execute_io_pread(fd, read_reqs, used_backend);
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
      return execute_io_pread(fd, read_reqs, used_backend);
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
      return execute_io_pread(fd, read_reqs, used_backend);
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
      return execute_io_pread(fd, read_reqs, used_backend);
    }
  }

  if (used_backend != nullptr) {
    *used_backend = ailego::IOBackendType::kLibAio;
  }
  return 0;
#elif defined(__APPLE__) && TARGET_OS_OSX
  (void)ctx;
  return execute_io_aio_suspend(fd, read_reqs, n_retries, used_backend);
#else
  (void)ctx;
  (void)n_retries;
  return execute_io_pread(fd, read_reqs, used_backend);
#endif
}

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
    LOG_ERROR("bad thread access; returning -1 as io_context_t");
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

  auto &backend = ailego::IOBackend::Instance();
  std::call_once(g_io_backend_log_once, [&backend] {
    if (backend.available() != ailego::IOBackendType::kPread) {
      LOG_INFO("DiskAnn I/O backend: %s (async I/O enabled)", backend.name());
    } else {
      LOG_WARN(
          "DiskAnn I/O backend: synchronous pread (no async I/O available)");
    }
  });
  if (backend.available() == ailego::IOBackendType::kPread) {
    lk.unlock();
    return;
  }
  int ret = LibAioLoader::Instance().io_setup(MAX_EVENTS, &ctx);
  if (ret != 0) {
    if (ret == -EAGAIN) {
      LOG_ERROR(
          "io_setup failed with EAGAIN: Consider increasing "
          "/proc/sys/fs/aio-max-nr");
    } else {
      LOG_ERROR("io_setup failed; returned: %d, %s", ret, ::strerror(-ret));
    }
  } else {
    LOG_INFO("allocating ctx: %lu", (uint64_t)ctx);
    ctx_map[thread_id] = ctx;
  }
  lk.unlock();
#elif defined(__APPLE__) && TARGET_OS_OSX
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(thread_id) != ctx_map.end()) {
    LOG_ERROR("multiple calls to register_thread from the same thread");
    return;
  }

  std::call_once(g_io_backend_log_once, [] {
    LOG_INFO("DiskAnn I/O backend: macOS POSIX AIO with aio_suspend");
  });
  ctx_map[thread_id] = 0;
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

  // io_destroy is a syscall; keep it outside the lock to avoid blocking others
  if (ailego::IOBackend::Instance().available() !=
      ailego::IOBackendType::kPread) {
    LibAioLoader::Instance().io_destroy(ctx);
  }
  LOG_INFO("returned ctx from thread");
#elif defined(__APPLE__) && TARGET_OS_OSX
  auto thread_id = std::this_thread::get_id();

  {
    std::lock_guard<std::mutex> lk(ctx_mut);
    auto it = ctx_map.find(thread_id);
    if (it == ctx_map.end()) {
      LOG_ERROR("deregister_thread: thread not registered");
      return;
    }
    ctx_map.erase(it);
  }

  LOG_INFO("deregistered POSIX AIO thread");
#endif
}

void LinuxAlignedFileReader::deregister_all_threads() {
#if (defined(__linux) || defined(__linux__))
  std::unique_lock<std::mutex> lk(ctx_mut);
  bool aio_available = ailego::IOBackend::Instance().available() !=
                       ailego::IOBackendType::kPread;
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    IOContext ctx = x->second;
    if (aio_available) {
      LibAioLoader::Instance().io_destroy(ctx);
    }
  }
  ctx_map.clear();
#elif defined(__APPLE__) && TARGET_OS_OSX
  std::unique_lock<std::mutex> lk(ctx_mut);
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

#if defined(__APPLE__) && TARGET_OS_OSX
  // macOS has no O_DIRECT. F_NOCACHE is its closest per-file equivalent: it
  // asks the kernel to minimize caching for I/O through this descriptor. This
  // is advisory rather than a guarantee that every read reaches the device.
  // Disable read-ahead as well because DiskAnn performs random reads.
  //
  // Do not mmap the entire index and call msync(MS_INVALIDATE) here. That does
  // not provide a reliable global cache eviction guarantee and makes open time
  // and virtual-address usage scale with the size of the index.
  if (this->file_desc != -1) {
    if (::fcntl(this->file_desc, F_NOCACHE, 1) == -1) {
      LOG_WARN(
          "fcntl(F_NOCACHE) failed for %s (errno=%d: %s); reads will use "
          "the page cache",
          fname.c_str(), errno, ::strerror(errno));
    } else {
      LOG_INFO("DiskAnn macOS: F_NOCACHE enabled for %s", fname.c_str());
    }

    if (::fcntl(this->file_desc, F_RDAHEAD, 0) == -1) {
      LOG_WARN("fcntl(F_RDAHEAD, 0) failed for %s (errno=%d: %s)",
               fname.c_str(), errno, ::strerror(errno));
    }
  }
#endif

  LOG_INFO("Opened file : %s", fname.c_str());
}

void LinuxAlignedFileReader::close() {
  if (file_desc >= 0) {
    ::close(file_desc);
    file_desc = -1;
  }
}

int LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                 IOContext &ctx, bool async,
                                 ailego::IOBackendType *used_backend) {
  if (async == true) {
    LOG_WARN("Async currently not supported");
  }

  if (this->file_desc == -1) {
    LOG_ERROR("Attempt to read from invalid file descriptor");
    return IndexError_Runtime;
  }

  int ret = execute_io(ctx, this->file_desc, read_reqs, 0, used_backend);

  return ret;
}

}  // namespace core
}  // namespace zvec
