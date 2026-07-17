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
#include <zvec/core/framework/index_logger.h>
#if defined(__APPLE__) || defined(__MACH__)
#include <fcntl.h>
#include <unistd.h>
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
#elif defined(__APPLE__) || defined(__MACH__)
  // Create a kqueue for this context. On macOS the kqueue is used to
  // monitor file descriptor readiness for async-style I/O.
  int kq = ::kqueue();
  if (kq == -1) {
    LOG_ERROR("kqueue() failed in setup_io_ctx; errno=%d, %s", errno,
              ::strerror(errno));
    return IndexError_Runtime;
  }
  ctx = kq;
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
#elif defined(__APPLE__) || defined(__MACH__)
  if (ctx >= 0) {
    ::close(ctx);
    ctx = -1;
  }
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

#if defined(__APPLE__) || defined(__MACH__)
// Execute batch I/O on macOS using kqueue to monitor file descriptor
// readiness and pread for actual data transfer.
//
// On macOS, regular file descriptors are almost always "readable", so
// kqueue's primary value here is providing the same async I/O interface
// as Linux's libaio. For each read request we:
//   1. Attempt a non-blocking pread (via O_NONBLOCK on the fd).
//   2. If EAGAIN, wait on kqueue for EVFILT_READ readiness, then retry.
//   3. Fall back to blocking pread if kqueue encounters an error.
//
// The kqueue fd is passed in as the IOContext. If no valid kqueue is
// available, we fall back to plain blocking pread.
static int execute_io_kqueue(int kq, int fd,
                             std::vector<AlignedRead> &read_reqs) {
  // If no kqueue available, fall back to blocking pread.
  if (kq < 0) {
    return execute_io_pread(fd, read_reqs);
  }

  // Register the file descriptor with the kqueue for read events.
  // EV_CLEAR gives edge-triggered semantics so we only get notified
  // when new data becomes available.
  struct kevent ke;
  EV_SET(&ke, fd, EVFILT_READ, EV_ADD | EV_CLEAR, 0, 0, nullptr);

  for (auto &req : read_reqs) {
    while (true) {
      ssize_t bytes_read = ::pread(fd, req.buf, req.len, req.offset);

      if (bytes_read > 0) {
        // Successfully read data; verify full read.
        if ((size_t)bytes_read != req.len) {
          // Partial read — retry for the remaining bytes.
          // Update offset and buffer to read the rest.
          char *buf_ptr = static_cast<char *>(req.buf) + bytes_read;
          uint64_t new_offset = req.offset + bytes_read;
          size_t remaining = req.len - bytes_read;
          while (remaining > 0) {
            ssize_t n = ::pread(fd, buf_ptr, remaining, new_offset);
            if (n < 0) {
              if (errno == EINTR) continue;
              LOG_ERROR("pread retry failed; errno=%d, %s", errno,
                        ::strerror(errno));
              return IndexError_Runtime;
            }
            if (n == 0) break;
            buf_ptr += n;
            new_offset += n;
            remaining -= n;
          }
          if (remaining > 0) {
            LOG_ERROR("pread short read after retry; remaining=%zu", remaining);
            return IndexError_Runtime;
          }
        }
        break;  // Success, move to next request.
      }

      if (bytes_read == 0) {
        // EOF — should not happen for a valid index file.
        LOG_ERROR("pread returned 0 (EOF); offset=%lu, len=%lu",
                  (unsigned long)req.offset, (unsigned long)req.len);
        return IndexError_Runtime;
      }

      // bytes_read == -1, error
      if (errno == EINTR) {
        continue;  // Retry on signal.
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // Data not ready — wait on kqueue for readability.
        struct kevent events[1];
        struct timespec ts;
        ts.tv_sec = 5;  // 5 second timeout as a safety net.
        ts.tv_nsec = 0;
        int n_ev = ::kevent(kq, &ke, 1, events, 1, &ts);
        if (n_ev < 0) {
          if (errno == EINTR) continue;
          // kqueue error — fall back to blocking pread for this request.
          LOG_WARN("kevent failed; errno=%d, %s, falling back to pread", errno,
                   ::strerror(errno));
          return execute_io_pread(fd, read_reqs);
        }
        if (n_ev == 0) {
          // Timeout — fall back to blocking pread.
          LOG_WARN("kqueue timeout, falling back to pread");
          return execute_io_pread(fd, read_reqs);
        }
        // Event triggered — retry pread.
        continue;
      }

      // Other error — fall back to blocking pread.
      LOG_ERROR("pread failed; errno=%d, %s, falling back to pread", errno,
                ::strerror(errno));
      return execute_io_pread(fd, read_reqs);
    }
  }

  return 0;
}
#endif  // __APPLE__

int execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs,
               uint64_t n_retries = 0) {
#if (defined(__linux) || defined(__linux__))
  if (ailego::IOBackend::Instance().available() ==
      ailego::IOBackendType::kPread) {
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
#elif defined(__APPLE__) || defined(__MACH__)
  // On macOS, use kqueue-based I/O. The IOContext (ctx) is a kqueue fd.
  (void)n_retries;
  return execute_io_kqueue(ctx, fd, read_reqs);
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
#elif defined(__APPLE__) || defined(__MACH__)
  auto thread_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(thread_id) != ctx_map.end()) {
    LOG_ERROR("multiple calls to register_thread from the same thread");
    return;
  }

  IOContext ctx = -1;
  int kq = ::kqueue();
  if (kq == -1) {
    LOG_ERROR("kqueue() failed in register_thread; errno=%d, %s", errno,
              ::strerror(errno));
  } else {
    LOG_INFO("allocating kqueue ctx: %d", kq);
    ctx = kq;
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

  // io_destroy is a syscall; keep it outside the lock to avoid blocking others
  if (ailego::IOBackend::Instance().available() !=
      ailego::IOBackendType::kPread) {
    LibAioLoader::Instance().io_destroy(ctx);
  }
  LOG_INFO("returned ctx from thread");
#elif defined(__APPLE__) || defined(__MACH__)
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

  if (ctx >= 0) {
    ::close(ctx);
  }
  LOG_INFO("returned kqueue ctx from thread");
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
#elif defined(__APPLE__) || defined(__MACH__)
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    IOContext ctx = x->second;
    if (ctx >= 0) {
      ::close(ctx);
    }
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

#if defined(__APPLE__) || defined(__MACH__)
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

}  // namespace core
}  // namespace zvec
