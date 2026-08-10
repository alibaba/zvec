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
#include <new>
#include <thread>
#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>

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
  std::call_once(g_io_backend_log_once, [] {
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
  });
#endif
}

int setup_io_ctx(IOContext &ctx) {
#if (defined(__linux) || defined(__linux__))
  log_diskann_io_backend();
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
    LOG_INFO("allocating ctx: %llu",
             static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ctx)));
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
                                 IOContext &ctx) {
  if (this->file_desc == -1) {
    LOG_ERROR("Attempt to read from invalid file descriptor");
    return IndexError_Runtime;
  }

  int ret = execute_io(ctx, this->file_desc, read_reqs);

  return ret;
}

BufferPoolAlignedFileReader::BufferPoolAlignedFileReader(
    std::shared_ptr<ailego::VecBufferPool> pool)
    : pool_(std::move(pool)) {}

BufferPoolAlignedFileReader::~BufferPoolAlignedFileReader() = default;

IOContext &BufferPoolAlignedFileReader::get_ctx() {
  return unused_ctx_;
}

void BufferPoolAlignedFileReader::register_thread() {}

void BufferPoolAlignedFileReader::deregister_thread() {}

void BufferPoolAlignedFileReader::deregister_all_threads() {}

void BufferPoolAlignedFileReader::open(const std::string &fname) {
  bypass_reader_.open(fname);
}

void BufferPoolAlignedFileReader::close() {
  bypass_reader_.close();
  pool_.reset();
}

int BufferPoolAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                      IOContext &ctx) {
  if (!pool_) {
    LOG_ERROR("BufferPoolAlignedFileReader: buffer pool is not available");
    return IndexError_Runtime;
  }
  if (read_reqs.empty()) return 0;

  try {
    struct UniquePage {
      ailego::block_id_t page_id;
      char *first_destination;
      char *cached_page{nullptr};
    };
    struct PageOccurrence {
      size_t unique_index;
      char *destination;
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
      if (offset % ailego::kVectorPageSize != 0 ||
          length % ailego::kVectorPageSize != 0 ||
          offset > pool_->file_size() || length > pool_->file_size() - offset) {
        return IndexError_InvalidArgument;
      }
      const size_t pages = length / ailego::kVectorPageSize;
      if (pages > std::numeric_limits<size_t>::max() - total_pages) {
        return IndexError_InvalidLength;
      }
      total_pages += pages;
    }

    std::vector<UniquePage> unique_pages;
    std::vector<PageOccurrence> occurrences;
    unique_pages.reserve(total_pages);
    occurrences.reserve(total_pages);
    for (const AlignedRead &req : read_reqs) {
      const size_t first_page =
          static_cast<size_t>(req.offset) / ailego::kVectorPageSize;
      const size_t pages =
          static_cast<size_t>(req.len) / ailego::kVectorPageSize;
      char *destination = static_cast<char *>(req.buf);
      for (size_t i = 0; i < pages; ++i) {
        const auto page_id = static_cast<ailego::block_id_t>(first_page + i);
        size_t unique_index = 0;
        while (unique_index < unique_pages.size() &&
               unique_pages[unique_index].page_id != page_id) {
          ++unique_index;
        }
        char *page_destination = destination + i * ailego::kVectorPageSize;
        if (unique_index == unique_pages.size()) {
          unique_pages.push_back(
              UniquePage{page_id, page_destination, nullptr});
        }
        occurrences.push_back(PageOccurrence{unique_index, page_destination});
      }
    }

    std::vector<ailego::block_id_t> admitted_ids;
    std::vector<size_t> admitted_indices;
    std::vector<char *> admitted_pages(unique_pages.size(), nullptr);
    std::vector<AlignedRead> bypass_requests;
    admitted_ids.reserve(unique_pages.size());
    admitted_indices.reserve(unique_pages.size());
    bypass_requests.reserve(unique_pages.size());

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
      if (pool_->should_admit_page(page.page_id)) {
        admitted_ids.push_back(page.page_id);
        admitted_indices.push_back(i);
      } else {
        bypass_requests.emplace_back(
            static_cast<uint64_t>(page.page_id) * ailego::kVectorPageSize,
            ailego::kVectorPageSize, page.first_destination);
      }
    }

    if (!admitted_ids.empty() &&
        !pool_->acquire_pages(admitted_ids.data(), admitted_ids.size(),
                              admitted_pages.data())) {
      release_cached_pages();
      return IndexError_ReadData;
    }
    for (size_t i = 0; i < admitted_ids.size(); ++i) {
      unique_pages[admitted_indices[i]].cached_page = admitted_pages[i];
    }

    if (!bypass_requests.empty()) {
#if defined(__linux__) || defined(__linux)
      // Buffer-pool hits need no DiskANN I/O context. Create it only when
      // admission first chooses direct AIO; the caller already owns and
      // destroys this context with its normal DiskANN context lifecycle.
      if (ctx == nullptr && setup_io_ctx(ctx) != 0) {
        release_cached_pages();
        return IndexError_Runtime;
      }
#endif
      const int read_ret = bypass_reader_.read(bypass_requests, ctx);
      if (read_ret != 0) {
        release_cached_pages();
        return read_ret;
      }
      pool_->record_bypass_read(bypass_requests.size() *
                                ailego::kVectorPageSize);
    }

    for (const PageOccurrence &occurrence : occurrences) {
      const UniquePage &page = unique_pages[occurrence.unique_index];
      if (page.cached_page != nullptr) {
        std::memcpy(occurrence.destination, page.cached_page,
                    ailego::kVectorPageSize);
      } else if (occurrence.destination != page.first_destination) {
        std::memcpy(occurrence.destination, page.first_destination,
                    ailego::kVectorPageSize);
      }
    }
    release_cached_pages();
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
}


}  // namespace core
}  // namespace zvec
