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
#include <mutex>
#include <new>
#include <thread>
#include <ailego/io/io_backend_def.h>
#include <zvec/ailego/buffer/vector_page_table.h>
#include <zvec/ailego/io/io_backend.h>
#include <zvec/ailego/logger/logger.h>
#if defined(__APPLE__) || defined(__MACH__)
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
     defined(__MACH__))
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

int setup_io_ctx(IOContext &ctx) {
  auto selected = ailego::IOBackend::Instance().available();
  ctx = new (std::nothrow) IoBackend();
  if (ctx == nullptr) {
    LOG_ERROR("Failed to allocate DiskAnn I/O context");
    return IndexError_NoMemory;
  }
  ctx->type = selected;

#if (defined(__linux) || defined(__linux__))
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

#if (defined(__linux) || defined(__linux__))
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
  // Guard against null or sentinel contexts.
  if (ctx == nullptr || ctx == (IOContext)-1) {
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
    LOG_INFO("allocating ctx: %p", static_cast<void *>(ctx));
  }
  ctx_map[thread_id] = ctx;
  lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread() {
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

  // Keep teardown outside the lock; async backends may block in syscalls.
  destroy_io_ctx(ctx);
  LOG_INFO("returned ctx from thread");
}

void LinuxAlignedFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    destroy_io_ctx(x->second);
  }
  ctx_map.clear();
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
  if (async) {
    LOG_WARN("Async currently not supported");
  }
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
                                      IOContext &ctx, bool /*async*/) {
  if (!pool_) {
    LOG_ERROR("BufferPoolAlignedFileReader: buffer pool is not available");
    return IndexError_Runtime;
  }
  if (read_reqs.empty()) return 0;

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

    std::vector<UniquePage> unique_pages;
    std::vector<PageOccurrence> occurrences;
    unique_pages.reserve(total_pages);
    occurrences.reserve(total_pages);
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
        occurrences.push_back(PageOccurrence{unique_index, destination,
                                             offset_in_page, copy_length,
                                             canonical_index});
        source_offset += copy_length;
        destination += copy_length;
        remaining -= copy_length;
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
        // A cache-admission race may exhaust the remaining budget after the
        // policy decision above. acquire_pages() rolls back all pins on
        // failure, so preserve query availability by reading the batch
        // directly instead of surfacing a capacity error to the caller.
        for (const size_t index : admitted_indices) {
          unique_pages[index].bypass_candidate = true;
        }
      }
    }

    // A rejected page may have become resident or started loading while the
    // admitted portion of this batch was populated. Rejoin that cache flight
    // instead of issuing duplicate direct I/O. This does not record another
    // admission observation.
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

    // Preserve contiguous DiskANN reads on the direct path. Duplicate slices
    // use their first destination as the canonical read target and are fanned
    // out after I/O completes. A native buffer-pool page may contain multiple
    // 4 KiB DiskANN sectors (for example, 16 KiB pages on Apple silicon).
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
    release_cached_pages();
    return 0;
  } catch (const std::bad_alloc &) {
    return IndexError_NoMemory;
  }
}

int BufferPoolAlignedFileReader::submit(PendingBatch &batch,
                                        std::vector<AlignedRead> &read_reqs,
                                        IOContext &ctx) {
  batch.n_submitted = 0;
  batch.n_reaped = 0;
  batch.used_pread = false;
#if defined(__linux__) || defined(__linux)
  batch.cbs.clear();
  batch.cb_ptrs.clear();
#endif

  const int ret = read(read_reqs, ctx);
  if (ret != 0) {
    return ret;
  }

  // Buffer-pool reads complete synchronously. Reporting them through the
  // completion API keeps the search pipeline identical to the direct reader.
  batch.used_pread = true;
  batch.n_submitted = static_cast<uint32_t>(read_reqs.size());
  return 0;
}

int BufferPoolAlignedFileReader::get_completed(
    PendingBatch &batch, IOContext & /*ctx*/, int /*min_completed*/,
    std::vector<uint32_t> &completed_indices) {
  completed_indices.clear();
  for (uint32_t i = batch.n_reaped; i < batch.n_submitted; ++i) {
    completed_indices.push_back(i);
  }
  batch.n_reaped = batch.n_submitted;
  return static_cast<int>(completed_indices.size());
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
      ctx->type == ailego::IOBackendType::kPread) {
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

}  // namespace core
}  // namespace zvec
