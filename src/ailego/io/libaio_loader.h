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

// dlopen-based wrapper for libaio.  Instead of linking against -laio at build
// time, the DiskAnn plugin loads libaio at runtime via dlopen()/dlsym().  This
// removes libaio as a hard build- and load-time dependency: the plugin .so no
// longer carries a NEEDED entry for libaio.so.1, so it can be dlopen'ed on
// hosts that don't have libaio installed.  Callers that actually exercise the
// async-I/O path will fall back to synchronous pread() when libaio is absent.
//
// All ABI-stable type and struct definitions (struct iocb, struct io_event,
// io_context_t, PADDED macros, io_prep_pread(), ...) now live in libaio_def.h —
// a private header that replaces <libaio.h> so the project has zero build-time
// dependency on libaio-dev.

#pragma once

#if (defined(__linux) || defined(__linux__)) && !defined(__ANDROID__)

#include <dlfcn.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>
#include <ailego/io/libaio_def.h>  // ABI-stable struct definitions (replaces <libaio.h>)
#include <zvec/ailego/buffer/block_eviction_queue.h>

// Function-pointer typedefs for the four libaio syscalls used by DiskAnn.
typedef int (*aio_setup_fn)(int maxevents, io_context_t *ctxp);
typedef int (*aio_destroy_fn)(io_context_t ctx);
typedef int (*aio_submit_fn)(io_context_t ctx, long nr, struct iocb *ios[]);
typedef int (*aio_getevents_fn)(io_context_t ctx, long min_nr, long nr,
                                struct io_event *events,
                                struct timespec *timeout);

// Runtime loader for libaio.  Thread-safe singleton that dlopen()'s libaio
// once and caches the function pointers.  If libaio is not present on the
// host, all pointers remain nullptr and callers should fall back to
// synchronous I/O (pread).
//
// Usage:
//   if (LibAioLoader::Instance().load()) {
//     LibAioLoader::Instance().io_setup(...);
//   }
class LibAioLoader {
 public:
  static LibAioLoader &Instance() {
    // Global thread pools may join workers during static destruction, after
    // this loader would be destroyed. Their thread-local AIO contexts still
    // need io_destroy(), so retain the loader and library until process exit.
    // Individual AIO contexts must still be destroyed by their owners.
    static LibAioLoader *const instance = new LibAioLoader();
    return *instance;
  }

  // Load (or confirm already loaded) libaio.  Returns true on success.
  // Thread-safe and idempotent.
  bool load() {
    if (available_.load(std::memory_order_acquire)) {
      return true;
    }
    std::call_once(once_, [this] { this->try_load(); });
    return available_.load(std::memory_order_relaxed);
  }

  bool is_available() const {
    return available_.load(std::memory_order_acquire);
  }

  struct RingMemory {
    size_t bytes{0};
    zvec::ailego::ProcessRssBudget::MaterializationStamp projection{};
  };

  int setup_budgeted(int maxevents, io_context_t &ctx, RingMemory &memory) {
    if (ctx || !io_setup || !io_destroy || maxevents <= 0) return -EINVAL;
    auto &pool = zvec::ailego::MemoryLimitPool::get_instance();
    const size_t page = pool.page_buffer_size();
    // io_setup returns a completion-ring address. Reserve its header page
    // before the syscall and admit the rest before any submission can use it.
    if (pool.under_process_rss_pressure() ||
        !pool.try_reserve_process_backing(page)) {
      return -ENOMEM;
    }

    io_context_t ring = nullptr;
    const int ret = io_setup(maxevents, &ring);
    if (ret != 0) {
      pool.finish_process_backing(page, false);
      return ret;
    }

    const size_t size = completion_mapping_size(ring, page);
    const size_t extra = size > page ? size - page : 0;
    if (size == 0 || !pool.try_reserve_process_backing(extra)) {
      if (destroy_ring(ring) == 0) {
        pool.finish_process_backing(page, false);
      } else {
        const bool accounted =
            !extra || pool.account_pending_process_backing(extra);
        memory.bytes = page + (accounted ? extra : 0);
        memory.projection = pool.finish_process_backing(memory.bytes, true);
        // Leave the rejected ring with its owner for teardown. A terminal
        // error prevents a caller from allocating another ring on every retry.
        ctx = ring;
        return -EIO;
      }
      return size == 0 ? -EINVAL : -ENOMEM;
    }

    memory = {size, pool.finish_process_backing(size, true)};
    ctx = ring;
    return 0;
  }

  int destroy_budgeted(io_context_t &ctx, RingMemory &memory) {
    if (!ctx) return 0;
    const int ret = destroy_ring(ctx);
    if (ret == 0) {
      if (memory.bytes) {
        zvec::ailego::MemoryLimitPool::get_instance()
            .release_unmapped_process_backing(memory.bytes, memory.projection);
      }
      ctx = nullptr;
      memory = {};
    }
    return ret;
  }

  // Function pointers — nullptr until load() succeeds.
  aio_setup_fn io_setup;
  aio_destroy_fn io_destroy;
  aio_submit_fn io_submit;
  aio_getevents_fn io_getevents;

  LibAioLoader(const LibAioLoader &) = delete;
  LibAioLoader &operator=(const LibAioLoader &) = delete;

 private:
  int destroy_ring(io_context_t ctx) const {
    int ret;
    do {
      ret = io_destroy(ctx);
    } while (ret == -EINTR);
    return ret;
  }

  static size_t completion_mapping_size(io_context_t ctx, size_t page) {
    // This header is the Linux aio_ring ABI used by libaio's io_getevents
    // fast path. Its event count reflects kernel and CPU-specific rounding.
    constexpr uint32_t kRingMagic = 0xa10a10a1;
    struct RingHeader {
      uint32_t id, event_count, head, tail;
      uint32_t magic, compat_features, incompat_features, header_bytes;
    };
    RingHeader header;
    if (!ctx || page < sizeof(header)) return 0;
    std::memcpy(&header, ctx, sizeof(header));
    const size_t events = header.event_count;
    const size_t header_bytes = header.header_bytes;
    if (header.magic != kRingMagic || header.incompat_features != 0 ||
        header_bytes < sizeof(header) || header_bytes > page || events == 0 ||
        events >
            (std::numeric_limits<size_t>::max() - header_bytes - (page - 1)) /
                sizeof(io_event)) {
      return 0;
    }
    const size_t bytes = header_bytes + events * sizeof(io_event);
    return ((bytes + page - 1) / page) * page;
  }

  LibAioLoader()
      : io_setup(nullptr),
        io_destroy(nullptr),
        io_submit(nullptr),
        io_getevents(nullptr) {}

  ~LibAioLoader() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  void try_load() {
    // On Ubuntu 24.04 the libaio package was renamed with the t64 suffix
    // (64-bit time_t transition), so probe both spellings.
    static constexpr const char *kSonames[] = {
        "libaio.so.1",
        "libaio.so.1t64",
    };

    for (const char *soname : kSonames) {
      void *h = dlopen(soname, RTLD_LAZY);
      if (h == nullptr) {
        continue;
      }

      io_setup = reinterpret_cast<aio_setup_fn>(dlsym(h, "io_setup"));
      io_destroy = reinterpret_cast<aio_destroy_fn>(dlsym(h, "io_destroy"));
      io_submit = reinterpret_cast<aio_submit_fn>(dlsym(h, "io_submit"));

      // io_getevents may be redirected to io_getevents_time64 on 32-bit
      // platforms compiled with _TIME_BITS=64.
      io_getevents =
          reinterpret_cast<aio_getevents_fn>(dlsym(h, "io_getevents"));
      if (io_getevents == nullptr) {
        io_getevents =
            reinterpret_cast<aio_getevents_fn>(dlsym(h, "io_getevents_time64"));
      }

      if (io_setup && io_destroy && io_submit && io_getevents) {
        handle_ = h;
        available_.store(true, std::memory_order_release);
        return;
      }

      // Some symbols missing — try the next soname.
      dlclose(h);
      io_setup = nullptr;
      io_destroy = nullptr;
      io_submit = nullptr;
      io_getevents = nullptr;
    }
  }

  std::once_flag once_;
  std::atomic<bool> available_{false};
  void *handle_{nullptr};
};

#endif  // __linux__
