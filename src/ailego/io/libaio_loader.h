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

#if defined(__linux) || defined(__linux__)

#include <dlfcn.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <ailego/io/libaio_def.h>  // ABI-stable struct definitions (replaces <libaio.h>)

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
    static LibAioLoader instance;
    return instance;
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

  // Function pointers — nullptr until load() succeeds.
  aio_setup_fn io_setup;
  aio_destroy_fn io_destroy;
  aio_submit_fn io_submit;
  aio_getevents_fn io_getevents;

 private:
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

  LibAioLoader(const LibAioLoader &) = delete;
  LibAioLoader &operator=(const LibAioLoader &) = delete;

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
