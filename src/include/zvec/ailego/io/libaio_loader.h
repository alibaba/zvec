#pragma once

#if defined(__linux) || defined(__linux__)

#include <dlfcn.h>
#include <atomic>
#include <mutex>
#include "libaio_def.h"

typedef int (*aio_setup_fn)(int maxevents, io_context_t *ctxp);
typedef int (*aio_destroy_fn)(io_context_t ctx);
typedef int (*aio_submit_fn)(io_context_t ctx, long nr, struct iocb *ios[]);
typedef int (*aio_getevents_fn)(io_context_t ctx, long min_nr, long nr,
                                struct io_event *events,
                                struct timespec *timeout);

// Shares the ZVEC_LIBAIO_LOADER_CLASS_DEFINED guard with the mirror header
// <ailego/io/libaio_loader.h>: the two declare a layout-identical class of the
// same (global) name, so a translation unit that pulls in both include roots
// must keep only the first definition it sees.
#ifndef ZVEC_LIBAIO_LOADER_CLASS_DEFINED
#define ZVEC_LIBAIO_LOADER_CLASS_DEFINED
class LibAioLoader {
 public:
  static LibAioLoader &Instance() {
    static LibAioLoader instance;
    return instance;
  }

  bool Load() {
    if (available_.load(std::memory_order_acquire)) {
      return true;
    }
    std::call_once(once_, [this] { this->TryLoad(); });
    return available_.load(std::memory_order_relaxed);
  }

  bool IsAvailable() const {
    return available_.load(std::memory_order_acquire);
  }

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

  ~LibAioLoader() = default;

  LibAioLoader(const LibAioLoader &) = delete;
  LibAioLoader &operator=(const LibAioLoader &) = delete;

  void TryLoad() {
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
#endif  // ZVEC_LIBAIO_LOADER_CLASS_DEFINED

#endif
