// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <zvec/export.h>
#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace zvec::ailego {

// Linux and Windows can reserve much more stack than ordinary workers touch.
// Plan for up to 512 KiB per worker; the process RSS controller covers growth
// beyond this allowance. macOS retains its full native default-stack charge.
// This is an admission allowance, not a change to the thread's stack limit.
inline size_t NativeWorkerStackAllowance(size_t stack_capacity) {
#if defined(__linux__) || defined(_WIN32)
  return std::min(stack_capacity, size_t{512} << 10);
#else
  return stack_capacity;
#endif
}

#if defined(_WIN32)
// Default native threads inherit the host executable's PE stack settings,
// including when zvec is loaded as a DLL. Parse only its mapped header region;
// do not charge the DLL's unrelated linker defaults or the calling thread's
// possibly custom stack. Initial commit must fit even when it exceeds the
// ordinary resident-stack allowance.
inline bool WindowsWorkerStackBytes(const void *image, size_t header_bytes,
                                    size_t page_size, size_t *bytes) noexcept {
  if (!image || !bytes || !page_size || header_bytes < sizeof(IMAGE_DOS_HEADER))
    return false;
  IMAGE_DOS_HEADER dos;
  std::memcpy(&dos, image, sizeof(dos));
  if (dos.e_magic != IMAGE_DOS_SIGNATURE || dos.e_lfanew < 0) return false;
  const size_t offset = static_cast<size_t>(dos.e_lfanew);
  if (offset > header_bytes || sizeof(IMAGE_NT_HEADERS) > header_bytes - offset)
    return false;
  IMAGE_NT_HEADERS nt;
  std::memcpy(&nt, static_cast<const char *>(image) + offset, sizeof(nt));
  if (nt.Signature != IMAGE_NT_SIGNATURE ||
      nt.FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER) ||
      nt.OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR_MAGIC ||
      !nt.OptionalHeader.SizeOfStackReserve)
    return false;
  const size_t allowance =
      std::max(NativeWorkerStackAllowance(nt.OptionalHeader.SizeOfStackReserve),
               static_cast<size_t>(nt.OptionalHeader.SizeOfStackCommit));
  const size_t padding = (page_size - allowance % page_size) % page_size;
  if (padding > std::numeric_limits<size_t>::max() - allowance) return false;
  *bytes = allowance + padding;
  return true;
}
#endif

inline size_t DefaultNativeWorkerStackBytes() {
#if defined(__APPLE__) || defined(__linux__)
  pthread_attr_t attributes;
  if (pthread_attr_init(&attributes) != 0) throw std::bad_alloc();
  size_t bytes = 0;
  const int result = pthread_attr_getstacksize(&attributes, &bytes);
  const int destroyed = pthread_attr_destroy(&attributes);
  if (result != 0 || destroyed != 0 || bytes == 0) throw std::bad_alloc();
  return NativeWorkerStackAllowance(bytes);
#elif defined(_WIN32)
  const auto image = ::GetModuleHandleW(nullptr);
  MEMORY_BASIC_INFORMATION region{};
  SYSTEM_INFO system{};
  ::GetSystemInfo(&system);
  size_t bytes = 0;
  if (!image || !::VirtualQuery(image, &region, sizeof(region)) ||
      region.State != MEM_COMMIT || region.Type != MEM_IMAGE ||
      !WindowsWorkerStackBytes(image, region.RegionSize, system.dwPageSize,
                               &bytes))
    throw std::bad_alloc();
  return bytes;
#else
  return 0;
#endif
}

// Admit temporary workers before creating them. Used only when a DB process
// RSS target is configured; standalone payload budgets keep their semantics.
// Share one concurrent high water across pools and retain started workers'
// backing after exit because libc/OS stack caches may keep those pages.
class ZVEC_AILEGO_API WorkerMemoryLease {
 public:
  explicit WorkerMemoryLease(size_t workers);
  ~WorkerMemoryLease();
  WorkerMemoryLease(const WorkerMemoryLease &) = delete;
  WorkerMemoryLease &operator=(const WorkerMemoryLease &) = delete;

  // Call after each successful native thread construction, before releasing
  // the creator's lease reference. Failure refunds never-started capacity.
  void worker_started() noexcept;

 private:
  size_t bytes_{0};
  size_t started_bytes_{0};
  size_t worker_bytes_{0};
};

}  // namespace zvec::ailego
