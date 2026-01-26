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

#include "file.h"
#include "ailego/internal/platform.h"
#if !defined(_WIN64) && !defined(_WIN32)
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#else
#include <Windows.h>
#endif

namespace zvec {
namespace ailego {

#if !defined(_WIN64) && !defined(_WIN32)

static inline int OpenSafely(const char *path, int flags) {
  int fd = open(path, flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  while (fd == -1 && errno == EINTR) {
    fd = open(path, flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  }
  return fd;
}

static inline void CloseSafely(int fd) {
  int ret = close(fd);
  while (ret == -1 && errno == EINTR) {
    ret = close(fd);
  }
}

static inline ssize_t ReadSafely(int fd, void *buf, size_t count) {
  ssize_t ret = read(fd, buf, count);
  while (ret == -1 && errno == EINTR) {
    ret = read(fd, buf, count);
  }
  return ret;
}

static inline ssize_t PreadSafely(int fd, void *buf, size_t count,
                                  ssize_t offset) {
  ssize_t ret = pread(fd, buf, count, offset);
  while (ret == -1 && errno == EINTR) {
    ret = pread(fd, buf, count, offset);
  }
  return ret;
}

static inline ssize_t WriteSafely(int fd, const void *buf, size_t count) {
  ssize_t ret = write(fd, buf, count);
  while (ret == -1 && errno == EINTR) {
    ret = write(fd, buf, count);
  }
  return ret;
}

static inline ssize_t PwriteSafely(int fd, const void *buf, size_t count,
                                   ssize_t offset) {
  ssize_t ret = pwrite(fd, buf, count, offset);
  while (ret == -1 && errno == EINTR) {
    ret = pwrite(fd, buf, count, offset);
  }
  return ret;
}

static inline size_t ReadAll(int fd, void *buf, size_t count) {
  size_t rdlen = 0;
  while (rdlen < count) {
    ssize_t ret = ReadSafely(fd, (char *)buf + rdlen, count - rdlen);
    if (ret <= 0) {
      break;
    }
    rdlen += ret;
  }
  return rdlen;
}

static inline size_t PreadAll(int fd, void *buf, size_t count, ssize_t offset) {
  size_t rdlen = 0;
  while (rdlen < count) {
    ssize_t ret =
        PreadSafely(fd, (char *)buf + rdlen, count - rdlen, offset + rdlen);
    if (ret <= 0) {
      break;
    }
    rdlen += ret;
  }
  return rdlen;
}

static inline size_t WriteAll(int fd, const void *buf, size_t count) {
  size_t wrlen = 0;
  while (wrlen < count) {
    ssize_t ret = WriteSafely(fd, (const char *)buf + wrlen, count - wrlen);
    if (ret <= 0) {
      break;
    }
    wrlen += ret;
  }
  return wrlen;
}

static inline size_t PwriteAll(int fd, const void *buf, size_t count,
                               ssize_t offset) {
  size_t wrlen = 0;
  while (wrlen < count) {
    ssize_t ret = PwriteSafely(fd, (const char *)buf + wrlen, count - wrlen,
                               offset + wrlen);
    if (ret <= 0) {
      break;
    }
    wrlen += ret;
  }
  return wrlen;
}

bool File::create(const char *path, size_t len, bool direct) {
  ailego_false_if_false(native_handle_ == File::InvalidHandle && path);

  // Try opening or creating a file
  int flags = O_RDWR | O_CREAT;
#ifdef O_DIRECT
  if (direct) {
    flags |= O_DIRECT;
  }
#else
  (void)direct;
#endif

  int fd = OpenSafely(path, flags);
  ailego_false_if_lt_zero(fd);

#ifdef F_NOCACHE
  // Direct IO canonical solution for Mac OSX
  if (direct) {
    ailego_false_if_ne_zero(fcntl(fd, F_NOCACHE, 1));
  }
#endif

  // Truncate the file to the specified size
  ailego_do_if_ne_zero(ftruncate(fd, len)) {
    CloseSafely(fd);
    return false;
  }

  read_only_ = false;
  native_handle_ = fd;
  return true;
}

bool File::open(const char *path, bool rdonly, bool direct) {
  ailego_false_if_false(native_handle_ == File::InvalidHandle && path);

  // Try opening the file
  int flags = rdonly ? O_RDONLY : O_RDWR;
#ifdef O_DIRECT
  if (direct) {
    flags |= O_DIRECT;
  }
#else
  (void)direct;
#endif

  int fd = OpenSafely(path, flags);
  ailego_false_if_lt_zero(fd);

#ifdef F_NOCACHE
  // Direct IO canonical solution for Mac OSX
  if (direct) {
    ailego_false_if_ne_zero(fcntl(fd, F_NOCACHE, 1));
  }
#endif

  read_only_ = rdonly;
  native_handle_ = fd;
  return true;
}

void File::close(void) {
  ailego_return_if_false(native_handle_ != File::InvalidHandle);
  CloseSafely(native_handle_);
  native_handle_ = File::InvalidHandle;
}

void File::reset(void) {
  ailego_return_if_false(native_handle_ != File::InvalidHandle);
  lseek(native_handle_, 0, SEEK_SET);
}

size_t File::write(const void *data, size_t len) {
  const size_t block_size = 0x40000000u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    size_t wrlen =
        WriteAll(native_handle_, (const uint8_t *)data + total, block_size);
    if (wrlen != block_size) {
      return (total + wrlen);
    }
    total += block_size;
  }
  if (len > 0) {
    total += WriteAll(native_handle_, (const uint8_t *)data + total, len);
  }
  return total;
}

size_t File::write(ssize_t off, const void *data, size_t len) {
  const size_t block_size = 0x40000000u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    size_t wrlen = PwriteAll(native_handle_, (const uint8_t *)data + total,
                             block_size, off + total);
    if (wrlen != block_size) {
      return (total + wrlen);
    }
    total += block_size;
  }
  if (len > 0) {
    total += PwriteAll(native_handle_, (const uint8_t *)data + total, len,
                       off + total);
  }
  return total;
}

size_t File::read(void *buf, size_t len) {
  const size_t block_size = 0x40000000u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    size_t rdlen = ReadAll(native_handle_, (uint8_t *)buf + total, block_size);
    if (rdlen != block_size) {
      return (total + rdlen);
    }
    total += block_size;
  }
  if (len > 0) {
    total += ReadAll(native_handle_, (uint8_t *)buf + total, len);
  }
  return total;
}

size_t File::read(ssize_t off, void *buf, size_t len) {
  const size_t block_size = 0x40000000u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    size_t rdlen = PreadAll(native_handle_, (uint8_t *)buf + total, block_size,
                            off + total);
    if (rdlen != block_size) {
      return (total + rdlen);
    }
    total += block_size;
  }
  if (len > 0) {
    total += PreadAll(native_handle_, (uint8_t *)buf + total, len, off + total);
  }
  return total;
}

bool File::flush(void) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);
  return (fsync(native_handle_) == 0);
}

bool File::seek(ssize_t off, Origin origin) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);
  ailego_false_if_false(lseek(native_handle_, off, (int)origin) != (off_t)-1);
  return true;
}

bool File::truncate(size_t len) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);
  ailego_false_if_ne_zero(ftruncate(native_handle_, (off_t)len));
  return true;
}

size_t File::size(void) const {
  struct stat fs;
  ailego_zero_if_false(native_handle_ != File::InvalidHandle &&
                       fstat(native_handle_, &fs) == 0);
  return (fs.st_size);
}

ssize_t File::offset(void) const {
  off_t off;
  ailego_zero_if_false(native_handle_ != File::InvalidHandle &&
                       (off = lseek(native_handle_, 0, SEEK_CUR)) != -1);
  return off;
}

void *File::MemoryMap(NativeHandle handle, ssize_t off, size_t len, int opts) {
  int prot =
      ((opts & File::MMAP_READONLY) ? PROT_READ : PROT_READ | PROT_WRITE);

#if defined(MAP_POPULATE)
  if (opts & File::MMAP_POPULATE) {
    prot |= MAP_POPULATE;
  }
#endif
  int flags = (opts & File::MMAP_SHARED) ? MAP_SHARED : MAP_PRIVATE;
#if defined(MAP_HUGETLB)
  if (opts & File::MMAP_HUGE_PAGE) {
    flags |= MAP_HUGETLB;
  }
#endif
  void *addr = mmap(nullptr, len, prot, flags, handle, off);
  ailego_null_if_false(addr != MAP_FAILED);

  if (opts & File::MMAP_LOCKED) {
    mlock(addr, len);
  }
  if (opts & File::MMAP_WARMUP) {
    File::MemoryWarmup(addr, len);
  }
  return addr;
}

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif

void *File::MemoryMap(size_t len, int opts) {
#if defined(MAP_ANONYMOUS)
  int prot =
      ((opts & File::MMAP_READONLY) ? PROT_READ : PROT_READ | PROT_WRITE);

#if defined(MAP_POPULATE)
  if (opts & File::MMAP_POPULATE) {
    prot |= MAP_POPULATE;
  }
#endif
  int flags = (opts & File::MMAP_SHARED) ? MAP_SHARED | MAP_ANONYMOUS
                                         : MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_HUGETLB)
  if (opts & File::MMAP_HUGE_PAGE) {
    flags |= MAP_HUGETLB;
  }
#endif
  void *addr = mmap(nullptr, len, prot, flags, -1, 0);
  ailego_null_if_false(addr != MAP_FAILED);
  return addr;
#else
  (void)len;
  (void)opts;
  return nullptr;
#endif  // MAP_ANONYMOUS
}

void *File::MemoryRemap(void *oldptr, size_t oldsize, void *newptr,
                        size_t newsize) {
#if defined(__linux) || defined(__linux__)
  return newptr ? mremap(oldptr, oldsize, newsize, MREMAP_FIXED, newptr)
                : mremap(oldptr, oldsize, newsize, MREMAP_MAYMOVE);
#elif defined(__NetBSD__)
  return newptr ? mremap(oldptr, oldsize, newptr, newsize, MAP_FIXED)
                : mremap(oldptr, oldsize, nullptr, newsize, 0);
#else
  (void)oldptr;
  (void)oldsize;
  (void)newptr;
  (void)newsize;
  errno = ENOTSUP;
  return nullptr;
#endif
}

void File::MemoryUnmap(void *addr, size_t len) {
  ailego_return_if_false(addr);
  munmap(addr, len);
}

bool File::MemoryFlush(void *addr, size_t len) {
  ailego_false_if_false(addr);
  return (msync(addr, len, MS_ASYNC) == 0);
}

bool File::MemoryLock(void *addr, size_t len) {
  ailego_false_if_false(addr && len);
  return (mlock(addr, len) == 0);
}

bool File::MemoryUnlock(void *addr, size_t len) {
  ailego_false_if_false(addr && len);
  return (munlock(addr, len) == 0);
}

#else

//! Create a local file
bool File::create(const char *path, size_t len, bool direct) {
  ailego_false_if_false(native_handle_ == File::InvalidHandle && path);

  // Try opening or creating the file
  HANDLE file_handle =
      CreateFileA(path, GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ, nullptr,
                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  ailego_false_if_false(file_handle != INVALID_HANDLE_VALUE);

  // Truncate the file to the specified size
  LARGE_INTEGER file_size;
  file_size.QuadPart = len;
  ailego_do_if_false(
      SetFilePointerEx(file_handle, file_size, nullptr, FILE_BEGIN) &&
      SetEndOfFile(file_handle)) {
    CloseHandle(file_handle);
    return false;
  }

  if (!direct) {
    // Reset the file pointer
    SetFilePointer(file_handle, 0, nullptr, FILE_BEGIN);
  } else {
    // Close and reopen file
    CloseHandle(file_handle);
    file_handle = CreateFileA(
        path, GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    ailego_false_if_false(file_handle != INVALID_HANDLE_VALUE);
  }

  read_only_ = false;
  native_handle_ = file_handle;
  return true;
}

//! Open a local file
bool File::open(const char *path, bool rdonly, bool direct) {
  ailego_false_if_false(native_handle_ == File::InvalidHandle && path);

  // Try opening the file
  DWORD flags = FILE_ATTRIBUTE_NORMAL;
  if (direct) {
    flags |= FILE_FLAG_NO_BUFFERING;
  }
  HANDLE file_handle =
      CreateFileA(path, (rdonly ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE),
                  FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
  ailego_false_if_false(file_handle != INVALID_HANDLE_VALUE);

  read_only_ = rdonly;
  native_handle_ = file_handle;
  return true;
}

void File::close(void) {
  ailego_return_if_false(native_handle_ != File::InvalidHandle);
  CloseHandle(native_handle_);
  native_handle_ = File::InvalidHandle;
}

void File::reset(void) {
  ailego_return_if_false(native_handle_ != File::InvalidHandle);
  SetFilePointer(native_handle_, 0, nullptr, FILE_BEGIN);
}

size_t File::write(const void *data, size_t len) {
  const DWORD block_size = 0x40000000u;
  DWORD wrlen = 0u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    if (!WriteFile(native_handle_, (const uint8_t *)data + total, block_size,
                   &wrlen, nullptr)) {
      return total;
    }
    if (wrlen != block_size) {
      return (total + wrlen);
    }
    total += block_size;
  }
  if (len > 0 && WriteFile(native_handle_, (const uint8_t *)data + total,
                           (DWORD)len, &wrlen, nullptr)) {
    total += wrlen;
  }
  return total;
}

size_t File::write(ssize_t off, const void *data, size_t len) {
  const DWORD block_size = 0x40000000u;
  DWORD wrlen = 0u;
  size_t total = 0u;

  OVERLAPPED overlapped;
  memset(&overlapped, 0, sizeof(OVERLAPPED));

  for (; len >= block_size; len -= block_size) {
    uint64_t current = off + total;
    overlapped.OffsetHigh = (DWORD)(current >> 32);
    overlapped.Offset = (DWORD)(current & 0xffffffffu);

    if (!WriteFile(native_handle_, (const uint8_t *)data + total, block_size,
                   &wrlen, &overlapped)) {
      return total;
    }
    if (wrlen != block_size) {
      return (total + wrlen);
    }
    total += block_size;
  }
  if (len > 0) {
    uint64_t current = off + total;
    overlapped.OffsetHigh = (DWORD)(current >> 32);
    overlapped.Offset = (DWORD)(current & 0xffffffffu);

    if (WriteFile(native_handle_, (const uint8_t *)data + total, (DWORD)len,
                  &wrlen, &overlapped)) {
      total += wrlen;
    }
  }
  return total;
}

size_t File::read(void *buf, size_t len) {
  const DWORD block_size = 0x40000000u;
  DWORD rdlen = 0u;
  size_t total = 0u;

  for (; len >= block_size; len -= block_size) {
    if (!ReadFile(native_handle_, (uint8_t *)buf + total, block_size, &rdlen,
                  nullptr)) {
      return total;
    }
    if (rdlen != block_size) {
      return (total + rdlen);
    }
    total += block_size;
  }
  if (len > 0 && ReadFile(native_handle_, (uint8_t *)buf + total, (DWORD)len,
                          &rdlen, nullptr)) {
    total += rdlen;
  }
  return total;
}

size_t File::read(ssize_t off, void *buf, size_t len) {
  const DWORD block_size = 0x40000000u;
  DWORD rdlen = 0u;
  size_t total = 0u;

  OVERLAPPED overlapped;
  memset(&overlapped, 0, sizeof(OVERLAPPED));

  for (; len >= block_size; len -= block_size) {
    uint64_t current = off + total;
    overlapped.OffsetHigh = (DWORD)(current >> 32);
    overlapped.Offset = (DWORD)(current & 0xffffffffu);

    if (!ReadFile(native_handle_, (uint8_t *)buf + total, block_size, &rdlen,
                  &overlapped)) {
      return total;
    }
    if (rdlen != block_size) {
      return (total + rdlen);
    }
    total += block_size;
  }
  if (len > 0) {
    uint64_t current = off + total;
    overlapped.OffsetHigh = (DWORD)(current >> 32);
    overlapped.Offset = (DWORD)(current & 0xffffffffu);

    if (ReadFile(native_handle_, (uint8_t *)buf + total, (DWORD)len, &rdlen,
                 &overlapped)) {
      total += rdlen;
    }
  }
  return total;
}

bool File::flush(void) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);
  return (!!FlushFileBuffers(native_handle_));
}

bool File::seek(ssize_t off, Origin origin) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);

  LARGE_INTEGER file_offset;
  file_offset.QuadPart = off;
  ailego_false_if_false(SetFilePointerEx(native_handle_, file_offset, nullptr,
                                         (DWORD)origin) != 0);
  return true;
}

bool File::truncate(size_t len) {
  ailego_false_if_false(native_handle_ != File::InvalidHandle);

  LARGE_INTEGER file_size, orig_file_size;
  file_size.QuadPart = 0;
  orig_file_size.QuadPart = 0;
  ailego_false_if_false(SetFilePointerEx(native_handle_, file_size,
                                         &orig_file_size, FILE_CURRENT));

  // Truncate the file to the specified size
  file_size.QuadPart = len;
  ailego_false_if_false(
      SetFilePointerEx(native_handle_, file_size, nullptr, FILE_BEGIN) &&
      SetEndOfFile(native_handle_));

  // Reset the file pointer
  SetFilePointerEx(native_handle_, orig_file_size, nullptr, FILE_BEGIN);
  return true;
}

size_t File::size(void) const {
  LARGE_INTEGER file_size;
  ailego_zero_if_false(native_handle_ != File::InvalidHandle &&
                       GetFileSizeEx(native_handle_, &file_size));
  return (size_t)file_size.QuadPart;
}

ssize_t File::offset(void) const {
  LARGE_INTEGER file_size;
  LARGE_INTEGER file_size_new;
  file_size.QuadPart = 0;
  ailego_zero_if_false(native_handle_ != File::InvalidHandle &&
                       SetFilePointerEx(native_handle_, file_size,
                                        &file_size_new, FILE_CURRENT));
  return (size_t)file_size_new.QuadPart;
}

void *File::MemoryMap(NativeHandle handle, ssize_t off, size_t len, int opts) {
  LARGE_INTEGER file_size;
  file_size.QuadPart = len;

  // Create map object
  HANDLE file_mapping = CreateFileMapping(
      handle, nullptr,
      ((opts & File::MMAP_READONLY) ? PAGE_READONLY : PAGE_READWRITE),
      file_size.HighPart, file_size.LowPart, nullptr);
  ailego_null_if_false(file_mapping != nullptr);

  DWORD desired_access = FILE_MAP_READ;
  if (!(opts & File::MMAP_READONLY)) {
    desired_access |= FILE_MAP_WRITE;
  }
  if (!(opts & File::MMAP_SHARED)) {
    desired_access |= FILE_MAP_COPY;
  }
  file_size.QuadPart = off;

  // Map the whole file to memory and close handle
  void *addr = MapViewOfFile(file_mapping, desired_access, file_size.HighPart,
                             file_size.LowPart, 0);
  CloseHandle(file_mapping);

  ailego_null_if_false(addr);
  if (opts & File::MMAP_LOCKED) {
    VirtualLock(addr, len);
  }
  if (opts & File::MMAP_WARMUP) {
    File::MemoryWarmup(addr, len);
  }
  return addr;
}

void *File::MemoryMap(size_t, int) {
  return nullptr;
}

void *File::MemoryRemap(void *, size_t, void *, size_t) {
  return nullptr;
}

void File::MemoryUnmap(void *addr, size_t /*len*/) {
  ailego_return_if_false(addr);
  UnmapViewOfFile(addr);
}

bool File::MemoryFlush(void *addr, size_t /*len*/) {
  ailego_false_if_false(addr);
  return (!!FlushViewOfFile(addr, 0));
}

bool File::MemoryLock(void *addr, size_t len) {
  ailego_false_if_false(addr && len);
  return (!!VirtualLock(addr, len));
}

bool File::MemoryUnlock(void *addr, size_t len) {
  ailego_false_if_false(addr && len);
  return (!!VirtualUnlock(addr, len));
}

static inline int getpagesize(void) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return info.dwPageSize;
}
#endif

void File::MemoryWarmup(void *addr, size_t len) {
  static int page_size = getpagesize();

  if (addr && len) {
    uint8_t *p = reinterpret_cast<uint8_t *>(addr);
    uint8_t *end = p + len;
    volatile uint8_t tmp = 0;

    while (p < end) {
      tmp ^= *p;
      p += page_size;
    }
  }
}

}  // namespace ailego
}  // namespace zvec