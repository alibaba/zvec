// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "buffer_storage_directory_pwrite.h"

namespace zvec::test {
namespace {
thread_local DirectorySplitPwriteHook pwrite_hook = nullptr;
thread_local DirectorySplitPreadHook pread_hook = nullptr;

ssize_t PwriteForDirectorySplit(int fd, const void *buffer, size_t count,
                                off_t offset) {
  // dyld leaves references within the interposing dylib bound to libSystem.
  if (pwrite_hook) return pwrite_hook(::pwrite, fd, buffer, count, offset);
  return ::pwrite(fd, buffer, count, offset);
}

struct InterposePwrite {
  decltype(&::pwrite) replacement;
  decltype(&::pwrite) original;
};

__attribute__((used, section("__DATA,__interpose")))
const InterposePwrite kInterposePwrite{PwriteForDirectorySplit, ::pwrite};

ssize_t PreadForDirectorySplit(int fd, void *buffer, size_t count,
                               off_t offset) {
  if (pread_hook) return pread_hook(::pread, fd, buffer, count, offset);
  return ::pread(fd, buffer, count, offset);
}

struct InterposePread {
  decltype(&::pread) replacement;
  decltype(&::pread) original;
};

__attribute__((used, section("__DATA,__interpose")))
const InterposePread kInterposePread{PreadForDirectorySplit, ::pread};
}  // namespace

void SetDirectorySplitPwriteHook(DirectorySplitPwriteHook hook) {
  pwrite_hook = hook;
}

void SetDirectorySplitPreadHook(DirectorySplitPreadHook hook) {
  pread_hook = hook;
}
}  // namespace zvec::test
