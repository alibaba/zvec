// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#include "buffer_storage_directory_pwrite.h"

namespace zvec::test {
namespace {
thread_local DirectorySplitPwriteHook pwrite_hook = nullptr;

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
}  // namespace

void SetDirectorySplitPwriteHook(DirectorySplitPwriteHook hook) {
  pwrite_hook = hook;
}
}  // namespace zvec::test
