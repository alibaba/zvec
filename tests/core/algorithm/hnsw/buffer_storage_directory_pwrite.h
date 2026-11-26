// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unistd.h>

namespace zvec::test {
using DirectorySplitPwriteHook = ssize_t (*)(decltype(&::pwrite), int,
                                             const void *, size_t, off_t);

void SetDirectorySplitPwriteHook(DirectorySplitPwriteHook hook);

using DirectorySplitPreadHook = ssize_t (*)(decltype(&::pread), int, void *,
                                            size_t, off_t);

void SetDirectorySplitPreadHook(DirectorySplitPreadHook hook);
}  // namespace zvec::test
