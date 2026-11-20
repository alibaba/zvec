// Copyright 2026 The Stile Authors
// SPDX-License-Identifier: Apache-2.0
// Adapted from Stile v0.1.0 for zvec.

#pragma once

#include <arrow/array.h>
#include <arrow/status.h>

namespace zvec::stile::internal {

// Check storage metadata before Arrow validation or any typed buffer access.
arrow::Status CheckReadable(const arrow::ArrayData &data);
}  // namespace zvec::stile::internal
