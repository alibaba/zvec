// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <arrow/array.h>
#include <arrow/status.h>

namespace zvec::stile::internal {

// Check storage metadata before Arrow validation or any typed buffer access.
arrow::Status CheckReadable(const arrow::ArrayData &data);
}  // namespace zvec::stile::internal
