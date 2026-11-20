// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0

#include "db/sqlengine/stile/validation.h"
#include <arrow/util/align_util.h>

namespace zvec::stile::internal {
namespace {
arrow::Status CheckCpu(const arrow::ArrayData &data, unsigned depth) {
  if (!data.type) return arrow::Status::Invalid("Missing Arrow array type");
  if (depth > 128)
    return arrow::Status::CapacityError("Arrow array nesting exceeds 128");
  for (const auto &buffer : data.buffers) {
    if (buffer && !buffer->is_cpu()) {
      return arrow::Status::NotImplemented(
          "Stile requires CPU-accessible buffers");
    }
    if (buffer && (buffer->size() < 0 || (buffer->size() && !buffer->data()))) {
      return arrow::Status::Invalid("Invalid CPU buffer");
    }
  }
  for (const auto &child : data.child_data) {
    if (!child) return arrow::Status::Invalid("Missing Arrow array child");
    ARROW_RETURN_NOT_OK(CheckCpu(*child, depth + 1));
  }
  if (data.dictionary) return CheckCpu(*data.dictionary, depth + 1);
  return arrow::Status::OK();
}
}  // namespace

arrow::Status CheckReadable(const arrow::ArrayData &data) {
  ARROW_RETURN_NOT_OK(CheckCpu(data, 0));
  if (!arrow::util::CheckAlignment(data, arrow::util::kValueAlignment)) {
    return arrow::Status::Invalid("Arrow buffer lacks natural value alignment");
  }
  return arrow::Status::OK();
}
}  // namespace zvec::stile::internal
