// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace gandiva {

// ABI shared by Gandiva's code generator and the caller. Evaluate exactly one
// row without an output bitmap or temporary storage proportional to batch size.
using ScalarEvalFunction = bool (*)(const uint8_t* const* buffers,
                                   const int64_t* offsets,
                                   const void* const* holders, int64_t row);

// One runtime parameter in Arrow's native primitive representation. Variable-
// width values use two int32 offsets and bytes owned by the bound query.
// Parameter data and validity always use element/bit zero, regardless of row.
struct alignas(8) ScalarParameterSlot {
  uint8_t data[8] = {};
  uint8_t validity = 1;
  const uint8_t* variable_data = nullptr;
  int32_t variable_offsets[2] = {};
};

}  // namespace gandiva
