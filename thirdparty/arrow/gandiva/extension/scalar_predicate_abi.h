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

#pragma once

#include <cstdint>

namespace gandiva {

// ABI shared by Gandiva's code generator and the caller. Evaluate exactly one
// row without an output bitmap or temporary storage proportional to batch size.
using ScalarEvalFunction = bool (*)(const uint8_t* const* buffers, const int64_t* offsets,
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
