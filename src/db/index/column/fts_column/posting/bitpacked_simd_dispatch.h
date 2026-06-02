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

#include <cstddef>
#include <cstdint>

namespace zvec::fts::simd {

// Function pointer types for SIMD-dispatched operations.
using MaxFunc = void (*)(const uint32_t *, const uint32_t *, const uint32_t *,
                         size_t, uint32_t, uint32_t &, uint32_t &, uint32_t &);
using PackFunc = void (*)(const uint32_t *, uint8_t, uint8_t *);
using UnpackFunc = void (*)(const uint8_t *, uint8_t, uint32_t *);
using PrefixSumFunc = void (*)(const uint32_t *, uint32_t, uint32_t,
                               uint32_t *);
using FindFirstGeFunc = size_t (*)(const uint32_t *, uint32_t, uint32_t,
                                   size_t);

/// Dispatch table populated once at startup via CPU feature detection.
struct DispatchTable {
  MaxFunc max_128;
  PackFunc pack_uint32_128;
  UnpackFunc unpack_uint32_128;
  PrefixSumFunc prefix_sum_128;
  FindFirstGeFunc find_first_ge;
};

/// Get the global dispatch table (initialized on first call).
const DispatchTable &get_dispatch();

}  // namespace zvec::fts::simd
