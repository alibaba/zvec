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

#include <array>
#include <cstddef>
#include <cstdint>

namespace zvec::turbo::sparse_internal {

// Keep shuffle controls as constant-initialized bytes. Global __m128i arrays
// initialized with intrinsics can emit AVX instructions in MSVC's DLL startup
// routines, before runtime CPU dispatch has selected a compatible kernel.
template <size_t ElementBytes>
constexpr auto MakeShuffleMasks() {
  static_assert(ElementBytes == 2 || ElementBytes == 4);
  constexpr size_t kElements = 16 / ElementBytes;
  std::array<std::array<uint8_t, 16>, 1U << kElements> masks{};
  for (size_t mask = 0; mask < masks.size(); ++mask) {
    for (size_t byte = 0; byte < 16; ++byte) {
      masks[mask][byte] = 0x81;  // PSHUFB zeroes bytes with the high bit set.
    }
    size_t output = 0;
    for (size_t element = 0; element < kElements; ++element) {
      if (mask & (1U << element)) {
        for (size_t byte = 0; byte < ElementBytes; ++byte) {
          masks[mask][output++] =
              static_cast<uint8_t>(element * ElementBytes + byte);
        }
      }
    }
  }
  return masks;
}

inline constexpr auto kFp16ShuffleMasks = MakeShuffleMasks<2>();
inline constexpr auto kFp32ShuffleMasks = MakeShuffleMasks<4>();

}  // namespace zvec::turbo::sparse_internal
