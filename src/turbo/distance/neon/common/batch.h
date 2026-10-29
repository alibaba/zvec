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
#include <type_traits>

namespace zvec::turbo::neon::detail {

// Compile-time row counts keep independent accumulators in registers. Partial
// groups use the same batch kernel, without padding pointers or single calls.
template <typename Kernel>
inline void for_each_batch4(size_t count, Kernel kernel) {
  size_t i = 0;
  for (; count - i >= 4; i += 4) {
    kernel(std::integral_constant<size_t, 4>{}, i);
  }
  switch (count - i) {
    case 3:
      kernel(std::integral_constant<size_t, 3>{}, i);
      break;
    case 2:
      kernel(std::integral_constant<size_t, 2>{}, i);
      break;
    case 1:
      kernel(std::integral_constant<size_t, 1>{}, i);
      break;
    default:
      break;
  }
}

}  // namespace zvec::turbo::neon::detail
