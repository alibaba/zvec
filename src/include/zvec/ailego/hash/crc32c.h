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

#include <zvec/ailego/container/vector.h>

namespace zvec {
namespace ailego {

/*! Crc32c Hash
 */
struct Crc32c {
  //! Compute the CRC32C checksum for the source data buffer
  static uint32_t Hash(const void *data, size_t len, uint32_t crc);

  //! Compute the CRC32C checksum for the source data buffer
  static inline uint32_t Hash(const void *data, size_t len) {
    return Hash(data, len, 0u);
  }
};

}  // namespace ailego
}  // namespace zvec
