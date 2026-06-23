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
#include <vector>
#include <ailego/math/fht.h>

namespace zvec {
namespace core {

// ============================================================================
// FhtKacRotatorImpl - O(d log d) FHT-based Kac random rotation
//
// Requires dimension % 4 == 0 (scalar tails handle SIMD remainder).
// When dimension is a power of 2, uses 4 rounds of (flip -> FHT -> rescale).
// When dimension is NOT a power of 2, uses kacs_walk reduction.
// ============================================================================

struct FhtKacRotatorImpl {
  std::vector<uint8_t> flip;
  size_t trunc_dim{0};
  float fac{0};

  static constexpr size_t kByteLen = 8;

  void init(size_t dim);
  void rotate(const float *in, float *out, size_t dim) const;
  void unrotate(const float *in, float *out, size_t dim) const;
  void save(char *data) const;
  void load(const char *data);
  size_t dump_bytes() const;
};

}  // namespace core
}  // namespace zvec
