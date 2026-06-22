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
#include <vector>

namespace zvec {
namespace core {

// ============================================================================
// MatrixRotatorImpl - O(d^2) random orthogonal matrix rotation
//
// No alignment requirement on dimension.  Uses a dim x dim square orthogonal
// matrix generated via Householder QR on a random Gaussian matrix.
// ============================================================================

struct MatrixRotatorImpl {
  std::vector<float> matrix;  // dim x dim, row-major

  void init(size_t dim);
  void rotate(const float *in, float *out, size_t dim) const;
  void unrotate(const float *in, float *out, size_t dim) const;
  void save(char *data) const;
  void load(const char *data);
  size_t dump_bytes() const;
};

}  // namespace core
}  // namespace zvec
