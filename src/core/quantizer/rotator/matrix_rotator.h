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
#include "rotator.h"

namespace zvec {
namespace core {

// ============================================================================
// MatrixRotator - O(d^2) random orthogonal matrix rotation
//
// No alignment requirement on dimension.  Uses a dim x dim square orthogonal
// matrix generated via Householder QR on a random Gaussian matrix.
// ============================================================================

class MatrixRotator : public Rotator {
 public:
  MatrixRotator() = default;
  ~MatrixRotator() override = default;

  // Virtual interface
  void rotate(const float *in, float *out) const override;
  void unrotate(const float *in, float *out) const override;
  RotatorType rotator_type() const override;

 protected:
  // Protected virtuals for base class factory/serialization
  int init_impl(size_t dim) override;
  size_t blob_bytes() const override;
  void save_blob(char *data) const override;
  void load_blob(const char *data) override;

 private:
  std::vector<float> matrix_;  // dim x dim, row-major
};

}  // namespace core
}  // namespace zvec
