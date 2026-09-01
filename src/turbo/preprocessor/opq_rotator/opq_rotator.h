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
#include <memory>
#include <string>
#include <vector>
// Rooted at src/ so this header stays includable from the PQ quantizer headers,
// which are themselves included from outside the turbo include root.
#include <turbo/preprocessor/preprocessor.h>

namespace zvec {
namespace turbo {

//! OPQ step 1 (fixed codebook -> rotation matrix): solves the orthogonal
//! Procrustes problem from (x, x_hat) pairs.  Owns no PQ knowledge; the
//! alternating loop is driven by the quantizer.
class OpqRotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<OpqRotator>;

  //! Matrix is initialized with a random orthogonal matrix (seeded), so
  //! apply() is valid even before the first fit().
  static Pointer create(int dim, uint64_t seed = 42);

  //! \p x      original-space training matrix, num x dim, packed.
  //! \p x_hat  codebook reconstruction of the same vectors in the ROTATED
  //!           space, num x dim, packed.
  int fit(const float *x, const float *x_hat, size_t num);

  // -- Preprocessor interface ------------------------------------------------

  int in_dim() const override {
    return dim_;
  }
  int out_dim() const override {
    return dim_;
  }

  void apply(const float *in, float *out) const override;
  void apply_inverse(const float *in, float *out) const override;

  //! No-op: OPQ cannot be trained from data alone, see the overload below.
  void train(const void *data, size_t num, size_t stride) override;

  //! \p data is the packed fp32 batch x, \p ctx the reconstruction x_hat
  //! (\p stride must be 0).
  void train(const void *data, void *ctx, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;
  int deserialize(const void *data, size_t len) override;

  RotateType rotate_type() const {
    return RotateType::kOpq;
  }

 private:
  OpqRotator() = default;

  int dim_{0};

  //! Rotation matrix R, row-major dim x dim: apply() = R * in,
  //! apply_inverse() = R^T * in.
  std::vector<float> matrix_;

  RotatorKernels kernels_{};
};

}  // namespace turbo
}  // namespace zvec
