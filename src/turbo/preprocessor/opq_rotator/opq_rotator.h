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

// ============================================================================
// OpqRotator - O(d^2) dense orthogonal rotation trained from the data
//
// OPQ alternates two orthogonal sub-problems:
//   1. fix the codebook, train the rotation matrix   <- this class
//   2. fix the rotation matrix, train the codebook   <- the PQ quantizer
//
// This class only implements step 1: given the original-space training vectors
// and their reconstruction in the rotated space, fit() solves the orthogonal
// Procrustes problem for the rotation matrix.  It owns no PQ knowledge (no
// sub-quantizer count, no centroid count, no k-means) and never calls back
// into the quantizer: the alternating loop is driven by whoever owns both the
// codebook and this rotator.
// ============================================================================

class OpqRotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<OpqRotator>;

  //! Create a fully-usable rotator for \p dim dimensions.  The matrix is
  //! initialized with a random orthogonal matrix (reproducible via \p seed),
  //! so apply() is a valid rotation even before the first fit().
  static Pointer create(int dim, uint64_t seed = 42);

  //! Typed convenience wrapper around the two-input train() overload: OPQ
  //! step 1, fix the codebook, train the rotation matrix.  Validates
  //! arguments and forwards, returning an error code.
  //! \p x     original-space (already normalized / centered) training matrix,
  //!          num x in_dim floats, packed.
  //! \p x_hat reconstruction of the same vectors produced by the caller's
  //!          current codebook in the ROTATED space, num x out_dim floats.
  //! Solves argmin_R sum_i ||R * x_i - x_hat_i||^2 subject to R^T R = I and
  //! replaces the current matrix with the solution.
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

  //! No-op, as for FhtRotator: OPQ cannot be trained from the data alone, its
  //! training entry point is the two-input train() below.
  void train(const void *data, size_t num, size_t stride) override;

  //! OPQ step 1 via the Preprocessor contract: \p data is the original-space
  //! batch x and \p ctx the reconstruction matrix x_hat, both packed fp32
  //! (\p stride must be 0).  Solves the orthogonal Procrustes problem and
  //! replaces the current matrix with the solution.
  void train(const void *data, void *ctx, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;
  int deserialize(const void *data, size_t len) override;

  //! Rotator type tag (kOpq = 2).
  RotateType rotate_type() const {
    return RotateType::kOpq;
  }

 private:
  OpqRotator() = default;

  int dim_{0};

  //! Rotation matrix R, row-major dim x dim.  apply() computes R * in,
  //! apply_inverse() computes R^T * in (R is orthogonal).
  std::vector<float> matrix_;

  //! ISA-dispatched rotate/unrotate kernels (dense GEMV).
  RotatorKernels kernels_{};
};

}  // namespace turbo
}  // namespace zvec
