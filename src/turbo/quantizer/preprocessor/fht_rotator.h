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
#include <vector>
#include "preprocessor.h"
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace turbo {

// ============================================================================
// FhtRotator - O(d log d) FHT-based Kac random rotation
//
// Works with any dimension (non-power-of-2 uses trunc_dim + KacsWalk).
// When dimension is a power of 2, uses 4 rounds of (flip -> FHT -> rescale).
// When dimension is NOT a power of 2, uses kacs_walk reduction.
// ============================================================================

class FhtRotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<FhtRotator>;

  //! Create an untrained rotator for \p in_dim dimensions.
  //! Call train() afterwards to generate the random flip-sign arrays.
  static Pointer create(int in_dim);

  //! Create and restore a rotator from a serialized blob (reads the type from
  //! the embedded RotatorSerHeader).  Returns nullptr on malformed input.
  static Pointer from_blob(const void *data, size_t len);

  // -- Preprocessor interface ------------------------------------------------

  int in_dim() const override { return in_dim_; }
  int out_dim() const override { return out_dim_; }

  void apply(const float *in, float *out) const override;
  void apply_inverse(const float *in, float *out) const override;

  //! Generate 4 rounds of random flip-sign arrays.
  //! For FhtRotator the training data is ignored; only \p in_dim() matters.
  void train(const void *data, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;
  int deserialize(const void *data, size_t len) override;

  //! Rotator type tag (kFht = 1).
  RotatorType rotator_type() const { return RotatorType::kFht; }

 private:
  FhtRotator() = default;

  //! Largest power of 2 <= dim.
  static size_t floor_pow2(size_t n);

  int in_dim_{0};
  int out_dim_{0};

  //! Packed flip-sign bits: 4 rounds, each ceil(in_dim / 8) bytes.
  std::vector<uint8_t> flip_;

  //! Bytes per round: ceil(in_dim / 8).
  size_t flip_offset_{0};

  //! Largest power of 2 <= in_dim (used for FHT length).
  size_t trunc_dim_{0};

  //! Rescale factor: 1 / sqrt(trunc_dim).
  float fac_{0};

  //! ISA-dispatched FHT kernels (cached at construction time).
  FhtFlipSignFunc flip_sign_fn_{nullptr};
  FhtKacsWalkFunc kacs_walk_fn_{nullptr};
  FhtKacsWalkFunc inv_kacs_walk_fn_{nullptr};
  FhtInplaceFunc inplace_fn_{nullptr};
  FhtVecRescaleFunc rescale_fn_{nullptr};

  static constexpr size_t kByteLen = 8;
};

}  // namespace turbo
}  // namespace zvec
