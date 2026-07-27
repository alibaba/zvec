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
#include "preprocessor/preprocessor.h"

namespace zvec {
namespace turbo {

// FHT context passed to ISA-level rotate/unrotate via void*.
// Layout (ISA code accesses by address, NOT by type):
//   offset  0: size_t flip_offset   (bytes per round)
//   offset  8: size_t trunc_dim     (largest power-of-2 <= in_dim)
//   offset 16: float  fac           (1 / sqrt(trunc_dim))
//   offset 20: uint8_t pad[4]       (explicit padding)
//   offset 24: uint8_t flip[]       (4 * flip_offset bytes, trailing data)
//
// Allocated as a single block: malloc(sizeof(FhtCtx) + 4 * flip_offset).
struct FhtCtx {
  size_t flip_offset;
  size_t trunc_dim;
  float fac;
  uint8_t pad_[4];
  uint8_t flip[];  // trailing flexible array (C++ extension)
};
static_assert(offsetof(FhtCtx, flip) == 24, "FhtCtx flip offset must be 24");
static_assert(sizeof(FhtCtx) == 24, "FhtCtx sizeof must be 24");

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

  //! Create a fully-initialized rotator for \p in_dim dimensions.
  //! Random flip-sign arrays are generated during creation; the returned
  //! object is immediately usable for apply() / apply_inverse().
  static Pointer create(int in_dim);

  //! Create and restore a rotator from a serialized blob (reads the type from
  //! the embedded RotatorSerHeader).  Returns nullptr on malformed input.
  static Pointer from_blob(const void *data, size_t len);

  // -- Preprocessor interface ------------------------------------------------

  int in_dim() const override {
    return in_dim_;
  }
  int out_dim() const override {
    return out_dim_;
  }

  void apply(const float *in, float *out) const override;
  void apply_inverse(const float *in, float *out) const override;

  //! No-op for FhtRotator.  Flip-sign arrays are generated in create().
  //! Provided for interface compatibility with the Preprocessor contract.
  void train(const void *data, size_t num, size_t stride) override;

  int serialize(std::string *out) const override;
  int deserialize(const void *data, size_t len) override;

  //! Rotator type tag (kFht = 1).
  RotateType rotate_type() const {
    return RotateType::kFht;
  }

  ~FhtRotator() override;

 private:
  FhtRotator() = default;

  //! Largest power of 2 <= dim.
  static size_t floor_pow2(size_t n);

  int in_dim_{0};
  int out_dim_{0};

  //! Bytes per round: ceil(in_dim / 8). Kept for serialization.
  size_t flip_offset_{0};

  //! ISA-dispatched rotate/unrotate kernels.
  RotatorKernels kernels_{};

  //! FHT state (flip_offset, trunc_dim, fac, flip[]) -- single allocation.
  FhtCtx *fht_ctx_{nullptr};

  static constexpr size_t kByteLen = 8;
};

}  // namespace turbo
}  // namespace zvec
