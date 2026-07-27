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

#include <memory>
#include <string>
#include <zvec/turbo/turbo.h>

namespace zvec {
namespace turbo {

//! Magic number ('ROTR') stamped at the start of a serialized rotator blob.
constexpr uint32_t kRotatorMagic = 0x52544F52u;
//! Current rotator serialization format version.
constexpr uint16_t kRotatorSerVersion = 1;

//! Self-describing, fixed-size header that prefixes every serialized rotator.
//! The type-specific payload (flip signs, rotation matrix, ...) follows
//! immediately after this header.
struct RotatorSerHeader {
  uint32_t magic;         // kRotatorMagic
  uint16_t version;       // kRotatorSerVersion
  uint16_t rotator_type;  // RotateType
  uint32_t in_dim;        // input dimensionality
  uint32_t out_dim;       // output dimensionality
  uint32_t payload_size;  // bytes following the header
  uint32_t reserved;      // 0, for future use / alignment
};
static_assert(sizeof(RotatorSerHeader) == 24,
              "RotatorSerHeader must be 24 bytes");

//! Abstract preprocessor interface.
//!
//! A Preprocessor applies a deterministic, invertible transform to each
//! vector (e.g. random rotation).  Concrete subclasses (FhtRotator, ...)
//! implement the actual algorithm.
class Preprocessor {
 public:
  using Pointer = std::shared_ptr<Preprocessor>;

  virtual ~Preprocessor() = default;

  //! Input dimensionality accepted by apply().
  virtual int in_dim() const = 0;

  //! Output dimensionality produced by apply().  A future preprocessor may
  //! change dimensionality (out_dim() != in_dim()), but FhtRotator keeps it
  //! unchanged: it operates on floor_pow2(in_dim) via the Hadamard transform
  //! plus a Kac's walk over the remainder, so out_dim() == in_dim().
  virtual int out_dim() const = 0;

  //! Forward transform: map an input vector to the preprocessed space.
  //! \p out must hold at least out_dim() elements.
  virtual void apply(const float *in, float *out) const = 0;

  //! Inverse transform: recover the original-space vector from a preprocessed
  //! one.  \p out must hold at least in_dim() elements.
  virtual void apply_inverse(const float *in, float *out) const = 0;

  //! Fit / initialize the preprocessor from a contiguous batch of training
  //! data.  For FhtRotator this generates the random flip-sign arrays.
  //! \p data  pointer to the first element of the batch.
  //! \p num   number of vectors in the batch.
  //! \p stride byte offset between consecutive vectors (0 => packed).
  virtual void train(const void *data, size_t num, size_t stride) = 0;

  //! Serialize the preprocessor into a self-contained blob
  //! (RotatorSerHeader + payload).
  virtual int serialize(std::string *out) const = 0;

  //! Deserialize the preprocessor from a raw, possibly mmap-backed buffer.
  virtual int deserialize(const void *data, size_t len) = 0;
};

}  // namespace turbo
}  // namespace zvec
