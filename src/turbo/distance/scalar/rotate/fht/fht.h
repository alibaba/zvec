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

namespace zvec::turbo::scalar {

//! Apply bitwise sign-flip mask to a float vector.
//! Each bit in \p flip controls one element: bit set means negate.
void fht_flip_sign(const uint8_t *flip, float *data, size_t dim);

//! Apply KacsWalk butterfly operation to non-power-of-2 FHT.
void fht_kacs_walk(float *data, size_t len);

//! Inverse KacsWalk butterfly operation.
void fht_inv_kacs_walk(float *data, size_t len);

//! In-place Fast Hadamard Transform on \p n elements (must be power-of-2).
void fht_inplace(float *data, size_t n);

//! Element-wise rescale: data[i] *= factor.
void fht_vec_rescale(float *data, size_t n, float factor);

//! Forward FHT rotation (compose flip -> FHT -> rescale, 4 rounds).
//! ctx is a FhtCtx* defined in preprocessor/fht_rotator/fht_rotator.h.
void fht_rotate(const float *in, float *out, size_t in_dim, size_t out_dim,
                void *ctx);

//! Inverse FHT rotation (undo 4 rounds in reverse order).
void fht_unrotate(const float *in, float *out, size_t in_dim, size_t out_dim,
                  void *ctx);

}  // namespace zvec::turbo::scalar
