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

namespace zvec::turbo::neon {

//! Apply bitwise sign-flip mask to a float vector (NEON).
void fht_flip_sign_neon(const uint8_t *flip, float *data, size_t dim);

//! Apply KacsWalk butterfly operation (NEON).
void fht_kacs_walk_neon(float *data, size_t len);

//! Inverse KacsWalk butterfly operation (NEON).
void fht_inv_kacs_walk_neon(float *data, size_t len);

//! Element-wise rescale: data[i] *= factor (NEON).
void fht_vec_rescale_neon(float *data, size_t n, float factor);

//! Forward FHT rotation (NEON). Inplace falls back to scalar.
void fht_rotate_neon(const float *in, float *out, size_t in_dim, size_t out_dim,
                     void *ctx);

//! Inverse FHT rotation (NEON). Inplace falls back to scalar.
void fht_unrotate_neon(const float *in, float *out, size_t in_dim,
                       size_t out_dim, void *ctx);

}  // namespace zvec::turbo::neon
