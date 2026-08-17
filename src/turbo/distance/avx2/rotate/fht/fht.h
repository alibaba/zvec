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

namespace zvec::turbo::avx2 {

//! Apply bitwise sign-flip mask to a float vector (AVX2).
void fht_flip_sign_avx2(const uint8_t *flip, float *data, size_t dim);

//! Apply KacsWalk butterfly operation (AVX2).
void fht_kacs_walk_avx2(float *data, size_t len);

//! Inverse KacsWalk butterfly operation (AVX2).
void fht_inv_kacs_walk_avx2(float *data, size_t len);

//! In-place Fast Hadamard Transform (AVX2, n must be power-of-2).
void fht_inplace_avx2(float *data, size_t n);

//! Element-wise rescale: data[i] *= factor (AVX2).
void fht_vec_rescale_avx2(float *data, size_t n, float factor);

//! Forward FHT rotation (AVX2).
void fht_rotate_avx2(const float *in, float *out, size_t in_dim, size_t out_dim,
                     void *ctx);

//! Inverse FHT rotation (AVX2).
void fht_unrotate_avx2(const float *in, float *out, size_t in_dim,
                       size_t out_dim, void *ctx);

}  // namespace zvec::turbo::avx2
