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

namespace zvec {
namespace ailego {

//! Flip the sign of elements based on a packed bit-array.
void fht_flip_sign(const uint8_t *flip, float *data, size_t dim);

//! Kac random walk: butterfly add/sub between first and second halves.
void fht_kacs_walk(float *data, size_t len);

//! Inverse Kac walk: undo butterfly add/sub with 0.5 factor.
void fht_inv_kacs_walk(float *data, size_t len);

//! In-place Fast Hadamard Transform on a power-of-2 length array.
void fht_inplace(float *data, size_t n);

//! Scale each element by a constant factor.
void fht_vec_rescale(float *data, size_t n, float factor);

}  // namespace ailego
}  // namespace zvec
