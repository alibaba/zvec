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

namespace zvec::turbo {

// Runtime-dispatched AVX512 kernels for the SymphonyQG query path. The build
// compiles symphony_qg_kernels_avx512.cc with an AVX512-capable -march; each
// entry point returns false when AVX512 is unavailable at runtime or the
// arguments do not meet the kernel's constraints, leaving outputs untouched
// so callers can fall back to the reference scalar path.

// Fused equivalent of rabitqlib's Lut::reset plus BatchQuery's correction
// terms: fills the 8-bit fastscan LUT (padded_dim * 4 bytes) and writes
// delta, sum_vl and k1xsumq. Requires padded_dim % 16 == 0.
bool symqg_lut_build(const float *rotated_query, size_t padded_dim,
                     uint8_t *lut, float *delta, float *sum_vl, float *k1xsumq);

// In-place Fast Hadamard Transform on a power-of-two length array; requires
// padded_dim % 16 == 0. Bit-identical to the scalar butterfly loop.
bool symqg_hadamard(float *v, size_t padded_dim);

// Element-wise multiply v[i] *= factors[i].
bool symqg_mul(const float *factors, float *v, size_t n);

// Fused FastScan accumulation, unsigned widening and score conversion for 32
// neighbors. Widen every 1024 dimensions to prevent uint16 overflow; retain
// the scalar scoring expression's rounding boundaries.
bool symqg_scan(const uint8_t *codes, const uint8_t *lut, size_t padded_dim,
                const float *add, const float *scale, float center_distance,
                float delta, float sum_vl, float k1xsumq, float *distances);

}  // namespace zvec::turbo
