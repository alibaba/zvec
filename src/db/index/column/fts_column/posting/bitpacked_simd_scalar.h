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

namespace zvec::fts::simd {

/// Scalar fallback: compute element-wise max of up to 128 uint32 values across
/// three arrays using a simple loop.
void scalar_max_128(const uint32_t *deltas, const uint32_t *tfs,
                    const uint32_t *doc_lens, size_t start, uint32_t count,
                    uint32_t &max_delta, uint32_t &max_tf, uint32_t &max_dl);

/// Scalar fallback: pack 128 uint32 values at \p bitwidth bits each into \p
/// out, producing the SAME interleaved byte layout as the SSE/AVX2 SIMD packer
/// so that indexes remain portable across architectures.
void scalar_pack_uint32_128(const uint32_t *in, uint8_t bitwidth, uint8_t *out);

/// Scalar fallback: unpack 128 uint32 values at \p bitwidth bits each from
/// \p in, reading the SAME interleaved byte layout as the SSE/AVX2 SIMD packer.
void scalar_unpack_uint32_128(const uint8_t *in, uint8_t bitwidth,
                              uint32_t *out);

/// Scalar fallback: compute prefix-sum over \p count delta values, producing
/// absolute doc_ids.
void scalar_prefix_sum_128(const uint32_t *deltas, uint32_t min_doc_id,
                           uint32_t count, uint32_t *out);

/// Scalar fallback: find the first index i in arr[start..size) where
/// arr[i] >= target using a linear scan.
size_t scalar_find_first_ge(const uint32_t *arr, uint32_t size, uint32_t target,
                            size_t start);

}  // namespace zvec::fts::simd
