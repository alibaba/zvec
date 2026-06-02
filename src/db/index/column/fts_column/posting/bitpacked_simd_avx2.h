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

/// Compute element-wise max of 128 uint32 values across three arrays using
/// AVX2 _mm256_max_epu32.  \p deltas must be 32-byte aligned; \p tfs and
/// \p doc_lens may be unaligned.
void avx2_max_128(const uint32_t *deltas, const uint32_t *tfs,
                  const uint32_t *doc_lens, size_t start, uint32_t count,
                  uint32_t &max_delta, uint32_t &max_tf, uint32_t &max_dl);

/// Pack 128 uint32 values at \p bitwidth bits each into \p out.
/// Falls back to SSE4.1 implementation (FastPForLib lacks AVX2 bitpacking).
void avx2_pack_uint32_128(const uint32_t *in, uint8_t bitwidth, uint8_t *out);

/// Unpack 128 uint32 values at \p bitwidth bits each from \p in.
/// Falls back to SSE4.1 implementation (FastPForLib lacks AVX2 bitpacking).
void avx2_unpack_uint32_128(const uint8_t *in, uint8_t bitwidth, uint32_t *out);

/// Compute prefix-sum over \p count (must be 128) delta values, producing
/// absolute doc_ids.  Uses AVX2 SIMD prefix-sum with carry propagation.
/// \p deltas must be 32-byte aligned; \p out must be 32-byte aligned.
void avx2_prefix_sum_128(const uint32_t *deltas, uint32_t min_doc_id,
                         uint32_t count, uint32_t *out);

/// Find the first index i in arr[start..size) where arr[i] >= target.
/// Uses AVX2 8-wide comparison with unsigned-to-signed trick.
/// \p arr must be 32-byte aligned.
size_t avx2_find_first_ge(const uint32_t *arr, uint32_t size, uint32_t target,
                          size_t start);

}  // namespace zvec::fts::simd
