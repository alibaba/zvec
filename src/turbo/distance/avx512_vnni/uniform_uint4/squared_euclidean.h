// Copyright 2025-present the zvec project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace zvec::turbo::avx512_vnni {

// `dimension` is the encoded byte count. Each byte stores two unsigned
// four-bit codes, with the low nibble first.
void uniform_squared_euclidean_uint4_distance(const void *lhs, const void *rhs,
                                              std::size_t dimension,
                                              float *distance);
void uniform_squared_euclidean_uint4_batch_distance(const void *const *vectors,
                                                    const void *query,
                                                    std::size_t count,
                                                    std::size_t dimension,
                                                    float *distances);

}  // namespace zvec::turbo::avx512_vnni
