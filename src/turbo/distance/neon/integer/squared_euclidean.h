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

namespace zvec::turbo::neon {

// Raw UINT8 and Uniform UINT7 use dim bytes, with no metadata tail.
void squared_euclidean_raw_uint8_distance(const void *, const void *, size_t,
                                          float *);
void squared_euclidean_raw_uint8_batch_distance(const void *const *,
                                                const void *, size_t, size_t,
                                                float *, const void *const *);
void uniform_squared_euclidean_uint7_distance(const void *, const void *,
                                              size_t, float *);
void uniform_squared_euclidean_uint7_batch_distance(const void *const *,
                                                    const void *, size_t,
                                                    size_t, float *,
                                                    const void *const *);

// Uniform UINT4's dimension is the packed byte count (two unsigned nibbles).
void uniform_squared_euclidean_uint4_distance(const void *, const void *,
                                              size_t, float *);
void uniform_squared_euclidean_uint4_batch_distance(const void *const *,
                                                    const void *, size_t,
                                                    size_t, float *,
                                                    const void *const *);

// Uniform UINT8: stored bodies are int8(raw - 128), followed by a uint32 norm.
// Single compares stored bodies; batch receives a preprocessed raw UINT8 query.
// Squared differences do not need the norm/extra_values or correction tail.
void uniform_squared_euclidean_uint8_distance(const void *, const void *,
                                              size_t, float *);
void uniform_squared_euclidean_uint8_batch_distance(const void *const *,
                                                    const void *, size_t,
                                                    size_t, float *,
                                                    const void *const *);
void uniform_squared_euclidean_uint8_query_preprocess(void *, size_t);

}  // namespace zvec::turbo::neon
