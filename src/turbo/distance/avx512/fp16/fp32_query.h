// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <cstddef>

namespace zvec::turbo::avx512 {

// FP16 stored vector, original FP32 query, and FP32 accumulation. Cosine
// computes 1 - dot, so normalization belongs to the caller, as for FP32.
void cosine_fp16_fp32_distance_avx512(const void *vector, const float *query,
                                      size_t dim, float *distance);
void squared_euclidean_fp16_fp32_distance_avx512(const void *vector,
                                                 const float *query, size_t dim,
                                                 float *distance);

}  // namespace zvec::turbo::avx512
