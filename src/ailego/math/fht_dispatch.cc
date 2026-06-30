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

#include <ailego/internal/cpu_features.h>
#include "fht.h"

namespace zvec {
namespace ailego {

// ISA-specific forward declarations (implementations in
// fht_scalar/sse/avx2/avx512/neon)
void fht_flip_sign_scalar(const uint8_t *flip, float *data, size_t dim);
void fht_kacs_walk_scalar(float *data, size_t len);
void fht_inv_kacs_walk_scalar(float *data, size_t len);
void fht_inplace_scalar(float *data, size_t n);
#if defined(__SSE2__)
void fht_flip_sign_sse(const uint8_t *flip, float *data, size_t dim);
void fht_kacs_walk_sse(float *data, size_t len);
void fht_inv_kacs_walk_sse(float *data, size_t len);
#endif
#if defined(__AVX2__)
void fht_flip_sign_avx2(const uint8_t *flip, float *data, size_t dim);
void fht_kacs_walk_avx2(float *data, size_t len);
void fht_inv_kacs_walk_avx2(float *data, size_t len);
void fht_inplace_avx2(float *data, size_t n);
#endif
#if defined(__AVX512F__)
void fht_flip_sign_avx512(const uint8_t *flip, float *data, size_t dim);
void fht_kacs_walk_avx512(float *data, size_t len);
void fht_inv_kacs_walk_avx512(float *data, size_t len);
void fht_inplace_avx512(float *data, size_t n);
#endif
#if defined(__ARM_NEON) && defined(__aarch64__)
void fht_flip_sign_neon(const uint8_t *flip, float *data, size_t dim);
void fht_kacs_walk_neon(float *data, size_t len);
void fht_inv_kacs_walk_neon(float *data, size_t len);
#endif

// ============================================================================
// Runtime dispatch entry points
// ============================================================================

void fht_flip_sign(const uint8_t *flip, float *data, size_t dim) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  fht_flip_sign_neon(flip, data, dim);
#else
#if defined(__AVX512F__)
  if (internal::CpuFeatures::static_flags_.AVX512F &&
      internal::CpuFeatures::static_flags_.AVX512DQ) {
    fht_flip_sign_avx512(flip, data, dim);
    return;
  }
#endif
#if defined(__AVX2__)
  if (internal::CpuFeatures::static_flags_.AVX2) {
    fht_flip_sign_avx2(flip, data, dim);
    return;
  }
#endif
#if defined(__SSE2__)
  if (internal::CpuFeatures::static_flags_.SSE2) {
    fht_flip_sign_sse(flip, data, dim);
    return;
  }
#endif
  fht_flip_sign_scalar(flip, data, dim);
#endif  // __ARM_NEON
}

void fht_kacs_walk(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  fht_kacs_walk_neon(data, len);
#else
#if defined(__AVX512F__)
  if (internal::CpuFeatures::static_flags_.AVX512F) {
    fht_kacs_walk_avx512(data, len);
    return;
  }
#endif
#if defined(__AVX2__)
  if (internal::CpuFeatures::static_flags_.AVX2) {
    fht_kacs_walk_avx2(data, len);
    return;
  }
#endif
#if defined(__SSE2__)
  if (internal::CpuFeatures::static_flags_.SSE2) {
    fht_kacs_walk_sse(data, len);
    return;
  }
#endif
  fht_kacs_walk_scalar(data, len);
#endif  // __ARM_NEON
}

void fht_inv_kacs_walk(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  fht_inv_kacs_walk_neon(data, len);
#else
#if defined(__AVX512F__)
  if (internal::CpuFeatures::static_flags_.AVX512F) {
    fht_inv_kacs_walk_avx512(data, len);
    return;
  }
#endif
#if defined(__AVX2__)
  if (internal::CpuFeatures::static_flags_.AVX2) {
    fht_inv_kacs_walk_avx2(data, len);
    return;
  }
#endif
#if defined(__SSE2__)
  if (internal::CpuFeatures::static_flags_.SSE2) {
    fht_inv_kacs_walk_sse(data, len);
    return;
  }
#endif
  fht_inv_kacs_walk_scalar(data, len);
#endif  // __ARM_NEON
}

void fht_inplace(float *data, size_t n) {
#if defined(__AVX512F__)
  if (internal::CpuFeatures::static_flags_.AVX512F) {
    fht_inplace_avx512(data, n);
    return;
  }
#endif
#if defined(__AVX2__)
  if (internal::CpuFeatures::static_flags_.AVX2) {
    fht_inplace_avx2(data, n);
    return;
  }
#endif
  fht_inplace_scalar(data, n);
}

}  // namespace ailego
}  // namespace zvec
