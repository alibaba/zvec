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

#if defined(__AVX512FP16__)
#include <immintrin.h>

namespace zvec::turbo::avx512_fp16::internal {

// Convert the two FP16 halves to FP32 before reducing. This preserves more
// precision than _mm512_reduce_add_ph, whose reduction is also performed in
// FP16.
inline float horizontal_add_fp16(__m512h value) {
  const __m512 low = _mm512_cvtxph_ps(_mm512_castph512_ph256(value));
  const __m512 high = _mm512_cvtxph_ps(
      _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(value), 1)));
  return _mm512_reduce_add_ps(_mm512_add_ps(low, high));
}

}  // namespace zvec::turbo::avx512_fp16::internal
#endif
