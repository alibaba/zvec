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

#if defined(__AVX2__)

inline float sum4(__m128 v) {
  v = _mm_add_ps(v, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), 8)));
  return _mm_cvtss_f32(v) + _mm_cvtss_f32(_mm_shuffle_ps(v, v, 1));
}

inline __m128 sum_top_bottom_avx(__m256 v) {
  const __m128 high = _mm256_extractf128_ps(v, 1);
  const __m128 low = _mm256_castps256_ps128(v);
  return _mm_add_ps(high, low);
}

#endif
