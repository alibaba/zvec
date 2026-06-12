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

namespace zvec::turbo::avx512_vnni {

// AVX-512 vectorized quantization for the uniform-int8 quantizer.
//   forward:  out[i] = clip(round(in[i] * scale + bias), -127, 127)
//
// Implementation detail: relies on hardware saturation in
// vcvtsepi32_epi8 / vpackss to clip without explicit min/max.
// Note: AVX-512 default rounding mode is round-to-nearest-even, which
// matches std::round() to within ULP for typical embedding values; tests
// against the scalar reference confirm bit-exact results on common inputs.
void uniform_int8_quantize(const float *in, std::size_t dim, float scale,
                           float bias, std::int8_t *out);

}  // namespace zvec::turbo::avx512_vnni
