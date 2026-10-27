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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
namespace zvec::core {
struct SQ8Center {
  // Shared shift is an immutable database-trained vector. Error is measured
  // against the actual FP32 reconstruction, including the rounded addition.
  static bool encode_shifted(const float *x, const float *shift, size_t n,
                             uint8_t *out, float &error, float &norm2,
                             float &offset, float &scale, float &projection,
                             float &projection_error) {
    std::vector<float> residual(n);
    for (size_t j = 0; j < n; ++j) residual[j] = x[j] - shift[j];
    if (!encode(residual.data(), n, out, error, norm2, offset, scale))
      return false;
    double err = 0, norm = 0, beta = 0, beta_abs = 0;
    for (size_t j = 0; j < n; ++j) {
      const float y = std::fma(float(out[j]), scale, offset) + shift[j];
      if (!std::isfinite(y)) return false;
      const double e = double(y) - x[j];
      err += e * e;
      norm += double(x[j]) * x[j];
      beta += double(shift[j]) * e;
      beta_abs += std::abs(double(shift[j]) * e);
    }
    error = std::nextafter(float(std::sqrt(err)),
                           std::numeric_limits<float>::infinity());
    norm2 = std::nextafter(float(norm), std::numeric_limits<float>::infinity());
    projection = float(beta);
    projection_error = std::nextafter(
        float(std::abs(double(projection) - beta) +
              4. * (n + 4) * std::numeric_limits<double>::epsilon() * beta_abs),
        std::numeric_limits<float>::infinity());
    return true;
  }
  static bool encode(const float *x, size_t n, uint8_t *out, float &error,
                     float &norm2, float &offset, float &scale) {
    offset = *std::min_element(x, x + n);
    float maximum = *std::max_element(x, x + n);
    scale = (maximum - offset) / 255.f;
    double err = 0, norm = 0;
    if (!std::isfinite(offset) || !std::isfinite(scale)) return false;
    for (size_t i = 0; i < n; ++i) {
      const float code =
          scale > 0 ? std::floor((x[i] - offset) / scale + .5f) : 0;
      out[i] = uint8_t(std::max(0.f, std::min(255.f, code)));
      const float y = std::fma(float(out[i]), scale, offset);
      double e = double(y) - x[i];
      err += e * e;
      norm += double(x[i]) * x[i];
    }
    error = std::nextafter(float(std::sqrt(err)),
                           std::numeric_limits<float>::infinity());
    norm2 = std::nextafter(float(norm), std::numeric_limits<float>::infinity());
    return true;
  }
};
}  // namespace zvec::core
