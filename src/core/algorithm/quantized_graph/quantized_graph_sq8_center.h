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
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  static void decode(const uint8_t *h, size_t n, float *out, float offset,
                     float scale) {
    for (size_t i = 0; i < n; ++i)
      out[i] = std::fma(float(h[i]), scale, offset);
  }
  __attribute__((target("avx512f,avx512bw,fma"))) static float distance(
      const float *q, const uint8_t *h, size_t n, float offset, float scale) {
    __m512 acc = _mm512_setzero_ps(), lo = _mm512_set1_ps(offset),
           step = _mm512_set1_ps(scale);
    size_t i = 0;
    __m512 accum[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                       _mm512_setzero_ps(), _mm512_setzero_ps()};
    for (; i + 64 <= n; i += 64) {
      for (size_t lane = 0; lane < 4; ++lane) {
        const size_t j = i + lane * 16;
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + j));
        __m512 v = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw)), step, lo);
        __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + j), v);
        accum[lane] = _mm512_fmadd_ps(diff, diff, accum[lane]);
      }
    }
    acc = _mm512_add_ps(_mm512_add_ps(accum[0], accum[1]),
                        _mm512_add_ps(accum[2], accum[3]));
    for (; i + 16 <= n; i += 16) {
      __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(h + i));
      __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(raw)),
                                 step, lo);
      __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + i), v);
      acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    float result = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) {
      float d = q[i] - std::fma(float(h[i]), scale, offset);
      result += d * d;
    }
    return result;
  }
  // Conditional lower bound for full FP32 L2 scoring under finite, normal-range
  // arithmetic, using the actual SQ8 reconstruction error.
  static float l2_lower_bound(float approximate, float error, size_t n) {
    // Direct sum-of-squared-differences kernels: gamma_(n+4) covers
    // rounded differences, products and a worst-case serial reduction.
    // Positive IEEE-754 doubles: one-bit steps provide outward rounding
    // without a libm call at each arithmetic operation.
    auto down = [](double x) {
      if (!(x > 0)) return 0.;
      uint64_t b;
      std::memcpy(&b, &x, 8);
      --b;
      std::memcpy(&x, &b, 8);
      return x;
    };
    auto up = [](double x) {
      uint64_t b;
      std::memcpy(&b, &x, 8);
      ++b;
      std::memcpy(&x, &b, 8);
      return x;
    };
    const double a = (n + 4) * (.5 * std::numeric_limits<float>::epsilon());
    const double gamma = up(a / (1. - a));
    const double tiny =
        4. * (n + 4) * double(std::numeric_limits<float>::min());
    const double s_lower =
        down(down(double(approximate) - tiny) / up(1. + gamma));
    const double radius = down(down(std::sqrt(s_lower)) - error);
    const double lower =
        down(down(down(1. - gamma) * down(radius * radius)) - tiny);
    return std::max(
        0.f,
        std::nextafter(float(lower), -std::numeric_limits<float>::infinity()));
  }
};
}  // namespace zvec::core
