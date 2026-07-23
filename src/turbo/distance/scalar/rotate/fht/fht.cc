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

#include "fht.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "common/fht_common.h"

namespace zvec::turbo::scalar {

void fht_flip_sign(const uint8_t *flip, float *data, size_t dim) {
  for (size_t i = 0; i < dim; ++i) {
    if (flip[i / 8] & (1u << (i % 8))) {
      data[i] = -data[i];
    }
  }
}

void fht_kacs_walk(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  for (size_t i = 0; i < half; ++i) {
    float x = data[i];
    float y = data[i + offset];
    data[i] = x + y;
    data[i + offset] = x - y;
  }
  if (base != 0) {
    data[half] *= std::sqrt(2.0f);
  }
}

void fht_inv_kacs_walk(float *data, size_t len) {
  size_t half = len / 2;
  size_t base = len % 2;
  size_t offset = base + half;
  if (base != 0) {
    data[half] *= std::sqrt(0.5f);
  }
  for (size_t i = 0; i < half; ++i) {
    float a = data[i];
    float b = data[i + offset];
    data[i] = (a + b) * 0.5f;
    data[i + offset] = (a - b) * 0.5f;
  }
}

void fht_inplace(float *data, size_t n) {
  for (size_t len = 1; len < n; len <<= 1) {
    for (size_t i = 0; i < n; i += len << 1) {
      for (size_t j = i; j < i + len; ++j) {
        float u = data[j];
        float v = data[j + len];
        data[j] = u + v;
        data[j + len] = u - v;
      }
    }
  }
}

void fht_vec_rescale(float *data, size_t n, float factor) {
  for (size_t i = 0; i < n; ++i) {
    data[i] *= factor;
  }
}

void fht_rotate(const float *in, float *out, size_t in_dim, size_t /*out_dim*/,
                void *ctx) {
  static constexpr FhtPrimitives kPrim = {fht_flip_sign, fht_inplace,
                                          fht_kacs_walk, fht_inv_kacs_walk,
                                          fht_vec_rescale};
  fht_rotate_impl(in, out, in_dim, ctx, kPrim);
}

void fht_unrotate(const float *in, float *out, size_t in_dim,
                  size_t /*out_dim*/, void *ctx) {
  static constexpr FhtPrimitives kPrim = {fht_flip_sign, fht_inplace,
                                          fht_kacs_walk, fht_inv_kacs_walk,
                                          fht_vec_rescale};
  fht_unrotate_impl(in, out, in_dim, ctx, kPrim);
}

}  // namespace zvec::turbo::scalar
