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

void fht_rotate(const float * /*in*/, float *out, size_t in_dim,
                size_t /*out_dim*/, void *ctx) {
  // FhtCtx layout: [0:flip_offset, 8:trunc_dim, 16:fac, 24:flip_data]
  auto *base = reinterpret_cast<const uint8_t *>(ctx);
  const size_t flip_offset = *reinterpret_cast<const size_t *>(base);
  const size_t trunc_dim = *reinterpret_cast<const size_t *>(base + 8);
  const float fac = *reinterpret_cast<const float *>(base + 16);
  const uint8_t *flip = base + 24;
  const size_t dim = in_dim;

  if (trunc_dim == dim) {
    fht_flip_sign(flip, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip + flip_offset, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip + 2 * flip_offset, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip + 3 * flip_offset, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);
    return;
  }

  // Non-power-of-2: 4 rounds with kacs_walk
  size_t start = dim - trunc_dim;
  float *trunc_ptr = out + start;

  fht_flip_sign(flip, out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  fht_flip_sign(flip + flip_offset, out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  fht_flip_sign(flip + 2 * flip_offset, out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  fht_flip_sign(flip + 3 * flip_offset, out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  fht_vec_rescale(out, dim, 0.25f);
}

void fht_unrotate(const float * /*in*/, float *out, size_t in_dim,
                  size_t /*out_dim*/, void *ctx) {
  auto *base = reinterpret_cast<const uint8_t *>(ctx);
  const size_t flip_offset = *reinterpret_cast<const size_t *>(base);
  const size_t trunc_dim = *reinterpret_cast<const size_t *>(base + 8);
  const float fac = *reinterpret_cast<const float *>(base + 16);
  const uint8_t *flip = base + 24;
  const size_t dim = in_dim;

  if (trunc_dim == dim) {
    for (int round = 3; round >= 0; --round) {
      fht_inplace(out, trunc_dim);
      fht_vec_rescale(out, trunc_dim, fac);
      fht_flip_sign(flip + static_cast<size_t>(round) * flip_offset, out, dim);
    }
    return;
  }

  // Non-power-of-2: undo final rescale(0.25) first
  fht_vec_rescale(out, dim, 4.0f);

  size_t start = dim - trunc_dim;
  float *trunc_ptr = out + start;

  fht_inv_kacs_walk(out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_flip_sign(flip + 3 * flip_offset, out, dim);

  fht_inv_kacs_walk(out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_flip_sign(flip + 2 * flip_offset, out, dim);

  fht_inv_kacs_walk(out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_flip_sign(flip + flip_offset, out, dim);

  fht_inv_kacs_walk(out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_flip_sign(flip, out, dim);
}

}  // namespace zvec::turbo::scalar
