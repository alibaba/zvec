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
#include <cstring>

namespace zvec::turbo {

/// ISA-level FHT primitive function pointers.
/// Each ISA fills in its own SIMD-optimized (or scalar-fallback) functions.
struct FhtPrimitives {
  void (*flip_sign)(const uint8_t *flip, float *data, size_t dim);
  void (*inplace)(float *data, size_t n);
  void (*kacs_walk)(float *data, size_t len);
  void (*inv_kacs_walk)(float *data, size_t len);
  void (*rescale)(float *data, size_t n, float factor);
};

/// FhtCtx memory layout (accessed by address, NOT by type):
///   offset  0: size_t flip_offset
///   offset  8: size_t trunc_dim
///   offset 16: float  fac
///   offset 20: uint8_t pad[4]
///   offset 24: uint8_t flip[]
inline void fht_rotate_impl(const float *in, float *out, size_t dim, void *ctx,
                            const FhtPrimitives &p) {
  if (out != in) {
    std::memcpy(out, in, sizeof(float) * dim);
  }
  float *data = out;
  auto *base = reinterpret_cast<const uint8_t *>(ctx);
  const size_t flip_offset = *reinterpret_cast<const size_t *>(base);
  const size_t trunc_dim = *reinterpret_cast<const size_t *>(base + 8);
  const float fac = *reinterpret_cast<const float *>(base + 16);
  const uint8_t *flip = base + 24;

  if (trunc_dim == dim) {
    for (size_t r = 0; r < 4; ++r) {
      p.flip_sign(flip + r * flip_offset, data, dim);
      p.inplace(data, trunc_dim);
      p.rescale(data, trunc_dim, fac);
    }
    return;
  }

  size_t start = dim - trunc_dim;
  float *trunc_ptr = data + start;

  p.flip_sign(flip, data, dim);
  p.inplace(data, trunc_dim);
  p.rescale(data, trunc_dim, fac);
  p.kacs_walk(data, dim);

  p.flip_sign(flip + flip_offset, data, dim);
  p.inplace(trunc_ptr, trunc_dim);
  p.rescale(trunc_ptr, trunc_dim, fac);
  p.kacs_walk(data, dim);

  p.flip_sign(flip + 2 * flip_offset, data, dim);
  p.inplace(data, trunc_dim);
  p.rescale(data, trunc_dim, fac);
  p.kacs_walk(data, dim);

  p.flip_sign(flip + 3 * flip_offset, data, dim);
  p.inplace(trunc_ptr, trunc_dim);
  p.rescale(trunc_ptr, trunc_dim, fac);
  p.kacs_walk(data, dim);

  p.rescale(data, dim, 0.25f);
}

inline void fht_unrotate_impl(const float *in, float *out, size_t dim,
                              void *ctx, const FhtPrimitives &p) {
  if (out != in) {
    std::memcpy(out, in, sizeof(float) * dim);
  }
  float *data = out;
  auto *base = reinterpret_cast<const uint8_t *>(ctx);
  const size_t flip_offset = *reinterpret_cast<const size_t *>(base);
  const size_t trunc_dim = *reinterpret_cast<const size_t *>(base + 8);
  const float fac = *reinterpret_cast<const float *>(base + 16);
  const uint8_t *flip = base + 24;

  if (trunc_dim == dim) {
    for (int round = 3; round >= 0; --round) {
      p.inplace(data, trunc_dim);
      p.rescale(data, trunc_dim, fac);
      p.flip_sign(flip + static_cast<size_t>(round) * flip_offset, data, dim);
    }
    return;
  }

  p.rescale(data, dim, 4.0f);

  size_t start = dim - trunc_dim;
  float *trunc_ptr = data + start;

  p.inv_kacs_walk(data, dim);
  p.inplace(trunc_ptr, trunc_dim);
  p.rescale(trunc_ptr, trunc_dim, fac);
  p.flip_sign(flip + 3 * flip_offset, data, dim);

  p.inv_kacs_walk(data, dim);
  p.inplace(data, trunc_dim);
  p.rescale(data, trunc_dim, fac);
  p.flip_sign(flip + 2 * flip_offset, data, dim);

  p.inv_kacs_walk(data, dim);
  p.inplace(trunc_ptr, trunc_dim);
  p.rescale(trunc_ptr, trunc_dim, fac);
  p.flip_sign(flip + flip_offset, data, dim);

  p.inv_kacs_walk(data, dim);
  p.inplace(data, trunc_dim);
  p.rescale(data, trunc_dim, fac);
  p.flip_sign(flip, data, dim);
}

}  // namespace zvec::turbo
