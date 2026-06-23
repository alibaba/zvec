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

#include "fht_rotator.h"
#include <cmath>
#include <cstring>
#include <random>
#include <ailego/internal/cpu_features.h>

namespace zvec {
namespace ailego {
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
} }  // namespace zvec::ailego

namespace zvec {
namespace core {

// ============================================================================
// Runtime dispatch entry points
// ============================================================================

void fht_flip_sign(const uint8_t *flip, float *data, size_t dim) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  ailego::fht_flip_sign_neon(flip, data, dim);
#else
#if defined(__AVX512F__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX512F &&
      ailego::internal::CpuFeatures::static_flags_.AVX512DQ) {
    ailego::fht_flip_sign_avx512(flip, data, dim);
    return;
  }
#endif
#if defined(__AVX2__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX2) {
    ailego::fht_flip_sign_avx2(flip, data, dim);
    return;
  }
#endif
#if defined(__SSE2__)
  if (ailego::internal::CpuFeatures::static_flags_.SSE2) {
    ailego::fht_flip_sign_sse(flip, data, dim);
    return;
  }
#endif
  ailego::fht_flip_sign_scalar(flip, data, dim);
#endif  // __ARM_NEON
}

void fht_kacs_walk(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  ailego::fht_kacs_walk_neon(data, len);
#else
#if defined(__AVX512F__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ailego::fht_kacs_walk_avx512(data, len);
    return;
  }
#endif
#if defined(__AVX2__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX2) {
    ailego::fht_kacs_walk_avx2(data, len);
    return;
  }
#endif
#if defined(__SSE2__)
  if (ailego::internal::CpuFeatures::static_flags_.SSE2) {
    ailego::fht_kacs_walk_sse(data, len);
    return;
  }
#endif
  ailego::fht_kacs_walk_scalar(data, len);
#endif  // __ARM_NEON
}

void fht_inv_kacs_walk(float *data, size_t len) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  ailego::fht_inv_kacs_walk_neon(data, len);
#else
#if defined(__AVX512F__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ailego::fht_inv_kacs_walk_avx512(data, len);
    return;
  }
#endif
#if defined(__AVX2__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX2) {
    ailego::fht_inv_kacs_walk_avx2(data, len);
    return;
  }
#endif
#if defined(__SSE2__)
  if (ailego::internal::CpuFeatures::static_flags_.SSE2) {
    ailego::fht_inv_kacs_walk_sse(data, len);
    return;
  }
#endif
  ailego::fht_inv_kacs_walk_scalar(data, len);
#endif  // __ARM_NEON
}

void fht_inplace(float *data, size_t n) {
#if defined(__AVX512F__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX512F) {
    ailego::fht_inplace_avx512(data, n);
    return;
  }
#endif
#if defined(__AVX2__)
  if (ailego::internal::CpuFeatures::static_flags_.AVX2) {
    ailego::fht_inplace_avx2(data, n);
    return;
  }
#endif
  ailego::fht_inplace_scalar(data, n);
}

void fht_vec_rescale(float *data, size_t n, float factor) {
  for (size_t i = 0; i < n; ++i) {
    data[i] *= factor;
  }
}

// ============================================================================
// FhtKacRotatorImpl method implementations
// ============================================================================

void FhtKacRotatorImpl::init(size_t dim) {
  flip.resize(4 * dim / kByteLen);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : flip) b = static_cast<uint8_t>(dist(gen));
}

void FhtKacRotatorImpl::rotate(const float *in, float *out, size_t dim) const {
  std::memcpy(out, in, sizeof(float) * dim);

  if (trunc_dim == dim) {
    // Exact power-of-2: 4 rounds of (flip -> FHT -> rescale)
    fht_flip_sign(flip.data(), out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip.data() + dim / kByteLen, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip.data() + 2 * dim / kByteLen, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    fht_flip_sign(flip.data() + 3 * dim / kByteLen, out, dim);
    fht_inplace(out, trunc_dim);
    fht_vec_rescale(out, trunc_dim, fac);

    return;
  }

  // Non-power-of-2 (64-aligned, e.g. 192, 320): 4 rounds with kacs_walk
  size_t start = dim - trunc_dim;
  float *trunc_ptr = out + start;

  // Round 1: FHT on [0, trunc_dim)
  fht_flip_sign(flip.data(), out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  // Round 2: FHT on [start, start + trunc_dim)
  fht_flip_sign(flip.data() + dim / kByteLen, out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  // Round 3: FHT on [0, trunc_dim)
  fht_flip_sign(flip.data() + 2 * dim / kByteLen, out, dim);
  fht_inplace(out, trunc_dim);
  fht_vec_rescale(out, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  // Round 4: FHT on [start, start + trunc_dim)
  fht_flip_sign(flip.data() + 3 * dim / kByteLen, out, dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  fht_kacs_walk(out, dim);

  // Final rescale: combine the 4 kacs_walk reductions
  fht_vec_rescale(out, dim, 0.25f);
}

void FhtKacRotatorImpl::unrotate(const float *in, float *out,
                                 size_t dim) const {
  // Copy input into working buffer
  std::vector<float> data(in, in + dim);

  if (trunc_dim == dim) {
    // Exact power-of-2: reverse 4 rounds in reverse order.
    const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
    for (int round = 3; round >= 0; --round) {
      fht_inplace(data.data(), trunc_dim);
      fht_vec_rescale(data.data(), trunc_dim, inv_fac);
      fht_flip_sign(flip.data() + round * dim / kByteLen, data.data(), dim);
    }
    std::memcpy(out, data.data(), dim * sizeof(float));
    return;
  }

  // Non-power-of-2: undo final rescale(0.25) first
  fht_vec_rescale(data.data(), dim, 4.0f);

  const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  size_t start = dim - trunc_dim;
  float *trunc_ptr = data.data() + start;

  // Undo Round 4 (FHT on [start, start+trunc_dim))
  fht_inv_kacs_walk(data.data(), dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, inv_fac);
  fht_flip_sign(flip.data() + 3 * dim / kByteLen, data.data(), dim);

  // Undo Round 3 (FHT on [0, trunc_dim))
  fht_inv_kacs_walk(data.data(), dim);
  fht_inplace(data.data(), trunc_dim);
  fht_vec_rescale(data.data(), trunc_dim, inv_fac);
  fht_flip_sign(flip.data() + 2 * dim / kByteLen, data.data(), dim);

  // Undo Round 2 (FHT on [start, start+trunc_dim))
  fht_inv_kacs_walk(data.data(), dim);
  fht_inplace(trunc_ptr, trunc_dim);
  fht_vec_rescale(trunc_ptr, trunc_dim, inv_fac);
  fht_flip_sign(flip.data() + dim / kByteLen, data.data(), dim);

  // Undo Round 1 (FHT on [0, trunc_dim))
  fht_inv_kacs_walk(data.data(), dim);
  fht_inplace(data.data(), trunc_dim);
  fht_vec_rescale(data.data(), trunc_dim, inv_fac);
  fht_flip_sign(flip.data(), data.data(), dim);

  std::memcpy(out, data.data(), dim * sizeof(float));
}

void FhtKacRotatorImpl::save(char *data) const {
  std::memcpy(data, flip.data(), flip.size());
}

void FhtKacRotatorImpl::load(const char *data) {
  std::memcpy(flip.data(), data, flip.size());
}

size_t FhtKacRotatorImpl::dump_bytes() const {
  return flip.size();
}

}  // namespace core
}  // namespace zvec
