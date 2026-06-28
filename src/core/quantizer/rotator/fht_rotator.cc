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
#include <zvec/core/framework/index_error.h>

namespace zvec {
namespace core {

namespace {

//! Largest power of 2 <= n (e.g. floor_pow2(97) = 64, floor_pow2(128) = 128).
size_t floor_pow2(size_t n) {
  if (n == 0) return 0;
  size_t p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

}  // anonymous namespace

// ============================================================================
// FhtRotator method implementations
// ============================================================================

int FhtRotator::init_impl(size_t dim) {
  if (dim == 0) {
    return IndexError_InvalidArgument;
  }
  trunc_dim = floor_pow2(dim);
  fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  flip_offset_ = (dim + kByteLen - 1) / kByteLen;
  flip.resize(4 * flip_offset_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : flip) b = static_cast<uint8_t>(dist(gen));
  return 0;
}

void FhtRotator::rotate(const float *in, float *out) const {
  const size_t dim = dimension_;
  std::memcpy(out, in, sizeof(float) * dim);

  if (trunc_dim == dim) {
    // Exact power-of-2: 4 rounds of (flip -> FHT -> rescale)
    ailego::fht_flip_sign(flip.data(), out, dim);
    ailego::fht_inplace(out, trunc_dim);
    ailego::fht_vec_rescale(out, trunc_dim, fac);

    ailego::fht_flip_sign(flip.data() + flip_offset_, out, dim);
    ailego::fht_inplace(out, trunc_dim);
    ailego::fht_vec_rescale(out, trunc_dim, fac);

    ailego::fht_flip_sign(flip.data() + 2 * flip_offset_, out, dim);
    ailego::fht_inplace(out, trunc_dim);
    ailego::fht_vec_rescale(out, trunc_dim, fac);

    ailego::fht_flip_sign(flip.data() + 3 * flip_offset_, out, dim);
    ailego::fht_inplace(out, trunc_dim);
    ailego::fht_vec_rescale(out, trunc_dim, fac);

    return;
  }

  // Non-power-of-2 (e.g. 97, 100, 192, 320): 4 rounds with kacs_walk
  size_t start = dim - trunc_dim;
  float *trunc_ptr = out + start;

  // Round 1: FHT on [0, trunc_dim)
  ailego::fht_flip_sign(flip.data(), out, dim);
  ailego::fht_inplace(out, trunc_dim);
  ailego::fht_vec_rescale(out, trunc_dim, fac);
  ailego::fht_kacs_walk(out, dim);

  // Round 2: FHT on [start, start + trunc_dim)
  ailego::fht_flip_sign(flip.data() + flip_offset_, out, dim);
  ailego::fht_inplace(trunc_ptr, trunc_dim);
  ailego::fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  ailego::fht_kacs_walk(out, dim);

  // Round 3: FHT on [0, trunc_dim)
  ailego::fht_flip_sign(flip.data() + 2 * flip_offset_, out, dim);
  ailego::fht_inplace(out, trunc_dim);
  ailego::fht_vec_rescale(out, trunc_dim, fac);
  ailego::fht_kacs_walk(out, dim);

  // Round 4: FHT on [start, start + trunc_dim)
  ailego::fht_flip_sign(flip.data() + 3 * flip_offset_, out, dim);
  ailego::fht_inplace(trunc_ptr, trunc_dim);
  ailego::fht_vec_rescale(trunc_ptr, trunc_dim, fac);
  ailego::fht_kacs_walk(out, dim);

  // Final rescale: combine the 4 kacs_walk reductions
  ailego::fht_vec_rescale(out, dim, 0.25f);
}

void FhtRotator::unrotate(const float *in, float *out) const {
  const size_t dim = dimension_;
  // Copy input into working buffer
  std::vector<float> data(in, in + dim);

  if (trunc_dim == dim) {
    // Exact power-of-2: reverse 4 rounds in reverse order.
    const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
    for (int round = 3; round >= 0; --round) {
      ailego::fht_inplace(data.data(), trunc_dim);
      ailego::fht_vec_rescale(data.data(), trunc_dim, inv_fac);
      ailego::fht_flip_sign(flip.data() + round * flip_offset_, data.data(),
                            dim);
    }
    std::memcpy(out, data.data(), dim * sizeof(float));
    return;
  }

  // Non-power-of-2: undo final rescale(0.25) first
  ailego::fht_vec_rescale(data.data(), dim, 4.0f);

  const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  size_t start = dim - trunc_dim;
  float *trunc_ptr = data.data() + start;

  // Undo Round 4 (FHT on [start, start+trunc_dim))
  ailego::fht_inv_kacs_walk(data.data(), dim);
  ailego::fht_inplace(trunc_ptr, trunc_dim);
  ailego::fht_vec_rescale(trunc_ptr, trunc_dim, inv_fac);
  ailego::fht_flip_sign(flip.data() + 3 * flip_offset_, data.data(), dim);

  // Undo Round 3 (FHT on [0, trunc_dim))
  ailego::fht_inv_kacs_walk(data.data(), dim);
  ailego::fht_inplace(data.data(), trunc_dim);
  ailego::fht_vec_rescale(data.data(), trunc_dim, inv_fac);
  ailego::fht_flip_sign(flip.data() + 2 * flip_offset_, data.data(), dim);

  // Undo Round 2 (FHT on [start, start+trunc_dim))
  ailego::fht_inv_kacs_walk(data.data(), dim);
  ailego::fht_inplace(trunc_ptr, trunc_dim);
  ailego::fht_vec_rescale(trunc_ptr, trunc_dim, inv_fac);
  ailego::fht_flip_sign(flip.data() + flip_offset_, data.data(), dim);

  // Undo Round 1 (FHT on [0, trunc_dim))
  ailego::fht_inv_kacs_walk(data.data(), dim);
  ailego::fht_inplace(data.data(), trunc_dim);
  ailego::fht_vec_rescale(data.data(), trunc_dim, inv_fac);
  ailego::fht_flip_sign(flip.data(), data.data(), dim);

  std::memcpy(out, data.data(), dim * sizeof(float));
}

RotatorType FhtRotator::rotator_type() const {
  return RotatorType::FhtKac;
}

void FhtRotator::save_blob(char *data) const {
  std::memcpy(data, flip.data(), flip.size());
}

void FhtRotator::load_blob(const char *data) {
  // Recompute derived fields from dimension_ (init_impl is not called
  // during open, so trunc_dim/fac/flip_offset_ must be restored here)
  trunc_dim = floor_pow2(dimension_);
  fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  flip_offset_ = (dimension_ + kByteLen - 1) / kByteLen;
  // Load flip data
  flip.resize(4 * flip_offset_);
  std::memcpy(flip.data(), data, flip.size());
}

size_t FhtRotator::blob_bytes() const {
  return flip.size();
}

}  // namespace core
}  // namespace zvec
