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
#include "../quantizer.h"
namespace zvec {
namespace turbo {

// ============================================================================
// FhtRotator method implementations
// ============================================================================

size_t FhtRotator::floor_pow2(size_t n) {
  if (n == 0) return 0;
  size_t p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

FhtRotator::Pointer FhtRotator::create(int dim) {
  if (dim <= 0) return nullptr;

  Pointer r(new FhtRotator());
  r->in_dim_ = dim;
  r->out_dim_ = dim;
  r->trunc_dim_ = floor_pow2(static_cast<size_t>(dim));
  r->fac_ = 1.0f / std::sqrt(static_cast<float>(r->trunc_dim_));
  r->flip_offset_ = (static_cast<size_t>(dim) + kByteLen - 1) / kByteLen;
  auto k = get_fht_kernels();
  r->flip_sign_fn_ = k.flip_sign;
  r->kacs_walk_fn_ = k.kacs_walk;
  r->inv_kacs_walk_fn_ = k.inv_kacs_walk;
  r->inplace_fn_ = k.inplace;
  r->rescale_fn_ = k.rescale;
  // flip_ stays empty until train() is called.
  return r;
}

FhtRotator::Pointer FhtRotator::from_blob(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return nullptr;

  const auto *hdr = reinterpret_cast<const RotatorSerHeader *>(data);
  if (hdr->magic != kRotatorMagic) return nullptr;
  if (hdr->version != kRotatorSerVersion) return nullptr;
  if (static_cast<RotatorType>(hdr->rotator_type) != RotatorType::kFht) {
    return nullptr;
  }

  Pointer r(new FhtRotator());
  const size_t expected_total =
      sizeof(RotatorSerHeader) + static_cast<size_t>(hdr->payload_size);
  if (len < expected_total) return nullptr;

  int rc = r->deserialize(data, len);
  if (rc != 0) return nullptr;
  return r;
}

void FhtRotator::train(const void * /*data*/, size_t /*num*/,
                       size_t /*stride*/) {
  if (in_dim_ <= 0) return;

  flip_offset_ = (static_cast<size_t>(in_dim_) + kByteLen - 1) / kByteLen;
  flip_.resize(4 * flip_offset_);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto &b : flip_) b = static_cast<uint8_t>(dist(gen));
}

// ---------------------------------------------------------------------------
// apply  (forward rotation)
// ---------------------------------------------------------------------------

void FhtRotator::apply(const float *in, float *out) const {
  const size_t dim = static_cast<size_t>(in_dim_);
  std::memcpy(out, in, sizeof(float) * dim);

  if (trunc_dim_ == dim) {
    // Exact power-of-2: 4 rounds of (flip -> FHT -> rescale)
    flip_sign_fn_(flip_.data(), out, dim);
    inplace_fn_(out, trunc_dim_);
    rescale_fn_(out, trunc_dim_, fac_);

    flip_sign_fn_(flip_.data() + flip_offset_, out, dim);
    inplace_fn_(out, trunc_dim_);
    rescale_fn_(out, trunc_dim_, fac_);

    flip_sign_fn_(flip_.data() + 2 * flip_offset_, out, dim);
    inplace_fn_(out, trunc_dim_);
    rescale_fn_(out, trunc_dim_, fac_);

    flip_sign_fn_(flip_.data() + 3 * flip_offset_, out, dim);
    inplace_fn_(out, trunc_dim_);
    rescale_fn_(out, trunc_dim_, fac_);

    return;
  }

  // Non-power-of-2: 4 rounds with kacs_walk
  size_t start = dim - trunc_dim_;
  float *trunc_ptr = out + start;

  // Round 1: FHT on [0, trunc_dim)
  flip_sign_fn_(flip_.data(), out, dim);
  inplace_fn_(out, trunc_dim_);
  rescale_fn_(out, trunc_dim_, fac_);
  kacs_walk_fn_(out, dim);

  // Round 2: FHT on [start, start + trunc_dim)
  flip_sign_fn_(flip_.data() + flip_offset_, out, dim);
  inplace_fn_(trunc_ptr, trunc_dim_);
  rescale_fn_(trunc_ptr, trunc_dim_, fac_);
  kacs_walk_fn_(out, dim);

  // Round 3: FHT on [0, trunc_dim)
  flip_sign_fn_(flip_.data() + 2 * flip_offset_, out, dim);
  inplace_fn_(out, trunc_dim_);
  rescale_fn_(out, trunc_dim_, fac_);
  kacs_walk_fn_(out, dim);

  // Round 4: FHT on [start, start + trunc_dim)
  flip_sign_fn_(flip_.data() + 3 * flip_offset_, out, dim);
  inplace_fn_(trunc_ptr, trunc_dim_);
  rescale_fn_(trunc_ptr, trunc_dim_, fac_);
  kacs_walk_fn_(out, dim);

  // Final rescale: combine the 4 kacs_walk reductions
  rescale_fn_(out, dim, 0.25f);
}

// ---------------------------------------------------------------------------
// apply_inverse  (inverse rotation)
// ---------------------------------------------------------------------------

void FhtRotator::apply_inverse(const float *in, float *out) const {
  const size_t dim = static_cast<size_t>(in_dim_);
  // Copy input into working buffer
  std::vector<float> data(in, in + dim);

  if (trunc_dim_ == dim) {
    // Exact power-of-2: reverse 4 rounds in reverse order.
    const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));
    for (int round = 3; round >= 0; --round) {
      inplace_fn_(data.data(), trunc_dim_);
      rescale_fn_(data.data(), trunc_dim_, inv_fac);
      flip_sign_fn_(flip_.data() + static_cast<size_t>(round) * flip_offset_,
                    data.data(), dim);
    }
    std::memcpy(out, data.data(), dim * sizeof(float));
    return;
  }

  // Non-power-of-2: undo final rescale(0.25) first
  rescale_fn_(data.data(), dim, 4.0f);

  const float inv_fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));
  size_t start = dim - trunc_dim_;
  float *trunc_ptr = data.data() + start;

  // Undo Round 4 (FHT on [start, start+trunc_dim))
  inv_kacs_walk_fn_(data.data(), dim);
  inplace_fn_(trunc_ptr, trunc_dim_);
  rescale_fn_(trunc_ptr, trunc_dim_, inv_fac);
  flip_sign_fn_(flip_.data() + 3 * flip_offset_, data.data(), dim);

  // Undo Round 3 (FHT on [0, trunc_dim))
  inv_kacs_walk_fn_(data.data(), dim);
  inplace_fn_(data.data(), trunc_dim_);
  rescale_fn_(data.data(), trunc_dim_, inv_fac);
  flip_sign_fn_(flip_.data() + 2 * flip_offset_, data.data(), dim);

  // Undo Round 2 (FHT on [start, start+trunc_dim))
  inv_kacs_walk_fn_(data.data(), dim);
  inplace_fn_(trunc_ptr, trunc_dim_);
  rescale_fn_(trunc_ptr, trunc_dim_, inv_fac);
  flip_sign_fn_(flip_.data() + flip_offset_, data.data(), dim);

  // Undo Round 1 (FHT on [0, trunc_dim))
  inv_kacs_walk_fn_(data.data(), dim);
  inplace_fn_(data.data(), trunc_dim_);
  rescale_fn_(data.data(), trunc_dim_, inv_fac);
  flip_sign_fn_(flip_.data(), data.data(), dim);

  std::memcpy(out, data.data(), dim * sizeof(float));
}

// ---------------------------------------------------------------------------
// serialize / deserialize
// ---------------------------------------------------------------------------

int FhtRotator::serialize(std::string *out) const {
  if (!out) return kErrInvalidArgument;
  if (flip_.empty()) return kErrRuntime;

  RotatorSerHeader hdr{};
  hdr.magic = kRotatorMagic;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotatorType::kFht);
  hdr.in_dim = static_cast<uint32_t>(in_dim_);
  hdr.out_dim = static_cast<uint32_t>(out_dim_);
  hdr.payload_size = static_cast<uint32_t>(flip_.size());
  hdr.reserved = 0;

  out->resize(sizeof(hdr) + flip_.size());
  std::memcpy(&(*out)[0], &hdr, sizeof(hdr));
  std::memcpy(&(*out)[sizeof(hdr)], flip_.data(), flip_.size());
  return 0;
}

int FhtRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return kErrInvalidArgument;

  const auto *hdr = reinterpret_cast<const RotatorSerHeader *>(data);
  if (hdr->magic != kRotatorMagic) return kErrUnsupported;
  if (hdr->version != kRotatorSerVersion) return kErrUnsupported;
  if (static_cast<RotatorType>(hdr->rotator_type) != RotatorType::kFht) {
    return kErrUnsupported;
  }

  const size_t total = sizeof(RotatorSerHeader) + hdr->payload_size;
  if (len < total) return kErrInvalidArgument;

  in_dim_ = static_cast<int>(hdr->in_dim);
  out_dim_ = static_cast<int>(hdr->out_dim);
  trunc_dim_ = floor_pow2(static_cast<size_t>(in_dim_));
  fac_ = 1.0f / std::sqrt(static_cast<float>(trunc_dim_));
  flip_offset_ = (static_cast<size_t>(in_dim_) + kByteLen - 1) / kByteLen;
  auto k = get_fht_kernels();
  flip_sign_fn_ = k.flip_sign;
  kacs_walk_fn_ = k.kacs_walk;
  inv_kacs_walk_fn_ = k.inv_kacs_walk;
  inplace_fn_ = k.inplace;
  rescale_fn_ = k.rescale;

  flip_.resize(hdr->payload_size);
  std::memcpy(flip_.data(),
              reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader),
              hdr->payload_size);

  return 0;
}

}  // namespace turbo
}  // namespace zvec
