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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

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

FhtRotator::~FhtRotator() {
  std::free(fht_ctx_);
}

FhtRotator::Pointer FhtRotator::create(int dim) {
  if (dim <= 0) return nullptr;

  Pointer r(new FhtRotator());
  r->in_dim_ = dim;
  r->out_dim_ = dim;
  r->flip_offset_ = (static_cast<size_t>(dim) + kByteLen - 1) / kByteLen;
  r->kernels_ = get_rotator_kernels(RotateType::kFht);

  const size_t trunc_dim = floor_pow2(static_cast<size_t>(dim));
  const float fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));
  const size_t flip_size = 4 * r->flip_offset_;

  // Single allocation: FhtCtx header + trailing flip data.
  r->fht_ctx_ = static_cast<FhtCtx *>(std::malloc(sizeof(FhtCtx) + flip_size));
  if (r->fht_ctx_ == nullptr) {
    // Allocation failed: bail out before dereferencing. Returning nullptr lets
    // the smart Pointer clean up the partially-built object (~FhtRotator frees
    // the null fht_ctx_ safely).
    return nullptr;
  }
  r->fht_ctx_->flip_offset = r->flip_offset_;
  r->fht_ctx_->trunc_dim = trunc_dim;
  r->fht_ctx_->fac = fac;

  // Generate 4 rounds of random flip-sign arrays.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < flip_size; ++i)
    r->fht_ctx_->flip[i] = static_cast<uint8_t>(dist(gen));

  return r;
}

FhtRotator::Pointer FhtRotator::from_blob(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return nullptr;

  // Copy the header into a properly aligned local before reading any field.
  // `data` may point to an unaligned byte buffer (e.g. std::string::data()),
  // so dereferencing a reinterpret_cast<const RotatorSerHeader *> directly
  // would be undefined behavior on architectures that require alignment.
  RotatorSerHeader hdr;
  std::memcpy(&hdr, data, sizeof(RotatorSerHeader));
  if (hdr.magic != kRotatorMagic) return nullptr;
  if (hdr.version != kRotatorSerVersion) return nullptr;
  if (static_cast<RotateType>(hdr.rotator_type) != RotateType::kFht) {
    return nullptr;
  }

  Pointer r(new FhtRotator());
  const size_t expected_total =
      sizeof(RotatorSerHeader) + static_cast<size_t>(hdr.payload_size);
  if (len < expected_total) return nullptr;

  int rc = r->deserialize(data, len);
  if (rc != 0) return nullptr;
  return r;
}

void FhtRotator::train(const void * /*data*/, size_t /*num*/,
                       size_t /*stride*/) {
  // No-op: flip-sign arrays are generated in create().
}

// ---------------------------------------------------------------------------
// apply  (forward rotation)
// ---------------------------------------------------------------------------

void FhtRotator::apply(const float *in, float *out) const {
  kernels_.rotate(in, out, static_cast<size_t>(in_dim_),
                  static_cast<size_t>(out_dim_), static_cast<void *>(fht_ctx_));
}

// ---------------------------------------------------------------------------
// apply_inverse  (inverse rotation)
// ---------------------------------------------------------------------------

void FhtRotator::apply_inverse(const float *in, float *out) const {
  kernels_.unrotate(in, out, static_cast<size_t>(in_dim_),
                    static_cast<size_t>(out_dim_),
                    static_cast<void *>(fht_ctx_));
}

// ---------------------------------------------------------------------------
// serialize / deserialize
// ---------------------------------------------------------------------------

int FhtRotator::serialize(std::string *out) const {
  if (!out) return kErrInvalidArgument;
  if (!fht_ctx_) return kErrRuntime;

  const size_t flip_size = 4 * flip_offset_;

  RotatorSerHeader hdr{};
  hdr.magic = kRotatorMagic;
  hdr.version = kRotatorSerVersion;
  hdr.rotator_type = static_cast<uint16_t>(RotateType::kFht);
  hdr.in_dim = static_cast<uint32_t>(in_dim_);
  hdr.out_dim = static_cast<uint32_t>(out_dim_);
  hdr.payload_size = static_cast<uint32_t>(flip_size);
  hdr.reserved = 0;

  out->resize(sizeof(hdr) + flip_size);
  std::memcpy(&(*out)[0], &hdr, sizeof(hdr));
  std::memcpy(&(*out)[sizeof(hdr)], fht_ctx_->flip, flip_size);
  return 0;
}

int FhtRotator::deserialize(const void *data, size_t len) {
  if (!data || len < sizeof(RotatorSerHeader)) return kErrInvalidArgument;

  // Copy the header into a properly aligned local before reading any field.
  // `data` may point to an unaligned byte buffer (e.g. std::string::data()),
  // so dereferencing a reinterpret_cast<const RotatorSerHeader *> directly
  // would be undefined behavior on architectures that require alignment.
  RotatorSerHeader hdr;
  std::memcpy(&hdr, data, sizeof(RotatorSerHeader));
  if (hdr.magic != kRotatorMagic) return kErrUnsupported;
  if (hdr.version != kRotatorSerVersion) return kErrUnsupported;
  if (static_cast<RotateType>(hdr.rotator_type) != RotateType::kFht) {
    return kErrUnsupported;
  }

  // Validate dimensions before any cast to int: must be strictly positive and
  // representable as int. FHT keeps dimensionality unchanged (in == out).
  if (hdr.in_dim == 0 ||
      hdr.in_dim > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return kErrInvalidArgument;
  }
  if (hdr.out_dim != hdr.in_dim) return kErrInvalidArgument;

  // Length check via subtraction to avoid size_t overflow on 32-bit
  // (len >= sizeof(header) was guaranteed above, so the subtraction is safe).
  if (hdr.payload_size > len - sizeof(RotatorSerHeader)) {
    return kErrInvalidArgument;
  }

  // Payload must hold exactly 4 rounds of ceil(in_dim/8) flip bytes. The
  // rotation kernels read 4 * flip_offset bytes, so any smaller payload would
  // read out of bounds; require an exact match.
  const size_t new_flip_offset =
      (static_cast<size_t>(hdr.in_dim) + kByteLen - 1) / kByteLen;
  const size_t expected_flip_size = 4 * new_flip_offset;
  if (hdr.payload_size != expected_flip_size) return kErrInvalidArgument;

  // Build the new context into a temporary first, so a failed allocation
  // leaves the existing object completely untouched (deserialize may target
  // an already-initialized rotator). Only commit member state once the
  // allocation and payload copy have both succeeded.
  const int new_in_dim = static_cast<int>(hdr.in_dim);
  const int new_out_dim = static_cast<int>(hdr.out_dim);
  const size_t trunc_dim = floor_pow2(static_cast<size_t>(new_in_dim));
  const float fac = 1.0f / std::sqrt(static_cast<float>(trunc_dim));

  FhtCtx *new_ctx =
      static_cast<FhtCtx *>(std::malloc(sizeof(FhtCtx) + expected_flip_size));
  if (new_ctx == nullptr) return kErrRuntime;
  new_ctx->flip_offset = new_flip_offset;
  new_ctx->trunc_dim = trunc_dim;
  new_ctx->fac = fac;
  std::memcpy(new_ctx->flip,
              reinterpret_cast<const char *>(data) + sizeof(RotatorSerHeader),
              expected_flip_size);

  // All allocations succeeded: commit new state and free the old context.
  std::free(fht_ctx_);
  fht_ctx_ = new_ctx;
  in_dim_ = new_in_dim;
  out_dim_ = new_out_dim;
  flip_offset_ = new_flip_offset;
  kernels_ = get_rotator_kernels(RotateType::kFht);

  return 0;
}

}  // namespace turbo
}  // namespace zvec
