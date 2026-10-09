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
#include <new>
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

void FhtRotator::init_context(size_t dim) {
  const size_t flip_offset = (dim + kByteLen - 1) / kByteLen;
  auto *ctx = static_cast<turbo::FhtCtx *>(
      std::malloc(sizeof(turbo::FhtCtx) + 4 * flip_offset));
  if (!ctx) throw std::bad_alloc();
  context_.reset(ctx);
  ctx->flip_offset = flip_offset;
  ctx->trunc_dim = floor_pow2(dim);
  ctx->fac = 1.0f / std::sqrt(static_cast<float>(ctx->trunc_dim));
  kernels_ = turbo::get_rotator_kernels(turbo::RotateType::kFht);
}

int FhtRotator::init_impl(size_t dim) {
  if (dim == 0) {
    return IndexError_InvalidArgument;
  }
  init_context(dim);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < blob_bytes(); ++i) {
    context_->flip[i] = static_cast<uint8_t>(dist(gen));
  }
  return 0;
}

void FhtRotator::rotate(const float *in, float *out) const {
  kernels_.rotate(in, out, dimension_, dimension_, context_.get());
}

void FhtRotator::unrotate(const float *in, float *out) const {
  kernels_.unrotate(in, out, dimension_, dimension_, context_.get());
}

RotatorType FhtRotator::rotator_type() const {
  return RotatorType::FhtKac;
}

void FhtRotator::save_blob(char *data) const {
  std::memcpy(data, context_->flip, blob_bytes());
}

void FhtRotator::load_blob(const char *data) {
  init_context(dimension_);
  std::memcpy(context_->flip, data, blob_bytes());
}

size_t FhtRotator::blob_bytes() const {
  return context_ ? 4 * context_->flip_offset : 0;
}

}  // namespace core
}  // namespace zvec
