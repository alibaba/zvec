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
#include <cstdlib>
#include <memory>
#include <turbo/preprocessor/fht_rotator/fht_rotator.h>
#include "rotator.h"

namespace zvec {
namespace core {

// ============================================================================
// FhtRotator - O(d log d) FHT-based Kac random rotation
//
// Works with any dimension (non-power-of-2 uses trunc_dim + KacsWalk).
// When dimension is a power of 2, uses 4 rounds of (flip -> FHT -> rescale).
// When dimension is NOT a power of 2, uses kacs_walk reduction.
// ============================================================================

class FhtRotator : public Rotator {
 public:
  FhtRotator() = default;
  ~FhtRotator() override = default;

  // Virtual interface
  void rotate(const float *in, float *out) const override;
  void unrotate(const float *in, float *out) const override;
  RotatorType rotator_type() const override;

 protected:
  // Protected virtuals for base class factory/serialization
  int init_impl(size_t dim) override;
  size_t blob_bytes() const override;
  void save_blob(char *data) const override;
  void load_blob(const char *data) override;

 private:
  void init_context(size_t dim);

  // Preserve the legacy flip-only payload while using Turbo's kernel context.
  std::unique_ptr<turbo::FhtCtx, decltype(&std::free)> context_{nullptr,
                                                                &std::free};
  turbo::RotatorKernels kernels_{};

  static constexpr size_t kByteLen = 8;
};

}  // namespace core
}  // namespace zvec
