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

#include <atomic>
#include <random>
#include <zvec/core/framework/index_context.h>
#include "turbo/quantizer/quantizer.h"

namespace zvec {
namespace core {

void IndexContext::apply_threshold() {
  float val = raw_threshold_;
  if (threshold_uses_quantizer_) {
    auto quantizer = index_quantizer_.lock();
    if (!quantizer) {
      // Fail closed until the context is rebound to a live backend. Never
      // mistake a caller-facing radius for an internal distance on expiry.
      threshold_ = -std::numeric_limits<float>::infinity();
      return;
    }
    if (quantizer->support_score_normalization()) {
      quantizer->denormalize_score(&val);
    }
  } else if (index_metric_ && index_metric_->support_normalize()) {
    index_metric_->denormalize(&val);
  }
  threshold_ = val;
}

uint32_t IndexContext::GenerateMagic() {
  static std::atomic_uint32_t magic_number{std::random_device()()};
  return magic_number.fetch_add(1);
}

}  // namespace core
}  // namespace zvec
