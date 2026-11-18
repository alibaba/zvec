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

#include <cfenv>
#include <cmath>
#include <cstddef>
#include <limits>

namespace zvec::fts {

// Upper bound for sums of nonnegative FP32 scores in any accumulation order.
// With n <= 2^20 and round-to-nearest, gamma_(n-1) <= n * FLT_EPSILON.
// Accumulate the input bounds upward in double, apply that proven rounding
// allowance, then round upward to float. Actual scoring is never changed.
class ScoreUpperBound {
 public:
  void add(float score) {
    if (!std::isfinite(score) || score < 0 || ++count_ > (1u << 20)) {
      sum_ = std::numeric_limits<double>::infinity();
      return;
    }
    if (score != 0) {
      sum_ =
          std::nextafter(sum_ + score, std::numeric_limits<double>::infinity());
    }
  }

  float value() const {
    if (std::fegetround() != FE_TONEAREST || !std::isfinite(sum_)) {
      return std::numeric_limits<float>::infinity();
    }
    if (sum_ == 0) {
      return 0;
    }
    const double factor = 1.0 + count_ * std::numeric_limits<float>::epsilon();
    const double upper =
        std::nextafter(sum_ * factor, std::numeric_limits<double>::infinity());
    if (upper > std::numeric_limits<float>::max()) {
      return std::numeric_limits<float>::infinity();
    }
    return std::nextafter(static_cast<float>(upper),
                          std::numeric_limits<float>::infinity());
  }

 private:
  double sum_{0};
  size_t count_{0};
};

}  // namespace zvec::fts
