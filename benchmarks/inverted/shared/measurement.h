// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "metrics.h"

namespace zvec::benchmarks::inverted {

// The caller defines a complete pool pass, including result consumption and
// release. No operation or result type is interpreted.
template <typename RunPool>
Measurement MeasurePool(size_t pool_size, double warmup_seconds,
                        double measure_seconds, RunPool &&run_pool) {
  if (!pool_size || !std::isfinite(warmup_seconds) || warmup_seconds < 0 ||
      !std::isfinite(measure_seconds) || measure_seconds <= 0) {
    throw std::invalid_argument("Invalid measurement parameters");
  }
  const auto warmup = Clock::now();
  do {
    run_pool();
  } while (Seconds(warmup, Clock::now()) < warmup_seconds);

  Measurement result;
  const auto begin = Clock::now();
  Clock::time_point end;
  do {
    run_pool();
    result.completed += pool_size;
    end = Clock::now();
  } while (Seconds(begin, end) < measure_seconds);
  result.elapsed_seconds = Seconds(begin, end);
  return result;
}

}  // namespace zvec::benchmarks::inverted
