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
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "metrics.h"

namespace zvec::benchmarks::inverted {

// The caller defines a complete pool pass, including result consumption and
// release, and returns its checksum. No operation or result type is
// interpreted.
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
    result.checksum += run_pool();
    result.completed += pool_size;
    end = Clock::now();
  } while (Seconds(begin, end) < measure_seconds);
  result.elapsed_seconds = Seconds(begin, end);
  return result;
}

inline std::vector<uint64_t> VerifiedCounts(const JsonObject &verification,
                                            size_t pool_size,
                                            uint32_t num_docs) {
  if (!verification["counts"].is_array())
    throw std::invalid_argument("Verification has no document counts");
  const auto counts = verification["counts"].as_array();
  if (counts.size() != pool_size)
    throw std::invalid_argument("Query pool was not fully verified");
  std::vector<uint64_t> result;
  for (const auto &count : counts) {
    if (!count.is_integer() || count.as_integer() < 0 ||
        static_cast<uint64_t>(count.as_integer()) > num_docs)
      throw std::invalid_argument("Invalid verified result count");
    result.push_back(count.as_integer());
  }
  return result;
}
}  // namespace zvec::benchmarks::inverted
