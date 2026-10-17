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

#include "utility/steady_clock_timer.h"

namespace zvec {
namespace core {

SteadyClockTimer::tick_t SteadyClockTimer::Now() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<tick_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

uint64_t SteadyClockTimer::ElapsedNs(tick_t start, tick_t end) {
  return end > start ? (end - start) : 0;
}

}  // namespace core
}  // namespace zvec
