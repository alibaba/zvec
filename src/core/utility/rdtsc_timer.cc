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

#include "utility/rdtsc_timer.h"
#include <ctime>

namespace zvec {
namespace core {

RdtscTimer::tick_t RdtscTimer::Now() {
#if ZVEC_CORE_HAS_TSC
  uint32_t lo = 0;
  uint32_t hi = 0;
  uint32_t aux = 0;
  __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) : :);
  return (static_cast<uint64_t>(hi) << 32) | lo;
#else
  return MonotonicRawNs();
#endif
}

uint64_t RdtscTimer::ElapsedNs(tick_t start, tick_t end) {
#if ZVEC_CORE_HAS_TSC
  if (end <= start) {
    return 0;
  }
  return static_cast<uint64_t>(static_cast<double>(end - start) * NsPerTick());
#else
  return end > start ? (end - start) : 0;
#endif
}

uint64_t RdtscTimer::MonotonicRawNs() {
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
         static_cast<uint64_t>(ts.tv_nsec);
}

double RdtscTimer::NsPerTick() {
  static const double ns_per_tick = CalibrateNsPerTick();
  return ns_per_tick;
}

double RdtscTimer::CalibrateNsPerTick() {
  constexpr uint64_t kMinCalibrationNs = 5 * 1000 * 1000;
  const uint64_t start_ns = MonotonicRawNs();
  const tick_t start_tick = Now();

  uint64_t end_ns = start_ns;
  while (end_ns - start_ns < kMinCalibrationNs) {
    end_ns = MonotonicRawNs();
  }

  const tick_t end_tick = Now();
  if (end_tick <= start_tick) {
    return 1.0;
  }
  return static_cast<double>(end_ns - start_ns) /
         static_cast<double>(end_tick - start_tick);
}

}  // namespace core
}  // namespace zvec
