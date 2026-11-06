#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
struct BenchmarkClock {
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<BenchmarkClock, duration>;
  static time_point now() {
    timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) std::abort();
    return time_point(duration(int64_t(value.tv_sec) * 1000000000 + value.tv_nsec));
  }
};
