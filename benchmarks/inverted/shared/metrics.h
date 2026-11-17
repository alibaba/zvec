// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include "json.h"

namespace zvec::benchmarks::inverted {

using Clock = std::chrono::steady_clock;

JsonObject MemorySnapshot();
JsonObject DiskUsage(const std::string &path);
JsonObject EnvironmentJson();
double Seconds(Clock::time_point begin, Clock::time_point end);

// Runs 100 iterations and returns the mean microseconds per query.
template <typename RunQueries>
double MeasureRepeatedQueries(size_t queries_per_iteration,
                              RunQueries &&run_queries) {
  constexpr size_t kIterations = 100;
  if (!queries_per_iteration) {
    throw std::invalid_argument("Query count per iteration must be positive");
  }
  const auto begin = Clock::now();
  for (size_t i = 0; i < kIterations; ++i) {
    run_queries();
  }
  const auto end = Clock::now();
  const auto elapsed_us =
      std::chrono::duration<double, std::micro>(end - begin).count();
  return elapsed_us / queries_per_iteration / kIterations;
}

JsonObject QueryReport(const std::string &name, double query_time_us);
}  // namespace zvec::benchmarks::inverted
