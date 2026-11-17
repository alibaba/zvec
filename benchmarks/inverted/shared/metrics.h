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
#include <cstdint>
#include <string>
#include "json.h"

namespace zvec::benchmarks::inverted {
using Clock = std::chrono::steady_clock;

JsonObject MemorySnapshot();
JsonObject DiskUsage(const std::string &path);
JsonObject EnvironmentJson();
double Seconds(Clock::time_point begin, Clock::time_point end);

struct Measurement {
  uint64_t completed{0};
  double elapsed_seconds{0};
};

// Optional report fields for query timing; cases add their own result fields.
JsonObject QueryReport(const std::string &name, const Measurement &measurement,
                       double warmup_seconds, double measure_seconds,
                       const JsonObject &before, const JsonObject &after);
}  // namespace zvec::benchmarks::inverted
