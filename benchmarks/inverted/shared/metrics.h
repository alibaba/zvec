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
#include <zvec/ailego/encoding/json.h>

namespace zvec::benchmarks::inverted {
using ailego::JsonArray;
using ailego::JsonObject;
using ailego::JsonValue;
using Clock = std::chrono::steady_clock;

JsonValue JsonStringValue(const std::string &value);
std::string JsonStringText(const JsonValue &value);
std::string Text(const JsonObject &object, const char *key);
uint64_t Number(const JsonObject &object, const char *key);
void WriteJson(const std::string &path, const JsonObject &object);
JsonObject ReadJson(const std::string &path);
JsonObject MemorySnapshot();
JsonObject DiskUsage(const std::string &path);
JsonObject EnvironmentJson();
JsonObject Report(const char *phase);
JsonArray BuildMetricNames();
double Seconds(Clock::time_point begin, Clock::time_point end);
std::string Hex(uint64_t value);

struct Measurement {
  uint64_t completed{0};
  uint64_t checksum{0};
  double elapsed_seconds{0};
};

// Optional report fields for query timing; cases add their own result fields.
JsonObject QueryReport(const std::string &name, const Measurement &measurement,
                       double warmup_seconds, double measure_seconds,
                       const JsonObject &before, const JsonObject &after);
}  // namespace zvec::benchmarks::inverted
