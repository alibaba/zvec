// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <cstdint>
#include <string>
#include "shared/metrics.h"

namespace zvec::benchmarks::inverted {

struct ArrayOptions {
  std::string profile;
  uint32_t num_docs{100000};
  uint64_t seed{20261115};
  std::string index_path;
  std::string build_report_path;
  std::string verification_path;
  std::string case_filter;
  double warmup_seconds{0.2};
  double measure_seconds{1.0};
  JsonObject dataset_overrides;
};

JsonObject ArrayCatalog(const ArrayOptions &options = {});
JsonObject RunArray(const std::string &mode, const ArrayOptions &options);

}  // namespace zvec::benchmarks::inverted
