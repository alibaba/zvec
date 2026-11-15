// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include "metrics.h"

namespace zvec::benchmarks::inverted {

struct RunOptions {
  std::string type;
  std::string profile;
  uint32_t num_docs{100000};
  uint64_t seed{20261115};
  std::string index;
  std::string manifest;
  std::string verification;
  std::string case_filter;
  double warmup_seconds{0.2};
  double measure_seconds{1.0};
};

JsonObject Catalog();
JsonObject RunBuild(const RunOptions &);
JsonObject RunVerify(const RunOptions &);
JsonObject RunQuery(const RunOptions &);

}  // namespace zvec::benchmarks::inverted
