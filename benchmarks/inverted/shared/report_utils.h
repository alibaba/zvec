// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <string>
#include <vector>
#include "metrics.h"

namespace zvec::benchmarks::inverted {

// Report helpers receive only their inputs; CLI options belong to each suite.
JsonObject LoadBuild(const std::string &build_report_path,
                     const std::string &index_path);
std::vector<JsonObject> SelectInputs(const JsonArray &cases,
                                     const std::string &filter);
JsonObject LoadVerification(const std::string &verification_path,
                            const JsonObject &build, const JsonObject &inputs);
JsonObject InputRecord(const std::string &name, const JsonArray &inputs);
void CheckRegenerated(const JsonObject &build, const std::string &fingerprint,
                      const JsonArray &inputs);

}  // namespace zvec::benchmarks::inverted
