// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <zvec/ailego/encoding/json.h>
#include "benchmark_case.h"

namespace zvec::benchmarks::inverted {

using ailego::JsonArray;
using ailego::JsonObject;
using ailego::JsonValue;

JsonValue JsonStringValue(const std::string &value);
std::string JsonStringText(const JsonValue &value);
JsonObject SpecJson(const DatasetSpec &spec);
DatasetSpec ParseSpec(const JsonObject &object);
JsonObject DatasetJson(const PreparedDataset &dataset);
JsonArray CasesJson(const std::vector<QueryCase> &cases);
std::vector<QueryCase> ParseCases(const JsonArray &array);
std::string QueryFingerprint(const QueryCase &query_case);
void WriteJson(const std::string &path, const JsonObject &object);
JsonObject ReadJson(const std::string &path);
JsonObject MemorySnapshot();
JsonObject DiskUsage(const std::string &path);
JsonObject EnvironmentJson();

}  // namespace zvec::benchmarks::inverted
