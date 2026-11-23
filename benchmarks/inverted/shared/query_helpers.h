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
#include "json.h"

namespace zvec::benchmarks::inverted {


JsonObject MakeQueryCase(const std::string &name, const JsonArray &inputs);

std::vector<JsonObject> ReadQueryInputs(const JsonArray &cases);


}  // namespace zvec::benchmarks::inverted
