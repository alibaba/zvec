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
#include <zvec/ailego/encoding/json.h>

namespace zvec::benchmarks::inverted {

using ailego::JsonArray;
using ailego::JsonObject;
using ailego::JsonValue;

JsonValue JsonStringValue(const std::string &value);
std::string JsonStringText(const JsonValue &value);
std::string Text(const JsonObject &object, const char *key);
uint64_t Number(const JsonObject &object, const char *key);
void WriteJson(const std::string &path, const JsonObject &object);
JsonObject ReadJson(const std::string &path);

}  // namespace zvec::benchmarks::inverted
