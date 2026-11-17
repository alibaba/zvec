// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "json.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <zvec/ailego/utility/file_helper.h>

namespace zvec::benchmarks::inverted {

JsonValue JsonStringValue(const std::string &value) {
  return JsonValue(ailego::JsonString(value).encode());
}

std::string JsonStringText(const JsonValue &value) {
  if (!value.is_string()) {
    throw std::invalid_argument("Expected JSON string");
  }
  return value.as_string().decode().as_stl_string();
}

std::string Text(const JsonObject &object, const char *key) {
  return JsonStringText(object[key]);
}

uint64_t Number(const JsonObject &object, const char *key) {
  if (!object[key].is_integer() || object[key].as_integer() < 0)
    throw std::invalid_argument(std::string("Expected nonnegative integer: ") +
                                key);
  return object[key].as_integer();
}

void WriteJson(const std::string &path, const JsonObject &object) {
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  ailego::FileHelper::OpenOfstream(output, path,
                                   std::ios::binary | std::ios::trunc);
  output << JsonValue(object).as_json_string().as_stl_string() << '\n';
  output.close();
}

JsonObject ReadJson(const std::string &path) {
  std::ifstream input;
  ailego::FileHelper::OpenIfstream(input, path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot read JSON file: " + path);
  std::ostringstream content;
  content << input.rdbuf();
  if (input.bad())
    throw std::runtime_error("Failed reading JSON file: " + path);
  JsonValue value;
  if (!value.parse(content.str()) || !value.is_object()) {
    throw std::runtime_error("Invalid JSON object: " + path);
  }
  return value.as_object();
}

}  // namespace zvec::benchmarks::inverted
