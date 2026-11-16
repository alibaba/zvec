// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include "metrics.h"

namespace zvec::benchmarks::inverted::test {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::random_device random;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("zvec-inverted-test-" + std::to_string(random()));
      if (std::filesystem::create_directory(path_)) return;
    }
    throw std::runtime_error("Cannot create isolated test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;

  const std::filesystem::path &path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

inline std::string JsonText(const JsonValue &value) {
  return value.as_json_string().as_stl_string();
}

inline JsonObject NamedRecord(const JsonArray &entries,
                              const std::string &name) {
  for (const auto &value : entries) {
    const auto entry = value.as_object();
    if (Text(entry, "name") == name) return entry;
  }
  throw std::runtime_error("Missing test record: " + name);
}

}  // namespace zvec::benchmarks::inverted::test
