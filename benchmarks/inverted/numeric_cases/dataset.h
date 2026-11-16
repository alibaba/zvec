// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "profile.h"

namespace zvec::benchmarks::inverted {

// Fixed for reproducible datasets; not a configurable profile field.
inline constexpr uint64_t kNumericSeed = 20261115;

struct NumericData {
  NumericProfile profile;
  uint64_t null_count{0};
  std::vector<uint8_t> is_null;
  std::vector<std::string> values;
  // Sorted observed value/count pairs; no storage proportional to the domain.
  std::vector<std::pair<int64_t, uint64_t>> frequencies;
  std::vector<int64_t> hot_values;
};

NumericData GenerateDataset(const NumericProfile &profile);
JsonObject DescribeDataset(const NumericData &dataset);

// Native scalar bytes accepted by the inverted-column API.
template <typename T>
std::string EncodeValue(T value) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  std::string bytes(sizeof(T), '\0');
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

template <typename T>
T DecodeValue(const std::string &bytes) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  if (bytes.size() != sizeof(T))
    throw std::invalid_argument("invalid native scalar payload size");
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

}  // namespace zvec::benchmarks::inverted
