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
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace zvec::benchmarks::inverted {

// Incremental FNV-1a fingerprint. Each caller chooses its fields and their
// order. Integers use little-endian bytes; strings include their length.
class Fingerprint {
 public:
  void AddByte(uint8_t value) {
    hash_ ^= value;
    hash_ *= UINT64_C(1099511628211);
  }

  void AddUint64(uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) {
      AddByte(static_cast<uint8_t>(value));
      value >>= 8;
    }
  }

  void AddString(std::string_view value) {
    AddUint64(value.size());
    for (unsigned char byte : value) AddByte(byte);
  }

  std::string Finish() const {
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash_;
    return result.str();
  }

 private:
  uint64_t hash_{UINT64_C(14695981039346656037)};
};

}  // namespace zvec::benchmarks::inverted
