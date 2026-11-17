// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace zvec::benchmarks::inverted {

// Explicit sampling and Fisher-Yates make results independent of the standard
// library's uniform_int_distribution and shuffle implementations.
class Random {
 public:
  explicit Random(uint64_t seed) : engine_(seed) {}

  uint64_t bounded(uint64_t bound) {
    if (bound == 0) throw std::invalid_argument("zero random bound");
    const uint64_t threshold = -bound % bound;
    uint64_t value;
    do {
      value = engine_();
    } while (value < threshold);
    return value % bound;
  }

  template <typename T>
  void shuffle(std::vector<T> &items) {
    for (size_t remaining = items.size(); remaining > 1; --remaining) {
      std::iter_swap(items.begin() + remaining - 1,
                     items.begin() + bounded(remaining));
    }
  }

 private:
  std::mt19937_64 engine_;
};

}  // namespace zvec::benchmarks::inverted
