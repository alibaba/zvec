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
#include <cstring>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include "benchmark_case.h"

namespace zvec::benchmarks::inverted::data {

constexpr size_t kQueryPoolSize = 8;
enum RowKind : uint8_t { kNull, kEmpty, kPlain, kDuplicate };

// Explicit bounded sampling and Fisher-Yates avoid library-dependent
// uniform_int_distribution/shuffle implementations in dataset generation.
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

  double unit() {
    return static_cast<double>(engine_() >> 11) * 0x1.0p-53;
  }

  template <typename T>
  void shuffle(std::vector<T> &items) {
    for (size_t remaining = items.size(); remaining > 1; --remaining) {
      std::swap(items[remaining - 1], items[bounded(remaining)]);
    }
  }

 private:
  std::mt19937_64 engine_;
};

inline std::vector<uint8_t> RowKinds(uint32_t count, bool arrays,
                                     bool duplicates, uint64_t seed) {
  if (count == 0) throw std::invalid_argument("num_docs must be positive");
  std::vector<uint8_t> kinds(count, kPlain);
  const uint32_t nulls = count / 10;
  const uint32_t empties = arrays ? count / 10 : 0;
  const uint32_t repeated = duplicates ? (count - nulls - empties) / 5 : 0;
  std::fill_n(kinds.begin(), nulls, kNull);
  std::fill_n(kinds.begin() + nulls, empties, kEmpty);
  std::fill_n(kinds.begin() + nulls + empties, repeated, kDuplicate);
  Random(seed ^ 0xa0761d6478bd642fULL).shuffle(kinds);
  return kinds;
}

template <typename T>
std::string NativeBytes(T value) {
  static_assert(std::is_integral_v<T>);
  std::string bytes(sizeof(T), '\0');
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

template <typename T>
T ReadNative(const std::string &bytes) {
  if (bytes.size() != sizeof(T)) {
    throw std::invalid_argument("invalid native scalar payload size");
  }
  T value;
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

inline int32_t ArrayElement(const std::string &bytes, size_t offset) {
  int32_t value;
  std::memcpy(&value, bytes.data() + offset * sizeof(value), sizeof(value));
  return value;
}

template <typename T>
QueryInput ScalarQuery(QueryOp op, T value) {
  return {op, {NativeBytes(value)}, 0};
}

template <typename T>
QueryInput ListQuery(QueryOp op, const std::vector<T> &values) {
  QueryInput query{op, {}, 0};
  query.values.reserve(values.size());
  for (T value : values) query.values.push_back(NativeBytes(value));
  return query;
}

template <typename T>
std::vector<T> Pool(const std::vector<T> &population) {
  const size_t count = std::min(kQueryPoolSize, population.size());
  std::vector<T> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.push_back(population[i * population.size() / count]);
  }
  return result;
}

inline void FinishStats(PreparedDataset &dataset) {
  dataset.stats.observed_cardinality =
      std::count_if(dataset.stats.term_doc_frequencies.begin(),
                    dataset.stats.term_doc_frequencies.end(),
                    [](uint64_t count) { return count != 0; });
}

inline void DeduplicateQueryPools(PreparedDataset &dataset) {
  for (auto &query_case : dataset.cases) {
    std::vector<QueryInput> unique;
    unique.reserve(query_case.inputs.size());
    for (auto &input : query_case.inputs) {
      if (input.op == QueryOp::kAny || input.op == QueryOp::kAll ||
          input.op == QueryOp::kNotAny || input.op == QueryOp::kNotAll) {
        std::sort(input.values.begin(), input.values.end());
      }
      const auto found = std::find_if(
          unique.begin(), unique.end(), [&input](const QueryInput &other) {
            return input.op == other.op && input.length == other.length &&
                   input.values == other.values;
          });
      if (found == unique.end()) unique.push_back(std::move(input));
    }
    query_case.inputs = std::move(unique);
  }
}

// Payloads intentionally retain the native representation accepted by the API;
// the runner reports byte order. The hash contains its own dataset metadata.
inline std::string Fingerprint(const PreparedDataset &dataset) {
  uint64_t hash = 14695981039346656037ULL;
  auto byte = [&hash](uint8_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
  };
  auto number = [&byte](uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) {
      byte(static_cast<uint8_t>(value));
      value >>= 8;
    }
  };
  auto string = [&byte, &number](const std::string &value) {
    number(value.size());
    for (unsigned char ch : value) byte(ch);
  };
  string("inverted-data-v1");
  string(dataset.spec.name);
  string(dataset.spec.type);
  string(dataset.spec.distribution);
  number(dataset.spec.cardinality);
  number(dataset.spec.array_length);
  number(dataset.num_docs);
  number(dataset.seed);
  for (uint32_t id = 0; id < dataset.num_docs; ++id) {
    byte(dataset.is_null[id]);
    const auto &value = dataset.values[id];
    number(value.size());
    for (unsigned char payload_byte : value) byte(payload_byte);
  }
  std::ostringstream result;
  result << std::hex << std::setfill('0') << std::setw(16) << hash;
  return result.str();
}

}  // namespace zvec::benchmarks::inverted::data
