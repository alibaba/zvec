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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <zvec/ailego/io/file.h>
#include "shared/json.h"
#include "profile.h"

namespace zvec::benchmarks::inverted {

struct NumericBatch {
  // Record i belongs to doc ID first_doc_id + i.
  uint32_t first_doc_id{0};
  // One NULL marker per record.
  std::vector<uint8_t> is_null;
  // Native scalar bytes ready for insert(); NULL records have empty strings.
  std::vector<std::string> values;
};

class NumericDataset {
 public:
  static NumericDataset Create(const NumericProfile &profile,
                               const std::string &directory);

  static NumericDataset Load(const std::string &directory);

  const NumericProfile &profile() const {
    return profile_;
  }
  const std::vector<JsonObject> &query_cases() const {
    return query_cases_;
  }


  bool read_next_batch(NumericBatch &batch);

  void rewind();

 private:
  NumericDataset(const std::string &directory, const JsonObject &metadata);

  // Workload configuration.
  NumericProfile profile_;
  // Query names and input values.
  std::vector<JsonObject> query_cases_;
  // Path to the document records in data.bin.
  std::string data_path_;
  // Read-only data file, opened on the first batch read.
  ailego::File file_;
  // Bytes occupied by each numeric value in data.bin.
  size_t bytes_per_value_;
  // Doc ID of the next record to read.
  uint32_t next_doc_id_{0};
  // Reusable raw-byte buffer for reading one batch of records.
  std::vector<char> buffer_;
};


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
