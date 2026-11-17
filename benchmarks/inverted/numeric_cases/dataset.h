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
#include "profile.h"

namespace zvec::benchmarks::inverted {

// Fixed for reproducible datasets; not a configurable profile field.
inline constexpr uint64_t kNumericSeed = 20261115;
// Maximum records per read/write; the final batch may be smaller.
inline constexpr uint32_t kNumericBatchSize = 32768;

// Temporary generation data, released before opening the index.
template <typename T>
struct NumericData {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  // Workload used to generate the records.
  NumericProfile profile;
  // Exact number of NULL documents.
  uint64_t null_count{0};
  // One marker per doc ID, including NULL documents.
  std::vector<uint8_t> is_null;
  // Non-NULL values in doc ID order; NULL documents have no slot here.
  std::vector<T> values;
  // Sorted hot set used for query selection; empty for uniform data.
  std::vector<T> hot_values;
};

template <typename T>
NumericData<T> GenerateDataset(const NumericProfile &profile);
// sorted_values contains all non-NULL values, including duplicates.
template <typename T>
JsonObject DescribeDataset(const NumericData<T> &dataset,
                           const std::vector<T> &sorted_values);

// Records follow doc ID order: one NULL byte, then native INT32/INT64 bytes.
// NULL value bytes are zero-filled. An existing file is never overwritten.
template <typename T>
void WriteDataset(const NumericData<T> &dataset, const std::string &path);

struct NumericBatch {
  // Record i belongs to doc ID first_doc_id + i.
  uint32_t first_doc_id{0};
  // One NULL marker per record.
  std::vector<uint8_t> is_null;
  // Native scalar bytes ready for insert(); NULL records have empty strings.
  std::vector<std::string> values;
};

class NumericDatasetReader {
 public:
  NumericDatasetReader(const NumericProfile &profile, const std::string &path);

  // Reuses batch storage. Returns false and empties the batch at end of file.
  bool ReadBatch(NumericBatch &batch);
  // Restarts reading from doc ID zero using the same file and buffer.
  void Reset();

 private:
  ailego::File file_;
  std::string path_;
  uint32_t num_docs_;
  size_t value_size_;
  uint32_t next_doc_id_{0};
  std::vector<char> buffer_;
};

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
