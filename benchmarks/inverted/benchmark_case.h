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
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <zvec/db/schema.h>

namespace zvec::benchmarks::inverted {

struct DatasetSpec {
  std::string name;
  std::string type;
  std::string distribution;
  uint32_t cardinality{0};
  uint32_t array_length{0};
  bool supplemental{false};
};

enum class QueryOp {
  kEq,
  kNe,
  kLt,
  kGe,
  kAny,
  kAll,
  kNotAny,
  kNotAll,
  kIsNull,
  kIsNotNull,
  kLengthEq
};

struct QueryInput {
  QueryOp op;
  // Native numeric bytes, exactly as accepted by the index interface.
  std::vector<std::string> values;
  uint32_t length{0};
};

struct QueryCase {
  std::string name;
  std::vector<QueryInput> inputs;
  bool from_documents{false};
};

struct DatasetStats {
  uint64_t null_count{0};
  uint64_t empty_count{0};
  uint64_t duplicate_count{0};
  uint64_t raw_elements{0};
  uint64_t distinct_elements{0};
  uint64_t observed_cardinality{0};
  std::vector<uint64_t> term_doc_frequencies;
  std::map<uint32_t, uint64_t> raw_length_histogram;
  std::map<uint32_t, uint64_t> unique_length_histogram;
};

struct PreparedDataset {
  DatasetSpec spec;
  uint32_t num_docs{0};
  uint64_t seed{0};
  std::vector<uint8_t> is_null;
  // Prepared before timing. Empty payload and NULL are distinguished by mask.
  std::vector<std::string> values;
  DatasetStats stats;
  std::string fingerprint;
  std::vector<QueryCase> cases;
};

struct TypeSuite {
  std::string name;
  DataType data_type;
  std::vector<DatasetSpec> profiles;
  std::function<std::vector<std::string>(const DatasetSpec &)> case_names;
  std::function<PreparedDataset(const DatasetSpec &, uint32_t, uint64_t)>
      prepare;
  // An independent typed predicate; never calls index or bitmap helpers.
  std::function<bool(const PreparedDataset &, uint32_t, const QueryInput &)>
      matches;
};

TypeSuite Int64Suite();
TypeSuite ArrayInt32Suite();
const std::vector<TypeSuite> &Suites();
const TypeSuite &FindSuite(const std::string &name);
const DatasetSpec &FindProfile(const TypeSuite &, const std::string &name);

}  // namespace zvec::benchmarks::inverted
