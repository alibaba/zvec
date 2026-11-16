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
#include <string>
#include <zvec/db/type.h>
#include "shared/metrics.h"

namespace zvec::benchmarks::inverted {


enum class NumericDistribution { UNIFORM, HOTSPOT };
enum class NumericDocCount : uint32_t {
  // 100,000 documents, including NULLs.
  ONE_HUNDRED_THOUSAND = 100000,
  // 10,000,000 documents, including NULLs.
  TEN_MILLION = 10000000,
};
enum class NumericValueRange { ZERO_TO_TYPE_MAX, ZERO_TO_30000 };
enum class NumericValueOrder { SHUFFLED, ASCENDING };
enum class NumericNullRatio { NONE, TEN_PERCENT, FIFTY_PERCENT };


struct NumericProfile {
  explicit NumericProfile(DataType type) : data_type(type) {}

  // Indexed data type; explicitly choose INT32 or INT64.
  DataType data_type;
  // Total document count, including NULLs: 100,000 or 10,000,000.
  NumericDocCount num_docs{NumericDocCount::ONE_HUNDRED_THOUSAND};
  // Value frequency distribution among non-NULL documents.
  NumericDistribution distribution{NumericDistribution::UNIFORM};
  // Inclusive nonnegative domain: [0, type maximum] or [0, 30000].
  NumericValueRange value_range{NumericValueRange::ZERO_TO_30000};
  // Non-NULL value order by doc ID; NULL positions stay unchanged.
  NumericValueOrder value_order{NumericValueOrder::SHUFFLED};
  // NULL fraction: none, 10%, or 50% of all documents.
  NumericNullRatio null_ratio{NumericNullRatio::TEN_PERCENT};
  // HOTSPOT only: number of distinct values in the hot set.
  uint32_t hot_value_count{100};
  // HOTSPOT only: fraction of non-NULL documents on hot values.
  double hot_doc_ratio{0.8};
};

// CLI preset/JSON adapters; generation and queries use the typed profile.
NumericProfile ResolveNumericProfile(DataType type, const std::string &preset,
                                     const JsonObject &overrides = {});
JsonObject NumericCatalog(DataType type, const std::string &preset = "",
                          const JsonObject &overrides = {});

// Workers share run_dir/{index,build.json,verify.json}. The caller saves each
// returned report; verification and timing restore the profile from build.json.
JsonObject BuildNumeric(const NumericProfile &profile,
                        const std::string &run_dir);
JsonObject VerifyNumeric(const std::string &run_dir,
                         const std::string &case_filter = "");
JsonObject QueryNumeric(const std::string &run_dir,
                        const std::string &case_name);


}  // namespace zvec::benchmarks::inverted
