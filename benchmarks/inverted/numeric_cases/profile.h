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
#include <zvec/db/type.h>
#include "shared/json.h"

namespace zvec::benchmarks::inverted {

enum class NumericDocCount : uint32_t {
  ONE_HUNDRED_THOUSAND = 100000,
  TEN_MILLION = 10000000
};
enum class NumericDistribution { UNIFORM, HOTSPOT_MODERATE, HOTSPOT_EXTREME };
enum class NumericValueRange { ZERO_TO_30000, ZERO_TO_TYPE_MAX };
enum class NumericValueOrder { ASCENDING, SHUFFLED };
enum class NumericNullRatio { NONE, TEN_PERCENT, FIFTY_PERCENT };


struct NumericProfile {
  explicit NumericProfile(DataType type) : data_type(type) {}

  // Indexed data type.
  DataType data_type;
  // Total document count, including NULLs.
  NumericDocCount num_docs{NumericDocCount::ONE_HUNDRED_THOUSAND};
  // Value frequency distribution among non-NULL documents.
  NumericDistribution distribution{NumericDistribution::UNIFORM};
  // Nonnegative domain.
  NumericValueRange value_range{NumericValueRange::ZERO_TO_30000};
  // Value order by doc ID.
  NumericValueOrder value_order{NumericValueOrder::ASCENDING};
  // NULL fraction of all documents.
  NumericNullRatio null_ratio{NumericNullRatio::NONE};
};

inline uint32_t DocumentCount(const NumericProfile &profile) {
  return static_cast<uint32_t>(profile.num_docs);
}

int64_t ValueMax(const NumericProfile &profile);
uint32_t NullPercent(NumericNullRatio ratio);

void ValidateProfile(const NumericProfile &profile);
JsonObject ProfileJson(const NumericProfile &profile);
NumericProfile ReadProfile(const JsonObject &description);

}  // namespace zvec::benchmarks::inverted
