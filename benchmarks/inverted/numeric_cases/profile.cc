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

#include "profile.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace zvec::benchmarks::inverted {


namespace {

std::string TypeName(DataType type) {
  if (type == DataType::INT32) return "int32";
  if (type == DataType::INT64) return "int64";
  throw std::invalid_argument("numeric supports int32 and int64");
}

DataType ParseType(const std::string &name) {
  if (name == "int32") return DataType::INT32;
  if (name == "int64") return DataType::INT64;
  throw std::invalid_argument("Unknown numeric data type: " + name);
}

std::string DistributionName(NumericDistribution distribution) {
  switch (distribution) {
    case NumericDistribution::UNIFORM:
      return "uniform";
    case NumericDistribution::HOTSPOT_MODERATE:
      return "hotspot_moderate";
    case NumericDistribution::HOTSPOT_EXTREME:
      return "hotspot_extreme";
  }
  throw std::invalid_argument("Unknown numeric distribution");
}

NumericDistribution ParseDistribution(const std::string &name) {
  if (name == "uniform") return NumericDistribution::UNIFORM;
  if (name == "hotspot_moderate") return NumericDistribution::HOTSPOT_MODERATE;
  if (name == "hotspot_extreme") return NumericDistribution::HOTSPOT_EXTREME;
  throw std::invalid_argument("Unknown numeric distribution: " + name);
}

std::string RangeName(NumericValueRange range) {
  switch (range) {
    case NumericValueRange::ZERO_TO_TYPE_MAX:
      return "0_to_type_max";
    case NumericValueRange::ZERO_TO_30000:
      return "0_to_30000";
  }
  throw std::invalid_argument("Unknown numeric value range");
}

NumericValueRange ParseRange(const std::string &name) {
  if (name == "0_to_type_max") return NumericValueRange::ZERO_TO_TYPE_MAX;
  if (name == "0_to_30000") return NumericValueRange::ZERO_TO_30000;
  throw std::invalid_argument("Unknown numeric value range: " + name);
}

std::string OrderName(NumericValueOrder order) {
  switch (order) {
    case NumericValueOrder::SHUFFLED:
      return "shuffled";
    case NumericValueOrder::ASCENDING:
      return "ascending";
  }
  throw std::invalid_argument("Unknown numeric value order");
}

NumericValueOrder ParseOrder(const std::string &name) {
  if (name == "shuffled") return NumericValueOrder::SHUFFLED;
  if (name == "ascending") return NumericValueOrder::ASCENDING;
  throw std::invalid_argument("Unknown numeric value order: " + name);
}

double Real(const JsonValue &value, const std::string &name) {
  if (!value.is_integer() && !value.is_float())
    throw std::invalid_argument(name + " must be numeric");
  const double number = value.as_float();
  if (!std::isfinite(number))
    throw std::invalid_argument(name + " must be finite");
  return number;
}

uint32_t Uint32(const JsonValue &value, const std::string &name) {
  if (!value.is_integer() || value.as_integer() < 0 ||
      static_cast<uint64_t>(value.as_integer()) > UINT32_MAX)
    throw std::invalid_argument(name + " must be a uint32");
  return value.as_integer();
}

}  // namespace


int64_t ValueMax(const NumericProfile &profile) {
  return profile.value_range == NumericValueRange::ZERO_TO_30000
             ? 30000
             : (profile.data_type == DataType::INT32 ? INT32_MAX : INT64_MAX);
}

uint32_t NullPercent(NumericNullRatio ratio) {
  switch (ratio) {
    case NumericNullRatio::NONE:
      return 0;
    case NumericNullRatio::TEN_PERCENT:
      return 10;
    case NumericNullRatio::FIFTY_PERCENT:
      return 50;
  }
  throw std::invalid_argument("Unknown numeric NULL ratio");
}

void ValidateProfile(const NumericProfile &profile) {
  if (profile.num_docs != NumericDocCount::ONE_HUNDRED_THOUSAND &&
      profile.num_docs != NumericDocCount::TEN_MILLION)
    throw std::invalid_argument("Numeric num_docs must be 100000 or 10000000");
  TypeName(profile.data_type);
  DistributionName(profile.distribution);
  RangeName(profile.value_range);
  OrderName(profile.value_order);
  NullPercent(profile.null_ratio);
}

JsonObject ProfileJson(const NumericProfile &profile) {
  JsonObject result;
  result.set("name", JsonStringValue(DistributionName(profile.distribution)));
  result.set("type", JsonStringValue(TypeName(profile.data_type)));
  result.set("num_docs", JsonValue(DocumentCount(profile)));
  result.set("distribution",
             JsonStringValue(DistributionName(profile.distribution)));
  result.set("value_range", JsonStringValue(RangeName(profile.value_range)));
  result.set("value_order", JsonStringValue(OrderName(profile.value_order)));
  result.set("value_min", JsonValue(0));
  result.set("value_max", JsonValue(ValueMax(profile)));
  result.set("null_ratio", JsonValue(NullPercent(profile.null_ratio) / 100.0));
  return result;
}

NumericProfile ReadProfile(const JsonObject &description) {
  NumericProfile profile(ParseType(Text(description, "type")));
  profile.num_docs =
      static_cast<NumericDocCount>(Uint32(description["num_docs"], "num_docs"));
  profile.distribution = ParseDistribution(Text(description, "distribution"));
  profile.value_range = ParseRange(Text(description, "value_range"));
  profile.value_order = ParseOrder(Text(description, "value_order"));
  const auto ratio = Real(description["null_ratio"], "null_ratio");
  if (ratio == 0)
    profile.null_ratio = NumericNullRatio::NONE;
  else if (ratio == 0.1)
    profile.null_ratio = NumericNullRatio::TEN_PERCENT;
  else if (ratio == 0.5)
    profile.null_ratio = NumericNullRatio::FIFTY_PERCENT;
  else
    throw std::invalid_argument("Numeric null_ratio must be 0, 0.1, or 0.5");
  ValidateProfile(profile);
  return profile;
}


}  // namespace zvec::benchmarks::inverted
