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

#include "numeric_cases.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "shared/fingerprint.h"
#include "shared/index_fixture.h"
#include "shared/measurement.h"
#include "shared/random.h"
#include "shared/report_utils.h"

namespace zvec::benchmarks::inverted {
namespace {

constexpr size_t kQueryPoolSize = 8;
constexpr uint64_t kSeed = 20261115;
constexpr double kWarmupSeconds = 0.2;
constexpr double kMeasureSeconds = 1.0;

std::string RunPath(const std::string &run_dir, const std::string &name) {
  if (run_dir.empty()) throw std::invalid_argument("run_dir must not be empty");
  return (std::filesystem::path(run_dir) / name).string();
}

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

uint32_t DocumentCount(const NumericProfile &profile) {
  return static_cast<uint32_t>(profile.num_docs);
}

std::string DistributionName(NumericDistribution distribution) {
  switch (distribution) {
    case NumericDistribution::UNIFORM:
      return "uniform";
    case NumericDistribution::HOTSPOT:
      return "hotspot";
  }
  throw std::invalid_argument("Unknown numeric distribution");
}

NumericDistribution ParseDistribution(const std::string &name) {
  if (name == "uniform") return NumericDistribution::UNIFORM;
  if (name == "hotspot") return NumericDistribution::HOTSPOT;
  throw std::invalid_argument("Unknown numeric distribution: " + name);
}

std::string RangeName(NumericValueRange range) {
  switch (range) {
    case NumericValueRange::ZERO_TO_TYPE_MAX:
      return "zero_to_type_max";
    case NumericValueRange::ZERO_TO_30000:
      return "zero_to_30000";
  }
  throw std::invalid_argument("Unknown numeric value range");
}

NumericValueRange ParseRange(const std::string &name) {
  if (name == "zero_to_type_max") return NumericValueRange::ZERO_TO_TYPE_MAX;
  if (name == "zero_to_30000") return NumericValueRange::ZERO_TO_30000;
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

int64_t ValueMax(const NumericProfile &profile) {
  return profile.value_range == NumericValueRange::ZERO_TO_30000
             ? 30000
             : (profile.data_type == DataType::INT32 ? INT32_MAX : INT64_MAX);
}

uint64_t DomainSize(const NumericProfile &profile) {
  // The nonnegative INT64 domain has 2^63 values; count it in uint64_t.
  return static_cast<uint64_t>(ValueMax(profile)) + 1;
}

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

std::vector<int64_t> SelectQueryTerms(const std::vector<int64_t> &terms) {
  const size_t count = std::min(kQueryPoolSize, terms.size());
  std::vector<int64_t> selected;
  selected.reserve(count);
  for (size_t i = 0; i < count; ++i)
    selected.push_back(terms[i * terms.size() / count]);
  return selected;
}

struct NumericData {
  NumericProfile profile;
  uint64_t null_count{0};
  std::vector<uint8_t> is_null;
  std::vector<std::string> values;
  // Sorted observed value/count pairs; no storage proportional to the domain.
  std::vector<std::pair<int64_t, uint64_t>> frequencies;
  std::vector<int64_t> hot_values;
  std::string fingerprint;
};

constexpr const char *kProfiles[]{"uniform", "hotspot"};
constexpr const char *kProfileFields[]{
    "num_docs",   "distribution",    "value_range",  "value_order",
    "null_ratio", "hot_value_count", "hot_doc_ratio"};

std::vector<std::string> CaseNames(const NumericProfile &profile) {
  std::vector<std::string> names =
      profile.distribution == NumericDistribution::UNIFORM
          ? std::vector<std::string>{"eq_hit"}
          : std::vector<std::string>{"eq_hot", "eq_cold"};
  for (const auto *name : {"eq_miss", "ne_hit", "lt_q01", "lt_q50", "lt_q90",
                           "ge_q50", "is_null", "is_not_null"})
    names.emplace_back(name);
  return names;
}

double Real(const JsonValue &value, const std::string &name) {
  if (!value.is_integer() && !value.is_float())
    throw std::invalid_argument(name + " must be numeric");
  const double number = value.is_integer()
                            ? static_cast<double>(value.as_integer())
                            : value.as_float();
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

void ValidateProfile(const NumericProfile &profile) {
  TypeName(profile.data_type);
  DistributionName(profile.distribution);
  RangeName(profile.value_range);
  OrderName(profile.value_order);
  NullPercent(profile.null_ratio);
  if (profile.num_docs != NumericDocCount::ONE_HUNDRED_THOUSAND &&
      profile.num_docs != NumericDocCount::TEN_MILLION)
    throw std::invalid_argument("Numeric num_docs must be 100000 or 10000000");
  if (!std::isfinite(profile.hot_doc_ratio))
    throw std::invalid_argument("hot_doc_ratio must be finite");
  if (profile.distribution == NumericDistribution::HOTSPOT &&
      (!profile.hot_value_count ||
       profile.hot_value_count >= DomainSize(profile) ||
       !(profile.hot_doc_ratio > 0 && profile.hot_doc_ratio < 1)))
    throw std::invalid_argument(
        "Hotspot needs hot and cold values and a document ratio in (0, 1)");
}

// Decode CLI/report fields once; workload functions use NumericProfile
// directly.
void ApplyOverrides(NumericProfile *profile, const JsonObject &overrides) {
  for (const auto &entry : overrides) {
    const auto key = entry.key().decode().as_stl_string();
    const auto value = entry.value();
    if (key == "num_docs") {
      profile->num_docs = static_cast<NumericDocCount>(Uint32(value, key));
    } else if (key == "distribution") {
      profile->distribution = ParseDistribution(JsonStringText(value));
    } else if (key == "value_range") {
      profile->value_range = ParseRange(JsonStringText(value));
    } else if (key == "value_order") {
      profile->value_order = ParseOrder(JsonStringText(value));
    } else if (key == "null_ratio") {
      const auto ratio = Real(value, key);
      if (ratio == 0)
        profile->null_ratio = NumericNullRatio::NONE;
      else if (ratio == 0.1)
        profile->null_ratio = NumericNullRatio::TEN_PERCENT;
      else if (ratio == 0.5)
        profile->null_ratio = NumericNullRatio::FIFTY_PERCENT;
      else
        throw std::invalid_argument(
            "Numeric null_ratio must be 0, 0.1, or 0.5");
    } else if (key == "hot_value_count") {
      profile->hot_value_count = Uint32(value, key);
    } else if (key == "hot_doc_ratio") {
      profile->hot_doc_ratio = Real(value, key);
    } else {
      throw std::invalid_argument("Unknown numeric profile field: " + key);
    }
  }
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
  result.set("hot_value_count", JsonValue(profile.hot_value_count));
  result.set("hot_doc_ratio", JsonValue(profile.hot_doc_ratio));
  result.set("supplemental", JsonValue(false));
  return result;
}

NumericProfile ReadProfile(const JsonObject &description) {
  NumericProfile profile(ParseType(Text(description, "type")));
  JsonObject fields;
  for (const auto *key : kProfileFields) fields.set(key, description[key]);
  ApplyOverrides(&profile, fields);
  ValidateProfile(profile);
  if (Number(description, "seed") != kSeed)
    throw std::invalid_argument("Numeric dataset uses a different fixed seed");
  return profile;
}

std::string DatasetFingerprint(const NumericData &dataset) {
  Fingerprint fingerprint;
  fingerprint.AddString("inverted-numeric-data-v2");
  fingerprint.AddString(
      JsonValue(ProfileJson(dataset.profile)).as_json_string().as_stl_string());
  fingerprint.AddUint64(kSeed);
  for (uint32_t id = 0; id < DocumentCount(dataset.profile); ++id) {
    fingerprint.AddByte(dataset.is_null[id]);
    fingerprint.AddString(dataset.values[id]);
  }
  return fingerprint.Finish();
}

std::vector<int64_t> SelectHotValues(const NumericProfile &profile) {
  // Floyd's sampling without replacement takes one draw per selected value,
  // even when the hot set nearly fills the domain. No full-domain array.
  Random random(kSeed ^ 0xe7037ed1a0b428dbULL);
  std::set<int64_t> selected;
  const auto size = DomainSize(profile);
  for (uint64_t last = size - profile.hot_value_count; last < size; ++last) {
    const auto value = static_cast<int64_t>(random.bounded(last + 1));
    selected.insert(selected.count(value) ? static_cast<int64_t>(last) : value);
  }
  return {selected.begin(), selected.end()};
}

int64_t ColdValue(uint64_t rank, const std::vector<int64_t> &hot_values) {
  // A cold rank counts only values outside the hot set. Each hot value at or
  // before that rank shifts the actual value by one; consecutive hot values
  // share the same compressed position. Binary search skips them together.
  size_t begin = 0, end = hot_values.size();
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2;
    if (static_cast<uint64_t>(hot_values[middle]) - middle <= rank)
      begin = middle + 1;
    else
      end = middle;
  }
  return static_cast<int64_t>(rank + begin);
}

template <typename T>
NumericData GenerateDataset(const NumericProfile &profile) {
  NumericData dataset{profile};
  dataset.null_count = static_cast<uint64_t>(DocumentCount(profile)) *
                       NullPercent(profile.null_ratio) / 100;
  dataset.is_null.resize(DocumentCount(profile), 0);
  std::fill_n(dataset.is_null.begin(), dataset.null_count, uint8_t{1});
  if (dataset.null_count)
    Random(kSeed ^ 0xa0761d6478bd642fULL).shuffle(dataset.is_null);
  const auto nonnull = DocumentCount(profile) - dataset.null_count;
  const bool hotspot = profile.distribution == NumericDistribution::HOTSPOT;
  if (hotspot) dataset.hot_values = SelectHotValues(profile);
  const uint64_t hot_rows =
      !hotspot ? 0
               : (profile.hot_doc_ratio == 0.8
                      ? nonnull * 4 / 5
                      : static_cast<uint64_t>(nonnull * profile.hot_doc_ratio));

  // bounded() uses integer rejection sampling, including the full 2^63 domain.
  Random random(kSeed ^ 0x8ebc6af09c88c6e3ULL);
  std::vector<int64_t> values(nonnull);
  for (uint64_t i = 0; i < nonnull; ++i) {
    if (!hotspot)
      values[i] = random.bounded(DomainSize(profile));
    else if (i < hot_rows)
      values[i] = dataset.hot_values[random.bounded(dataset.hot_values.size())];
    else
      values[i] = ColdValue(
          random.bounded(DomainSize(profile) - dataset.hot_values.size()),
          dataset.hot_values);
  }
  auto sorted = values;
  std::sort(sorted.begin(), sorted.end());
  for (auto value : sorted) {
    if (dataset.frequencies.empty() ||
        dataset.frequencies.back().first != value)
      dataset.frequencies.emplace_back(value, 1);
    else
      ++dataset.frequencies.back().second;
  }
  if (profile.value_order == NumericValueOrder::ASCENDING)
    values.swap(sorted);
  else
    Random(kSeed ^ 0x589965cc75374cc3ULL).shuffle(values);
  dataset.values.resize(DocumentCount(profile));
  size_t position = 0;
  for (uint32_t id = 0; id < DocumentCount(profile); ++id)
    if (!dataset.is_null[id])
      dataset.values[id] = EncodeValue(static_cast<T>(values[position++]));
  dataset.fingerprint = DatasetFingerprint(dataset);
  return dataset;
}

template <typename T>
T RangeBoundary(const NumericData &dataset, uint32_t percentile) {
  const uint64_t target =
      (DocumentCount(dataset.profile) - dataset.null_count) * percentile;
  uint64_t cumulative = 0, best_distance = UINT64_MAX;
  T boundary = 0;
  auto consider = [&](int64_t value) {
    const auto scaled = cumulative * 100;
    const auto distance = scaled > target ? scaled - target : target - scaled;
    if (distance < best_distance) {
      best_distance = distance;
      boundary = static_cast<T>(value);
    }
  };
  for (const auto &[value, count] : dataset.frequencies) {
    consider(value);
    cumulative += count;
  }
  const auto largest = dataset.frequencies.back().first;
  if (largest < std::numeric_limits<T>::max()) consider(largest + 1);
  return boundary;
}

template <typename T>
JsonArray SelectQueryInputs(const NumericData &dataset) {
  std::vector<int64_t> observed, hot, cold;
  for (const auto &[value, count] : dataset.frequencies) {
    observed.push_back(value);
    if (std::binary_search(dataset.hot_values.begin(), dataset.hot_values.end(),
                           value))
      hot.push_back(value);
    else
      cold.push_back(value);
  }
  JsonArray cases;
  auto points = [&](const std::string &name,
                    const std::vector<int64_t> &values) {
    JsonArray inputs;
    for (auto value : SelectQueryTerms(values)) inputs.push(JsonValue(value));
    cases.push(InputRecord(name, inputs));
  };
  if (dataset.profile.distribution == NumericDistribution::UNIFORM) {
    points("eq_hit", observed);
  } else {
    // Tiny workloads may round one group to zero documents; query its known
    // values anyway and report the actual zero count rather than inventing
    // hits.
    points("eq_hot", hot.empty()
                         ? std::vector<int64_t>{dataset.hot_values.front()}
                         : hot);
    points("eq_cold",
           cold.empty() ? std::vector<int64_t>{ColdValue(0, dataset.hot_values)}
                        : cold);
  }
  JsonArray misses;
  for (int64_t i = 1; i <= static_cast<int64_t>(kQueryPoolSize); ++i)
    misses.push(JsonValue(-i));
  cases.push(InputRecord("eq_miss", misses));
  points("ne_hit", hot.empty() ? observed : hot);
  for (const auto &[name, percentile] :
       std::vector<std::pair<std::string, uint32_t>>{
           {"lt_q01", 1}, {"lt_q50", 50}, {"lt_q90", 90}, {"ge_q50", 50}}) {
    JsonArray inputs;
    inputs.push(JsonValue(RangeBoundary<T>(dataset, percentile)));
    cases.push(InputRecord(name, inputs));
  }
  JsonArray null_input;
  null_input.push(JsonValue());
  cases.push(InputRecord("is_null", null_input));
  cases.push(InputRecord("is_not_null", null_input));
  return cases;
}

JsonObject DescribeDataset(const NumericData &dataset) {
  auto description = ProfileJson(dataset.profile);
  description.set("seed", JsonValue(kSeed));
  description.set("data_fingerprint", JsonStringValue(dataset.fingerprint));
  JsonObject stats, percentiles;
  Fingerprint frequencies;
  std::vector<uint64_t> counts;
  uint64_t hot_docs = 0;
  for (const auto &[value, count] : dataset.frequencies) {
    frequencies.AddUint64(static_cast<uint64_t>(value));
    frequencies.AddUint64(count);
    counts.push_back(count);
    if (std::binary_search(dataset.hot_values.begin(), dataset.hot_values.end(),
                           value))
      hot_docs += count;
  }
  std::sort(counts.begin(), counts.end());
  percentiles.set("min", JsonValue(counts.front()));
  percentiles.set("p50", JsonValue(counts[(counts.size() - 1) / 2]));
  percentiles.set("p95", JsonValue(counts[(counts.size() - 1) * 95 / 100]));
  percentiles.set("max", JsonValue(counts.back()));
  stats.set("observed_cardinality",
            JsonValue(static_cast<uint64_t>(counts.size())));
  stats.set("observed_term_df", percentiles);
  stats.set("observed_value_min", JsonValue(dataset.frequencies.front().first));
  stats.set("observed_value_max", JsonValue(dataset.frequencies.back().first));
  stats.set("value_frequencies_fingerprint",
            JsonStringValue(frequencies.Finish()));
  stats.set("null_count", JsonValue(dataset.null_count));
  JsonArray hot_values;
  for (auto value : dataset.hot_values) hot_values.push(JsonValue(value));
  stats.set("hot_values", hot_values);
  stats.set("hot_doc_count", JsonValue(hot_docs));
  description.set("stats", stats);
  return description;
}

template <typename T>
JsonObject ExecuteNumericCase(const std::string &index_path,
                              const JsonObject &record,
                              const NumericData *dataset,
                              InvertedColumnIndexer *column,
                              const JsonObject &verification,
                              uint32_t num_docs = 0) {
  const auto name = Text(record, "name");
  const auto inputs = record["inputs"].as_array();
  const bool null_query = name == "is_null" || name == "is_not_null";
  std::vector<T> operands;
  std::vector<std::string> encoded;
  if (null_query) {
    if (inputs.size() != 1 || !inputs[0].is_null()) {
      throw std::invalid_argument(
          "NULL query requires one null parameter marker");
    }
  } else {
    for (const auto &input : inputs) {
      if (!input.is_integer() ||
          input.as_integer() < std::numeric_limits<T>::min() ||
          input.as_integer() > std::numeric_limits<T>::max()) {
        throw std::invalid_argument(
            "Numeric query operand exceeds data type bounds");
      }
      operands.push_back(static_cast<T>(input.as_integer()));
      encoded.push_back(EncodeValue(operands.back()));
    }
  }
  std::vector<uint64_t> expected;
  JsonObject before;
  std::unique_ptr<IndexFixture> query_index;
  if (!dataset) {
    const auto type =
        std::is_same_v<T, int32_t> ? DataType::INT32 : DataType::INT64;
    expected = VerifiedCounts(verification, inputs.size(), num_docs);
    before = MemorySnapshot();
    query_index = std::make_unique<IndexFixture>(type, index_path, true);
    column = &query_index->column();
  }
  auto execute = [&](auto search, auto predicate, bool range = false) {
    if (dataset) {
      JsonArray counts, fingerprints, ratios, nonnull_ratios;
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto verified =
            VerifyIds(DocumentCount(dataset->profile), search(i),
                      [&](uint32_t id) { return predicate(id, i); });
        counts.push(JsonValue(verified.count));
        fingerprints.push(JsonStringValue(verified.fingerprint));
        ratios.push(JsonValue(static_cast<double>(verified.count) /
                              DocumentCount(dataset->profile)));
        if (range)
          nonnull_ratios.push(
              DocumentCount(dataset->profile) == dataset->null_count
                  ? JsonValue()
                  : JsonValue(static_cast<double>(verified.count) /
                              (DocumentCount(dataset->profile) -
                               dataset->null_count)));
      }
      JsonObject result;
      result.set("name", JsonStringValue(name));
      result.set("query_fingerprint", record["query_fingerprint"]);
      result.set("counts", counts);
      result.set("id_fingerprints", fingerprints);
      result.set("match_ratios", ratios);
      if (range) result.set("nonnull_match_ratios", nonnull_ratios);
      return result;
    }
    const auto measurement =
        MeasurePool(inputs.size(), kWarmupSeconds, kMeasureSeconds, [&]() {
          uint64_t checksum = 0;
          for (size_t i = 0; i < inputs.size(); ++i) {
            auto result = search(i);
            if (!result)
              throw std::runtime_error("Numeric query failed: " + name);
            const auto count = result->count();
            if (count != expected[i]) {
              throw std::runtime_error(
                  "Numeric query count changed during timing: " + name);
            }
            checksum += count;
          }
          return checksum;
        });
    const auto after = MemorySnapshot();
    query_index->Close();
    auto result = QueryReport(name, inputs, measurement, kWarmupSeconds,
                              kMeasureSeconds, before, after);
    JsonArray counts;
    for (auto count : expected) counts.push(JsonValue(count));
    result.set("match_counts", counts);
    return result;
  };
  if (name == "is_null") {
    return execute(
        [&](size_t) { return column->search_null(); },
        [&](uint32_t id, size_t) { return dataset->is_null[id] != 0; });
  }
  if (name == "is_not_null") {
    return execute(
        [&](size_t) { return column->search_non_null(); },
        [&](uint32_t id, size_t) { return dataset->is_null[id] == 0; });
  }
  if (name == "eq_hit" || name == "eq_hot" || name == "eq_cold" ||
      name == "eq_miss") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::EQ); },
        [&](uint32_t id, size_t i) {
          return !dataset->is_null[id] &&
                 DecodeValue<T>(dataset->values[id]) == operands[i];
        });
  }
  if (name == "ne_hit") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::NE); },
        [&](uint32_t id, size_t i) {
          return !dataset->is_null[id] &&
                 DecodeValue<T>(dataset->values[id]) != operands[i];
        });
  }
  if (name == "lt_q01" || name == "lt_q50" || name == "lt_q90") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::LT); },
        [&](uint32_t id, size_t i) {
          return !dataset->is_null[id] &&
                 DecodeValue<T>(dataset->values[id]) < operands[i];
        },
        true);
  }
  if (name == "ge_q50") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::GE); },
        [&](uint32_t id, size_t i) {
          return !dataset->is_null[id] &&
                 DecodeValue<T>(dataset->values[id]) >= operands[i];
        },
        true);
  }
  throw std::invalid_argument("Unknown numeric query case: " + name);
}

template <typename T>
JsonObject Build(const NumericProfile &profile, const std::string &index_path) {
  auto dataset = GenerateDataset<T>(profile);
  const auto inputs = SelectQueryInputs<T>(dataset);
  const auto before = MemorySnapshot();
  IndexFixture index(profile.data_type, index_path, false);
  auto &column = index.column();
  const auto begin = Clock::now();
  for (uint32_t id = 0; id < DocumentCount(dataset.profile); ++id) {
    auto status = dataset.is_null[id] ? column.insert_null(id)
                                      : column.insert(id, dataset.values[id]);
    if (!status.ok())
      throw std::runtime_error("Numeric insert failed at doc " +
                               std::to_string(id));
  }
  const auto inserted = Clock::now();
  index.Seal();
  const auto sealed = Clock::now();
  const auto after = MemorySnapshot();
  index.Close();
  auto result = Report("build");
  result.set("dataset", DescribeDataset(dataset));
  result.set("cases", inputs);
  result.set(
      "index_path",
      JsonStringValue(std::filesystem::weakly_canonical(index_path).string()));
  result.set("insert_seconds", JsonValue(Seconds(begin, inserted)));
  result.set("seal_seconds", JsonValue(Seconds(inserted, sealed)));
  result.set("build_seconds", JsonValue(Seconds(begin, sealed)));
  result.set("docs_per_second", JsonValue(DocumentCount(dataset.profile) /
                                          Seconds(begin, sealed)));
  const auto disk = DiskUsage(index_path);
  result.set("index_file_bytes", disk["index_file_bytes"]);
  result.set("index_allocated_bytes", disk["index_allocated_bytes"]);
  result.set("bytes_per_doc",
             JsonValue(static_cast<double>(Number(disk, "index_file_bytes")) /
                       DocumentCount(dataset.profile)));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  result.set("metric_names", BuildMetricNames());
  return result;
}

template <typename T>
JsonObject Verify(const NumericProfile &profile, const JsonObject &build,
                  const std::string &index_path,
                  const std::string &case_filter) {
  auto dataset = GenerateDataset<T>(profile);
  CheckRegenerated(build, dataset.fingerprint, SelectQueryInputs<T>(dataset));
  IndexFixture index(profile.data_type, index_path, true);
  JsonArray checked;
  for (const auto &record :
       SelectInputs(build["cases"].as_array(), case_filter))
    checked.push(ExecuteNumericCase<T>(index_path, record, &dataset,
                                       &index.column(), JsonObject()));
  auto result = Report("verify");
  result.set("passed", JsonValue(true));
  result.set("data_fingerprint", JsonStringValue(dataset.fingerprint));
  result.set("cases", checked);
  return result;
}
}  // namespace

NumericProfile ResolveNumericProfile(DataType type, const std::string &preset,
                                     const JsonObject &overrides) {
  NumericProfile profile(type);
  profile.distribution = ParseDistribution(preset);
  ApplyOverrides(&profile, overrides);
  if (profile.distribution != ParseDistribution(preset))
    throw std::invalid_argument(
        "Select the distribution with --profile, not a conflicting config "
        "override");
  ValidateProfile(profile);
  return profile;
}

JsonObject NumericCatalog(DataType type, const std::string &preset,
                          const JsonObject &overrides) {
  TypeName(type);
  if (!preset.empty()) ParseDistribution(preset);
  JsonObject result;
  result.set("name", JsonStringValue("numeric"));
  JsonArray profiles;
  for (const auto *name : kProfiles) {
    if (!preset.empty() && preset != name) continue;
    const auto profile = ResolveNumericProfile(type, name, overrides);
    auto record = ProfileJson(profile);
    record["name"] = JsonStringValue(name);
    JsonArray cases;
    for (const auto &name : CaseNames(profile))
      cases.push(JsonStringValue(name));
    record.set("cases", cases);
    profiles.push(record);
  }
  result.set("profiles", profiles);
  return result;
}

JsonObject BuildNumeric(const NumericProfile &profile,
                        const std::string &run_dir) {
  ValidateProfile(profile);
  const auto index_path = RunPath(run_dir, "index");
  if (std::filesystem::exists(index_path))
    throw std::invalid_argument("Numeric build requires a new index directory");
  if (profile.data_type == DataType::INT32)
    return Build<int32_t>(profile, index_path);
  return Build<int64_t>(profile, index_path);
}

JsonObject VerifyNumeric(const std::string &run_dir,
                         const std::string &case_filter) {
  const auto index_path = RunPath(run_dir, "index");
  const auto build = LoadBuild(RunPath(run_dir, "build.json"), index_path);
  const auto profile = ReadProfile(build["dataset"].as_object());
  if (profile.data_type == DataType::INT32)
    return Verify<int32_t>(profile, build, index_path, case_filter);
  return Verify<int64_t>(profile, build, index_path, case_filter);
}

JsonObject QueryNumeric(const std::string &run_dir,
                        const std::string &case_name) {
  const auto index_path = RunPath(run_dir, "index");
  const auto build = LoadBuild(RunPath(run_dir, "build.json"), index_path);
  const auto profile = ReadProfile(build["dataset"].as_object());
  const auto selected = SelectInputs(build["cases"].as_array(), case_name);
  if (selected.size() != 1)
    throw std::invalid_argument("query requires exactly one case");
  const auto verified = LoadVerification(RunPath(run_dir, "verify.json"), build,
                                         selected.front());
  auto result = profile.data_type == DataType::INT32
                    ? ExecuteNumericCase<int32_t>(index_path, selected.front(),
                                                  nullptr, nullptr, verified,
                                                  DocumentCount(profile))
                    : ExecuteNumericCase<int64_t>(index_path, selected.front(),
                                                  nullptr, nullptr, verified,
                                                  DocumentCount(profile));
  result.set("data_fingerprint",
             build["dataset"].as_object()["data_fingerprint"]);
  return result;
}
}  // namespace zvec::benchmarks::inverted
