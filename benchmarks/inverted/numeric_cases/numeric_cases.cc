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
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "shared/index_fixture.h"
#include "shared/measurement.h"
#include "shared/report_utils.h"
#include "dataset.h"


namespace zvec::benchmarks::inverted {


namespace {

constexpr size_t kQueryPoolSize = 8;
constexpr double kWarmupSeconds = 0.2;
constexpr double kMeasureSeconds = 1.0;

std::string RunPath(const std::string &run_dir, const std::string &name) {
  if (run_dir.empty()) throw std::invalid_argument("run_dir must not be empty");
  return (std::filesystem::path(run_dir) / name).string();
}

std::vector<int64_t> SelectQueryTerms(const std::vector<int64_t> &terms) {
  const size_t count = std::min(kQueryPoolSize, terms.size());
  std::vector<int64_t> selected;
  selected.reserve(count);
  for (size_t i = 0; i < count; ++i)
    selected.push_back(terms[i * terms.size() / count]);
  return selected;
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
    points("eq_hot", hot);
    points("eq_cold", cold);
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
      JsonArray counts, ratios, nonnull_ratios;
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto count =
            VerifyIds(DocumentCount(dataset->profile), search(i),
                      [&](uint32_t id) { return predicate(id, i); });
        counts.push(JsonValue(count));
        ratios.push(JsonValue(static_cast<double>(count) /
                              DocumentCount(dataset->profile)));
        if (range)
          nonnull_ratios.push(DocumentCount(dataset->profile) ==
                                      dataset->null_count
                                  ? JsonValue()
                                  : JsonValue(static_cast<double>(count) /
                                              (DocumentCount(dataset->profile) -
                                               dataset->null_count)));
      }
      JsonObject result;
      result.set("name", JsonStringValue(name));
      result.set("counts", counts);
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
    auto result = QueryReport(name, measurement, kWarmupSeconds,
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
  auto dataset = GenerateDataset(profile);
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
                  const std::string &index_path) {
  if (Number(build["dataset"].as_object(), "seed") != kNumericSeed)
    throw std::invalid_argument("Numeric dataset uses a different fixed seed");
  auto dataset = GenerateDataset(profile);
  IndexFixture index(profile.data_type, index_path, true);
  JsonArray checked;
  for (const auto &record : ReadQueryInputs(build["cases"].as_array()))
    checked.push(ExecuteNumericCase<T>(index_path, record, &dataset,
                                       &index.column(), JsonObject()));
  auto result = Report("verify");
  result.set("passed", JsonValue(true));
  result.set("cases", checked);
  return result;
}


}  // namespace


JsonObject BuildNumeric(const NumericProfile &profile,
                        const std::string &run_dir) {
  ValidateProfile(profile);
  const auto index_path = RunPath(run_dir, "index");
  if (std::filesystem::exists(index_path))
    throw std::invalid_argument(
        "index directory already exists: '" + index_path +
        "'\n  use a new --run_dir to start another run");
  if (profile.data_type == DataType::INT32)
    return Build<int32_t>(profile, index_path);
  return Build<int64_t>(profile, index_path);
}

JsonObject VerifyNumeric(const std::string &run_dir) {
  const auto index_path = RunPath(run_dir, "index");
  const auto build = LoadBuild(RunPath(run_dir, "build.json"), index_path);
  const auto profile = ReadProfile(build["dataset"].as_object());
  if (profile.data_type == DataType::INT32)
    return Verify<int32_t>(profile, build, index_path);
  return Verify<int64_t>(profile, build, index_path);
}

JsonObject QueryNumeric(const std::string &run_dir) {
  const auto index_path = RunPath(run_dir, "index");
  const auto build = LoadBuild(RunPath(run_dir, "build.json"), index_path);
  const auto profile = ReadProfile(build["dataset"].as_object());
  const auto inputs = ReadQueryInputs(build["cases"].as_array());
  // Require successful verification of every case before starting measurement.
  std::vector<JsonObject> verified;
  for (const auto &record : inputs)
    verified.push_back(
        LoadVerification(RunPath(run_dir, "verify.json"), record));
  JsonArray cases;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto query = profile.data_type == DataType::INT32
                     ? ExecuteNumericCase<int32_t>(
                           index_path, inputs[i], nullptr, nullptr, verified[i],
                           DocumentCount(profile))
                     : ExecuteNumericCase<int64_t>(
                           index_path, inputs[i], nullptr, nullptr, verified[i],
                           DocumentCount(profile));
    cases.push(query);
  }
  auto result = Report("query");
  result.set("cases", cases);
  return result;
}


}  // namespace zvec::benchmarks::inverted
