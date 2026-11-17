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

template <typename T, typename Predicate>
std::vector<T> SelectQueryTerms(const std::vector<T> &sorted,
                                Predicate &&matches) {
  // Select evenly spaced distinct-value ranks without storing all candidates.
  size_t terms = 0;
  for (size_t i = 0; i < sorted.size(); ++i)
    if ((i == 0 || sorted[i] != sorted[i - 1]) && matches(sorted[i])) ++terms;
  const size_t count = std::min(kQueryPoolSize, terms);
  std::vector<T> selected;
  selected.reserve(count);
  if (!count) return selected;
  size_t rank = 0;
  for (size_t i = 0; i < sorted.size(); ++i) {
    if ((i > 0 && sorted[i] == sorted[i - 1]) || !matches(sorted[i])) continue;
    if (rank == selected.size() * terms / count) {
      selected.push_back(sorted[i]);
      if (selected.size() == count) break;
    }
    ++rank;
  }
  return selected;
}

template <typename T>
T RangeBoundary(const std::vector<T> &sorted, uint32_t percentile) {
  const uint64_t target = static_cast<uint64_t>(sorted.size()) * percentile;
  uint64_t best_distance = UINT64_MAX;
  T boundary = 0;
  auto consider = [&](T value, uint64_t cumulative) {
    const auto scaled = cumulative * 100;
    const auto distance = scaled > target ? scaled - target : target - scaled;
    if (distance < best_distance) {
      best_distance = distance;
      boundary = value;
    }
  };
  // A distinct value's first position is the number of documents below it.
  for (size_t i = 0; i < sorted.size(); ++i)
    if (i == 0 || sorted[i] != sorted[i - 1]) consider(sorted[i], i);
  const auto largest = sorted.back();
  if (largest < std::numeric_limits<T>::max())
    consider(largest + 1, sorted.size());
  return boundary;
}

template <typename T>
JsonArray SelectQueryInputs(const NumericData<T> &dataset,
                            const std::vector<T> &sorted) {
  JsonArray cases;
  auto points = [&](const std::string &name, const std::vector<T> &values) {
    JsonArray inputs;
    for (auto value : values) inputs.push(JsonValue(value));
    cases.push(InputRecord(name, inputs));
  };
  std::vector<T> hits;
  if (dataset.profile.distribution == NumericDistribution::UNIFORM) {
    hits = SelectQueryTerms(sorted, [](T) { return true; });
    points("eq_hit", hits);
  } else {
    auto is_hot = [&](T value) {
      return std::binary_search(dataset.hot_values.begin(),
                                dataset.hot_values.end(), value);
    };
    hits = SelectQueryTerms(sorted, is_hot);
    points("eq_hot", hits);
    points("eq_cold",
           SelectQueryTerms(sorted, [&](T value) { return !is_hot(value); }));
  }
  JsonArray misses;
  for (int64_t i = 1; i <= static_cast<int64_t>(kQueryPoolSize); ++i)
    misses.push(JsonValue(-i));
  cases.push(InputRecord("eq_miss", misses));
  points("ne_hit", hits);
  for (const auto &[name, percentile] :
       std::vector<std::pair<std::string, uint32_t>>{
           {"lt_q01", 1}, {"lt_q50", 50}, {"lt_q90", 90}, {"ge_q50", 50}}) {
    JsonArray inputs;
    inputs.push(JsonValue(RangeBoundary(sorted, percentile)));
    cases.push(InputRecord(name, inputs));
  }
  JsonArray null_input;
  null_input.push(JsonValue());
  cases.push(InputRecord("is_null", null_input));
  cases.push(InputRecord("is_not_null", null_input));
  return cases;
}

// Stream original records; the predicate does not consult the index.
template <typename Predicate>
uint64_t VerifyNumericIds(NumericDatasetReader &reader, NumericBatch &batch,
                          const InvertedSearchResult::Ptr &actual,
                          Predicate &&matches) {
  if (!actual) throw std::runtime_error("Index query failed");
  reader.Reset();
  auto iterator = actual->create_iterator();
  uint64_t count = 0;
  while (reader.ReadBatch(batch)) {
    for (size_t i = 0; i < batch.values.size(); ++i) {
      if (!matches(batch.is_null[i] != 0, batch.values[i])) continue;
      const auto id = batch.first_doc_id + static_cast<uint32_t>(i);
      if (!iterator->valid() || iterator->doc_id() != id)
        throw std::runtime_error("ID mismatch at doc " + std::to_string(id));
      ++count;
      iterator->next();
    }
  }
  if (iterator->valid() || actual->count() != count)
    throw std::runtime_error("Extra IDs or incorrect result count");
  return count;
}

template <typename T>
JsonObject ExecuteNumericCase(const NumericProfile &profile,
                              const std::string &index_path,
                              const JsonObject &record,
                              NumericDatasetReader *reader,
                              InvertedColumnIndexer *column,
                              const JsonObject &verification) {
  const auto num_docs = DocumentCount(profile);
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
  if (!reader) {
    expected = VerifiedCounts(verification, inputs.size(), num_docs);
    before = MemorySnapshot();
    query_index =
        std::make_unique<IndexFixture>(profile.data_type, index_path, true);
    column = &query_index->column();
  }
  auto execute = [&](auto search, auto predicate, bool range = false) {
    if (reader) {
      const auto null_count = static_cast<uint64_t>(num_docs) *
                              NullPercent(profile.null_ratio) / 100;
      NumericBatch batch;
      JsonArray counts, ratios, nonnull_ratios;
      for (size_t i = 0; i < inputs.size(); ++i) {
        const auto count =
            VerifyNumericIds(*reader, batch, search(i),
                             [&](bool is_null, const std::string &value) {
                               return predicate(is_null, value, i);
                             });
        counts.push(JsonValue(count));
        ratios.push(JsonValue(static_cast<double>(count) / num_docs));
        if (range)
          nonnull_ratios.push(num_docs == null_count
                                  ? JsonValue()
                                  : JsonValue(static_cast<double>(count) /
                                              (num_docs - null_count)));
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
        [&](bool is_null, const std::string &, size_t) { return is_null; });
  }
  if (name == "is_not_null") {
    return execute(
        [&](size_t) { return column->search_non_null(); },
        [&](bool is_null, const std::string &, size_t) { return !is_null; });
  }
  if (name == "eq_hit" || name == "eq_hot" || name == "eq_cold" ||
      name == "eq_miss") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::EQ); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) == operands[i];
        });
  }
  if (name == "ne_hit") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::NE); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) != operands[i];
        });
  }
  if (name == "lt_q01" || name == "lt_q50" || name == "lt_q90") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::LT); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) < operands[i];
        },
        true);
  }
  if (name == "ge_q50") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::GE); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) >= operands[i];
        },
        true);
  }
  throw std::invalid_argument("Unknown numeric query case: " + name);
}

template <typename T>
JsonObject Build(const NumericProfile &profile, const std::string &index_path,
                 const std::string &data_path) {
  JsonObject description;
  JsonArray inputs;
  // No generation arrays survive into the insertion phase.
  {
    auto dataset = GenerateDataset<T>(profile);
    std::vector<T> sorted_copy;
    const auto *sorted = &dataset.values;
    if (profile.value_order == NumericValueOrder::SHUFFLED) {
      sorted_copy = dataset.values;
      std::sort(sorted_copy.begin(), sorted_copy.end());
      sorted = &sorted_copy;
    }
    description = DescribeDataset(dataset, *sorted);
    inputs = SelectQueryInputs(dataset, *sorted);
    WriteDataset(dataset, data_path);
  }
  NumericDatasetReader reader(profile, data_path);
  NumericBatch batch;
  batch.is_null.resize(kNumericBatchSize);
  batch.values.resize(kNumericBatchSize);
  const auto before = MemorySnapshot();
  IndexFixture index(profile.data_type, index_path, false);
  auto &column = index.column();
  double insert_seconds = 0;
  while (reader.ReadBatch(batch)) {
    const auto begin = Clock::now();
    for (size_t i = 0; i < batch.values.size(); ++i) {
      const auto id = batch.first_doc_id + static_cast<uint32_t>(i);
      auto status = batch.is_null[i] ? column.insert_null(id)
                                     : column.insert(id, batch.values[i]);
      if (!status.ok())
        throw std::runtime_error("Numeric insert failed at doc " +
                                 std::to_string(id));
    }
    insert_seconds += Seconds(begin, Clock::now());
  }
  const auto seal_begin = Clock::now();
  index.Seal();
  const auto seal_seconds = Seconds(seal_begin, Clock::now());
  const auto build_seconds = insert_seconds + seal_seconds;
  const auto after = MemorySnapshot();
  index.Close();
  auto result = Report("build");
  result.set("dataset", description);
  result.set("cases", inputs);
  result.set(
      "index_path",
      JsonStringValue(std::filesystem::weakly_canonical(index_path).string()));
  result.set("insert_seconds", JsonValue(insert_seconds));
  result.set("seal_seconds", JsonValue(seal_seconds));
  result.set("build_seconds", JsonValue(build_seconds));
  result.set("docs_per_second",
             JsonValue(DocumentCount(profile) / build_seconds));
  const auto disk = DiskUsage(index_path);
  result.set("index_file_bytes", disk["index_file_bytes"]);
  result.set("index_allocated_bytes", disk["index_allocated_bytes"]);
  result.set("bytes_per_doc",
             JsonValue(static_cast<double>(Number(disk, "index_file_bytes")) /
                       DocumentCount(profile)));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  result.set("metric_names", BuildMetricNames());
  return result;
}

template <typename T>
JsonObject Verify(const NumericProfile &profile, const JsonObject &build,
                  const std::string &index_path, const std::string &data_path) {
  NumericDatasetReader reader(profile, data_path);
  IndexFixture index(profile.data_type, index_path, true);
  JsonArray checked;
  for (const auto &record : ReadQueryInputs(build["cases"].as_array()))
    checked.push(ExecuteNumericCase<T>(profile, index_path, record, &reader,
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
  const auto data_path = RunPath(run_dir, "data.bin");
  if (std::filesystem::exists(index_path))
    throw std::invalid_argument(
        "index directory already exists: '" + index_path +
        "'\n  use a new --run_dir to start another run");
  if (std::filesystem::exists(data_path))
    throw std::invalid_argument(
        "dataset file already exists: '" + data_path +
        "'\n  use a new --run_dir to start another run");
  if (profile.data_type == DataType::INT32)
    return Build<int32_t>(profile, index_path, data_path);
  return Build<int64_t>(profile, index_path, data_path);
}

JsonObject VerifyNumeric(const std::string &run_dir) {
  const auto index_path = RunPath(run_dir, "index");
  const auto build = LoadBuild(RunPath(run_dir, "build.json"), index_path);
  const auto profile = ReadProfile(build["dataset"].as_object());
  const auto data_path = RunPath(run_dir, "data.bin");
  if (profile.data_type == DataType::INT32)
    return Verify<int32_t>(profile, build, index_path, data_path);
  return Verify<int64_t>(profile, build, index_path, data_path);
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
    auto query =
        profile.data_type == DataType::INT32
            ? ExecuteNumericCase<int32_t>(profile, index_path, inputs[i],
                                          nullptr, nullptr, verified[i])
            : ExecuteNumericCase<int64_t>(profile, index_path, inputs[i],
                                          nullptr, nullptr, verified[i]);
    cases.push(query);
  }
  auto result = Report("query");
  result.set("cases", cases);
  return result;
}


}  // namespace zvec::benchmarks::inverted
