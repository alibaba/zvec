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
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <zvec/ailego/utility/file_helper.h>
#include "shared/index_fixture.h"
#include "shared/json.h"
#include "shared/metrics.h"
#include "dataset.h"


namespace zvec::benchmarks::inverted {


namespace {

// Stream original records; the predicate does not consult the index.
template <typename Predicate>
void VerifyNumericIds(NumericDataset &reader, NumericBatch &batch,
                      const InvertedSearchResult::Ptr &actual,
                      Predicate &&matches) {
  if (!actual) throw std::runtime_error("Index query failed");
  reader.rewind();
  auto iterator = actual->create_iterator();
  uint64_t count = 0;
  while (reader.read_next_batch(batch)) {
    for (size_t i = 0; i < batch.values.size(); ++i) {
      if (!matches(batch.is_null[i] != 0, batch.values[i])) continue;
      const auto id = batch.first_doc_id + static_cast<uint32_t>(i);
      if (!iterator->valid())
        throw std::runtime_error("Missing ID at doc " + std::to_string(id));
      if (iterator->doc_id() != id)
        throw std::runtime_error("Expected doc ID " + std::to_string(id) +
                                 ", got " + std::to_string(iterator->doc_id()));
      ++count;
      iterator->next();
    }
  }
  if (iterator->valid())
    throw std::runtime_error("Unexpected doc ID " +
                             std::to_string(iterator->doc_id()));
  if (actual->count() != count)
    throw std::runtime_error("Incorrect result count");
}

// Verify: pass non-null reader and column, both borrowed from the caller.
// reader scans the original dataset to verify the results returned by column.
// Query: pass nullptr for both. This function opens and owns a read-only index
// for this case, then assigns column for measurement.
template <typename T>
JsonObject ExecuteNumericCase(const NumericProfile &profile,
                              const std::string &index_path,
                              const JsonObject &record, NumericDataset *reader,
                              InvertedColumnIndexer *column) {
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
  JsonObject before;
  std::unique_ptr<IndexFixture> query_index;
  if (!reader) {
    before = MemorySnapshot();
    query_index =
        std::make_unique<IndexFixture>(profile.data_type, index_path, true);
    column = &query_index->column();
  }
  auto execute = [&](auto search, auto predicate) {
    if (reader) {
      NumericBatch batch;
      for (size_t i = 0; i < inputs.size(); ++i) {
        try {
          VerifyNumericIds(*reader, batch, search(i),
                           [&](bool is_null, const std::string &value) {
                             return predicate(is_null, value, i);
                           });
        } catch (const std::exception &error) {
          throw std::runtime_error(
              "verification failed for case '" + name + "', input " +
              (null_query ? "NULL" : std::to_string(operands[i])) + ": " +
              error.what());
        }
      }
      return JsonObject();
    }
    std::vector<uint64_t> match_counts(inputs.size());
    bool collect_counts = true;
    const auto mean_ns_per_query = MeasureRepeatedQueries(inputs.size(), [&]() {
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto result = search(i);
        if (!result) throw std::runtime_error("Numeric query failed: " + name);
        // Keep result counting in the timed workload without validation.
        const volatile auto count = result->count();
        if (collect_counts) match_counts[i] = count;
      }
      // Record counts only during the first measured iteration.
      collect_counts = false;
    });
    const auto after = MemorySnapshot();
    query_index->close();
    auto result = QueryReport(name, mean_ns_per_query, before, after);
    JsonArray counts;
    for (auto count : match_counts) counts.push(JsonValue(count));
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
  if (name == "eq_hit" || name == "eq_hottest" || name == "eq_hot" ||
      name == "eq_cold" || name == "eq_miss") {
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
  if (name == "lt_q01" || name == "lt_q11" || name == "lt_q33" ||
      name == "lt_q50" || name == "lt_q77" || name == "lt_q90" ||
      name == "lt_q99") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::LT); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) < operands[i];
        });
  }
  if (name == "gt_q01" || name == "gt_q11" || name == "gt_q33" ||
      name == "gt_q50" || name == "gt_q77" || name == "gt_q90" ||
      name == "gt_q99") {
    return execute(
        [&](size_t i) { return column->search(encoded[i], CompareOp::GT); },
        [&](bool is_null, const std::string &value, size_t i) {
          return !is_null && DecodeValue<T>(value) > operands[i];
        });
  }
  throw std::invalid_argument("Unknown numeric query case: " + name);
}

JsonObject Build(const std::string &index_path, const std::string &run_dir) {
  auto dataset = NumericDataset::Load(run_dir);
  const auto &profile = dataset.profile();
  NumericBatch batch;
  IndexFixture index(profile.data_type, index_path, false);
  auto &column = index.column();
  const auto insert_begin = Clock::now();
  while (dataset.read_next_batch(batch)) {
    for (size_t i = 0; i < batch.values.size(); ++i) {
      const auto id = batch.first_doc_id + static_cast<uint32_t>(i);
      auto status = batch.is_null[i] ? column.insert_null(id)
                                     : column.insert(id, batch.values[i]);
      if (!status.ok())
        throw std::runtime_error("Numeric insert failed at doc " +
                                 std::to_string(id));
    }
  }
  const auto optimize_begin = Clock::now();
  index.seal();
  const auto optimize_end = Clock::now();
  const auto insert_seconds = Seconds(insert_begin, optimize_begin);
  const auto optimize_seconds = Seconds(optimize_begin, optimize_end);
  const auto total_seconds = insert_seconds + optimize_seconds;
  const auto memory = MemorySnapshot();
  index.close();
  JsonObject result;
  result.set("environment", EnvironmentJson());
  result.set(
      "index_path",
      JsonStringValue(std::filesystem::weakly_canonical(index_path).string()));
  const auto disk = DiskUsage(index_path);
  result.set("index_file_mib", disk["index_file_mib"]);
  result.set("insert_seconds", JsonValue(insert_seconds));
  result.set("optimize_seconds", JsonValue(optimize_seconds));
  result.set("total_seconds", JsonValue(total_seconds));
  JsonObject process_memory;
  process_memory.set("peak_rss_mib", memory["peak_rss_mib"]);
  process_memory.set("final_rss_mib", memory["rss_mib"]);
  result.set("process_memory", process_memory);
  return result;
}

template <typename T>
void Verify(NumericDataset &dataset, const std::string &index_path) {
  const auto &profile = dataset.profile();
  IndexFixture index(profile.data_type, index_path, true);
  for (const auto &record : dataset.query_cases())
    ExecuteNumericCase<T>(profile, index_path, record, &dataset,
                          &index.column());
}


}  // namespace


JsonObject BuildNumeric(const std::string &run_dir) {
  const auto index_path = ailego::FileHelper::PathJoin(run_dir, "index");
  if (std::filesystem::exists(index_path))
    throw std::invalid_argument(
        "index directory already exists: '" + index_path +
        "'\n  use a new --run_dir to start another run");
  return Build(index_path, run_dir);
}

void VerifyNumeric(const std::string &run_dir) {
  const auto index_path = ailego::FileHelper::PathJoin(run_dir, "index");
  const auto build =
      ReadJson(ailego::FileHelper::PathJoin(run_dir, "build.json"));
  if (std::filesystem::weakly_canonical(Text(build, "index_path")) !=
      std::filesystem::weakly_canonical(index_path))
    throw std::invalid_argument("Index path does not match build report");
  auto dataset = NumericDataset::Load(run_dir);
  const auto &profile = dataset.profile();
  if (profile.data_type == DataType::INT32)
    return Verify<int32_t>(dataset, index_path);
  return Verify<int64_t>(dataset, index_path);
}

JsonObject QueryNumeric(const std::string &run_dir) {
  const auto index_path = ailego::FileHelper::PathJoin(run_dir, "index");
  const auto build =
      ReadJson(ailego::FileHelper::PathJoin(run_dir, "build.json"));
  if (std::filesystem::weakly_canonical(Text(build, "index_path")) !=
      std::filesystem::weakly_canonical(index_path))
    throw std::invalid_argument("Index path does not match build report");

  auto dataset = NumericDataset::Load(run_dir);
  const auto &profile = dataset.profile();
  const auto &inputs = dataset.query_cases();

  JsonArray cases;
  for (const auto &record : inputs) {
    auto query = profile.data_type == DataType::INT32
                     ? ExecuteNumericCase<int32_t>(profile, index_path, record,
                                                   nullptr, nullptr)
                     : ExecuteNumericCase<int64_t>(profile, index_path, record,
                                                   nullptr, nullptr);
    cases.push(query);
  }
  JsonObject result;
  result.set("cases", cases);
  return result;
}


}  // namespace zvec::benchmarks::inverted
