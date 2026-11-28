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

CompareOp NumericCompareOp(const std::string &name) {
  if (name == "is_null") return CompareOp::IS_NULL;
  if (name == "is_not_null") return CompareOp::IS_NOT_NULL;
  if (name == "eq_hit" || name == "eq_hottest" || name == "eq_hot" ||
      name == "eq_cold" || name == "eq_miss")
    return CompareOp::EQ;
  if (name == "ne_hit") return CompareOp::NE;
  if (name == "lt_q01" || name == "lt_q11" || name == "lt_q33" ||
      name == "lt_q50" || name == "lt_q77" || name == "lt_q90" ||
      name == "lt_q99")
    return CompareOp::LT;
  if (name == "gt_q01" || name == "gt_q11" || name == "gt_q33" ||
      name == "gt_q50" || name == "gt_q77" || name == "gt_q90" ||
      name == "gt_q99")
    return CompareOp::GT;
  throw std::invalid_argument("Unknown numeric query case: " + name);
}

// NULL queries validate their marker and have no numeric operands.
template <typename T>
std::vector<T> ReadNumericOperands(const JsonArray &inputs, CompareOp op) {
  if (op == CompareOp::IS_NULL || op == CompareOp::IS_NOT_NULL) {
    if (inputs.size() != 1 || !inputs[0].is_null()) {
      throw std::invalid_argument(
          "NULL query requires one null parameter marker");
    }
    return {};
  }
  std::vector<T> operands;
  for (const auto &input : inputs) {
    if (!input.is_integer() ||
        input.as_integer() < std::numeric_limits<T>::min() ||
        input.as_integer() > std::numeric_limits<T>::max()) {
      throw std::invalid_argument(
          "Numeric query operand exceeds data type bounds");
    }
    operands.push_back(static_cast<T>(input.as_integer()));
  }
  return operands;
}

template <typename T>
void VerifyNumericCase(const JsonObject &record, NumericDataset &dataset,
                       InvertedColumnIndexer &column) {
  const auto name = Text(record, "name");
  const auto inputs = record["inputs"].as_array();
  const auto op = NumericCompareOp(name);
  const auto operands = ReadNumericOperands<T>(inputs, op);
  NumericBatch batch;
  for (size_t i = 0; i < inputs.size(); ++i) {
    try {
      if (op == CompareOp::IS_NULL) {
        VerifyNumericIds(
            dataset, batch, column.search_null(),
            [](bool is_null, const std::string &) { return is_null; });
      } else if (op == CompareOp::IS_NOT_NULL) {
        VerifyNumericIds(
            dataset, batch, column.search_non_null(),
            [](bool is_null, const std::string &) { return !is_null; });
      } else {
        const auto result = column.search(EncodeValue(operands[i]), op);
        VerifyNumericIds(
            dataset, batch, result,
            [&](bool is_null, const std::string &bytes) {
              if (is_null) return false;
              const auto value = DecodeValue<T>(bytes);
              switch (op) {
                case CompareOp::EQ:
                  return value == operands[i];
                case CompareOp::NE:
                  return value != operands[i];
                case CompareOp::LT:
                  return value < operands[i];
                case CompareOp::GT:
                  return value > operands[i];
                default:
                  throw std::invalid_argument("Unsupported numeric comparison");
              }
            });
      }
    } catch (const std::exception &error) {
      throw std::runtime_error(
          "verification failed for case '" + name + "', input " +
          (operands.empty() ? "NULL" : std::to_string(operands[i])) + ": " +
          error.what());
    }
  }
}

JsonObject Build(const std::string &index_path, const std::string &run_dir) {
  auto dataset = NumericDataset::Load(run_dir);
  const auto &profile = dataset.profile();
  NumericBatch batch;
  IndexFixture index(profile.data_type, index_path, false);
  auto &column = index.column();
  double insert_seconds = 0.0;
  while (dataset.read_next_batch(batch)) {
    const auto insert_begin = Clock::now();
    for (size_t i = 0; i < batch.values.size(); ++i) {
      const auto id = batch.first_doc_id + static_cast<uint32_t>(i);
      auto status = batch.is_null[i] ? column.insert_null(id)
                                     : column.insert(id, batch.values[i]);
      if (!status.ok())
        throw std::runtime_error("Numeric insert failed at doc " +
                                 std::to_string(id));
    }
    insert_seconds += Seconds(insert_begin, Clock::now());
  }
  const auto optimize_begin = Clock::now();
  index.seal();
  const auto optimize_end = Clock::now();
  const auto optimize_seconds = Seconds(optimize_begin, optimize_end);
  const auto total_seconds = insert_seconds + optimize_seconds;
  const auto memory = MemorySnapshot();
  index.close();
  JsonObject result;
  result.set("environment", EnvironmentJson());
  result.set("index_path",
             JsonStringValue(ailego::FileHelper::PathToUtf8(
                 std::filesystem::weakly_canonical(
                     ailego::FileHelper::PathFromUtf8(index_path)))));
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
    VerifyNumericCase<T>(record, dataset, index.column());
}

template <typename T>
JsonObject QueryNumericCase(DataType data_type, const std::string &index_path,
                            const JsonObject &record) {
  const auto name = Text(record, "name");
  const auto inputs = record["inputs"].as_array();
  const auto op = NumericCompareOp(name);
  const auto operands = ReadNumericOperands<T>(inputs, op);
  std::vector<std::string> encoded;
  for (auto operand : operands) encoded.push_back(EncodeValue(operand));

  IndexFixture index(data_type, index_path, true);
  auto &column = index.column();
  std::vector<uint64_t> match_counts(inputs.size());
  bool collect_counts = true;
  auto measure = [&](auto search) {
    return MeasureRepeatedQueries(inputs.size(), [&]() {
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto result = search(i);
        if (!result) throw std::runtime_error("Numeric query failed: " + name);
        const volatile auto count = result->count();
        if (collect_counts) match_counts[i] = count;
      }
      collect_counts = false;
    });
  };
  double query_time_us;
  if (op == CompareOp::IS_NULL) {
    query_time_us = measure([&](size_t) { return column.search_null(); });
  } else if (op == CompareOp::IS_NOT_NULL) {
    query_time_us = measure([&](size_t) { return column.search_non_null(); });
  } else {
    query_time_us =
        measure([&](size_t i) { return column.search(encoded[i], op); });
  }
  index.close();
  auto result = QueryReport(name, query_time_us);
  JsonArray counts;
  for (auto count : match_counts) counts.push(JsonValue(count));
  result.set("match_counts", counts);
  return result;
}


}  // namespace


JsonObject BuildNumeric(const std::string &run_dir) {
  const auto index_path = ailego::FileHelper::PathJoin(run_dir, "index");
  if (ailego::File::IsExist(index_path))
    throw std::invalid_argument(
        "index directory already exists: '" + index_path +
        "'\n  use a new --run_dir to start another run");
  return Build(index_path, run_dir);
}

void VerifyNumeric(const std::string &run_dir) {
  const auto index_path = ailego::FileHelper::PathJoin(run_dir, "index");
  const auto build =
      ReadJson(ailego::FileHelper::PathJoin(run_dir, "build.json"));
  if (!ailego::File::IsSame(Text(build, "index_path"), index_path))
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
  if (!ailego::File::IsSame(Text(build, "index_path"), index_path))
    throw std::invalid_argument("Index path does not match build report");
  auto dataset = NumericDataset::Load(run_dir);
  const auto &profile = dataset.profile();
  const auto &inputs = dataset.query_cases();
  JsonArray cases;
  for (const auto &record : inputs) {
    auto query =
        profile.data_type == DataType::INT32
            ? QueryNumericCase<int32_t>(profile.data_type, index_path, record)
            : QueryNumericCase<int64_t>(profile.data_type, index_path, record);
    cases.push(query);
  }
  JsonObject result;
  result.set("cases", cases);
  return result;
}


}  // namespace zvec::benchmarks::inverted
