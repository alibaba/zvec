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

#include "dataset.h"
#include <algorithm>
#include <limits>
#include <set>
#include <utility>
#include <zvec/ailego/utility/file_helper.h>
#include <zvec/db/schema.h>
#include "shared/json.h"
#include "shared/query_helpers.h"
#include "shared/random.h"

namespace zvec::benchmarks::inverted {


namespace {

// Fixed for reproducible datasets; not a configurable profile field.
constexpr uint64_t kNumericSeed = 19991230;
constexpr size_t kQueryPoolSize = 7;
constexpr uint32_t kNumericBatchSize = 32768;


template <typename T>
struct NumericData {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  // Configuration used to generate this dataset.
  NumericProfile profile;
  // Total number of generated documents, including NULLs.
  uint32_t doc_count{0};
  // Number of NULL documents.
  uint64_t null_count{0};
  // Whether each document is NULL, indexed by doc ID.
  std::vector<bool> is_null;
  // Non-NULL values in doc ID order; NULL documents have no slot here.
  std::vector<T> values;
  // Sorted hot values for generation and query selection; empty for uniform.
  std::vector<T> hot_values;
};

template <typename T>
std::vector<T> SelectHotValues(uint64_t value_max, uint32_t count) {
  // Floyd's sampling without replacement takes one draw per selected value,
  // even when the hot set nearly fills the domain. No full-domain array.
  Random random(kNumericSeed ^ 0xe7037ed1a0b428dbULL);
  std::set<T> selected;
  for (uint64_t last = value_max - count; last < value_max; ++last) {
    const auto value = static_cast<T>(random.bounded(last + 1));
    selected.insert(selected.count(value) ? static_cast<T>(last) : value);
  }
  return {selected.begin(), selected.end()};
}

template <typename T>
T SelectHottestValue(const std::vector<T> &sorted,
                     const std::vector<T> &hot_values) {
  T hottest = hot_values.front();
  size_t max_count = 0;
  // Hot values are sorted, so equal counts keep the smallest value.
  for (const auto value : hot_values) {
    const auto range = std::equal_range(sorted.begin(), sorted.end(), value);
    const auto count = static_cast<size_t>(range.second - range.first);
    if (count > max_count) {
      hottest = value;
      max_count = count;
    }
  }
  return hottest;
}

template <typename T>
T ColdValue(uint64_t rank, const std::vector<T> &hot_values) {
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
  return static_cast<T>(rank + begin);
}

template <typename T>
NumericData<T> GenerateDataset(const NumericProfile &profile) {
  NumericData<T> dataset{profile};
  dataset.doc_count = DocumentCount(profile);

  // Set the NULL count and randomly distribute NULLs across doc IDs.
  dataset.null_count = static_cast<uint64_t>(dataset.doc_count) *
                       NullPercent(profile.null_ratio) / 100;
  dataset.is_null.resize(dataset.doc_count, false);
  std::fill_n(dataset.is_null.begin(), dataset.null_count, true);
  if (dataset.null_count) {
    Random(kNumericSeed ^ 0xa0761d6478bd642fULL).shuffle(dataset.is_null);
  }
  const auto nonnull = dataset.doc_count - dataset.null_count;

  // Generate non-NULL values using the configured distribution.
  Random random(kNumericSeed ^ 0x8ebc6af09c88c6e3ULL);
  auto &values = dataset.values;
  values.resize(nonnull);
  if (profile.distribution == NumericDistribution::UNIFORM) {
    for (auto &value : values) value = random.bounded(ValueMax(profile));
  } else {
    uint32_t hot_value_count = 0;
    uint32_t hot_doc_percent = 0;
    switch (profile.distribution) {
      case NumericDistribution::HOTSPOT_MODERATE:
        hot_value_count = 3000;
        hot_doc_percent = 50;
        break;
      case NumericDistribution::HOTSPOT_EXTREME:
        hot_value_count = 300;
        hot_doc_percent = 90;
        break;
      default:
        throw std::invalid_argument("Expected a hotspot distribution");
    }
    dataset.hot_values = SelectHotValues<T>(ValueMax(profile), hot_value_count);
    const uint64_t hot_rows = nonnull * hot_doc_percent / 100;
    for (uint64_t i = 0; i < nonnull; ++i) {
      if (i < hot_rows)
        values[i] =
            dataset.hot_values[random.bounded(dataset.hot_values.size())];
      else
        values[i] = ColdValue(
            random.bounded(ValueMax(profile) - dataset.hot_values.size()),
            dataset.hot_values);
    }
  }

  // Sort or shuffle non-NULL values according to the configured order.
  if (profile.value_order == NumericValueOrder::ASCENDING) {
    std::sort(values.begin(), values.end());
  } else {
    Random(kNumericSeed ^ 0x589965cc75374cc3ULL).shuffle(values);
  }
  return dataset;
}

template <typename T>
JsonObject DescribeDataset(const NumericData<T> &dataset,
                           const std::vector<T> &sorted_values) {
  JsonObject description;
  description.set("profile", ProfileJson(dataset.profile));
  description.set("seed", JsonValue(kNumericSeed));
  JsonObject stats, percentiles;
  size_t cardinality = 1;
  for (size_t i = 1; i < sorted_values.size(); ++i)
    cardinality += sorted_values[i] != sorted_values[i - 1];
  std::vector<uint32_t> counts;
  counts.reserve(cardinality);
  for (size_t i = 0; i < sorted_values.size(); ++i) {
    if (i == 0 || sorted_values[i] != sorted_values[i - 1])
      counts.push_back(1);
    else
      ++counts.back();
  }
  std::sort(counts.begin(), counts.end());
  percentiles.set("min", JsonValue(counts.front()));
  percentiles.set("p50", JsonValue(counts[(counts.size() - 1) / 2]));
  percentiles.set("p95", JsonValue(counts[(counts.size() - 1) * 95 / 100]));
  percentiles.set("max", JsonValue(counts.back()));
  stats.set("distinct_value_count",
            JsonValue(static_cast<uint64_t>(counts.size())));
  stats.set("documents_per_value", percentiles);
  stats.set("min_value", JsonValue(sorted_values.front()));
  stats.set("max_value", JsonValue(sorted_values.back()));
  stats.set("null_count", JsonValue(dataset.null_count));
  description.set("stats", stats);
  return description;
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
  auto add_case = [&](const std::string &name, const std::vector<T> &values) {
    JsonArray inputs;
    for (auto value : values) inputs.push(JsonValue(value));
    cases.push(MakeQueryCase(name, inputs));
  };
  std::vector<T> hits;
  if (dataset.profile.distribution == NumericDistribution::UNIFORM) {
    hits = SelectQueryTerms(sorted, [](T) { return true; });
    add_case("eq_hit", hits);
  } else {
    auto is_hot = [&](T value) {
      return std::binary_search(dataset.hot_values.begin(),
                                dataset.hot_values.end(), value);
    };
    add_case("eq_hottest", {SelectHottestValue(sorted, dataset.hot_values)});
    hits = SelectQueryTerms(sorted, is_hot);
    add_case("eq_hot", hits);
    add_case("eq_cold",
             SelectQueryTerms(sorted, [&](T value) { return !is_hot(value); }));
  }
  JsonArray misses;
  for (int64_t i = 1; i <= static_cast<int64_t>(kQueryPoolSize); ++i)
    misses.push(JsonValue(-i));
  cases.push(MakeQueryCase("eq_miss", misses));
  add_case("ne_hit", hits);
  for (const auto &[quantile, percentile] :
       std::vector<std::pair<std::string, uint32_t>>{{"q01", 1},
                                                     {"q11", 11},
                                                     {"q33", 33},
                                                     {"q50", 50},
                                                     {"q77", 77},
                                                     {"q90", 90},
                                                     {"q99", 99}}) {
    const auto boundary = RangeBoundary(sorted, percentile);
    add_case("lt_" + quantile, {boundary});
    add_case("gt_" + quantile, {boundary});
  }
  JsonArray null_input;
  null_input.push(JsonValue());
  cases.push(MakeQueryCase("is_null", null_input));
  cases.push(MakeQueryCase("is_not_null", null_input));
  return cases;
}

template <typename T>
void WriteDataset(const NumericData<T> &dataset, const std::string &path) {
  const auto num_docs = dataset.doc_count;
  if (dataset.is_null.size() != num_docs ||
      dataset.values.size() + dataset.null_count != num_docs)
    throw std::invalid_argument(
        "numeric dataset size does not match document count");
  if (ailego::File::IsExist(path))
    throw std::invalid_argument("dataset file already exists: '" + path + "'");
  ailego::File file;
  if (!file.create(path, 0))
    throw std::runtime_error("cannot create dataset file: '" + path + "'");

  constexpr auto value_size = sizeof(T);
  const auto record_size = 1 + value_size;
  std::vector<char> buffer(kNumericBatchSize * record_size);
  size_t position = 0;
  for (uint32_t first = 0; first < num_docs;) {
    const auto count = std::min(kNumericBatchSize, num_docs - first);
    for (uint32_t i = 0; i < count; ++i) {
      const auto id = first + i;
      auto *record = buffer.data() + i * record_size;
      record[0] = dataset.is_null[id] ? 1 : 0;
      if (dataset.is_null[id]) {
        std::memset(record + 1, 0, value_size);
      } else {
        const auto value = dataset.values[position++];
        std::memcpy(record + 1, &value, value_size);
      }
    }
    const auto bytes = count * record_size;
    if (file.write(buffer.data(), bytes) != bytes)
      throw std::runtime_error("cannot write dataset file: '" + path + "'");
    first += count;
  }
  if (!file.flush())
    throw std::runtime_error("cannot flush dataset file: '" + path + "'");
  file.close();
}

template <typename T>
void CreateDatasetFiles(const NumericProfile &profile,
                        const std::string &directory) {
  auto dataset = GenerateDataset<T>(profile);
  // Sorted values for statistics and query selection.
  const auto *sorted = &dataset.values;
  std::vector<T> sorted_copy;
  if (profile.value_order == NumericValueOrder::SHUFFLED) {
    sorted_copy = dataset.values;
    std::sort(sorted_copy.begin(), sorted_copy.end());
    sorted = &sorted_copy;
  }
  auto metadata = DescribeDataset(dataset, *sorted);
  metadata.set("queries", SelectQueryInputs(dataset, *sorted));
  WriteDataset(dataset, ailego::FileHelper::PathJoin(directory, "data.bin"));
  WriteJson(ailego::FileHelper::PathJoin(directory, "dataset.json"), metadata);
}

}  // namespace


NumericDataset NumericDataset::Create(const NumericProfile &profile,
                                      const std::string &directory) {
  ValidateProfile(profile);
  if (!ailego::File::IsDirectory(directory))
    throw std::invalid_argument("expected an existing directory: '" +
                                directory + "'");
  for (const auto *name : {"data.bin", "dataset.json"}) {
    const auto path = ailego::FileHelper::PathJoin(directory, name);
    if (ailego::File::IsExist(path))
      throw std::invalid_argument(
          "dataset file already exists: '" + path +
          "'\n  use a new --run_dir to start another run");
  }
  if (profile.data_type == DataType::INT32) {
    CreateDatasetFiles<int32_t>(profile, directory);
  } else {
    CreateDatasetFiles<int64_t>(profile, directory);
  }
  return Load(directory);
}

NumericDataset NumericDataset::Load(const std::string &directory) {
  if (!ailego::File::IsDirectory(directory))
    throw std::invalid_argument("expected an existing directory: '" +
                                directory + "'");
  const auto metadata =
      ReadJson(ailego::FileHelper::PathJoin(directory, "dataset.json"));
  if (!metadata["profile"].is_object() || !metadata["queries"].is_array())
    throw std::invalid_argument("invalid dataset metadata");
  return NumericDataset(directory, metadata);
}

NumericDataset::NumericDataset(const std::string &directory,
                               const JsonObject &metadata)
    : profile_(ReadProfile(metadata["profile"].as_object())),
      query_cases_(ReadQueryInputs(metadata["queries"].as_array())),
      data_path_(ailego::FileHelper::PathJoin(directory, "data.bin")),
      bytes_per_value_(FieldSchema::get_element_data_size(profile_.data_type)) {
}

bool NumericDataset::read_next_batch(NumericBatch &batch) {
  const auto num_docs = DocumentCount(profile_);
  if (!file_.is_valid()) {
    if (!file_.open(data_path_, true))
      throw std::runtime_error("cannot open dataset file: '" + data_path_ +
                               "'");
    const auto expected_size =
        static_cast<uint64_t>(num_docs) * (1 + bytes_per_value_);
    if (file_.size() != expected_size) {
      file_.close();
      throw std::runtime_error("dataset file size does not match profile: '" +
                               data_path_ + "'");
    }
    buffer_.resize(kNumericBatchSize * (1 + bytes_per_value_));
  }
  const auto count = std::min(kNumericBatchSize, num_docs - next_doc_id_);
  batch.first_doc_id = next_doc_id_;
  batch.is_null.resize(count);
  batch.values.resize(count);
  if (!count) return false;

  const auto record_size = 1 + bytes_per_value_;
  const auto bytes = count * record_size;
  if (file_.read(buffer_.data(), bytes) != bytes)
    throw std::runtime_error("cannot read dataset file: '" + data_path_ + "'");
  for (uint32_t i = 0; i < count; ++i) {
    const auto *record = buffer_.data() + i * record_size;
    const auto is_null = static_cast<uint8_t>(record[0]);
    if (is_null > 1)
      throw std::runtime_error("invalid NULL marker at doc " +
                               std::to_string(next_doc_id_ + i) +
                               " in dataset file: '" + data_path_ + "'");
    batch.is_null[i] = is_null;
    if (is_null)
      batch.values[i].clear();
    else
      batch.values[i].assign(record + 1, bytes_per_value_);
  }
  next_doc_id_ += count;
  return true;
}

void NumericDataset::rewind() {
  if (file_.is_valid() && !file_.seek(0, ailego::File::Origin::Begin))
    throw std::runtime_error("cannot rewind dataset file: '" + data_path_ +
                             "'");
  next_doc_id_ = 0;
}


}  // namespace zvec::benchmarks::inverted
