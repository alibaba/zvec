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
#include "shared/random.h"
#include "shared/report_utils.h"

namespace zvec::benchmarks::inverted {


namespace {

// Fixed for reproducible datasets; not a configurable profile field.
constexpr uint64_t kNumericSeed = 20261115;
constexpr size_t kQueryPoolSize = 8;
constexpr uint32_t kNumericBatchSize = 32768;

// Temporary generation data, released before opening the index.
template <typename T>
struct NumericData {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>);
  // Workload used to generate the records.
  NumericProfile profile;
  // Exact number of NULL documents.
  uint64_t null_count{0};
  // One marker per doc ID, including NULL documents.
  std::vector<uint8_t> is_null;
  // Non-NULL values in doc ID order; NULL documents have no slot here.
  std::vector<T> values;
  // Sorted hot set used for query selection; empty for uniform data.
  std::vector<T> hot_values;
};

uint64_t DomainSize(const NumericProfile &profile) {
  // The nonnegative INT64 domain has 2^63 values; count it in uint64_t.
  return static_cast<uint64_t>(ValueMax(profile)) + 1;
}

struct HotspotPreset {
  uint32_t value_count;
  uint32_t doc_percent;  // Percentage of non-NULL documents on hot values.
};

HotspotPreset HotspotFor(NumericDistribution distribution) {
  switch (distribution) {
    case NumericDistribution::HOTSPOT_MODERATE:
      return {3000, 50};
    case NumericDistribution::HOTSPOT_EXTREME:
      return {300, 90};
    case NumericDistribution::UNIFORM:
      break;
  }
  throw std::invalid_argument("Expected a hotspot distribution");
}

template <typename T>
std::vector<T> SelectHotValues(const NumericProfile &profile, uint32_t count) {
  // Floyd's sampling without replacement takes one draw per selected value,
  // even when the hot set nearly fills the domain. No full-domain array.
  Random random(kNumericSeed ^ 0xe7037ed1a0b428dbULL);
  std::set<T> selected;
  const auto size = DomainSize(profile);
  for (uint64_t last = size - count; last < size; ++last) {
    const auto value = static_cast<T>(random.bounded(last + 1));
    selected.insert(selected.count(value) ? static_cast<T>(last) : value);
  }
  return {selected.begin(), selected.end()};
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
  dataset.null_count = static_cast<uint64_t>(DocumentCount(profile)) *
                       NullPercent(profile.null_ratio) / 100;
  dataset.is_null.resize(DocumentCount(profile), 0);
  std::fill_n(dataset.is_null.begin(), dataset.null_count, uint8_t{1});
  if (dataset.null_count)
    Random(kNumericSeed ^ 0xa0761d6478bd642fULL).shuffle(dataset.is_null);
  const auto nonnull = DocumentCount(profile) - dataset.null_count;

  // bounded() uses integer rejection sampling, including the full 2^63 domain.
  Random random(kNumericSeed ^ 0x8ebc6af09c88c6e3ULL);
  auto &values = dataset.values;
  values.resize(nonnull);
  if (profile.distribution == NumericDistribution::UNIFORM) {
    for (auto &value : values) value = random.bounded(DomainSize(profile));
  } else {
    const auto hotspot = HotspotFor(profile.distribution);
    dataset.hot_values = SelectHotValues<T>(profile, hotspot.value_count);
    const uint64_t hot_rows = nonnull * hotspot.doc_percent / 100;
    for (uint64_t i = 0; i < nonnull; ++i) {
      if (i < hot_rows)
        values[i] =
            dataset.hot_values[random.bounded(dataset.hot_values.size())];
      else
        values[i] = ColdValue(
            random.bounded(DomainSize(profile) - dataset.hot_values.size()),
            dataset.hot_values);
    }
  }
  if (profile.value_order == NumericValueOrder::ASCENDING)
    std::sort(values.begin(), values.end());
  else
    Random(kNumericSeed ^ 0x589965cc75374cc3ULL).shuffle(values);
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
  stats.set("observed_cardinality",
            JsonValue(static_cast<uint64_t>(counts.size())));
  stats.set("observed_term_df", percentiles);
  stats.set("observed_value_min", JsonValue(sorted_values.front()));
  stats.set("observed_value_max", JsonValue(sorted_values.back()));
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

template <typename T>
void WriteDataset(const NumericData<T> &dataset, const std::string &path) {
  const auto num_docs = DocumentCount(dataset.profile);
  if (dataset.is_null.size() != num_docs ||
      dataset.values.size() + dataset.null_count != num_docs)
    throw std::invalid_argument("numeric dataset size does not match profile");
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
  std::vector<T> sorted_copy;
  const auto *sorted = &dataset.values;
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
