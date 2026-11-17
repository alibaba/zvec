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
#include <filesystem>
#include <set>
#include <zvec/db/schema.h>
#include "shared/random.h"

namespace zvec::benchmarks::inverted {


namespace {

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

}  // namespace


template <typename T>
NumericData<T> GenerateDataset(const NumericProfile &profile) {
  ValidateProfile(profile);
  if ((profile.data_type == DataType::INT32) != std::is_same_v<T, int32_t>)
    throw std::invalid_argument("numeric profile does not match value type");
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
  auto description = ProfileJson(dataset.profile);
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

template <typename T>
void WriteDataset(const NumericData<T> &dataset, const std::string &path) {
  ValidateProfile(dataset.profile);
  const auto num_docs = DocumentCount(dataset.profile);
  if (dataset.is_null.size() != num_docs ||
      dataset.values.size() + dataset.null_count != num_docs)
    throw std::invalid_argument("numeric dataset size does not match profile");
  if (std::filesystem::exists(path))
    throw std::invalid_argument("dataset file already exists: '" + path + "'");
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
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

// These are the only scalar types supported by NumericProfile.
template NumericData<int32_t> GenerateDataset<int32_t>(const NumericProfile &);
template NumericData<int64_t> GenerateDataset<int64_t>(const NumericProfile &);
template JsonObject DescribeDataset(const NumericData<int32_t> &,
                                    const std::vector<int32_t> &);
template JsonObject DescribeDataset(const NumericData<int64_t> &,
                                    const std::vector<int64_t> &);
template void WriteDataset(const NumericData<int32_t> &, const std::string &);
template void WriteDataset(const NumericData<int64_t> &, const std::string &);

NumericDatasetReader::NumericDatasetReader(const NumericProfile &profile,
                                           const std::string &path)
    : path_(path),
      num_docs_(DocumentCount(profile)),
      value_size_(FieldSchema::get_element_data_size(profile.data_type)),
      buffer_(kNumericBatchSize * (1 + value_size_)) {
  ValidateProfile(profile);
  if (!file_.open(path_, true))
    throw std::runtime_error("cannot open dataset file: '" + path_ + "'");
  const auto expected_size =
      static_cast<uint64_t>(num_docs_) * (1 + value_size_);
  if (file_.size() != expected_size)
    throw std::runtime_error("dataset file size does not match profile: '" +
                             path_ + "'");
}

bool NumericDatasetReader::ReadBatch(NumericBatch &batch) {
  const auto count = std::min(kNumericBatchSize, num_docs_ - next_doc_id_);
  batch.first_doc_id = next_doc_id_;
  batch.is_null.resize(count);
  batch.values.resize(count);
  if (!count) return false;

  const auto record_size = 1 + value_size_;
  const auto bytes = count * record_size;
  if (file_.read(buffer_.data(), bytes) != bytes)
    throw std::runtime_error("cannot read dataset file: '" + path_ + "'");
  for (uint32_t i = 0; i < count; ++i) {
    const auto *record = buffer_.data() + i * record_size;
    const auto is_null = static_cast<uint8_t>(record[0]);
    if (is_null > 1)
      throw std::runtime_error("invalid NULL marker at doc " +
                               std::to_string(next_doc_id_ + i) +
                               " in dataset file: '" + path_ + "'");
    batch.is_null[i] = is_null;
    if (is_null)
      batch.values[i].clear();
    else
      batch.values[i].assign(record + 1, value_size_);
  }
  next_doc_id_ += count;
  return true;
}

void NumericDatasetReader::Reset() {
  if (!file_.seek(0, ailego::File::Origin::Begin))
    throw std::runtime_error("cannot rewind dataset file: '" + path_ + "'");
  next_doc_id_ = 0;
}


}  // namespace zvec::benchmarks::inverted
