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
#include <set>
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

std::vector<int64_t> SelectHotValues(const NumericProfile &profile,
                                     uint32_t count) {
  // Floyd's sampling without replacement takes one draw per selected value,
  // even when the hot set nearly fills the domain. No full-domain array.
  Random random(kNumericSeed ^ 0xe7037ed1a0b428dbULL);
  std::set<int64_t> selected;
  const auto size = DomainSize(profile);
  for (uint64_t last = size - count; last < size; ++last) {
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
NumericData GenerateTypedDataset(const NumericProfile &profile) {
  NumericData dataset{profile};
  dataset.null_count = static_cast<uint64_t>(DocumentCount(profile)) *
                       NullPercent(profile.null_ratio) / 100;
  dataset.is_null.resize(DocumentCount(profile), 0);
  std::fill_n(dataset.is_null.begin(), dataset.null_count, uint8_t{1});
  if (dataset.null_count)
    Random(kNumericSeed ^ 0xa0761d6478bd642fULL).shuffle(dataset.is_null);
  const auto nonnull = DocumentCount(profile) - dataset.null_count;

  // bounded() uses integer rejection sampling, including the full 2^63 domain.
  Random random(kNumericSeed ^ 0x8ebc6af09c88c6e3ULL);
  std::vector<int64_t> values(nonnull);
  if (profile.distribution == NumericDistribution::UNIFORM) {
    for (auto &value : values) value = random.bounded(DomainSize(profile));
  } else {
    const auto hotspot = HotspotFor(profile.distribution);
    dataset.hot_values = SelectHotValues(profile, hotspot.value_count);
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
    Random(kNumericSeed ^ 0x589965cc75374cc3ULL).shuffle(values);
  dataset.values.resize(DocumentCount(profile));
  size_t position = 0;
  for (uint32_t id = 0; id < DocumentCount(profile); ++id)
    if (!dataset.is_null[id])
      dataset.values[id] = EncodeValue(static_cast<T>(values[position++]));
  return dataset;
}

}  // namespace


NumericData GenerateDataset(const NumericProfile &profile) {
  if (profile.data_type == DataType::INT32)
    return GenerateTypedDataset<int32_t>(profile);
  return GenerateTypedDataset<int64_t>(profile);
}

JsonObject DescribeDataset(const NumericData &dataset) {
  auto description = ProfileJson(dataset.profile);
  description.set("seed", JsonValue(kNumericSeed));
  JsonObject stats, percentiles;
  std::vector<uint64_t> counts;
  for (const auto &entry : dataset.frequencies) counts.push_back(entry.second);
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
  stats.set("null_count", JsonValue(dataset.null_count));
  description.set("stats", stats);
  return description;
}


}  // namespace zvec::benchmarks::inverted
