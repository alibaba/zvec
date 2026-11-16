// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "array_numeric_cases.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#include <zvec/db/type.h>
#include "shared/fingerprint.h"
#include "shared/index_fixture.h"
#include "shared/measurement.h"
#include "shared/random.h"
#include "shared/report_utils.h"

namespace zvec::benchmarks::inverted {
namespace {

struct ArrayConfig {
  std::string name;
  std::string distribution;
  uint32_t cardinality{1000};
  uint32_t array_length{8};
  double null_ratio{0.1};
  double empty_ratio{0.1};
  double duplicate_ratio{0.2};
  bool supplemental{false};
};

struct ArrayData {
  ArrayConfig config;
  uint32_t num_docs{0};
  uint64_t seed{0};
  std::vector<uint8_t> is_null;
  std::vector<uint8_t> kinds;
  std::vector<std::string> values;
  std::vector<uint64_t> frequencies;
  uint64_t null_count{0}, empty_count{0}, duplicate_count{0};
  uint64_t raw_elements{0}, distinct_elements{0};
  std::map<uint32_t, uint64_t> raw_lengths, unique_lengths;
  std::string fingerprint;
};

constexpr size_t kQueryPoolSize = 8;
enum RowKind : uint8_t { kNull, kEmpty, kPlain, kDuplicate };

std::vector<uint8_t> GenerateRowKinds(const ArrayConfig &config, uint32_t count,
                                      uint64_t seed) {
  if (count == 0) throw std::invalid_argument("num_docs must be positive");
  std::vector<uint8_t> kinds(count, kPlain);
  if (!std::isfinite(config.null_ratio) || config.null_ratio < 0 ||
      config.null_ratio >= 1 || !std::isfinite(config.empty_ratio) ||
      config.empty_ratio < 0 || config.null_ratio + config.empty_ratio >= 1 ||
      !std::isfinite(config.duplicate_ratio) || config.duplicate_ratio < 0 ||
      config.duplicate_ratio > 1)
    throw std::invalid_argument("Invalid array row-kind ratios");
  const uint32_t nulls = static_cast<uint32_t>(count * config.null_ratio);
  const uint32_t empties = static_cast<uint32_t>(count * config.empty_ratio);
  const uint32_t repeated =
      config.array_length > 1
          ? static_cast<uint32_t>((count - nulls - empties) *
                                  config.duplicate_ratio)
          : 0;
  std::fill_n(kinds.begin(), nulls, kNull);
  std::fill_n(kinds.begin() + nulls, empties, kEmpty);
  std::fill_n(kinds.begin() + nulls + empties, repeated, kDuplicate);
  Random(seed ^ 0xa0761d6478bd642fULL).shuffle(kinds);
  return kinds;
}

std::string EncodeMember(int32_t value) {
  std::string bytes(sizeof(value), '\0');
  std::memcpy(bytes.data(), &value, sizeof(value));
  return bytes;
}

int32_t ReadElement(const std::string &bytes, size_t offset) {
  int32_t value;
  std::memcpy(&value, bytes.data() + offset * sizeof(value), sizeof(value));
  return value;
}

std::vector<int32_t> SampleTerms(const std::vector<int32_t> &population) {
  const size_t count = std::min(kQueryPoolSize, population.size());
  std::vector<int32_t> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i)
    result.push_back(population[i * population.size() / count]);
  return result;
}

std::string DatasetFingerprint(const ArrayData &dataset) {
  const auto &config = dataset.config;
  Fingerprint fingerprint;
  fingerprint.AddString("inverted-data-v1");
  fingerprint.AddString(config.name);
  fingerprint.AddString("array_int32");
  fingerprint.AddString(config.distribution);
  fingerprint.AddUint64(config.cardinality);
  fingerprint.AddUint64(config.array_length);
  fingerprint.AddUint64(dataset.num_docs);
  fingerprint.AddUint64(dataset.seed);
  for (uint32_t id = 0; id < dataset.num_docs; ++id) {
    fingerprint.AddByte(dataset.is_null[id]);
    fingerprint.AddString(dataset.values[id]);
  }
  return fingerprint.Finish();
}

JsonObject LengthHistogram(const std::map<uint32_t, uint64_t> &histogram) {
  JsonObject result;
  for (const auto &[length, count] : histogram)
    result.set(std::to_string(length).c_str(), JsonValue(count));
  return result;
}

std::vector<ArrayConfig> Profiles() {
  return {{"random_l8", "random"},
          {"window_l8", "window"},
          {"random_l1", "random", 1000, 1, 0.1, 0.1, 0.2, true},
          {"random_l32", "random", 1000, 32, 0.1, 0.1, 0.2, true}};
}

ArrayConfig Profile(const std::string &name) {
  for (const auto &profile : Profiles())
    if (profile.name == name) return profile;
  throw std::invalid_argument("Unknown array profile: " + name);
}

double Ratio(const JsonValue &value) {
  if (!value.is_integer() && !value.is_float())
    throw std::invalid_argument("Array ratio must be numeric");
  const double result =
      value.is_integer() ? value.as_integer() : value.as_float();
  if (!std::isfinite(result) || result < 0 || result > 1)
    throw std::invalid_argument("Array ratio must be within [0, 1]");
  return result;
}

ArrayConfig Configure(ArrayConfig config, const JsonObject &overrides) {
  for (const auto &entry : overrides) {
    const auto key = entry.key().decode().as_stl_string();
    const auto &value = entry.value();
    if (key == "distribution") {
      config.distribution = JsonStringText(value);
    } else if (key == "cardinality" || key == "array_length") {
      if (!value.is_integer() || value.as_integer() <= 0 ||
          static_cast<uint64_t>(value.as_integer()) > UINT32_MAX)
        throw std::invalid_argument("Array size must be a positive uint32");
      if (key == "cardinality")
        config.cardinality = value.as_integer();
      else
        config.array_length = value.as_integer();
    } else if (key == "null_ratio") {
      config.null_ratio = Ratio(value);
    } else if (key == "empty_ratio") {
      config.empty_ratio = Ratio(value);
    } else if (key == "duplicate_ratio") {
      config.duplicate_ratio = Ratio(value);
    } else {
      throw std::invalid_argument("Unknown array configuration: " + key);
    }
  }
  if ((config.distribution != "random" && config.distribution != "window") ||
      config.cardinality < 8 || config.cardinality > INT32_MAX - 17 ||
      config.array_length > config.cardinality ||
      (config.array_length > 1 && config.array_length % 2 &&
       config.duplicate_ratio > 0) ||
      config.null_ratio + config.empty_ratio >= 1)
    throw std::invalid_argument("Invalid array dataset configuration");
  return config;
}

JsonObject ConfigJson(const ArrayConfig &config) {
  JsonObject result;
  result.set("name", JsonStringValue(config.name));
  result.set("type", JsonStringValue("array_int32"));
  result.set("distribution", JsonStringValue(config.distribution));
  result.set("cardinality", JsonValue(config.cardinality));
  result.set("array_length", JsonValue(config.array_length));
  result.set("null_ratio", JsonValue(config.null_ratio));
  result.set("empty_ratio", JsonValue(config.empty_ratio));
  result.set("duplicate_ratio", JsonValue(config.duplicate_ratio));
  result.set("supplemental", JsonValue(config.supplemental));
  return result;
}

ArrayConfig ReadConfig(const JsonObject &description) {
  if (Text(description, "type") != "array_int32")
    throw std::invalid_argument("Expected ARRAY_INT32 dataset");
  JsonObject config;
  for (const char *key : {"distribution", "cardinality", "array_length",
                          "null_ratio", "empty_ratio", "duplicate_ratio"})
    config.set(key, description[key]);
  return Configure(Profile(Text(description, "name")), config);
}

std::vector<std::string> CaseNames(const ArrayConfig &config) {
  std::vector<std::string> names{"any_1_hit", "any_8_global", "all_2_adjacent"};
  if (config.array_length > 1) names.emplace_back("all_m_witness");
  for (const auto *name :
       {"all_8_global", "all_hit_miss", "not_any_2", "not_all_2", "is_null",
        "is_not_null", "len_eq_zero", "len_eq_raw"})
    names.emplace_back(name);
  return names;
}

ArrayData GenerateData(const ArrayConfig &config, uint32_t num_docs,
                       uint64_t seed) {
  ArrayData dataset;
  dataset.config = config;
  dataset.num_docs = num_docs;
  dataset.seed = seed;
  dataset.kinds = GenerateRowKinds(config, num_docs, seed);
  dataset.is_null.resize(num_docs, 0);
  dataset.values.resize(num_docs);
  dataset.frequencies.resize(config.cardinality, 0);
  std::vector<uint32_t> rows;
  for (uint32_t id = 0; id < num_docs; ++id) {
    if (dataset.kinds[id] == kNull) {
      dataset.is_null[id] = 1;
      ++dataset.null_count;
    } else if (dataset.kinds[id] == kEmpty) {
      ++dataset.empty_count;
      ++dataset.raw_lengths[0];
      ++dataset.unique_lengths[0];
    } else {
      rows.push_back(id);
      if (dataset.kinds[id] == kDuplicate) ++dataset.duplicate_count;
    }
  }
  Random(seed ^ 0x589965cc75374cc3ULL).shuffle(rows);
  Random values(seed ^ 0x8ebc6af09c88c6e3ULL);
  std::vector<uint32_t> seen(config.cardinality, 0);
  std::vector<int32_t> members;
  for (size_t position = 0; position < rows.size(); ++position) {
    const uint32_t id = rows[position];
    const bool repeated = dataset.kinds[id] == kDuplicate;
    const uint32_t unique_length =
        repeated ? config.array_length / 2 : config.array_length;
    members.clear();
    if (config.distribution == "window") {
      const uint32_t start = position % config.cardinality;
      for (uint32_t i = 0; i < unique_length; ++i)
        members.push_back(
            static_cast<int32_t>((start + i) % config.cardinality));
    } else {
      while (members.size() < unique_length) {
        const uint32_t term = values.bounded(config.cardinality);
        if (seen[term] == id + 1) continue;
        seen[term] = id + 1;
        members.push_back(term);
      }
    }
    auto &bytes = dataset.values[id];
    bytes.resize(config.array_length * sizeof(int32_t));
    size_t output = 0;
    for (int32_t member : members) {
      ++dataset.frequencies[member];
      for (uint32_t copy = 0; copy < (repeated ? 2U : 1U); ++copy) {
        std::memcpy(bytes.data() + output * sizeof(int32_t), &member,
                    sizeof(member));
        ++output;
      }
    }
    dataset.raw_elements += config.array_length;
    dataset.distinct_elements += unique_length;
    ++dataset.raw_lengths[config.array_length];
    ++dataset.unique_lengths[unique_length];
  }
  dataset.fingerprint = DatasetFingerprint(dataset);
  return dataset;
}

JsonObject Describe(const ArrayData &dataset) {
  auto result = ConfigJson(dataset.config);
  result.set("num_docs", JsonValue(dataset.num_docs));
  result.set("seed", JsonValue(dataset.seed));
  result.set("data_fingerprint", JsonStringValue(dataset.fingerprint));
  auto stats = FrequencyStats(dataset.frequencies);
  stats.set("null_count", JsonValue(dataset.null_count));
  stats.set("empty_count", JsonValue(dataset.empty_count));
  stats.set("duplicate_count", JsonValue(dataset.duplicate_count));
  stats.set("raw_elements", JsonValue(dataset.raw_elements));
  stats.set("distinct_elements", JsonValue(dataset.distinct_elements));
  stats.set("raw_length_histogram", LengthHistogram(dataset.raw_lengths));
  stats.set("unique_length_histogram", LengthHistogram(dataset.unique_lengths));
  result.set("stats", stats);
  return result;
}

// Keep member ordering identical to the index interface's native-byte inputs.
JsonArray MemberInputs(std::vector<std::vector<int32_t>> pool) {
  JsonArray inputs;
  std::vector<std::vector<int32_t>> unique;
  for (auto &terms : pool) {
    std::sort(terms.begin(), terms.end(), [](int32_t a, int32_t b) {
      return EncodeMember(a) < EncodeMember(b);
    });
    if (std::find(unique.begin(), unique.end(), terms) != unique.end())
      continue;
    unique.push_back(terms);
    JsonArray values;
    for (int32_t term : terms) values.push(JsonValue(term));
    inputs.push(values);
  }
  if (inputs.empty())
    throw std::runtime_error("Array case has no query inputs");
  return inputs;
}

JsonArray PrepareQueryInputs(const ArrayData &dataset) {
  const auto &config = dataset.config;
  std::vector<std::vector<int32_t>> single, missing, global, adjacent,
      witnesses;
  std::vector<int32_t> observed;
  for (uint32_t term = 0; term < config.cardinality; ++term)
    if (dataset.frequencies[term]) observed.push_back(term);
  for (int32_t term : SampleTerms(observed)) {
    single.push_back({term});
    missing.push_back({term, static_cast<int32_t>(config.cardinality + 17)});
  }
  Random random(dataset.seed ^ 0x1d8e4e27c47d124fULL);
  for (size_t i = 0; i < kQueryPoolSize; ++i) {
    std::vector<int32_t> terms;
    while (terms.size() < std::min<uint32_t>(8, config.cardinality)) {
      const auto term =
          static_cast<int32_t>(random.bounded(config.cardinality));
      if (std::find(terms.begin(), terms.end(), term) == terms.end())
        terms.push_back(term);
    }
    global.push_back(std::move(terms));
    const auto anchor =
        static_cast<int32_t>(i * config.cardinality / kQueryPoolSize);
    adjacent.push_back(
        {anchor, static_cast<int32_t>((anchor + 1) % config.cardinality)});
  }
  JsonArray cases;
  cases.push(InputRecord("any_1_hit", MemberInputs(single)));
  cases.push(InputRecord("any_8_global", MemberInputs(global)));
  cases.push(InputRecord("all_2_adjacent", MemberInputs(adjacent)));
  if (config.array_length > 1) {
    auto source_kind = kPlain;
    size_t source_count =
        std::count(dataset.kinds.begin(), dataset.kinds.end(), source_kind);
    // A configuration with 100% duplicates still has valid witness documents.
    if (!source_count) {
      source_kind = kDuplicate;
      source_count = dataset.duplicate_count;
    }
    const size_t count = std::min(kQueryPoolSize, source_count);
    size_t ordinal = 0, selected = 0;
    for (uint32_t id = 0; id < dataset.num_docs && selected < count; ++id) {
      if (dataset.kinds[id] != source_kind) continue;
      if (ordinal == selected * source_count / count) {
        std::vector<int32_t> terms;
        for (uint32_t i = 0; i < std::min<uint32_t>(8, config.array_length);
             ++i) {
          const auto term = ReadElement(dataset.values[id], i);
          if (std::find(terms.begin(), terms.end(), term) == terms.end())
            terms.push_back(term);
        }
        witnesses.push_back(std::move(terms));
        ++selected;
      }
      ++ordinal;
    }
    cases.push(InputRecord("all_m_witness", MemberInputs(witnesses)));
  }
  cases.push(InputRecord("all_8_global", MemberInputs(global)));
  cases.push(InputRecord("all_hit_miss", MemberInputs(missing)));
  cases.push(InputRecord("not_any_2", MemberInputs(adjacent)));
  cases.push(InputRecord("not_all_2", MemberInputs(adjacent)));
  JsonArray no_operand, zero, length;
  no_operand.push(JsonValue());
  zero.push(JsonValue(0));
  length.push(JsonValue(config.array_length));
  cases.push(InputRecord("is_null", no_operand));
  cases.push(InputRecord("is_not_null", no_operand));
  cases.push(InputRecord("len_eq_zero", zero));
  cases.push(InputRecord("len_eq_raw", length));
  return cases;
}

bool Contains(const std::string &bytes, int32_t term) {
  for (size_t i = 0; i < bytes.size() / sizeof(int32_t); ++i)
    if (ReadElement(bytes, i) == term) return true;
  return false;
}

template <typename Search, typename Predicate>
JsonObject VerifyOrMeasureQuery(const std::string &mode,
                                const ArrayOptions &options,
                                const std::string &name,
                                const JsonArray &inputs, uint32_t num_docs,
                                const JsonObject &verification,
                                InvertedColumnIndexer *column, Search search,
                                Predicate matches) {
  if (mode == "verify") {
    JsonArray counts, fingerprints, ratios;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto checked =
          VerifyIds(num_docs, search(*column, i),
                    [&](uint32_t id) { return matches(id, i); });
      if ((name == "all_m_witness" && !checked.count) ||
          (name == "all_hit_miss" && checked.count))
        throw std::runtime_error("Array case expectation failed: " + name);
      counts.push(JsonValue(checked.count));
      fingerprints.push(JsonStringValue(checked.fingerprint));
      ratios.push(JsonValue(static_cast<double>(checked.count) / num_docs));
    }
    JsonObject result;
    result.set("name", JsonStringValue(name));
    result.set("query_fingerprint",
               JsonStringValue(InputFingerprint(name, inputs)));
    result.set("counts", counts);
    result.set("id_fingerprints", fingerprints);
    result.set("match_ratios", ratios);
    return result;
  }
  const auto expected = VerifiedCounts(verification, inputs.size(), num_docs);
  auto before = MemorySnapshot();
  IndexFixture index(DataType::ARRAY_INT32, options.index_path, true);
  const auto measured = MeasurePool(
      inputs.size(), options.warmup_seconds, options.measure_seconds, [&]() {
        uint64_t checksum = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
          auto result = search(index.column(), i);
          if (!result)
            throw std::runtime_error("Array query failed during timing: " +
                                     name);
          const auto count = result->count();
          if (count != expected[i])
            throw std::runtime_error("Array result count changed: " + name);
          checksum += count;
        }
        return checksum;
      });
  auto after = MemorySnapshot();
  index.Close();
  auto result = QueryReport(name, inputs, measured, options.warmup_seconds,
                            options.measure_seconds, before, after);
  result.set("match_counts", verification["counts"]);
  return result;
}

JsonObject ExecuteQuery(const std::string &mode, const ArrayOptions &options,
                        const JsonObject &record, uint32_t num_docs,
                        const JsonObject &verification, const ArrayData *data,
                        InvertedColumnIndexer *column) {
  const auto name = Text(record, "name");
  const auto inputs = record["inputs"].as_array();
  auto run = [&](auto search, auto matches) {
    return VerifyOrMeasureQuery(mode, options, name, inputs, num_docs,
                                verification, column, search, matches);
  };
  if (name == "is_null" || name == "is_not_null") {
    if (inputs.size() != 1 || !inputs[0].is_null())
      throw std::invalid_argument("NULL case takes no operand");
    if (name == "is_null")
      return run([](auto &index, size_t) { return index.search_null(); },
                 [&](uint32_t id, size_t) { return data->is_null[id] != 0; });
    return run([](auto &index, size_t) { return index.search_non_null(); },
               [&](uint32_t id, size_t) { return data->is_null[id] == 0; });
  }
  if (name == "len_eq_zero" || name == "len_eq_raw") {
    std::vector<uint32_t> lengths;
    for (const auto &value : inputs) {
      if (!value.is_integer() || value.as_integer() < 0 ||
          static_cast<uint64_t>(value.as_integer()) > UINT32_MAX)
        throw std::invalid_argument("Array length must be uint32");
      lengths.push_back(value.as_integer());
    }
    return run(
        [&](auto &index, size_t i) {
          return index.search_array_len(lengths[i], CompareOp::EQ);
        },
        [&](uint32_t id, size_t i) {
          return !data->is_null[id] &&
                 data->values[id].size() / sizeof(int32_t) == lengths[i];
        });
  }
  // Decode and encode the small operand pool before opening or timing the
  // index.
  std::vector<std::vector<int32_t>> operands;
  std::vector<std::vector<std::string>> encoded;
  for (const auto &input : inputs) {
    if (!input.is_array()) throw std::invalid_argument("Expected member array");
    const auto values = input.as_array();
    if (values.empty()) throw std::invalid_argument("Member array is empty");
    std::vector<int32_t> terms;
    std::vector<std::string> bytes;
    for (const auto &value : values) {
      if (!value.is_integer() || value.as_integer() < INT32_MIN ||
          value.as_integer() > INT32_MAX)
        throw std::invalid_argument("Member must be INT32");
      terms.push_back(value.as_integer());
      bytes.push_back(EncodeMember(terms.back()));
    }
    operands.push_back(std::move(terms));
    encoded.push_back(std::move(bytes));
  }
  auto any = [&](uint32_t id, size_t i) {
    for (auto term : operands[i])
      if (Contains(data->values[id], term)) return true;
    return false;
  };
  auto all = [&](uint32_t id, size_t i) {
    for (auto term : operands[i])
      if (!Contains(data->values[id], term)) return false;
    return true;
  };
  if (name == "any_1_hit" || name == "any_8_global")
    return run(
        [&](auto &index, size_t i) {
          return index.multi_search(encoded[i], CompareOp::CONTAIN_ANY);
        },
        [&](uint32_t id, size_t i) {
          return !data->is_null[id] && any(id, i);
        });
  if (name == "all_2_adjacent" || name == "all_m_witness" ||
      name == "all_8_global" || name == "all_hit_miss")
    return run(
        [&](auto &index, size_t i) {
          return index.multi_search(encoded[i], CompareOp::CONTAIN_ALL);
        },
        [&](uint32_t id, size_t i) {
          return !data->is_null[id] && all(id, i);
        });
  if (name == "not_any_2")
    return run(
        [&](auto &index, size_t i) {
          return index.multi_search(encoded[i], CompareOp::NOT_CONTAIN_ANY);
        },
        [&](uint32_t id, size_t i) {
          return !data->is_null[id] && !any(id, i);
        });
  if (name == "not_all_2")
    return run(
        [&](auto &index, size_t i) {
          return index.multi_search(encoded[i], CompareOp::NOT_CONTAIN_ALL);
        },
        [&](uint32_t id, size_t i) {
          return !data->is_null[id] && !all(id, i);
        });
  throw std::invalid_argument("Unknown array case: " + name);
}

JsonObject Build(const ArrayOptions &options) {
  if (std::filesystem::exists(options.index_path))
    throw std::invalid_argument("Array index directory already exists");
  const auto config =
      Configure(Profile(options.profile), options.dataset_overrides);
  // Dataset generation and query preparation are outside build measurements.
  auto dataset = GenerateData(config, options.num_docs, options.seed);
  const auto cases = PrepareQueryInputs(dataset);
  auto before = MemorySnapshot();
  IndexFixture index(DataType::ARRAY_INT32, options.index_path, false);
  auto &column = index.column();
  const auto begin = Clock::now();
  for (uint32_t id = 0; id < dataset.num_docs; ++id) {
    const auto status = dataset.is_null[id]
                            ? column.insert_null(id)
                            : column.insert(id, dataset.values[id]);
    if (!status.ok())
      throw std::runtime_error("Array insert failed at doc " +
                               std::to_string(id));
  }
  const auto inserted = Clock::now();
  index.Seal();
  const auto sealed = Clock::now();
  auto after = MemorySnapshot();
  index.Close();
  auto result = Report("build");
  result.set("dataset", Describe(dataset));
  result.set("cases", cases);
  result.set(
      "index_path",
      JsonStringValue(
          std::filesystem::weakly_canonical(options.index_path).string()));
  result.set("insert_seconds", JsonValue(Seconds(begin, inserted)));
  result.set("seal_seconds", JsonValue(Seconds(inserted, sealed)));
  result.set("build_seconds", JsonValue(Seconds(begin, sealed)));
  result.set("docs_per_second",
             JsonValue(dataset.num_docs / Seconds(begin, sealed)));
  const auto disk = DiskUsage(options.index_path);
  result.set("index_file_bytes", disk["index_file_bytes"]);
  result.set("index_allocated_bytes", disk["index_allocated_bytes"]);
  result.set("bytes_per_doc",
             JsonValue(static_cast<double>(Number(disk, "index_file_bytes")) /
                       dataset.num_docs));
  result.set("memory_before", before);
  result.set("memory_after", after);
  result.set("environment", EnvironmentJson());
  result.set("metric_names", BuildMetricNames());
  return result;
}
}  // namespace

JsonObject ArrayCatalog(const ArrayOptions &options) {
  JsonArray profiles;
  for (auto config : Profiles()) {
    if (!options.profile.empty() && config.name != options.profile) continue;
    config = Configure(config, options.dataset_overrides);
    auto profile = ConfigJson(config);
    JsonArray names;
    for (const auto &name : CaseNames(config))
      names.push(JsonStringValue(name));
    profile.set("cases", names);
    profiles.push(profile);
  }
  JsonObject result;
  result.set("name", JsonStringValue("array"));
  result.set("profiles", profiles);
  return result;
}

JsonObject RunArray(const std::string &mode, const ArrayOptions &options) {
  if (mode == "build") return Build(options);
  if (mode != "verify" && mode != "query")
    throw std::invalid_argument("Unknown array benchmark mode: " + mode);
  if (!options.dataset_overrides.empty())
    throw std::invalid_argument("Array config overrides apply only to build");
  const auto build = LoadBuild(options.build_report_path, options.index_path);
  const auto description = build["dataset"].as_object();
  const auto config = ReadConfig(description);
  const auto num_docs = Number(description, "num_docs");
  if (!num_docs || num_docs > UINT32_MAX)
    throw std::invalid_argument("Invalid array document count");
  const auto cases = build["cases"].as_array();
  const auto selected = SelectInputs(cases, options.case_filter);
  if (mode == "verify") {
    // Regenerate raw rows for the independent oracle; reuse one index for all
    // cases.
    auto dataset = GenerateData(config, num_docs, Number(description, "seed"));
    CheckRegenerated(build, dataset.fingerprint, PrepareQueryInputs(dataset));
    IndexFixture index(DataType::ARRAY_INT32, options.index_path, true);
    JsonArray checked;
    for (const auto &record : selected)
      checked.push(ExecuteQuery(mode, options, record, num_docs, {}, &dataset,
                                &index.column()));
    auto result = Report("verify");
    result.set("passed", JsonValue(true));
    result.set("data_fingerprint", JsonStringValue(dataset.fingerprint));
    result.set("cases", checked);
    return result;
  }
  if (options.case_filter.empty() || selected.size() != 1)
    throw std::invalid_argument("Array query requires exactly one case");
  // Measurement loads only persisted query inputs and verified counts.
  const auto verification =
      LoadVerification(options.verification_path, build, selected.front());
  auto result = ExecuteQuery(mode, options, selected.front(), num_docs,
                             verification, nullptr, nullptr);
  result.set("data_fingerprint", description["data_fingerprint"]);
  return result;
}
}  // namespace zvec::benchmarks::inverted
