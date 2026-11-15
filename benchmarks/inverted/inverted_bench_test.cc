// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <algorithm>
#include <cstring>
#include <numeric>
#include <set>
#include <gtest/gtest.h>
#include "benchmark_case.h"
#include "metrics.h"

namespace zvec::benchmarks::inverted {
namespace {

constexpr uint64_t kSeed = 20261115;

TEST(InvertedBenchmarkReports, PreserveBinaryQueriesAndEscapedStrings) {
  const std::string text =
      "a path with \"quotes\", a backslash \\ and\na newline";
  ailego::JsonValue parsed;
  JsonObject object;
  object.set("text", JsonStringValue(text));
  ASSERT_TRUE(
      parsed.parse(ailego::JsonValue(object).as_json_string().as_stl_string()));
  EXPECT_EQ(JsonStringText(parsed.as_object()["text"]), text);

  const QueryCase original{"binary_query",
                           {{QueryOp::kEq, {std::string("\0\xff\"\\", 4)}, 0}},
                           false};
  auto encoded = CasesJson({original});
  ASSERT_TRUE(parsed.parse(
      ailego::JsonValue(encoded).as_json_string().as_stl_string()));
  const auto restored = ParseCases(parsed.as_array());
  ASSERT_EQ(restored.size(), 1);
  EXPECT_EQ(restored.front().inputs.front().values,
            original.inputs.front().values);
  EXPECT_EQ(QueryFingerprint(restored.front()), QueryFingerprint(original));

  auto changed = encoded[0].as_object();
  changed["query_fingerprint"] =
      JsonStringValue("not-the-original-fingerprint");
  JsonArray corrupted;
  corrupted.push(changed);
  EXPECT_THROW(ParseCases(corrupted), std::invalid_argument);
}

template <typename T>
std::string Bytes(T value) {
  std::string result(sizeof(value), '\0');
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

std::string ArrayBytes(std::initializer_list<int32_t> values) {
  std::string result;
  for (int32_t value : values) result += Bytes(value);
  return result;
}

std::vector<uint32_t> MatchingIds(const TypeSuite &suite,
                                  const PreparedDataset &dataset,
                                  const QueryInput &query) {
  std::vector<uint32_t> ids;
  for (uint32_t id = 0; id < dataset.num_docs; ++id) {
    if (suite.matches(dataset, id, query)) ids.push_back(id);
  }
  return ids;
}

TEST(InvertedBenchmarkOracle, Int64UsesSignedValuesAndExcludesNullFromNe) {
  auto suite = Int64Suite();
  PreparedDataset dataset;
  dataset.num_docs = 4;
  dataset.is_null = {0, 0, 1, 0};
  dataset.values = {Bytes<int64_t>(-5), Bytes<int64_t>(0), "",
                    Bytes<int64_t>(7)};
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kEq, {Bytes<int64_t>(0)}, 0}),
            (std::vector<uint32_t>{1}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kNe, {Bytes<int64_t>(0)}, 0}),
            (std::vector<uint32_t>{0, 3}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kLt, {Bytes<int64_t>(0)}, 0}),
            (std::vector<uint32_t>{0}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kGe, {Bytes<int64_t>(0)}, 0}),
            (std::vector<uint32_t>{1, 3}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kIsNull, {}, 0}),
            (std::vector<uint32_t>{2}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kIsNotNull, {}, 0}),
            (std::vector<uint32_t>{0, 1, 3}));
}

TEST(InvertedBenchmarkOracle, ArraysSeparateNullEmptyMultiplicityAndLength) {
  auto suite = ArrayInt32Suite();
  PreparedDataset dataset;
  dataset.num_docs = 5;
  dataset.is_null = {1, 0, 0, 0, 0};
  dataset.values = {"", "", ArrayBytes({1, 1, 2, 2}), ArrayBytes({2, 3}),
                    ArrayBytes({-7})};
  const std::vector<std::string> pair{Bytes<int32_t>(1), Bytes<int32_t>(2)};
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kAny, pair, 0}),
            (std::vector<uint32_t>{2, 3}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kAll, pair, 0}),
            (std::vector<uint32_t>{2}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kNotAny, pair, 0}),
            (std::vector<uint32_t>{1, 4}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kNotAll, pair, 0}),
            (std::vector<uint32_t>{1, 3, 4}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kLengthEq, {}, 0}),
            (std::vector<uint32_t>{1}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kLengthEq, {}, 4}),
            (std::vector<uint32_t>{2}));
  EXPECT_EQ(
      MatchingIds(suite, dataset,
                  {QueryOp::kAll, {Bytes<int32_t>(1), Bytes<int32_t>(1)}, 0}),
      (std::vector<uint32_t>{2}));
  EXPECT_EQ(MatchingIds(suite, dataset, {QueryOp::kIsNotNull, {}, 0}),
            (std::vector<uint32_t>{1, 2, 3, 4}));
}

TEST(InvertedBenchmarkData, UniformAndHotspotHaveExactQuotas) {
  const auto suite = Int64Suite();
  auto uniform = suite.prepare(suite.profiles[0], 1000, kSeed);
  EXPECT_EQ(uniform.stats.null_count, 100);
  EXPECT_EQ(uniform.stats.observed_cardinality, 100);
  for (auto count : uniform.stats.term_doc_frequencies) EXPECT_EQ(count, 9);
  auto hotspot = suite.prepare(suite.profiles[2], 100000, kSeed);
  EXPECT_EQ(hotspot.stats.null_count, 10000);
  EXPECT_EQ(hotspot.stats.observed_cardinality, 10000);
  EXPECT_EQ(std::count(hotspot.stats.term_doc_frequencies.begin(),
                       hotspot.stats.term_doc_frequencies.end(), 72),
            1000);
  EXPECT_EQ(std::count(hotspot.stats.term_doc_frequencies.begin(),
                       hotspot.stats.term_doc_frequencies.end(), 2),
            9000);
  EXPECT_EQ(
      std::accumulate(hotspot.stats.term_doc_frequencies.begin(),
                      hotspot.stats.term_doc_frequencies.end(), uint64_t{0}),
      90000);
}

TEST(InvertedBenchmarkData,
     ArrayRawLengthAndDocumentFrequencyIgnoreDuplicates) {
  const auto suite = ArrayInt32Suite();
  for (const auto &profile : suite.profiles) {
    SCOPED_TRACE(profile.name);
    const auto dataset = suite.prepare(profile, 1000, kSeed);
    EXPECT_EQ(dataset.stats.null_count, 100);
    EXPECT_EQ(dataset.stats.empty_count, 100);
    EXPECT_EQ(dataset.stats.duplicate_count,
              profile.array_length == 1 ? 0 : 160);
    EXPECT_EQ(dataset.stats.raw_elements, 800U * profile.array_length);
    EXPECT_EQ(dataset.stats.distinct_elements,
              (profile.array_length == 1 ? 800U : 720U) * profile.array_length);
    std::vector<uint64_t> observed_df(profile.cardinality, 0);
    uint64_t nulls = 0, empties = 0, repeats = 0;
    for (uint32_t id = 0; id < dataset.num_docs; ++id) {
      if (dataset.is_null[id]) {
        ++nulls;
        EXPECT_TRUE(dataset.values[id].empty());
        continue;
      }
      const auto &bytes = dataset.values[id];
      if (bytes.empty()) {
        ++empties;
        continue;
      }
      ASSERT_EQ(bytes.size(), profile.array_length * sizeof(int32_t));
      std::multiset<int32_t> raw;
      for (size_t offset = 0; offset < bytes.size();
           offset += sizeof(int32_t)) {
        int32_t value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        raw.insert(value);
      }
      const std::set<int32_t> unique(raw.begin(), raw.end());
      if (unique.size() != profile.array_length) {
        ++repeats;
        EXPECT_EQ(unique.size(), profile.array_length / 2);
        for (int32_t value : unique) EXPECT_EQ(raw.count(value), 2);
      }
      for (int32_t value : unique) {
        ASSERT_GE(value, 0);
        ASSERT_LT(static_cast<uint32_t>(value), profile.cardinality);
        ++observed_df[value];
      }
    }
    EXPECT_EQ(nulls, dataset.stats.null_count);
    EXPECT_EQ(empties, dataset.stats.empty_count);
    EXPECT_EQ(repeats, dataset.stats.duplicate_count);
    EXPECT_EQ(observed_df, dataset.stats.term_doc_frequencies);
    EXPECT_EQ(dataset.stats.raw_length_histogram.at(0), 100);
    EXPECT_EQ(dataset.stats.raw_length_histogram.at(profile.array_length), 800);
  }
}

TEST(InvertedBenchmarkData, StableDataAndDeclaredQueryMatrix) {
  for (const auto &suite : {Int64Suite(), ArrayInt32Suite()}) {
    for (const auto &profile : suite.profiles) {
      SCOPED_TRACE(profile.name);
      const auto first = suite.prepare(profile, 200, kSeed);
      const auto second = suite.prepare(profile, 200, kSeed);
      EXPECT_EQ(first.values, second.values);
      EXPECT_EQ(first.is_null, second.is_null);
      EXPECT_EQ(first.fingerprint, second.fingerprint);
      EXPECT_EQ(first.fingerprint.size(), 16);
      EXPECT_NE(first.fingerprint,
                suite.prepare(profile, 200, kSeed + 1).fingerprint);
      std::vector<std::string> actual_names;
      for (const auto &query_case : first.cases) {
        actual_names.push_back(query_case.name);
        ASSERT_FALSE(query_case.inputs.empty());
        ASSERT_LE(query_case.inputs.size(), 8);
        for (size_t i = 0; i < query_case.inputs.size(); ++i) {
          const auto &left = query_case.inputs[i];
          for (size_t j = i + 1; j < query_case.inputs.size(); ++j) {
            const auto &right = query_case.inputs[j];
            EXPECT_FALSE(left.op == right.op && left.length == right.length &&
                         left.values == right.values);
          }
        }
        for (const auto &input : query_case.inputs) {
          const auto ids = MatchingIds(suite, first, input);
          if (query_case.from_documents || query_case.name == "eq_hit" ||
              query_case.name == "eq_hot" || query_case.name == "eq_cold" ||
              query_case.name == "any_1_hit")
            EXPECT_FALSE(ids.empty());
          if (query_case.name == "eq_miss" ||
              query_case.name == "all_hit_miss") {
            EXPECT_TRUE(ids.empty());
          }
        }
      }
      EXPECT_EQ(actual_names, suite.case_names(profile));
      if (profile.array_length == 1) {
        EXPECT_EQ(std::count(actual_names.begin(), actual_names.end(),
                             "all_m_witness"),
                  0);
      }
    }
  }
}

TEST(InvertedBenchmarkData, WitnessPoolsDeduplicateEquivalentMemberSets) {
  const auto suite = ArrayInt32Suite();
  auto profile = suite.profiles[1];
  profile.cardinality = 8;
  const auto dataset = suite.prepare(profile, 1000, kSeed);
  for (const auto &query_case : dataset.cases) {
    if (query_case.name == "all_m_witness" ||
        query_case.name == "all_8_global" ||
        query_case.name == "any_8_global") {
      ASSERT_EQ(query_case.inputs.size(), 1);
      EXPECT_TRUE(std::is_sorted(query_case.inputs[0].values.begin(),
                                 query_case.inputs[0].values.end()));
    }
    if (query_case.name == "all_m_witness") {
      EXPECT_EQ(MatchingIds(suite, dataset, query_case.inputs[0]).size(), 640);
    }
  }
}

}  // namespace
}  // namespace zvec::benchmarks::inverted
