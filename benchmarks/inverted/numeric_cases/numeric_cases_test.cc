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
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "shared/index_fixture.h"
#include "shared/test_utils.h"

namespace zvec::benchmarks::inverted {
namespace {

using test::JsonText;
using test::NamedRecord;

static_assert(!std::is_default_constructible_v<NumericProfile>,
              "A numeric workload must explicitly select its data type");

JsonObject BuildAndVerify(const NumericProfile &profile,
                          const std::filesystem::path &directory,
                          JsonObject *build_out = nullptr) {
  const auto build = BuildNumeric(profile, directory.string());
  WriteJson((directory / "build.json").string(), build);
  const auto verified = VerifyNumeric(directory.string());
  WriteJson((directory / "verify.json").string(), verified);
  if (build_out) *build_out = build;
  return verified;
}

std::string Encode(DataType type, int64_t value) {
  if (type == DataType::INT32) {
    const auto narrow = static_cast<int32_t>(value);
    return std::string(reinterpret_cast<const char *>(&narrow), sizeof(narrow));
  }
  return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

// In an ascending workload, range results form prefixes/suffixes of the
// non-NULL document IDs, and each EQ result forms one contiguous segment.
bool QueryResultsAreOrdered(const NumericProfile &profile,
                            const std::filesystem::path &directory,
                            const JsonObject &build) {
  IndexFixture index(profile.data_type, (directory / "index").string(), true);
  const auto nonnull = index.column().search_non_null();
  if (!nonnull) throw std::runtime_error("Cannot read non-NULL rows");
  std::vector<uint32_t> ids;
  for (auto it = nonnull->create_iterator(); it->valid(); it->next())
    ids.push_back(it->doc_id());
  const auto cases = build["cases"].as_array();
  bool ordered = true;
  for (const auto &value : cases) {
    const auto record = value.as_object();
    const auto name = Text(record, "name");
    const bool equal = name.find("eq_") == 0;
    const bool less = name.find("lt_") == 0;
    const bool greater = name.find("ge_") == 0;
    if (!equal && !less && !greater) continue;
    const auto inputs = record["inputs"].as_array();
    for (const auto &input : inputs) {
      const auto result =
          index.column().search(Encode(profile.data_type, input.as_integer()),
                                equal  ? CompareOp::EQ
                                : less ? CompareOp::LT
                                       : CompareOp::GE);
      if (!result) throw std::runtime_error("Cannot inspect query order");
      size_t first = ids.size(), last = 0, count = 0;
      for (size_t i = 0; i < ids.size(); ++i) {
        const bool contains = result->contains(ids[i]);
        if (contains) {
          first = std::min(first, i);
          last = i;
          ++count;
        }
        if (less) ordered &= contains == (i < result->count());
        if (greater) ordered &= contains == (i + result->count() >= ids.size());
      }
      if (equal && count) ordered &= last - first + 1 == count;
    }
  }
  return ordered;
}

TEST(InvertedBenchmarkNumeric,
     ProfileMatrixPreservesValuesAndNullsAcrossOrders) {
  test::TemporaryDirectory directory;
  unsigned run = 0;
  for (const auto type : {DataType::INT32, DataType::INT64}) {
    for (const auto range : {NumericValueRange::ZERO_TO_TYPE_MAX,
                             NumericValueRange::ZERO_TO_30000}) {
      for (const auto distribution :
           {NumericDistribution::UNIFORM, NumericDistribution::HOTSPOT_MODERATE,
            NumericDistribution::HOTSPOT_EXTREME}) {
        const uint32_t hot_value_count =
            distribution == NumericDistribution::UNIFORM
                ? 0
                : (distribution == NumericDistribution::HOTSPOT_MODERATE ? 3000
                                                                         : 300);
        const uint32_t hot_percent =
            distribution == NumericDistribution::UNIFORM
                ? 0
                : (distribution == NumericDistribution::HOTSPOT_MODERATE ? 50
                                                                         : 90);
        for (const auto null_ratio :
             {NumericNullRatio::NONE, NumericNullRatio::TEN_PERCENT,
              NumericNullRatio::FIFTY_PERCENT}) {
          NumericProfile profile(type);
          profile.num_docs = NumericDocCount::ONE_HUNDRED_THOUSAND;
          const auto num_docs = static_cast<uint32_t>(profile.num_docs);
          profile.value_range = range;
          profile.distribution = distribution;
          profile.null_ratio = null_ratio;
          const uint32_t null_count =
              null_ratio == NumericNullRatio::NONE
                  ? 0
                  : num_docs /
                        (null_ratio == NumericNullRatio::TEN_PERCENT ? 10 : 2);
          const uint32_t nonnull = num_docs - null_count;
          const int64_t maximum = range == NumericValueRange::ZERO_TO_30000
                                      ? 30000
                                  : type == DataType::INT32 ? INT32_MAX
                                                            : INT64_MAX;
          JsonObject shuffled_build, shuffled_verify;
          for (const auto order :
               {NumericValueOrder::SHUFFLED, NumericValueOrder::ASCENDING}) {
            profile.value_order = order;
            const auto path = directory.path() / std::to_string(run++);
            SCOPED_TRACE(path.string());
            JsonObject build;
            const auto verified = BuildAndVerify(profile, path, &build);
            const auto description = build["dataset"].as_object();
            const auto stats = description["stats"].as_object();
            const auto hot_values = stats["hot_values"].as_array();
            EXPECT_EQ(Number(description, "num_docs"), num_docs);
            EXPECT_EQ(Number(description, "seed"), 20261115);
            EXPECT_EQ(description["value_min"].as_integer(), 0);
            EXPECT_EQ(description["value_max"].as_integer(), maximum);
            EXPECT_EQ(Number(stats, "null_count"), null_count);
            EXPECT_GT(Number(stats, "observed_cardinality"), 0);
            EXPECT_LE(Number(stats, "observed_cardinality"), nonnull);
            EXPECT_GE(stats["observed_value_min"].as_integer(), 0);
            EXPECT_LE(stats["observed_value_max"].as_integer(), maximum);
            EXPECT_FALSE(Text(stats, "value_frequencies_fingerprint").empty());
            // Query workers must not load the entire distinct-value histogram.
            EXPECT_FALSE(stats["observed_values"].is_array());
            EXPECT_FALSE(stats["term_doc_frequencies"].is_array());
            std::vector<int64_t> hot;
            for (const auto &value : hot_values)
              hot.push_back(value.as_integer());
            EXPECT_TRUE(std::is_sorted(hot.begin(), hot.end()));
            EXPECT_EQ(std::adjacent_find(hot.begin(), hot.end()), hot.end());
            EXPECT_EQ(hot.size(), hot_value_count);
            if (distribution == NumericDistribution::UNIFORM) {
              EXPECT_TRUE(description["hot_value_count"].is_null());
              EXPECT_TRUE(description["hot_doc_ratio"].is_null());
            } else {
              EXPECT_EQ(Number(description, "hot_value_count"),
                        hot_value_count);
              EXPECT_DOUBLE_EQ(description["hot_doc_ratio"].as_float(),
                               hot_percent / 100.0);
            }
            uint64_t hot_docs = 0;
            {
              IndexFixture index(type, (path / "index").string(), true);
              for (const auto value : hot) {
                EXPECT_GE(value, 0);
                EXPECT_LE(value, maximum);
                const auto result =
                    index.column().search(Encode(type, value), CompareOp::EQ);
                ASSERT_TRUE(result);
                hot_docs += result->count();
              }
            }
            EXPECT_EQ(hot_docs,
                      static_cast<uint64_t>(nonnull) * hot_percent / 100);
            EXPECT_EQ(Number(stats, "hot_doc_count"), hot_docs);
            // Wide-range sampling must not truncate to 32 bits or double's
            // exact integer range, nor enumerate a small prefix of the domain.
            if (type == DataType::INT64 &&
                range == NumericValueRange::ZERO_TO_TYPE_MAX)
              EXPECT_GT(stats["observed_value_max"].as_integer(), INT64_C(1)
                                                                      << 53);
            else
              EXPECT_GT(stats["observed_value_max"].as_integer(), maximum / 2);
            const auto misses =
                NamedRecord(verified["cases"].as_array(), "eq_miss");
            const auto miss_counts = misses["counts"].as_array();
            for (const auto &count : miss_counts)
              EXPECT_EQ(count.as_integer(), 0);
            const auto miss_inputs =
                NamedRecord(build["cases"].as_array(), "eq_miss")["inputs"]
                    .as_array();
            for (const auto &value : miss_inputs)
              EXPECT_LT(value.as_integer(), 0);
            const auto nulls =
                NamedRecord(verified["cases"].as_array(), "is_null");
            EXPECT_EQ(nulls["counts"].as_array()[0].as_integer(), null_count);
            if (order == NumericValueOrder::SHUFFLED) {
              EXPECT_FALSE(QueryResultsAreOrdered(profile, path, build));
              shuffled_build = build;
              shuffled_verify = verified;
            } else {
              EXPECT_TRUE(QueryResultsAreOrdered(profile, path, build));
              EXPECT_EQ(
                  JsonText(description["stats"]),
                  JsonText(shuffled_build["dataset"].as_object()["stats"]));
              EXPECT_EQ(JsonText(build["cases"]),
                        JsonText(shuffled_build["cases"]));
              const auto previous_nulls =
                  NamedRecord(shuffled_verify["cases"].as_array(), "is_null");
              EXPECT_EQ(JsonText(nulls["id_fingerprints"]),
                        JsonText(previous_nulls["id_fingerprints"]));
              const auto cases = verified["cases"].as_array();
              for (const auto &value : cases) {
                const auto result = value.as_object();
                const auto previous = NamedRecord(
                    shuffled_verify["cases"].as_array(), Text(result, "name"));
                EXPECT_EQ(JsonText(result["counts"]),
                          JsonText(previous["counts"]));
              }
              EXPECT_NE(Text(description, "data_fingerprint"),
                        Text(shuffled_build["dataset"].as_object(),
                             "data_fingerprint"));
            }
          }
        }
      }
    }
  }
}

TEST(InvertedBenchmarkNumeric, SmallRangeTypesShareLogicalDataAndIdResults) {
  test::TemporaryDirectory directory;
  for (const auto distribution :
       {NumericDistribution::UNIFORM, NumericDistribution::HOTSPOT_MODERATE,
        NumericDistribution::HOTSPOT_EXTREME}) {
    JsonObject previous_build, previous_verify;
    for (const auto type : {DataType::INT32, DataType::INT64}) {
      NumericProfile profile(type);
      profile.distribution = distribution;
      const auto path =
          directory.path() / (std::to_string(static_cast<int>(distribution)) +
                              "-" + std::to_string(static_cast<int>(type)));
      JsonObject build;
      const auto verified = BuildAndVerify(profile, path, &build);
      if (!previous_build.empty()) {
        EXPECT_EQ(JsonText(build["dataset"].as_object()["stats"]),
                  JsonText(previous_build["dataset"].as_object()["stats"]));
        EXPECT_EQ(JsonText(build["cases"]), JsonText(previous_build["cases"]));
        EXPECT_EQ(JsonText(verified["cases"]),
                  JsonText(previous_verify["cases"]));
      }
      previous_build = build;
      previous_verify = verified;
    }
  }
}

TEST(InvertedBenchmarkNumeric,
     DocumentCountPresetsAreDescribedWithoutBuilding) {
  EXPECT_EQ(NumericProfile(DataType::INT64).num_docs,
            NumericDocCount::ONE_HUNDRED_THOUSAND);
  for (const auto type : {DataType::INT32, DataType::INT64}) {
    for (const auto &[preset, count] :
         std::vector<std::pair<NumericDocCount, uint32_t>>{
             {NumericDocCount::ONE_HUNDRED_THOUSAND, 100000},
             {NumericDocCount::TEN_MILLION, 10000000}}) {
      NumericProfile profile(type);
      profile.num_docs = preset;
      EXPECT_EQ(Number(DescribeNumericProfile(profile), "num_docs"), count);
    }
  }
  for (uint32_t count : {0U, 1U, 257U, 1234U, 100001U, 10000001U, UINT32_MAX}) {
    NumericProfile profile(DataType::INT64);
    profile.num_docs = static_cast<NumericDocCount>(count);
    EXPECT_THROW(DescribeNumericProfile(profile), std::invalid_argument)
        << count;
  }
}

TEST(InvertedBenchmarkNumeric, SameCountsWithDifferentIdsAreRejected) {
  test::TemporaryDirectory directory;
  const auto original = directory.path() / "original";
  const auto rotated = directory.path() / "rotated";
  NumericProfile profile(DataType::INT64);
  profile.null_ratio = NumericNullRatio::NONE;
  const auto num_docs = static_cast<uint32_t>(profile.num_docs);
  JsonObject build;
  BuildAndVerify(profile, original, &build);
  // Move each selected EQ posting to different IDs without changing any of
  // its counts. The remaining rows use a value outside the generated domain.
  const auto inputs =
      NamedRecord(build["cases"].as_array(), "eq_hit")["inputs"].as_array();
  std::vector<int64_t> rows(num_docs, 30001);
  {
    IndexFixture source(profile.data_type, (original / "index").string(), true);
    for (const auto &input : inputs) {
      const auto value = input.as_integer();
      const auto result = source.column().search(
          Encode(profile.data_type, value), CompareOp::EQ);
      ASSERT_TRUE(result);
      for (auto it = result->create_iterator(); it->valid(); it->next())
        rows[(it->doc_id() + 1) % num_docs] = value;
    }
    IndexFixture target(profile.data_type, (rotated / "index").string(), false);
    for (uint32_t id = 0; id < num_docs; ++id)
      ASSERT_TRUE(
          target.column().insert(id, Encode(profile.data_type, rows[id])).ok());
    target.Seal();
  }
  {
    IndexFixture source(profile.data_type, (original / "index").string(), true);
    IndexFixture target(profile.data_type, (rotated / "index").string(), true);
    for (const auto &input : inputs) {
      const auto bytes = Encode(profile.data_type, input.as_integer());
      const auto expected = source.column().search(bytes, CompareOp::EQ);
      const auto actual = target.column().search(bytes, CompareOp::EQ);
      ASSERT_TRUE(expected);
      ASSERT_TRUE(actual);
      EXPECT_EQ(expected->count(), actual->count());
      bool differs = false;
      for (uint32_t id = 0; id < num_docs; ++id)
        differs |= expected->contains(id) != actual->contains(id);
      EXPECT_TRUE(differs);
    }
  }
  build["index_path"] = JsonStringValue((rotated / "index").string());
  WriteJson((rotated / "build.json").string(), build);
  EXPECT_THROW(VerifyNumeric(rotated.string()), std::runtime_error);
}

TEST(InvertedBenchmarkNumeric, QueryRejectsCorruptedVerificationCounts) {
  test::TemporaryDirectory directory;
  NumericProfile profile(DataType::INT64);
  auto verified = BuildAndVerify(profile, directory.path());
  auto cases = verified["cases"].as_array();
  for (size_t i = 0; i < cases.size(); ++i) {
    auto entry = cases[i].as_object();
    if (Text(entry, "name") != "eq_hit") continue;
    auto counts = entry["counts"].as_array();
    counts[0] = JsonValue(counts[0].as_integer() + 1);
    entry["counts"] = counts;
    cases[i] = entry;
  }
  verified["cases"] = cases;
  WriteJson((directory.path() / "verify.json").string(), verified);
  EXPECT_THROW(QueryNumeric(directory.path().string()), std::runtime_error);
}

TEST(InvertedBenchmarkNumeric, QueryMeasuresEveryWorkloadCase) {
  test::TemporaryDirectory directory;
  NumericProfile profile(DataType::INT32);
  profile.null_ratio = NumericNullRatio::TEN_PERCENT;
  JsonObject build;
  const auto verified = BuildAndVerify(profile, directory.path(), &build);
  const auto measured = QueryNumeric(directory.path().string());
  const auto fingerprint =
      Text(build["dataset"].as_object(), "data_fingerprint");
  EXPECT_EQ(Text(measured, "phase"), "query");
  EXPECT_EQ(Text(measured, "data_fingerprint"), fingerprint);
  const auto cases = measured["cases"].as_array();
  std::vector<std::string> names;
  for (const auto &value : cases) {
    const auto query = value.as_object();
    const auto name = Text(query, "case");
    SCOPED_TRACE(name);
    names.push_back(name);
    const auto expected = NamedRecord(verified["cases"].as_array(), name);
    EXPECT_EQ(Text(query, "data_fingerprint"), fingerprint);
    EXPECT_EQ(Text(query, "query_fingerprint"),
              Text(expected, "query_fingerprint"));
    EXPECT_EQ(JsonText(query["match_counts"]), JsonText(expected["counts"]));
    EXPECT_GT(Number(query, "completed_queries"), 0);
    EXPECT_GT(query["elapsed_seconds"].as_float(), 0);
    EXPECT_GT(query["qps"].as_float(), 0);
    EXPECT_FALSE(query["metric_names"].as_array().empty());
  }
  std::sort(names.begin(), names.end());
  const std::vector<std::string> expected_names{
      "eq_hit", "eq_miss", "ge_q50", "is_not_null", "is_null",
      "lt_q01", "lt_q50",  "lt_q90", "ne_hit"};
  EXPECT_EQ(names, expected_names);
}

TEST(InvertedBenchmarkNumeric, QueryRequiresVerificationForEveryCase) {
  test::TemporaryDirectory directory;
  NumericProfile profile(DataType::INT64);
  const auto verified = BuildAndVerify(profile, directory.path());
  const auto original_cases = verified["cases"].as_array();
  ASSERT_GT(original_cases.size(), 1);

  // Remove the final case: earlier cases must not make a partial verification
  // sufficient for a full query run.
  auto incomplete = verified;
  JsonArray cases;
  for (size_t i = 0; i + 1 < original_cases.size(); ++i)
    cases.push(original_cases[i]);
  incomplete["cases"] = cases;
  WriteJson((directory.path() / "verify.json").string(), incomplete);
  EXPECT_THROW(QueryNumeric(directory.path().string()), std::invalid_argument);

  auto mismatched = verified;
  cases = original_cases;
  auto last = cases[cases.size() - 1].as_object();
  last["query_fingerprint"] = JsonStringValue("different-query");
  cases[cases.size() - 1] = last;
  mismatched["cases"] = cases;
  WriteJson((directory.path() / "verify.json").string(), mismatched);
  EXPECT_THROW(QueryNumeric(directory.path().string()), std::invalid_argument);
}

TEST(InvertedBenchmarkNumeric, RejectsInvalidSavedProfile) {
  test::TemporaryDirectory directory;
  NumericProfile profile(DataType::INT32);
  JsonObject build;
  BuildAndVerify(profile, directory.path(), &build);
  const auto original = build["dataset"].as_object();
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, JsonValue>>{
           {"type", JsonStringValue("array_int32")},
           {"seed", JsonValue(20261116)},
           {"num_docs", JsonValue(1234)},
           {"num_docs", JsonValue(-1)},
           {"num_docs", JsonValue(1.5)},
           {"num_docs", JsonValue(static_cast<int64_t>(UINT32_MAX) + 1)},
           {"distribution", JsonStringValue("hotspot")},
           {"distribution", JsonValue()},
           {"value_range", JsonStringValue("unknown")},
           {"value_order", JsonStringValue("descending")},
           {"null_ratio", JsonValue(0.25)},
           {"null_ratio", JsonStringValue("0.1")},
           {"null_ratio", JsonValue()}}) {
    SCOPED_TRACE(key + ": " + JsonText(value));
    auto description = original;
    description[key.c_str()] = value;
    build["dataset"] = description;
    WriteJson((directory.path() / "build.json").string(), build);
    EXPECT_THROW(VerifyNumeric(directory.path().string()),
                 std::invalid_argument);
    EXPECT_THROW(QueryNumeric(directory.path().string()),
                 std::invalid_argument);
  }
}

TEST(InvertedBenchmarkNumeric, RejectsInvalidProfilesBeforeCreatingIndex) {
  test::TemporaryDirectory directory;
  const auto path = (directory.path() / "invalid").string();
  NumericProfile defaults(DataType::INT32);
  const auto rejects = [&](const NumericProfile &profile) {
    EXPECT_THROW(DescribeNumericProfile(profile), std::invalid_argument);
    EXPECT_THROW(BuildNumeric(profile, path), std::invalid_argument);
    EXPECT_FALSE(
        std::filesystem::exists(std::filesystem::path(path) / "index"));
  };
  auto profile = defaults;
  profile.data_type = DataType::ARRAY_INT32;
  rejects(profile);
  profile = defaults;
  for (const auto count : {0U, 1U, 257U, 1234U, UINT32_MAX}) {
    profile.num_docs = static_cast<NumericDocCount>(count);
    rejects(profile);
  }
  profile = defaults;
  profile.distribution = static_cast<NumericDistribution>(-1);
  rejects(profile);
  profile = defaults;
  profile.value_range = static_cast<NumericValueRange>(-1);
  rejects(profile);
  profile = defaults;
  profile.value_order = static_cast<NumericValueOrder>(-1);
  rejects(profile);
  profile = defaults;
  profile.null_ratio = static_cast<NumericNullRatio>(-1);
  rejects(profile);
  EXPECT_NO_THROW(BuildAndVerify(defaults, path));
  EXPECT_THROW(BuildNumeric(defaults, path), std::invalid_argument);
}

TEST(InvertedBenchmarkNumeric, DescriptionIncludesTypedWorkloadAndAllCases) {
  for (const auto type : {DataType::INT32, DataType::INT64}) {
    for (const auto &[name, distribution] :
         std::vector<std::pair<std::string, NumericDistribution>>{
             {"uniform", NumericDistribution::UNIFORM},
             {"hotspot_moderate", NumericDistribution::HOTSPOT_MODERATE},
             {"hotspot_extreme", NumericDistribution::HOTSPOT_EXTREME}}) {
      SCOPED_TRACE(name);
      NumericProfile profile(type);
      profile.num_docs = NumericDocCount::TEN_MILLION;
      profile.distribution = distribution;
      profile.value_range = NumericValueRange::ZERO_TO_TYPE_MAX;
      profile.value_order = NumericValueOrder::SHUFFLED;
      profile.null_ratio = NumericNullRatio::FIFTY_PERCENT;
      const auto description = DescribeNumericProfile(profile);
      EXPECT_EQ(Text(description, "name"), name);
      EXPECT_EQ(Text(description, "distribution"), name);
      EXPECT_EQ(Text(description, "type"),
                type == DataType::INT32 ? "int32" : "int64");
      EXPECT_EQ(Number(description, "num_docs"), 10000000);
      EXPECT_EQ(Text(description, "value_range"), "0_to_type_max");
      EXPECT_EQ(description["value_min"].as_integer(), 0);
      EXPECT_EQ(description["value_max"].as_integer(),
                type == DataType::INT32 ? INT32_MAX : INT64_MAX);
      EXPECT_EQ(Text(description, "value_order"), "shuffled");
      EXPECT_DOUBLE_EQ(description["null_ratio"].as_float(), 0.5);
      EXPECT_FALSE(description["supplemental"].as_bool());
      const bool uniform = distribution == NumericDistribution::UNIFORM;
      if (uniform) {
        EXPECT_TRUE(description["hot_value_count"].is_null());
        EXPECT_TRUE(description["hot_doc_ratio"].is_null());
      } else {
        const bool moderate =
            distribution == NumericDistribution::HOTSPOT_MODERATE;
        EXPECT_EQ(Number(description, "hot_value_count"),
                  moderate ? 3000 : 300);
        EXPECT_DOUBLE_EQ(description["hot_doc_ratio"].as_float(),
                         moderate ? 0.5 : 0.9);
      }
      std::vector<std::string> cases;
      const auto case_names = description["cases"].as_array();
      for (const auto &value : case_names)
        cases.push_back(JsonStringText(value));
      std::vector<std::string> expected{"eq_miss", "ne_hit",     "lt_q01",
                                        "lt_q50",  "lt_q90",     "ge_q50",
                                        "is_null", "is_not_null"};
      if (uniform) {
        expected.push_back("eq_hit");
      } else {
        expected.push_back("eq_hot");
        expected.push_back("eq_cold");
      }
      std::sort(cases.begin(), cases.end());
      std::sort(expected.begin(), expected.end());
      EXPECT_EQ(cases, expected);
    }
  }
}

}  // namespace
}  // namespace zvec::benchmarks::inverted
