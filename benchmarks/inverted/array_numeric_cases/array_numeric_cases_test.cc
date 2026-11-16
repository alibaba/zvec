// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "array_numeric_cases.h"
#include <stdexcept>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "shared/test_utils.h"

namespace zvec::benchmarks::inverted {
namespace {

ArrayOptions Options(const test::TemporaryDirectory &directory,
                     const std::string &profile, const std::string &tag) {
  ArrayOptions options;
  options.profile = profile;
  options.num_docs = 2000;
  options.index_path = (directory.path() / (tag + "-index")).string();
  options.build_report_path =
      (directory.path() / (tag + "-build.json")).string();
  options.verification_path =
      (directory.path() / (tag + "-verify.json")).string();
  return options;
}

JsonObject BuildAndVerify(ArrayOptions &options,
                          JsonObject *build_out = nullptr) {
  const auto build = RunArray("build", options);
  WriteJson(options.build_report_path, build);
  options.dataset_overrides = JsonObject();
  const auto verified = RunArray("verify", options);
  WriteJson(options.verification_path, verified);
  if (build_out) *build_out = build;
  return verified;
}

TEST(InvertedBenchmarkArray,
     NullEmptyDuplicateAndWitnessSemanticsStayDistinct) {
  test::TemporaryDirectory directory;
  auto options = Options(directory, "random_l8", "array");
  JsonObject build;
  const auto verified = BuildAndVerify(options, &build);
  const auto stats = build["dataset"].as_object()["stats"].as_object();
  EXPECT_EQ(Number(stats, "null_count"), 200);
  EXPECT_EQ(Number(stats, "empty_count"), 200);
  EXPECT_EQ(Number(stats, "duplicate_count"), 320);
  EXPECT_EQ(Number(stats, "raw_elements"), 12800);
  EXPECT_EQ(Number(stats, "distinct_elements"), 11520);
  const auto cases = verified["cases"].as_array();
  EXPECT_EQ(
      test::NamedRecord(cases, "is_null")["counts"].as_array()[0].as_integer(),
      200);
  EXPECT_EQ(test::NamedRecord(cases, "is_not_null")["counts"]
                .as_array()[0]
                .as_integer(),
            1800);
  EXPECT_EQ(test::NamedRecord(cases, "len_eq_zero")["counts"]
                .as_array()[0]
                .as_integer(),
            200);
  EXPECT_EQ(test::NamedRecord(cases, "len_eq_raw")["counts"]
                .as_array()[0]
                .as_integer(),
            1600);
  for (const auto *name : {"not_any_2", "not_all_2"}) {
    const auto counts = test::NamedRecord(cases, name)["counts"].as_array();
    for (const auto &count : counts) EXPECT_GE(count.as_integer(), 200);
  }
  const auto misses =
      test::NamedRecord(cases, "all_hit_miss")["counts"].as_array();
  for (const auto &count : misses) EXPECT_EQ(count.as_integer(), 0);
  const auto witnesses =
      test::NamedRecord(cases, "all_m_witness")["counts"].as_array();
  for (const auto &count : witnesses) EXPECT_GT(count.as_integer(), 0);
}

TEST(InvertedBenchmarkArray,
     CustomLengthAndDuplicateRatiosResolveTheirOwnCases) {
  test::TemporaryDirectory directory;
  for (const auto &[length, duplicate_ratio] :
       std::vector<std::pair<uint32_t, double>>{{1, 0.2}, {3, 0.0}, {8, 1.0}}) {
    auto options = Options(directory, "random_l8", std::to_string(length));
    options.dataset_overrides.set("array_length", JsonValue(length));
    options.dataset_overrides.set("duplicate_ratio",
                                  JsonValue(duplicate_ratio));
    const auto catalog = ArrayCatalog(options);
    const auto profiles = catalog["profiles"].as_array();
    ASSERT_EQ(profiles.size(), 1);
    const auto declared = profiles[0].as_object()["cases"].as_array();
    bool has_witness = false;
    for (const auto &name : declared)
      has_witness |= JsonStringText(name) == "all_m_witness";
    EXPECT_EQ(has_witness, length > 1);
    JsonObject build;
    const auto verified = BuildAndVerify(options, &build);
    if (length > 1) {
      const auto counts = test::NamedRecord(verified["cases"].as_array(),
                                            "all_m_witness")["counts"]
                              .as_array();
      for (const auto &count : counts) EXPECT_GT(count.as_integer(), 0);
    }
    EXPECT_EQ(build["cases"].as_array().size(), declared.size());
  }
}

TEST(InvertedBenchmarkArray, RejectsNumericBuildReports) {
  test::TemporaryDirectory directory;
  auto options = Options(directory, "random_l8", "array");
  JsonObject build;
  BuildAndVerify(options, &build);
  auto dataset = build["dataset"].as_object();
  dataset["type"] = JsonStringValue("int64");
  build["dataset"] = dataset;
  WriteJson(options.build_report_path, build);
  options.case_filter = "any_1_hit";
  EXPECT_THROW(RunArray("verify", options), std::invalid_argument);
  EXPECT_THROW(RunArray("query", options), std::invalid_argument);
}

TEST(InvertedBenchmarkArray, RejectsInvalidRowRatios) {
  ArrayOptions options;
  options.profile = "random_l8";
  options.dataset_overrides.set("null_ratio", JsonValue(0.8));
  options.dataset_overrides.set("empty_ratio", JsonValue(0.3));
  EXPECT_THROW(ArrayCatalog(options), std::invalid_argument);
}

}  // namespace
}  // namespace zvec::benchmarks::inverted
