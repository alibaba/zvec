// Copyright 2025-present the zvec project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <gtest/gtest.h>
#include "measurement.h"
#include "report_utils.h"
#include "test_utils.h"

namespace zvec::benchmarks::inverted {
namespace {

TEST(InvertedBenchmarkReports, VerificationDoesNotRequireDocumentCounts) {
  test::TemporaryDirectory directory;
  JsonArray inputs;
  inputs.push(JsonStringValue("quoted \"value\"\n"));
  const auto record = InputRecord("ratio", inputs);
  JsonArray records;
  records.push(record);
  EXPECT_EQ(SelectInputs(records, "ratio").size(), 1);
  auto damaged = record;
  damaged["query_fingerprint"] = JsonStringValue("changed");
  JsonArray bad;
  bad.push(damaged);
  EXPECT_THROW(SelectInputs(bad, ""), std::invalid_argument);

  const auto verification_path =
      (directory.path() / "ratio-verify.json").string();
  JsonObject dataset;
  dataset.set("data_fingerprint", JsonStringValue("dataset-hash"));
  JsonObject build;
  build.set("dataset", dataset);
  auto checked = Report("verify");
  checked.set("passed", JsonValue(true));
  checked.set("data_fingerprint", JsonStringValue("dataset-hash"));
  JsonObject ratio;
  ratio.set("name", JsonStringValue("ratio"));
  ratio.set("query_fingerprint", record["query_fingerprint"]);
  ratio.set("expected_ratio", JsonValue(0.5));
  JsonArray results;
  results.push(ratio);
  checked.set("cases", results);
  WriteJson(verification_path, checked);
  EXPECT_DOUBLE_EQ(
      LoadVerification(verification_path, build, record)["expected_ratio"]
          .as_float(),
      0.5);
}

TEST(InvertedBenchmarkMeasurement, WarmupIsExcludedFromCompletionAndChecksum) {
  uint64_t calls = 0;
  const auto timing = MeasurePool(3, 0, 0.001, [&]() {
    ++calls;
    return uint64_t{7};
  });
  EXPECT_GT(timing.completed, 0);
  EXPECT_EQ(timing.completed % 3, 0);
  EXPECT_EQ(calls, timing.completed / 3 + 1);
  EXPECT_EQ(timing.checksum, timing.completed / 3 * 7);
  EXPECT_GE(timing.elapsed_seconds, 0.001);
  EXPECT_THROW(MeasurePool(0, 0, 0.001, [] { return uint64_t{0}; }),
               std::invalid_argument);
}

}  // namespace
}  // namespace zvec::benchmarks::inverted
