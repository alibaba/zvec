// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <omega/model_manager.h>
#include <omega/search_context.h>
#include <fstream>
#include <string>
#include <vector>

namespace omega {

TEST(OmegaSearchContextTest, DefaultStateWithoutModelDoesNotEarlyStop) {
  SearchContext ctx(nullptr, nullptr, 0.95f, 3, 5);

  int hops = -1;
  int comparisons = -1;
  int collected_gt = -1;
  ctx.GetStats(&hops, &comparisons, &collected_gt);

  EXPECT_EQ(hops, 0);
  EXPECT_EQ(comparisons, 0);
  EXPECT_EQ(collected_gt, 0);
  EXPECT_EQ(ctx.GetK(), 3);
  EXPECT_EQ(ctx.GetKTrain(), 1);
  EXPECT_EQ(ctx.GetNextPredictionCmps(), 50);
  EXPECT_EQ(ctx.GetPredictionBatchMinInterval(), 10);
  EXPECT_FALSE(ctx.ShouldTrackTraversalWindow());
  EXPECT_FALSE(ctx.ShouldStopEarly());
  EXPECT_FALSE(ctx.EarlyStopHit());
  auto prediction_stats = ctx.GetPredictionProfileStats();
  EXPECT_EQ(prediction_stats.checks, 0U);
  EXPECT_EQ(prediction_stats.model_calls, 0U);
  EXPECT_EQ(prediction_stats.decision_time_ns, 0U);
  EXPECT_EQ(prediction_stats.model_time_ns, 0U);
}

TEST(OmegaSearchContextTest, ExplicitKTrainIsClampedAndExposed) {
  SearchContext ctx(nullptr, nullptr, 0.95f, 100, 5, 8);
  EXPECT_EQ(ctx.GetKTrain(), 8);

  SearchContext clamped_ctx(nullptr, nullptr, 0.95f, 100, 5, 0);
  EXPECT_EQ(clamped_ctx.GetKTrain(), 1);

  SearchContext training_clamped_ctx(nullptr, nullptr, 0.95f, 100, 5, 8);
  training_clamped_ctx.EnableTrainingMode(1, std::vector<int>{11}, 0);
  EXPECT_EQ(training_clamped_ctx.GetKTrain(), 1);
}

TEST(OmegaSearchContextTest, TrainingModeCollectsRecordsAndGtCmps) {
  SearchContext ctx(nullptr, nullptr, 0.95f, 2, 4);
  ctx.EnableTrainingMode(7, std::vector<int>{11, 22}, 2);
  ctx.SetDistStart(0.5f);

  EXPECT_TRUE(ctx.ShouldTrackTraversalWindow());

  ctx.ReportHop();
  EXPECT_FALSE(ctx.ReportVisitCandidate(11, 0.1f, true));
  ctx.ReportHop();
  EXPECT_FALSE(ctx.ReportVisitCandidate(22, 0.2f, true));

  const auto& records = ctx.GetTrainingRecords();
  ASSERT_EQ(records.size(), 2U);
  EXPECT_EQ(records[0].query_id, 7);
  EXPECT_EQ(records[0].cmps_visited, 1);
  EXPECT_EQ(records[0].hops_visited, 1);
  EXPECT_EQ(records[0].label, 0);
  ASSERT_EQ(records[0].traversal_window_stats.size(), 7U);

  EXPECT_EQ(records[1].query_id, 7);
  EXPECT_EQ(records[1].cmps_visited, 2);
  EXPECT_EQ(records[1].hops_visited, 2);
  EXPECT_EQ(records[1].label, 1);
  ASSERT_EQ(records[1].traversal_window_stats.size(), 7U);

  const auto& gt_cmps = ctx.GetGtCmpsPerRank();
  ASSERT_EQ(gt_cmps.size(), 2U);
  EXPECT_EQ(gt_cmps[0], 1);
  EXPECT_EQ(gt_cmps[1], 2);
  EXPECT_EQ(ctx.GetTotalCmps(), 2);
}

TEST(OmegaSearchContextTest, ReportVisitCandidatesReturnsPredictionPointAtBoundary) {
  SearchContext ctx(nullptr, nullptr, 0.95f, 2, 3);

  std::vector<SearchContext::VisitCandidate> warmup;
  warmup.reserve(49);
  for (int i = 0; i < 49; ++i) {
    warmup.push_back({100 + i, 10.0f + static_cast<float>(i), i < 2});
  }

  EXPECT_FALSE(ctx.ReportVisitCandidates(warmup.data(), warmup.size()));
  EXPECT_EQ(ctx.GetTotalCmps(), 49);
  EXPECT_EQ(ctx.GetTopCandidateCountForHook(), 2);

  EXPECT_TRUE(ctx.ReportVisitCandidate(999, 0.01f, true));
  EXPECT_EQ(ctx.GetTotalCmps(), 50);
  EXPECT_EQ(ctx.GetTopCandidateCountForHook(), 2);

  ctx.Reset();
  EXPECT_EQ(ctx.GetTotalCmps(), 0);
  EXPECT_EQ(ctx.GetTopCandidateCountForHook(), 0);
  EXPECT_EQ(ctx.GetNextPredictionCmps(), 50);
}

TEST(OmegaModelManagerTest, MissingModelFileFailsClearly) {
  ModelManager manager;
  EXPECT_FALSE(manager.LoadModel("this/path/should/not/exist"));
  EXPECT_FALSE(manager.IsLoaded());
  EXPECT_EQ(manager.GetModel(), nullptr);
}

}  // namespace omega
