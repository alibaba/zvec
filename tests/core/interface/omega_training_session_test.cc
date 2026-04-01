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
#include "core/interface/indexes/omega_training_session.h"

namespace zvec::core_interface {

TEST(OmegaTrainingSessionTest, StartFailsWithoutStreamer) {
  OmegaTrainingSession session(nullptr);
  TrainingSessionConfig config;
  config.topk = 3;
  config.ground_truth = {{1, 2, 3}};

  auto status = session.Start(config);
  EXPECT_FALSE(status.ok());
}

TEST(OmegaTrainingSessionTest, ConsumeArtifactsAggregatesRecordsAndGtCmps) {
  OmegaTrainingSession session(nullptr);

  QueryTrainingArtifacts first;
  first.training_query_id = 0;
  first.total_cmps = 13;
  first.gt_cmps_per_rank = {3, 7, 11};
  first.records.push_back(
      TrainingRecord{0, 1, 3, 0.1f, 0.2f, std::vector<float>(7, 1.0f), 1});

  QueryTrainingArtifacts second;
  second.training_query_id = 2;
  second.total_cmps = 21;
  second.gt_cmps_per_rank = {5, 9, 15};
  second.records.push_back(
      TrainingRecord{2, 4, 8, 0.3f, 0.4f, std::vector<float>(7, 2.0f), 0});

  session.CollectQueryArtifacts(std::move(first));
  session.CollectQueryArtifacts(std::move(second));

  TrainingArtifacts artifacts = session.ConsumeArtifacts();
  ASSERT_EQ(artifacts.records.size(), 2U);
  EXPECT_EQ(artifacts.records[0].query_id, 0);
  EXPECT_EQ(artifacts.records[1].query_id, 2);

  ASSERT_EQ(artifacts.gt_cmps_data.topk, 3U);
  ASSERT_EQ(artifacts.gt_cmps_data.num_queries, 3U);
  ASSERT_EQ(artifacts.gt_cmps_data.gt_cmps.size(), 3U);
  ASSERT_EQ(artifacts.gt_cmps_data.total_cmps.size(), 3U);

  EXPECT_EQ(artifacts.gt_cmps_data.gt_cmps[0][0], 3);
  EXPECT_EQ(artifacts.gt_cmps_data.gt_cmps[0][2], 11);
  EXPECT_EQ(artifacts.gt_cmps_data.total_cmps[0], 13);

  EXPECT_EQ(artifacts.gt_cmps_data.gt_cmps[1][0], 0);
  EXPECT_EQ(artifacts.gt_cmps_data.total_cmps[1], 0);

  EXPECT_EQ(artifacts.gt_cmps_data.gt_cmps[2][1], 9);
  EXPECT_EQ(artifacts.gt_cmps_data.total_cmps[2], 21);

  TrainingArtifacts drained = session.ConsumeArtifacts();
  EXPECT_TRUE(drained.records.empty());
  EXPECT_TRUE(drained.gt_cmps_data.gt_cmps.empty());
}

TEST(OmegaTrainingSessionTest, ConsumeArtifactsUsesConfiguredShapeWhenAvailable) {
  OmegaTrainingSession session(nullptr);

  QueryTrainingArtifacts only;
  only.training_query_id = 1;
  only.total_cmps = 8;
  only.gt_cmps_per_rank = {2, 4};
  session.CollectQueryArtifacts(std::move(only));

  TrainingArtifacts inferred = session.ConsumeArtifacts();
  ASSERT_EQ(inferred.gt_cmps_data.num_queries, 2U);
  ASSERT_EQ(inferred.gt_cmps_data.topk, 2U);
  EXPECT_EQ(inferred.gt_cmps_data.total_cmps[1], 8);
}

}  // namespace zvec::core_interface
