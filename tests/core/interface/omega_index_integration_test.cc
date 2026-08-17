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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tests/test_util.h"
#include "zvec/core/interface/index.h"
#include "zvec/core/interface/index_factory.h"
#include "zvec/core/interface/index_param_builders.h"

namespace zvec::core_interface {
namespace {

constexpr uint32_t kDimension = 16;
constexpr uint32_t kTrainQueryDocId = 77;
const std::string kIndexPath = "OmegaIndexIntegrationTest/test.index";

void SetEnvVar(const char *name, const std::string &value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void UnsetEnvVar(const char *name) {
#ifdef _WIN32
  _putenv_s(name, "");
#else
  unsetenv(name);
#endif
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, std::string value) : name_(name) {
    SetEnvVar(name_.c_str(), value);
  }
  ~ScopedEnvVar() {
    UnsetEnvVar(name_.c_str());
  }

 private:
  std::string name_;
};

BaseIndexParam::Pointer CreateOmegaIndexParam() {
  return OmegaIndexParamBuilder()
      .WithMetricType(MetricType::kInnerProduct)
      .WithDataType(DataType::DT_FP32)
      .WithDimension(kDimension)
      .WithIsSparse(false)
      .WithM(8)
      .WithEFConstruction(64)
      .Build();
}

BaseIndexParam::Pointer CreateTrainableOmegaIndexParam() {
  return OmegaIndexParamBuilder()
      .WithMetricType(MetricType::kInnerProduct)
      .WithDataType(DataType::DT_FP32)
      .WithDimension(kDimension)
      .WithIsSparse(false)
      .WithM(8)
      .WithEFConstruction(64)
      .WithMinVectorThreshold(1)
      .WithNumTrainingQueries(32)
      .WithEFTraining(64)
      .WithEFGroundTruth(64)
      .Build();
}

void PopulateIndex(const Index::Pointer &index, uint32_t doc_count) {
  for (uint32_t doc_id = 0; doc_id < doc_count; ++doc_id) {
    std::vector<float> values(kDimension, static_cast<float>(doc_id) / 10.0f);
    values[0] = 1.0f + static_cast<float>(doc_id);

    VectorData vector_data;
    vector_data.vector = DenseVector{values.data()};
    ASSERT_EQ(index->Add(vector_data, doc_id), 0);
  }
}

std::vector<float> MakeRandomUnitVector(std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> values(kDimension, 0.0f);
  float norm_sq = 0.0f;
  for (auto &value : values) {
    value = dist(rng);
    norm_sq += value * value;
  }

  const float norm = std::sqrt(norm_sq);
  if (norm > 0.0f) {
    for (auto &value : values) {
      value /= norm;
    }
  }
  return values;
}

std::vector<std::vector<float>> PopulateRandomIndex(const Index::Pointer &index,
                                                    uint32_t doc_count) {
  std::mt19937 rng(42);
  std::vector<std::vector<float>> dataset;
  dataset.reserve(doc_count);
  for (uint32_t doc_id = 0; doc_id < doc_count; ++doc_id) {
    dataset.push_back(MakeRandomUnitVector(rng));
    VectorData vector_data;
    vector_data.vector = DenseVector{dataset.back().data()};
    if (index->Add(vector_data, doc_id) != 0) {
      ADD_FAILURE() << "failed to add random training document " << doc_id;
      return {};
    }
  }
  return dataset;
}

VectorData MakeQuery(float base) {
  static std::vector<float> values;
  values.assign(kDimension, base);
  values[0] = 1.0f;
  return VectorData{DenseVector{values.data()}};
}

class OmegaIndexIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zvec::test_util::RemoveTestPath("OmegaIndexIntegrationTest");
  }

  void TearDown() override {
    zvec::test_util::RemoveTestPath("OmegaIndexIntegrationTest");
  }
};

TEST_F(OmegaIndexIntegrationTest,
       SearchFallsBackWithoutModelAndReturnsResults) {
  auto index = IndexFactory::CreateAndInitIndex(*CreateOmegaIndexParam());
  ASSERT_NE(index, nullptr);
  ASSERT_EQ(index->Open(kIndexPath, {StorageOptions::StorageType::kMMAP, true}),
            0);

  PopulateIndex(index, 8);
  ASSERT_EQ(index->Train(), 0);

  auto query_param = OmegaQueryParamBuilder()
                         .with_topk(3)
                         .with_fetch_vector(true)
                         .with_ef_search(32)
                         .with_target_recall(0.90f)
                         .build();

  auto query = MakeQuery(0.0f);
  SearchResult result;
  ASSERT_EQ(index->Search(query, query_param, &result), 0);
  ASSERT_EQ(result.doc_list_.size(), 3U);
  EXPECT_EQ(result.doc_list_[0].key(), 7U);
  EXPECT_TRUE(result.training_records_.empty());
  EXPECT_TRUE(result.gt_cmps_per_rank_.empty());

  ASSERT_EQ(index->Close(), 0);
}

TEST_F(OmegaIndexIntegrationTest,
       TrainingSessionCollectsArtifactsThroughIndexSearch) {
  auto index = IndexFactory::CreateAndInitIndex(*CreateOmegaIndexParam());
  ASSERT_NE(index, nullptr);
  ASSERT_EQ(index->Open(kIndexPath, {StorageOptions::StorageType::kMMAP, true}),
            0);

  PopulateIndex(index, 8);
  ASSERT_EQ(index->Train(), 0);

  auto *training_capable = index->GetTrainingCapability();
  ASSERT_NE(training_capable, nullptr);

  auto session = training_capable->CreateTrainingSession();
  ASSERT_NE(session, nullptr);

  TrainingSessionConfig config;
  config.topk = 1;
  config.k_train = 1;
  config.ground_truth = {{0}};
  ASSERT_TRUE(session->Start(config).ok());
  session->BeginQuery(0);

  auto query_param = OmegaQueryParamBuilder()
                         .with_topk(3)
                         .with_ef_search(32)
                         .with_training_query_id(0)
                         .with_target_recall(0.95f)
                         .build();

  auto query = MakeQuery(0.0f);
  SearchResult result;
  ASSERT_EQ(index->Search(query, query_param, &result), 0);
  EXPECT_EQ(result.training_query_id_, 0);
  EXPECT_FALSE(result.training_records_.empty());
  ASSERT_EQ(result.gt_cmps_per_rank_.size(), 1U);
  EXPECT_GT(result.total_cmps_, 0);

  QueryTrainingArtifacts artifacts;
  artifacts.records = result.training_records_;
  artifacts.gt_cmps_per_rank = result.gt_cmps_per_rank_;
  artifacts.total_cmps = result.total_cmps_;
  artifacts.training_query_id = result.training_query_id_;
  session->CollectQueryArtifacts(std::move(artifacts));

  TrainingArtifacts consumed = session->ConsumeArtifacts();
  ASSERT_FALSE(consumed.records.empty());
  ASSERT_EQ(consumed.gt_cmps_data.num_queries, 1U);
  ASSERT_EQ(consumed.gt_cmps_data.topk, 1U);
  ASSERT_EQ(consumed.gt_cmps_data.gt_cmps.size(), 1U);
  ASSERT_EQ(consumed.gt_cmps_data.total_cmps.size(), 1U);
  EXPECT_EQ(consumed.gt_cmps_data.total_cmps[0], result.total_cmps_);

  session->Finish();
  ASSERT_EQ(index->Close(), 0);
}

TEST_F(OmegaIndexIntegrationTest, TrainBuildsModelFilesForCoreWorkflow) {
  auto index =
      IndexFactory::CreateAndInitIndex(*CreateTrainableOmegaIndexParam());
  ASSERT_NE(index, nullptr);
  ASSERT_EQ(index->Open(kIndexPath, {StorageOptions::StorageType::kMMAP, true}),
            0);

  auto dataset = PopulateRandomIndex(index, 128);
  ASSERT_EQ(index->Train(), 0);

  EXPECT_TRUE(std::filesystem::exists(
      "OmegaIndexIntegrationTest/omega_model/model.txt"));
  EXPECT_TRUE(std::filesystem::exists(
      "OmegaIndexIntegrationTest/omega_model/threshold_table.txt"));
  EXPECT_TRUE(std::filesystem::exists(
      "OmegaIndexIntegrationTest/omega_model/training_queries.bin"));

  auto query_param = OmegaQueryParamBuilder()
                         .with_topk(5)
                         .with_fetch_vector(false)
                         .with_ef_search(64)
                         .with_target_recall(0.95f)
                         .build();

  auto query = VectorData{DenseVector{dataset[kTrainQueryDocId].data()}};
  SearchResult result;
  const std::string profile_path =
      "OmegaIndexIntegrationTest/omega_prediction_profile.jsonl";
  {
    ScopedEnvVar profile_enabled("ZVEC_OMEGA_PROFILE_PREDICTION", "1");
    ScopedEnvVar profile_output("ZVEC_OMEGA_PROFILE_OUTPUT", profile_path);
    ASSERT_EQ(index->Search(query, query_param, &result), 0);
  }
  ASSERT_EQ(result.doc_list_.size(), 5U);

  ASSERT_TRUE(std::filesystem::exists(profile_path));
  std::ifstream profile_file(profile_path);
  std::string profile_line;
  ASSERT_TRUE(static_cast<bool>(std::getline(profile_file, profile_line)));
  EXPECT_NE(profile_line.find("\"model_calls\""), std::string::npos);
  EXPECT_NE(profile_line.find("\"model_time_ns\""), std::string::npos);
  EXPECT_NE(profile_line.find("\"decision_time_ns\""), std::string::npos);
  EXPECT_NE(profile_line.find("\"saved_comparisons\""), std::string::npos);
  EXPECT_NE(profile_line.find("\"baseline_comparisons\""), std::string::npos);

  ASSERT_EQ(index->Close(), 0);
}

}  // namespace
}  // namespace zvec::core_interface
