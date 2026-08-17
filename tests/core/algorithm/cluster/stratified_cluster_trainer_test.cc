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

#include <cstring>
#include <limits>
#include <gtest/gtest.h>
#include <zvec/core/framework/index_factory.h>
#include <zvec/core/framework/index_trainer.h>
#include "algorithm/cluster/cluster_params.h"
#include "algorithm/cluster/holder_cluster.h"

using zvec::ailego::Params;

namespace zvec {
namespace core {
namespace {

struct ClusterCalls {
  int holder_result{0};
  size_t holder_calls{0};
  size_t mount_calls{0};
  std::vector<float> values;
};

ClusterCalls calls;

// Record the actual trainer input independently of KMeans' implementation.
class RecordingHolderCluster : public IndexCluster, public HolderCluster {
 public:
  int init(const IndexMeta &, const Params &) override {
    return 0;
  }
  int cleanup() override {
    return 0;
  }
  int reset() override {
    calls.values.clear();
    return 0;
  }
  int update(const Params &) override {
    return 0;
  }
  void suggest(uint32_t) override {}
  int mount(IndexFeatures::Pointer features) override {
    ++calls.mount_calls;
    calls.values.clear();
    for (size_t i = 0; i < features->count(); ++i) {
      append(features->element(i));
    }
    return 0;
  }
  int cluster(IndexThreads::Pointer, CentroidList &cents) override {
    if (calls.values.empty()) {
      return IndexError_InvalidArgument;
    }
    cents.emplace_back(&calls.values.front(), sizeof(float));
    cents.back().set_follows(calls.values.size());
    return 0;
  }
  int classify(IndexThreads::Pointer, CentroidList &) override {
    return IndexError_NotImplemented;
  }
  int label(IndexThreads::Pointer, const CentroidList &,
            std::vector<uint32_t> *) override {
    return IndexError_NotImplemented;
  }
  int cluster_holder(IndexThreads::Pointer threads, IndexHolder::Pointer holder,
                     CentroidList &cents) override {
    ++calls.holder_calls;
    if (calls.holder_result != 0) {
      return calls.holder_result;
    }
    for (auto iter = holder->create_iterator(); iter->is_valid();
         iter->next()) {
      append(iter->data());
    }
    return cluster(std::move(threads), cents);
  }

 private:
  static void append(const void *data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(value));
    calls.values.push_back(value);
  }
};

INDEX_FACTORY_REGISTER_CLUSTER(RecordingHolderCluster);

class TrainerTestHolder : public MultiPassIndexHolder<IndexMeta::DT_FP32> {
 public:
  TrainerTestHolder() : MultiPassIndexHolder(1) {
    for (uint32_t i = 0; i < 16; ++i) {
      zvec::ailego::NumericalVector<float> value(1);
      value[0] = static_cast<float>(i);
      emplace(i, std::move(value));
    }
  }
  size_t count() const override {
    return reported_count;
  }
  IndexHolder::Iterator::Pointer create_iterator() override {
    ++iterator_calls;
    return MultiPassIndexHolder::create_iterator();
  }
  size_t reported_count{16};
  size_t iterator_calls{0};
};

class StratifiedTrainerInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls = {};
  }
  IndexTrainer::Pointer make_trainer(uint32_t sample_count,
                                     float ratio = 0.0f) {
    auto trainer = IndexFactory::CreateTrainer("StratifiedClusterTrainer");
    Params params;
    params.set(STRATIFIED_TRAINER_CLUSTER_COUNT, "1");
    params.set(STRATIFIED_TRAINER_CLASS_NAME, "RecordingHolderCluster");
    params.set(STRATIFIED_TRAINER_SAMPLE_COUNT, sample_count);
    params.set(STRATIFIED_TRAINER_SAMPLE_RATIO, ratio);
    EXPECT_NE(nullptr, trainer);
    if (trainer) {
      EXPECT_EQ(0, trainer->init(IndexMeta(IndexMeta::DT_FP32, 1), params));
    }
    return trainer;
  }
  IndexThreads::Pointer threads =
      std::make_shared<SingleQueueIndexThreads>(1, false);
};

TEST_F(StratifiedTrainerInputTest, FullSamplesUseHolderAndPreserveOrder) {
  for (uint32_t sample_count : {0u, 16u, 32u}) {
    SCOPED_TRACE(sample_count);
    calls = {};
    auto trainer = make_trainer(sample_count);
    ASSERT_NE(nullptr, trainer);
    auto holder = std::make_shared<TrainerTestHolder>();
    ASSERT_EQ(0, trainer->train(threads, holder));
    EXPECT_EQ(1u, calls.holder_calls);
    EXPECT_EQ(0u, calls.mount_calls);
    EXPECT_EQ(16u, trainer->stats().trained_count());
    ASSERT_EQ(16u, calls.values.size());
    for (size_t i = 0; i < calls.values.size(); ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(i), calls.values[i]);
    }
  }
}

TEST_F(StratifiedTrainerInputTest, SamplingKeepsTheReservoirSequence) {
  for (bool use_ratio : {false, true}) {
    SCOPED_TRACE(use_ratio);
    calls = {};
    auto trainer = make_trainer(use_ratio ? 0u : 4u, use_ratio ? 0.25f : 0.0f);
    ASSERT_NE(nullptr, trainer);
    auto holder = std::make_shared<TrainerTestHolder>();
    SampleIndexFeatures<CompactIndexFeatures> expected(
        IndexMeta(IndexMeta::DT_FP32, 1), 4);
    for (auto iter = holder->create_iterator(); iter->is_valid();
         iter->next()) {
      expected.emplace(iter->data());
    }
    ASSERT_EQ(0, trainer->train(threads, holder));
    EXPECT_EQ(0u, calls.holder_calls);
    EXPECT_EQ(1u, calls.mount_calls);
    EXPECT_EQ(4u, trainer->stats().trained_count());
    ASSERT_EQ(expected.count(), calls.values.size());
    for (size_t i = 0; i < calls.values.size(); ++i) {
      float value = 0.0f;
      std::memcpy(&value, expected.element(i), sizeof(value));
      EXPECT_FLOAT_EQ(value, calls.values[i]);
    }
  }
}

TEST_F(StratifiedTrainerInputTest, UnsupportedHolderFallsBackButErrorsDoNot) {
  for (int result : {IndexError_NotImplemented, IndexError_InvalidArgument,
                     IndexError_NoMemory}) {
    SCOPED_TRACE(result);
    calls = {};
    calls.holder_result = result;
    auto trainer = make_trainer(32);
    ASSERT_NE(nullptr, trainer);
    auto holder = std::make_shared<TrainerTestHolder>();
    const bool fallback = result == IndexError_NotImplemented;
    EXPECT_EQ(fallback ? 0 : result, trainer->train(threads, holder));
    EXPECT_EQ(1u, calls.holder_calls);
    EXPECT_EQ(fallback ? 1u : 0u, calls.mount_calls);
    EXPECT_EQ(fallback ? 1u : 0u, holder->iterator_calls);
    if (fallback) {
      EXPECT_EQ(16u, trainer->stats().trained_count());
    }
  }
}

TEST_F(StratifiedTrainerInputTest, UnknownSizeMaterializesActualSampleCount) {
  for (uint32_t sample_count : {0u, 4u, 32u}) {
    SCOPED_TRACE(sample_count);
    calls = {};
    auto trainer = make_trainer(sample_count);
    ASSERT_NE(nullptr, trainer);
    auto holder = std::make_shared<TrainerTestHolder>();
    holder->reported_count = std::numeric_limits<size_t>::max();
    ASSERT_EQ(0, trainer->train(threads, holder));
    EXPECT_EQ(0u, calls.holder_calls);
    EXPECT_EQ(1u, calls.mount_calls);
    EXPECT_EQ(sample_count == 4u ? 4u : 16u, trainer->stats().trained_count());
  }
}

TEST_F(StratifiedTrainerInputTest, FullRatiosAvoidOverflowAndUseHolder) {
  for (float ratio : {1.0f, 2.0f, std::numeric_limits<float>::max()}) {
    SCOPED_TRACE(ratio);
    calls = {};
    auto trainer = make_trainer(0, ratio);
    ASSERT_NE(nullptr, trainer);
    ASSERT_EQ(0,
              trainer->train(threads, std::make_shared<TrainerTestHolder>()));
    EXPECT_EQ(1u, calls.holder_calls);
    EXPECT_EQ(0u, calls.mount_calls);
    EXPECT_EQ(16u, trainer->stats().trained_count());
  }
}

TEST_F(StratifiedTrainerInputTest, InvalidRatiosDoNotConsumeInput) {
  for (float ratio : {-1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    SCOPED_TRACE(ratio);
    auto trainer = make_trainer(0, ratio);
    ASSERT_NE(nullptr, trainer);
    auto holder = std::make_shared<TrainerTestHolder>();
    EXPECT_EQ(IndexError_InvalidArgument, trainer->train(threads, holder));
    EXPECT_EQ(0u, holder->iterator_calls);
  }
  auto trainer = make_trainer(4, 0.25f);
  ASSERT_NE(nullptr, trainer);
  auto holder = std::make_shared<TrainerTestHolder>();
  holder->reported_count = std::numeric_limits<size_t>::max();
  EXPECT_EQ(IndexError_InvalidArgument, trainer->train(threads, holder));
  EXPECT_EQ(0u, holder->iterator_calls);
  EXPECT_EQ(0u, calls.holder_calls);
  EXPECT_EQ(0u, calls.mount_calls);
}

}  // namespace
}  // namespace core
}  // namespace zvec
