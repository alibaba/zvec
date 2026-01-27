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
#include "ivf_builder.h"
#include <future>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

class IVFBuilderTest : public testing::Test {
 protected:
  void SetUp();
  void TearDown();

  void prepare_index_holder(uint32_t base_key, uint32_t num);

  IndexMeta index_meta_;
  Params params_;
  uint32_t dimension_;
  IndexHolder::Pointer holder_;
  IndexThreads::Pointer threads_{};
};

void IVFBuilderTest::SetUp() {
  dimension_ = 8U;

  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());

  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "8");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  std::mt19937 gen((std::random_device())());
  bool v = std::uniform_int_distribution<size_t>(0, 1)(gen);
  if (v) {
    threads_ = std::make_shared<SingleQueueIndexThreads>();
  }
}

void IVFBuilderTest::TearDown() {}

void IVFBuilderTest::prepare_index_holder(uint32_t base_key, uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 1.0f * i;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

TEST_F(IVFBuilderTest, TestInitSuccess) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
}

TEST_F(IVFBuilderTest, TestInitFailedWithInvalidMetric) {
  IVFBuilder builder;
  index_meta_.set_metric("invalid", 0, Params());
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(IndexError_NoExist, ret);
}

TEST_F(IVFBuilderTest, TestInitFailedWithInvalidCentroidsNum) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(IndexError_InvalidArgument, ret);
}

TEST_F(IVFBuilderTest, TestTrainWithHolder1Level) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_GT(centroid_index->centroids_count(), 0u);
}

TEST_F(IVFBuilderTest, TestTrainWithHolder2Level) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 8);
}

TEST_F(IVFBuilderTest, TestTrainWithTrainer2Level) {
  IndexTrainer::Pointer trainer =
      IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_TRUE(!!trainer);

  prepare_index_holder(0, 1000);

  Params params;
  params.set("proxima.stratified.trainer.cluster_count", "4*2");
  ASSERT_EQ(0, trainer->init(index_meta_, params));
  ASSERT_EQ(0, trainer->train(threads_, holder_));

  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);


  ret = builder.train(trainer);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 8);
}

TEST_F(IVFBuilderTest, TestTrainWithTrainer1Level) {
  IVFBuilder builder;

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  IndexTrainer::Pointer trainer =
      IndexFactory::CreateTrainer("StratifiedClusterTrainer");
  ASSERT_TRUE(!!trainer);

  prepare_index_holder(0, 1000);

  Params params1;
  params1.set("proxima.stratified.trainer.cluster_count", "4");
  ASSERT_EQ(0, trainer->init(index_meta_, params1));
  ASSERT_EQ(0, trainer->train(threads_, holder_));

  ret = builder.train(trainer);
  EXPECT_EQ(0, ret);

  auto centroid_index = builder.centroid_index();
  EXPECT_EQ(centroid_index->centroids_count(), 4);
}

TEST_F(IVFBuilderTest, TestBuildWith2Level) {
  IVFBuilder builder;

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  EXPECT_EQ((size_t)1000, builder.stats().built_count());
}

TEST_F(IVFBuilderTest, TestBuildWith1Level) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  EXPECT_EQ((size_t)1000, builder.stats().built_count());
}

TEST_F(IVFBuilderTest, TestDump) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

#if 0
TEST_F(IVFBuilderTest, TestBuildWithNoEnoughMemory)
{
    IVFBuilder builder;
    Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

    dimension_ = 256;
    index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);

    int ret = builder.init(index_meta_, params);
    EXPECT_EQ(0, ret);

    prepare_index_holder(0, 1000);

    ret = builder.train(threads_, holder_);
    EXPECT_EQ(0, ret);

    ret = builder.build(threads_, holder_);
    EXPECT_EQ(IndexError_IndexFull, ret);
}
#endif

TEST_F(IVFBuilderTest, TestBuildWithEnoughMemory) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

#if 0
TEST_F(IVFBuilderTest, TestBuildWithRowMajorAndNoEnoughMemory)
{
    IVFBuilder builder;
    Params params;
    params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
    params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

    dimension_ = 256;
    index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
    index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

    int ret = builder.init(index_meta_, params);
    EXPECT_EQ(0, ret);

    prepare_index_holder(0, 1000);

    ret = builder.train(threads_, holder_);
    EXPECT_EQ(0, ret);

    ret = builder.build(threads_, holder_);
    EXPECT_EQ(IndexError_IndexFull, ret);
}
#endif

TEST_F(IVFBuilderTest, TestBuildWithRowMajorAndMemory) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);

  prepare_index_holder(0, 1000);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

TEST_F(IVFBuilderTest, TestBuildWithEmptyCentroid) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MajorOrder::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  size_t doc_cnt = 10;

  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 1.0f;
    }
    holder->emplace(i, vec);
  }
  holder_.reset(holder);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  ret = dumper->create("path");
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)10, builder.stats().built_count());
  EXPECT_EQ((size_t)10, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
}

TEST_F(IVFBuilderTest, TestTrainClusterParams) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  prepare_index_holder(0, 1000);
  EXPECT_EQ(0, builder.init(index_meta_, params));
  EXPECT_EQ(0, builder.train(threads_, holder_));
  EXPECT_EQ(0, builder.build(threads_, holder_));

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("MemoryDumper");
  EXPECT_EQ(0, dumper->create("test.index"));
  EXPECT_EQ(0, builder.dump(dumper));
}

TEST_F(IVFBuilderTest, TestIndexThreads) {
  IndexBuilder::Pointer builder1 = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder1, nullptr);
  IndexBuilder::Pointer builder2 = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder2, nullptr);

  size_t dim = 128UL;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  std::srand(Realtime::MilliSeconds());
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2*2");
  ASSERT_EQ(0, builder1->init(meta, params));
  ASSERT_EQ(0, builder2->init(meta, params));

  auto threads =
      std::make_shared<SingleQueueIndexThreads>(std::rand() % 4, false);
  auto build_index1 = [&]() {
    ASSERT_EQ(0, builder1->train(threads, holder));
    ASSERT_EQ(0, builder1->build(threads, holder));
  };
  auto build_index2 = [&]() {
    ASSERT_EQ(0, builder2->train(threads, holder));
    ASSERT_EQ(0, builder2->build(threads, holder));
  };

  auto t1 = std::async(std::launch::async, build_index1);
  auto t2 = std::async(std::launch::async, build_index2);
  t1.wait();
  t2.wait();


  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  std::string path = "./hc_index";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder1->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder2->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats1 = builder1->stats();
  ASSERT_EQ(doc_cnt, stats1.built_count());
  auto &stats2 = builder2->stats();
  ASSERT_EQ(doc_cnt, stats2.built_count());
}