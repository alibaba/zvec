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
#include "ivf_searcher.h"
#include <future>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "zvec/core/framework/index_framework.h"
#include "ivf_builder.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

class IVFSearcherTest : public testing::Test {
 public:
 protected:
  void SetUp();
  void TearDown();
  void prepare_index_holder(uint32_t base_key, uint32_t num);

  void prepare_rand_index_holder(uint32_t base_key, uint32_t num);

  void prepare_fp16_index_holder(uint32_t base_key, uint32_t num);

  void prepare_fp32_index_holder(uint32_t base_key, uint32_t num);

  void prepare_binary_index_holder(uint32_t base_key, uint32_t num);

  void prepare_int8_index_holder(uint32_t base_key, uint32_t num);

  void prepare_same_index_holder(uint32_t base_key, uint32_t num);

  IndexMeta index_meta_;
  Params params_;
  uint32_t dimension_;
  IndexHolder::Pointer holder_;
  std::string index_path_;
  IndexThreads::Pointer threads_{};
};

void IVFSearcherTest::SetUp() {
  dimension_ = 8U;

  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());

  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "4*2");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");
  index_path_ = "./ivf_searcher.index";
  std::mt19937 gen((std::random_device())());
  bool v = std::uniform_int_distribution<size_t>(0, 1)(gen);
  if (v) {
    threads_ = std::make_shared<SingleQueueIndexThreads>();
  }
}

void IVFSearcherTest::TearDown() {
  File::RemovePath(index_path_);
}

void IVFSearcherTest::prepare_index_holder(uint32_t base_key, uint32_t num) {
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

void IVFSearcherTest::prepare_rand_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = std::rand() % 1000 * 1.0;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_fp32_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 0.01f * i;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_fp16_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 0.01f * i;
    }
    holder->emplace(key + i, vec);
  }

  IndexConverter::Pointer conveter =
      IndexFactory::CreateConverter("HalfFloatConverter");
  conveter->init(index_meta_, Params());
  IndexHolder::Pointer new_holder(holder);
  conveter->transform(new_holder);
  holder_ = conveter->result();
}

void IVFSearcherTest::prepare_int8_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_INT8> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<int8_t> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = (int8_t)(i % 128);
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_binary_index_holder(uint32_t base_key,
                                                  uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_BINARY32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    BinaryVector<uint32_t> vec(dimension_);
    for (size_t j = 0; j < dimension_ && j < i; ++j) {
      vec.set(j);
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

void IVFSearcherTest::prepare_same_index_holder(uint32_t base_key,
                                                uint32_t num) {
  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  uint32_t key = base_key;
  for (size_t i = 0; i < num; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = 8;
    }
    holder->emplace(key + i, vec);
  }

  holder_.reset(holder);
}

TEST_F(IVFSearcherTest, TestInit) {
  IVFSearcher searcher;
  int ret = searcher.init(params_);
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSimple) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 33);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      ASSERT_EQ((uint64_t)32 - i, result[i].key());
      ASSERT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)32 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSimpleCosine) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "1");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineNormalizeConverter");
  ASSERT_TRUE(converter != nullptr);
  auto original_index_meta = index_meta_;
  original_index_meta.set_metric("Cosine", 0, Params());
  EXPECT_EQ(0, converter->init(original_index_meta, converter_params));
  IndexMeta index_meta = converter->meta();
  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);
  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  int ret = builder.init(index_meta, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 33);
  converter->transform(holder_);
  auto holder = converter->result();

  EXPECT_EQ(0, builder.train(threads_, holder));
  EXPECT_EQ(0, builder.build(threads_, holder));
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  EXPECT_EQ(0, dumper->create(index_path_));

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f + i);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }
  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf search
  {
    size_t topk = 33;
    context->set_topk(topk);
    
    std::string new_vec;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->convert(query.data(), qmeta, &new_vec, &new_meta));

    ret = searcher.search_bf_impl(new_vec.data(), new_meta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < 1; ++i) {
      // ASSERT_EQ(29, result[i].key());
      EXPECT_NEAR(0, result[i].score(), 1e-2);
    }
  }
  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithBuildMemory) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  //    params_.set("proxima.hc.builder.thread_count", 1);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      ASSERT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithFilter) {
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  context->set_filter([](uint64_t key) {
    if (key > 0) {
      return true;
    }
    return false;
  });
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

///////////////////////////  row major ////////////////////////////////
TEST_F(IVFSearcherTest, TestRowMajorFloatWithBuildMemory) {
  index_meta_.set_major_order(IndexMeta::MO_ROW);
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFloatWithFilter) {
  index_meta_.set_major_order(IndexMeta::MO_ROW);
  IVFBuilder builder;
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  context->set_filter([](uint64_t key) {
    if (key > 0) {
      return true;
    }
    return false;
  });
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)1, result.size());
    for (size_t i = 0; i < 1; ++i) {
      EXPECT_EQ((uint64_t)0, result[i].key());
      EXPECT_EQ((float)999 * 999 * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)1, result.size());
      EXPECT_EQ((uint64_t)0, result[0].key());
      EXPECT_EQ((float)q * q * dimension_, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFloatWith1LevelAndBuildMemory) {
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "10");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWith1LevelAndBuildMemory) {
  IVFBuilder builder;
  Params build_params;
  build_params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "10");
  build_params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  prepare_index_holder(0, 1000);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorInt8WithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 12;
  index_meta_.set_meta(IndexMeta::DataType::DT_INT8, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 128;
  prepare_int8_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<int8_t> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(127);
  }

  size_t qnum = 63;
  std::vector<int8_t> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorInt8WithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 12;
  index_meta_.set_meta(IndexMeta::DataType::DT_INT8, dimension_);
  index_meta_.set_metric("SquaredEuclidean", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 128;
  prepare_int8_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<int8_t> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(127);
  }

  size_t qnum = 63;
  std::vector<int8_t> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_INT8, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)127 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorBinaryWithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_BINARY32, dimension_);
  index_meta_.set_metric("Hamming", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 257;
  prepare_binary_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  BinaryVector<uint32_t> query(dimension_);
  for (size_t i = 0; i < dimension_; ++i) {
    query.set(i);
  }

  size_t qnum = 63;
  BinaryVector<uint32_t> query1(dimension_ * qnum);
  for (size_t i = 0; i < qnum; ++i) {
    for (size_t j = 0; j < dimension_ && j < i; ++j) {
      query1.set(i * dimension_ + j);
    }
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)256 - i, result[i].key());
      EXPECT_EQ((float)i, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)256 - i, result[i].key());
      EXPECT_EQ((float)i, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorBinaryWithBuildMemory) {
  IVFBuilder builder;
  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_BINARY32, dimension_);
  index_meta_.set_metric("Hamming", 0, Params());
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  size_t fnum = 257;
  prepare_binary_index_holder(0, fnum);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, dumper->close());
  EXPECT_EQ((size_t)fnum, builder.stats().built_count());
  EXPECT_EQ((size_t)fnum, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  BinaryVector<uint32_t> query(dimension_);
  for (size_t i = 0; i < dimension_; ++i) {
    query.set(i);
  }

  size_t qnum = 63;
  BinaryVector<uint32_t> query1(dimension_ * qnum);
  for (size_t i = 0; i < qnum; ++i) {
    for (size_t j = 0; j < dimension_ && j < i; ++j) {
      query1.set(i * dimension_ + j);
    }
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_BINARY32, dimension_);

  // single bf serch
  {
    size_t topk = 128;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)256 - i, result[i].key());
      EXPECT_EQ((float)i, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)256 - i, result[i].key());
      EXPECT_EQ((float)i, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSearchWithEmptyCentroid) {
  IVFBuilder builder;
  Params params;
  params.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "3*3");
  params.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster*KmeansCluster");

  dimension_ = 256;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  int ret = builder.init(index_meta_, params);
  EXPECT_EQ(0, ret);
  size_t doc_cnt = 10;

  MultiPassIndexHolder<IndexMeta::DataType::DT_FP32> *holder =
      new MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>(dimension_);
  for (size_t i = 0; i < doc_cnt; ++i) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = i % 5;
    }
    holder->emplace(i, vec);
  }
  holder_.reset(holder);

  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);

  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);

  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");

  std::string path = "searcher_empty_centroid.index";
  ret = dumper->create(path);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ(0, ret);
  EXPECT_EQ((size_t)10, builder.stats().built_count());
  EXPECT_EQ((size_t)10, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  dumper->close();

  IVFSearcher searcher;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(path, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(999.0f);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    auto key1 = result[0].key();
    EXPECT_TRUE(key1 == 4ul || key1 == 9ul);
  }

  // single knn search
  {
    size_t topk = 3;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    auto key1 = result[0].key();
    auto key2 = result[1].key();
    auto key3 = result[2].key();
    EXPECT_TRUE(key1 == 4ul || key1 == 9ul);
    EXPECT_TRUE(key2 == 4ul || key2 == 9ul);
    EXPECT_TRUE(key3 == 3ul || key3 == 8ul);
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFp16WithBuildMemory) {
  const float epsilon = 1e-2;
  dimension_ = 8;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_COLUMN);

  prepare_fp16_index_holder(0, 1000);
  IVFBuilder builder;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP16, dimension_);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta1(IndexMeta::DataType::DT_FP32, dimension_);

  std::string query_buf;
  query_buf.resize(dimension_ * sizeof(uint16_t));
  std::string query1_buf;
  query1_buf.resize(dimension_ * sizeof(uint16_t) * qnum);

  IndexReformer::Pointer reformer =
      IndexFactory::CreateReformer("HalfFloatReformer");
  IndexQueryMeta qmeta;
  reformer->transform(query.data(), qmeta1, &query_buf, &qmeta);
  reformer->transform(query1.data(), qmeta1, qnum, &query1_buf, &qmeta);
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestRowMajorFp16WithBuildMemory) {
  const float epsilon = 1e-2;
  dimension_ = 8;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP32, dimension_);
  index_meta_.set_major_order(IndexMeta::MO_ROW);

  prepare_fp16_index_holder(0, 1000);
  IVFBuilder builder;
  index_meta_.set_meta(IndexMeta::DataType::DT_FP16, dimension_);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)1000, builder.stats().built_count());
  EXPECT_EQ((size_t)1000, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta1(IndexMeta::DataType::DT_FP32, dimension_);

  std::string query_buf;
  query_buf.resize(dimension_ * sizeof(uint16_t));
  std::string query1_buf;
  query1_buf.resize(dimension_ * sizeof(uint16_t) * qnum);

  IndexReformer::Pointer reformer =
      IndexFactory::CreateReformer("HalfFloatReformer");
  IndexQueryMeta qmeta;
  reformer->transform(query.data(), qmeta1, &query_buf, &qmeta);
  reformer->transform(query1.data(), qmeta1, qnum, &query1_buf, &qmeta);
  // single bf serch
  {
    size_t topk = 1000;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query_buf.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1_buf.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithHnswGraphType) {
  IVFBuilder builder;
  params_.set("proxima.ivf.builder.graph_type", "hnsw");
  params_.set("proxima.ivf.builder.graph_ef", 200);
  params_.set("proxima.ivf.builder.graph_scan_ratio", 1.0);
  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithSsgGraphType) {
  IVFBuilder builder;
  params_.set("proxima.ivf.builder.graph_type", "ssg");
  params_.set("proxima.ivf.builder.graph_ef", 200);
  params_.set("proxima.ivf.builder.graph_scan_ratio", 1.0);

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithInt8Converter) {
  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "Int8QuantizerConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back((total - 1) * 1.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)(total - 1) - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)999 - i, result[i].key());
      EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithFloat16Quantizer) {
  const float epsilon = 1e-2;

  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_fp32_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }

  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestColumnMajorFloatWithConverterAndQuantizer) {
  const float epsilon = 1e-2;
  IVFBuilder builder;
  auto build_params = params_;
  build_params.set(PARAM_IVF_BUILDER_CONVERTER_CLASS, "Int8QuantizerConverter");
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "HalfFloatConverter");
  int ret = builder.init(index_meta_, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_fp32_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(-0.1f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_ * 0.01);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 100;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((uint64_t)i, result[i].key());
      EXPECT_NEAR((float)(0.01f * i + 0.1) * (0.01f * i + 0.1) * dimension_ /
                      result[i].score(),
                  1, epsilon);
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      EXPECT_EQ((uint64_t)q, result[0].key());
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestQuantizedPerCentroid) {
  IVFBuilder builder;
  auto build_params = params_;
  auto meta = index_meta_;
  meta.set_metric("InnerProduct", 0, Params());
  build_params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  build_params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
  int ret = builder.init(meta, build_params);
  EXPECT_EQ(0, ret);
  int total = 1000;
  prepare_index_holder(0, total);
  ret = builder.train(threads_, holder_);
  ASSERT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)total, builder.stats().built_count());
  EXPECT_EQ((size_t)total, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(500.0f);
  }

  size_t qnum = 63;
  std::vector<float> query1;
  for (size_t i = 1; i <= dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = (size_t)total;
    context->set_topk(topk);
    context->set_filter([](uint64_t) { return false; });
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      ASSERT_NEAR((uint64_t)(total - 1) - i, result[i].key(), 150);
      float expect = (float)result[i].key() * 500.0f * dimension_;
      ASSERT_NEAR(expect, std::abs(result[i].score()), expect * 0.2 + 500000);
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_NEAR((uint64_t)(total - 1) - q, result[0].key(), 100);
      // EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 10;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_NEAR((uint64_t)total - i - 1, result[i].key(), 100);
      // EXPECT_EQ((float)i * i * dimension_, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((size_t)topk, result.size());
      ASSERT_NEAR((uint64_t)(total - 1) - q, result[0].key(), 100);
      // EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

TEST_F(IVFSearcherTest, TestSharedContext) {
  size_t dim = dimension_;
  auto gen_holder = [&](int start, size_t doc_cnt) {
    auto holder =
        make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
    uint64_t key = start;
    for (size_t i = 0; i < doc_cnt; i++) {
      NumericalVector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i;
      }
      key += 3;
      holder->emplace(key, vec);
    }
    return holder;
  };
  auto gen_index = [&](int start, size_t docs, std::string path) {
    auto holder = gen_holder(start, docs);
    IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
    Params params;
    params.set("proxima.ivf.builder.centroid_count", "16");
    builder->init(index_meta_, params);
    builder->train(holder);
    builder->build(holder);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    dumper->create(path);
    builder->dump(dumper);
    dumper->close();

    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("IVFSearcher");
    auto name = rand() % 2 ? "FileReadStorage" : "MMapFileReadStorage";
    auto container = IndexFactory::CreateStorage(name);
    bool alone_file_handle = std::rand() % 2;
    bool lock_hot = std::rand() % 2;
    params.set("proxima.file.read_storage.alone_file_handle",
               alone_file_handle);
    params.set("proxima.file.read_storage.lock_hot_in_memory", lock_hot);
    container->init(params);
    container->open(path, false);
    searcher->init(Params());
    searcher->load(container, IndexMetric::Pointer());
    return searcher;
  };

  srand(Realtime::MilliSeconds());
  size_t docs1 = rand() % 500 + 100;
  size_t docs2 = rand() % 5000 + 100;
  size_t docs3 = rand() % 50000 + 100;
  auto path1 = "unittest-index/TestSharedContext.index1";
  auto path2 = "unittest-index/TestSharedContext.index2";
  auto path3 = "unittest-index/TestSharedContext.index3";
  auto searcher1 = gen_index(0, docs1, path1);
  auto searcher2 = gen_index(1, docs2, path2);
  auto searcher3 = gen_index(2, docs3, path3);

  srand(Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto do_test = [&]() {
    IndexSearcher::Context::Pointer ctx;
    switch (rand() % 3) {
      case 0:
        ctx = searcher1->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
      case 1:
        ctx = searcher2->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
      case 2:
        ctx = searcher3->create_context();
        if (rand() % 2 == 0) {
          ctx->set_filter([](uint64_t) { return false; });
        }
        break;
    }
    ctx->set_topk(10);

    int ret = 0;
    for (int i = 0; i < 100; ++i) {
      NumericalVector<float> query(dim);
      for (size_t j = 0; j < dim; ++j) {
        query[j] = i + 0.1f;
      }

      auto code = rand() % 6;
      switch (code) {
        case 0:
          ret = searcher1->search_impl(query.data(), qmeta, ctx);
          break;
        case 1:
          ret = searcher2->search_impl(query.data(), qmeta, ctx);
          break;
        case 2:
          ret = searcher3->search_impl(query.data(), qmeta, ctx);
          break;
        case 3:
          ret = searcher1->search_bf_impl(query.data(), qmeta, ctx);
          break;
        case 4:
          ret = searcher2->search_bf_impl(query.data(), qmeta, ctx);
          break;
        case 5:
          ret = searcher3->search_bf_impl(query.data(), qmeta, ctx);
          break;
      }

      ASSERT_EQ(0, ret);
      auto &results = ctx->result();
      EXPECT_EQ(10, results.size());
      for (int k = 0; k < 10; ++k) {
        EXPECT_EQ(code % 3, results[k].key() % 3);
      }
    }
  };
  auto t1 = std::async(std::launch::async, do_test);
  auto t2 = std::async(std::launch::async, do_test);
  t1.wait();
  t2.wait();
}

TEST_F(IVFSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  size_t dim = 16;
  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = "IVFSearcherTest.TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  ASSERT_EQ(0, builder->cleanup());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(Params()));

  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  float radius = 1000.0f;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  EXPECT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }
  File::RemovePath(path);
}

TEST_F(IVFSearcherTest, TestProvider) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  std::vector<uint64_t> keys(doc_cnt);
  srand(Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](key_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = keys[i];
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  ASSERT_EQ(0, builder->init(index_meta_, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
  for (size_t i = 0; i < keys.size(); ++i) {
    const float *d1 =
        reinterpret_cast<const float *>(provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d1[j], keys[i]);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size());
  ASSERT_EQ(index_meta_.data_type(), provider->data_type());
}

TEST_F(IVFSearcherTest, TestProviderInt8) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  std::vector<key_t> keys(doc_cnt);
  srand(Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](key_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = keys[i];
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  params.set("proxima.ivf.builder.retain_original_features", false);
  auto meta = index_meta_;
  meta.set_metric("InnerProduct", 0, Params());
  params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  params.set(PARAM_IVF_BUILDER_QUANTIZE_BY_CENTROID, true);
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_TRUE(!!provider);
  ASSERT_EQ(IndexMeta::DataType::DT_INT8, provider->data_type());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto d1 = reinterpret_cast<const int8_t *>(provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_LT(d1[j], 255);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    const int8_t *d = reinterpret_cast<const int8_t *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_LT(d[j], 255);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size() * 4);
}

TEST_F(IVFSearcherTest, TestSearcherReuse) {
  auto build_index = [](IndexMeta &meta, size_t base, size_t doc_cnt,
                        std::string &path) {
    IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
    ASSERT_NE(builder, nullptr);
    IndexHolder::Pointer holder;
    if (meta.data_type() == IndexMeta::DataType::DT_INT8) {
      auto h = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_INT8>>(
          meta.dimension());
      for (size_t i = base; i < doc_cnt; i++) {
        NumericalVector<int8_t> vec(meta.dimension());
        for (size_t j = 0; j < meta.dimension(); ++j) {
          vec[j] = i;
        }
        ASSERT_TRUE(h->emplace(i, vec));
      }
      holder = h;
    } else if (meta.data_type() == IndexMeta::DataType::DT_FP32) {
      auto h = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
          meta.dimension());
      for (size_t i = base; i < doc_cnt; i++) {
        NumericalVector<float> vec(meta.dimension());
        for (size_t j = 0; j < meta.dimension(); ++j) {
          vec[j] = i;
        }
        ASSERT_TRUE(h->emplace(i, vec));
      }
      holder = h;
    }
    Params params;
    LOG_DEBUG("Build index %s count=%zu", path.c_str(), holder->count());
    params.set("proxima.ivf.builder.centroid_count", "10");
    ASSERT_EQ(0, builder->init(meta, params));
    ASSERT_EQ(0, builder->train(holder));
    ASSERT_EQ(0, builder->build(holder));
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    ASSERT_NE(dumper, nullptr);
    ASSERT_EQ(0, dumper->create(path));
    ASSERT_EQ(0, builder->dump(dumper));
    ASSERT_EQ(0, dumper->close());
    ASSERT_EQ(0, builder->cleanup());
  };

  auto path1 = index_path_ + "/index1";
  auto path2 = index_path_ + "/index2";
  IndexMeta meta1(IndexMeta::DataType::DT_INT8, 16);
  IndexMeta meta2(IndexMeta::DataType::DT_FP32, 31);
  build_index(meta1, 10, 200, path1);
  build_index(meta2, 2000, 3000, path2);

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_INT8, searcher->meta().data_type());
  ASSERT_EQ(190UL, searcher->stats().loaded_count());
  ASSERT_EQ(190UL, provider->count());
  ASSERT_EQ("IVFSearcher", provider->owner_class());
  for (size_t i = 10; i < 200ul; ++i) {
    const int8_t *d1 =
        reinterpret_cast<const int8_t *>(provider->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < meta1.dimension(); ++j) {
      ASSERT_EQ(d1[j], (int8_t)i);
    }
  }
  ASSERT_EQ(meta1.dimension(), provider->dimension());
  ASSERT_EQ(meta1.element_size(), provider->element_size());
  ASSERT_EQ(meta1.data_type(), provider->data_type());
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(0, searcher->cleanup());

  auto container2 = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container2->open(path2, false));
  ASSERT_EQ(0, searcher->init(searcherParams));
  ASSERT_EQ(0, searcher->load(container2, IndexMetric::Pointer()));

  auto provider2 = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, searcher->meta().data_type());
  for (size_t i = 2000; i < 3000ul; ++i) {
    const float *d1 = reinterpret_cast<const float *>(provider2->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < meta2.dimension(); ++j) {
      ASSERT_FLOAT_EQ(d1[j], i);
    }
  }
  ASSERT_EQ(meta2.dimension(), provider2->dimension());
  ASSERT_EQ(meta2.element_size(), provider2->element_size());
  ASSERT_EQ(meta2.data_type(), provider2->data_type());
  ASSERT_EQ(1000UL, provider2->count());
  ASSERT_EQ(1000UL, searcher->stats().loaded_count());
}

TEST_F(IVFSearcherTest, TestInt8QuantizerWithL2) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder = make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(
      dimension_);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  Params params;
  params.set("proxima.ivf.builder.centroid_count", "20");
  params.set("proxima.ivf.builder.store_original_features", true);
  auto meta = index_meta_;
  params.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");
  ASSERT_EQ(0, builder->init(meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = index_path_ + "/TestQuantizer";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  ASSERT_EQ(IndexMeta::DataType::DT_FP32, provider->data_type());
  for (size_t i = 0; i < doc_cnt; ++i) {
    const float *d1 = reinterpret_cast<const float *>(provider->get_vector(i));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d1[j], i);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dimension_; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dimension_, provider->dimension());
  ASSERT_EQ(index_meta_.element_size(), provider->element_size());
  ASSERT_EQ(index_meta_.data_type(), provider->data_type());

  auto context = searcher->create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);
  size_t topk = 1;
  context->set_topk(topk);
  context->set_filter([](uint64_t) { return false; });
  for (size_t i = 0; i < doc_cnt; i += 20) {
    NumericalVector<float> query(dimension_);
    for (size_t j = 0; j < dimension_; ++j) {
      query[j] = i;
    }
    int ret = searcher->search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);
    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    ASSERT_NEAR(i, result[0].key(), 100);
  }
}

TEST_F(IVFSearcherTest, TestMipsEuclideanMetric) {
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("MipsSquaredEuclidean", 0, Params());
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("IVFBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      std::make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  const size_t COUNT = 10000UL;
  for (size_t i = 0; i < COUNT; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 100.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  Params builder_params;
  builder_params.set("proxima.ivf.builder.centroid_count", 1024);
  ASSERT_EQ(0, builder->init(meta, builder_params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  std::string path = "IVFTestMipsEuclideanMetric";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher = IndexFactory::CreateSearcher("IVFSearcher");
  ASSERT_NE(searcher, nullptr);
  Params params;
  params.set("proxima.ivf.searcher.scan_ratio", 0.1f);
  ASSERT_EQ(0, searcher->init(params));

  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->open(path, false));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 10;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  EXPECT_EQ(results.size(), topk);
  EXPECT_NEAR((uint64_t)(COUNT - 1), results[0].key(), 10);
  File::RemovePath(path);
}

TEST_F(IVFSearcherTest, TestSameValue) {
  IVFBuilder builder;
  //    index_meta_.set_major_order(IndexMeta::MO_ROW);
  params_.set(PARAM_IVF_BUILDER_CENTROID_COUNT, "2");
  params_.set(PARAM_IVF_BUILDER_CLUSTER_CLASS, "KmeansCluster");
  params_.set(PARAM_IVF_BUILDER_QUANTIZER_CLASS, "Int8QuantizerConverter");

  int ret = builder.init(index_meta_, params_);
  EXPECT_EQ(0, ret);
  prepare_same_index_holder(0, 33);
  ret = builder.train(threads_, holder_);
  EXPECT_EQ(0, ret);
  ret = builder.build(threads_, holder_);
  EXPECT_EQ(0, ret);
  IndexDumper::Pointer dumper = IndexFactory::CreateDumper("FileDumper");
  ret = dumper->create(index_path_);
  EXPECT_EQ(0, ret);

  ret = builder.dump(dumper);
  EXPECT_EQ((size_t)33, builder.stats().built_count());
  EXPECT_EQ((size_t)33, builder.stats().dumped_count());
  EXPECT_EQ((size_t)0, builder.stats().discarded_count());
  EXPECT_EQ(0, dumper->close());

  IVFSearcher searcher;
  Params params;
  params.set(PARAM_IVF_SEARCHER_SCAN_RATIO, 1.0);
  params.set(PARAM_IVF_SEARCHER_BRUTE_FORCE_THRESHOLD, 1);

  ret = searcher.init(params);
  EXPECT_EQ(0, ret);

  IndexStorage::Pointer container =
      IndexFactory::CreateStorage("MMapFileReadStorage");
  EXPECT_TRUE(!!container);

  Params container_params;
  container_params.set("proxima.mmap_file.container.memory_warmup", true);
  container->init(container_params);
  ret = container->open(index_path_, false);
  EXPECT_EQ(0, ret);

  ret = searcher.load(container, IndexMetric::Pointer());
  EXPECT_EQ(0, ret);

  std::vector<float> query;
  for (size_t i = 0; i < dimension_; ++i) {
    query.push_back(32.0f);
  }

  size_t qnum = 33;
  std::vector<float> query1;
  for (size_t i = 0; i < dimension_ * qnum; ++i) {
    query1.push_back(i / dimension_);
  }


  auto context = searcher.create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension_);

  // single bf serch
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      // std::cout << "i: " << i << ", key: " << result[i].key() << ", score: "
      // << result[i].score() << std::endl;
      ASSERT_EQ(0, result[i].score());
    }
  }

  // batch bf serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_bf_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  // single knn search
  {
    size_t topk = 33;
    context->set_topk(topk);
    ret = searcher.search_impl(query.data(), qmeta, context);
    EXPECT_EQ(0, ret);

    const IndexDocumentList &result = context->result(0);
    EXPECT_EQ((size_t)topk, result.size());
    for (size_t i = 0; i < topk; ++i) {
      EXPECT_EQ((float)0, result[i].score());
    }
  }

  // batch knn serch
  {
    size_t topk = 1;
    context->set_topk(topk);
    ret = searcher.search_impl(query1.data(), qmeta, qnum, context);
    EXPECT_EQ(0, ret);

    for (size_t q = 0; q < qnum; ++q) {
      const IndexDocumentList &result = context->result(q);
      EXPECT_EQ((float)0, result[0].score());
    }
  }

  ret = searcher.unload();
  EXPECT_EQ(0, ret);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif