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

#include "hnsw_rabitq_builder.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <future>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_framework.h"
#include "zvec/core/framework/index_logger.h"
#include "zvec/core/framework/index_provider.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 128;

class HnswRabitqBuilderTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string HnswRabitqBuilderTest::_dir("hnswRabitqBuilderTest");
shared_ptr<IndexMeta> HnswRabitqBuilderTest::_index_meta_ptr;

void HnswRabitqBuilderTest::SetUp(void) {
  IndexLoggerBroker::SetLevel(0);
  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswRabitqBuilderTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  // system(cmdBuf);
}

TEST_F(HnswRabitqBuilderTest, TestGeneral) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_GT(stats.built_costtime(), 0UL);
}

TEST_F(HnswRabitqBuilderTest, TestLoad) {
  // Load index with searcher and verify search
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  // Perform search verification
  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  auto context = searcher->create_context();
  ASSERT_NE(context, nullptr);
  context->set_topk(10);

  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);
}

TEST_F(HnswRabitqBuilderTest, TestDimensions) {
  std::vector<size_t> dimensions = {1,    2,    4,    8,    16,   32,   33,
                                    63,   64,   128,  256,  512,  1024, 2047,
                                    2048, 2049, 4095, 4096, 4097, 8192, 16384};
  size_t doc_cnt = 100;

  for (size_t test_dim : dimensions) {
    std::cout << "Testing dimension: " << test_dim << std::endl;

    IndexMeta index_meta(IndexMeta::DataType::DT_FP32, test_dim);
    index_meta.set_metric("SquaredEuclidean", 0, ailego::Params());

    IndexBuilder::Pointer builder =
        IndexFactory::CreateBuilder("HnswRabitqBuilder");
    ASSERT_NE(builder, nullptr) << "dim=" << test_dim;

    ailego::Params params;
    params.set("proxima.rabitq.num_clusters", 16UL);
    params.set("proxima.rabitq.total_bits", 2UL);
    params.set("proxima.hnsw_rabitq.general.dimension", test_dim);

    int ret = builder->init(index_meta, params);

    // dimension <= 63 or >= 4096: init() should return -31
    if (test_dim <= 63 || test_dim >= 4096) {
      ASSERT_EQ(-31, ret) << "expected init to fail with -31, dim=" << test_dim;
      std::cout << "Dimension " << test_dim
                << " correctly rejected with ret=" << ret << std::endl;
      continue;
    }

    // Valid dimensions: verify full build succeeds
    ASSERT_EQ(0, ret) << "init failed, dim=" << test_dim;

    auto holder =
        make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(
            test_dim);
    for (size_t i = 0; i < doc_cnt; i++) {
      NumericalVector<float> vec(test_dim);
      for (size_t j = 0; j < test_dim; ++j) {
        vec[j] = static_cast<float>(i * test_dim + j) / 1000.0f;
      }
      ASSERT_TRUE(holder->emplace(i, std::move(vec))) << "dim=" << test_dim;
    }

    ret = builder->train(holder);
    ASSERT_EQ(0, ret) << "train failed, dim=" << test_dim;

    ret = builder->build(holder);
    ASSERT_EQ(0, ret) << "build failed, dim=" << test_dim;

    auto &stats = builder->stats();
    ASSERT_EQ(doc_cnt, stats.built_count()) << "dim=" << test_dim;

    std::cout << "Dimension " << test_dim << " passed, built "
              << stats.built_count() << " docs" << std::endl;
  }
}

TEST_F(HnswRabitqBuilderTest, TestMemquota) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);
  params.set("proxima.hnsw_rabitq.builder.memory_quota", 100000UL);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(IndexError_NoMemory, builder->build(holder));
}

TEST_F(HnswRabitqBuilderTest, TestIndexThreads) {
  IndexBuilder::Pointer builder1 =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder1, nullptr);
  IndexBuilder::Pointer builder2 =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder2, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  std::srand(ailego::Realtime::MilliSeconds());
  auto threads =
      std::make_shared<SingleQueueIndexThreads>(std::rand() % 4, false);
  ASSERT_EQ(0, builder1->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder2->init(*_index_meta_ptr, params));

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

  string path = _dir + "/TestIndexThreads";
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

TEST_F(HnswRabitqBuilderTest, TestCosine) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  converter->transform(holder);

  auto converted_holder = converter->result();
  converted_holder = convert_holder_to_provider(converted_holder);

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  ASSERT_EQ(0, builder->init(index_meta, params));

  ASSERT_EQ(0, builder->train(converted_holder));

  ASSERT_EQ(0, builder->build(converted_holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestCosine";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_GT(stats.built_costtime(), 0UL);
}

TEST_F(HnswRabitqBuilderTest, TestCleanupAndRebuild) {
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestCleanupAndRebuild";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_GT(stats.built_costtime(), 0UL);

  // Cleanup and rebuild with more documents
  ASSERT_EQ(0, builder->cleanup());

  auto holder2 =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt2 = 2000UL;
  for (size_t i = 0; i < doc_cnt2; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i * dim + j) / 1000.0f;
    }
    ASSERT_TRUE(holder2->emplace(i, std::move(vec)));
  }

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder2));
  ASSERT_EQ(0, builder->build(holder2));

  auto dumper2 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper2, nullptr);
  ASSERT_EQ(0, dumper2->create(path));
  ASSERT_EQ(0, builder->dump(dumper2));
  ASSERT_EQ(0, dumper2->close());

  ASSERT_EQ(0UL, stats.trained_count());
  ASSERT_EQ(doc_cnt2, stats.built_count());
  ASSERT_EQ(doc_cnt2, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_EQ(0UL, stats.trained_costtime());
  ASSERT_GT(stats.built_costtime(), 0UL);
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
