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

#include "hnsw_rabitq_searcher.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cstdio>
#include <random>
#include <gtest/gtest.h>
#include <zvec/ailego/container/params.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_framework.h"
#include "zvec/core/framework/index_logger.h"
#include "hnsw_rabitq_builder.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace std;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 128;

class HnswRabitqSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string HnswRabitqSearcherTest::_dir("HnswRabitqSearcherTest");
shared_ptr<IndexMeta> HnswRabitqSearcherTest::_index_meta_ptr;

void HnswRabitqSearcherTest::SetUp(void) {
  IndexLoggerBroker::SetLevel(0);
  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswRabitqSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  // system(cmdBuf);
}

TEST_F(HnswRabitqSearcherTest, TestBasicSearch) {
  // Build index first
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
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

  string path = _dir + "/TestBasicSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  // Perform search
  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  auto context = searcher->create_context();
  ASSERT_TRUE(!!context);
  context->set_topk(10);

  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &result = context->result(0);
  ASSERT_GT(result.size(), 0UL);
  ASSERT_LE(result.size(), 10UL);

  // Verify results are sorted by distance
  for (size_t i = 1; i < result.size(); ++i) {
    ASSERT_LE(result[i - 1].score(), result[i].score());
  }
}

TEST_F(HnswRabitqSearcherTest, DISABLED_TestRnnSearch) {
  // Build index first
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
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

  string path = _dir + "/TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher with radius search
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = 0.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto context = searcher->create_context();
  ASSERT_NE(context, nullptr);

  size_t topk = 50;
  context->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &results = context->result(0);
  ASSERT_EQ(topk, results.size());

  // Test with radius threshold
  float radius = results[topk / 2].score();
  context->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test reset threshold
  context->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswRabitqSearcherTest, DISABLED_TestSearchInnerProduct) {
  // Build index with InnerProduct metric
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_TRUE(holder->emplace(i, std::move(vec)));
  }

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("InnerProduct", 0, ailego::Params());

  ailego::Params params;
  params.set("proxima.rabitq.num_clusters", 16UL);
  params.set("proxima.rabitq.total_bits", 2UL);
  params.set("proxima.hnsw_rabitq.general.dimension", dim);

  ASSERT_EQ(0, builder->init(index_meta, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestSearchInnerProduct";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = 1.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto context = searcher->create_context();
  ASSERT_TRUE(!!context);

  size_t topk = 50;
  context->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));

  const auto &results = context->result(0);
  ASSERT_EQ(topk, results.size());

  // Test with radius threshold (note: InnerProduct uses negative scores)
  float radius = -results[topk / 2].score();
  context->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    LOG_ERROR("radius: %f, score: %f", radius, results[k].score());
    EXPECT_GE(radius, results[k].score());
  }

  // Test reset threshold
  context->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(query_vec.data(), query_meta, 1, context));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(-radius, results[topk - 1].score());
}

TEST_F(HnswRabitqSearcherTest, TestSearchCosine) {
  // Build index with Cosine metric
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dist(gen);
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

  string path = _dir + "/TestSearchCosine";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = 1.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);
  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  std::string new_query;
  IndexQueryMeta new_meta;
  ASSERT_EQ(0, reformer->transform(query_vec.data(), query_meta, &new_query,
                                   &new_meta));

  auto context = searcher->create_context();
  ASSERT_TRUE(!!context);

  size_t topk = 50;
  context->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, 1, context));

  const auto &results = context->result(0);
  ASSERT_EQ(topk, results.size());

  // Test with radius threshold
  float radius = 0.5f;
  context->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, 1, context));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test reset threshold
  context->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, 1, context));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswRabitqSearcherTest, TestMultipleQueries) {
  // Build index first
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
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

  string path = _dir + "/TestMultipleQueries";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher with multiple queries
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  // Test with different query vectors
  for (size_t query_id = 0; query_id < 5; ++query_id) {
    NumericalVector<float> query_vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      query_vec[j] = static_cast<float>(query_id * dim + j) / 1000.0f;
    }

    IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

    auto context = searcher->create_context();
    ASSERT_TRUE(!!context);
    context->set_topk(20);

    ASSERT_EQ(0,
              searcher->search_impl(query_vec.data(), query_meta, 1, context));

    const auto &result = context->result(0);
    ASSERT_GT(result.size(), 0UL);
    ASSERT_LE(result.size(), 20UL);

    // Verify results are sorted
    for (size_t i = 1; i < result.size(); ++i) {
      ASSERT_LE(result[i - 1].score(), result[i].score());
    }
  }
}

TEST_F(HnswRabitqSearcherTest, TestDifferentTopK) {
  // Build index first
  IndexBuilder::Pointer builder =
      IndexFactory::CreateBuilder("HnswRabitqBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexProvider<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
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

  string path = _dir + "/TestDifferentTopK";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // Test searcher with different topk values
  auto searcher = IndexFactory::CreateSearcher("HnswRabitqSearcher");
  ASSERT_NE(searcher, nullptr);

  ailego::Params search_params;
  search_params.set("proxima.hnsw_rabitq.searcher.ef", 100UL);
  ASSERT_EQ(0, searcher->init(search_params));

  auto loader = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_NE(loader, nullptr);
  ASSERT_EQ(0, loader->init(ailego::Params()));
  ASSERT_EQ(0, loader->open(path, false));

  ASSERT_EQ(0, searcher->load(loader, nullptr));

  NumericalVector<float> query_vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    query_vec[j] = static_cast<float>(j) / 1000.0f;
  }

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, dim);

  // Test with different topk values
  std::vector<size_t> topk_values = {1, 5, 10, 20, 50, 100};
  for (size_t topk : topk_values) {
    auto context = searcher->create_context();
    ASSERT_TRUE(!!context);
    context->set_topk(topk);

    ASSERT_EQ(0,
              searcher->search_impl(query_vec.data(), query_meta, 1, context));

    const auto &result = context->result(0);
    ASSERT_GT(result.size(), 0UL);
    ASSERT_LE(result.size(), topk);

    // Verify results are sorted
    for (size_t i = 1; i < result.size(); ++i) {
      ASSERT_LE(result[i - 1].score(), result[i].score());
    }
  }
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
