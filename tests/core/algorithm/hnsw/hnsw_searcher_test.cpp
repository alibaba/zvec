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

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cstdio>
#include <future>
#include <ailego/math/distance.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "zvec/core/framework/index_builder.h"
#include "zvec/core/framework/index_factory.h"
#include "zvec/core/framework/index_meta.h"
#include "hnsw_params.h"

using namespace std;
using namespace testing;
using namespace zvec::ailego;

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

namespace zvec {
namespace core {

constexpr size_t static dim = 16;

class HnswSearcherTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string HnswSearcherTest::_dir("HnswSearcherTest/");
shared_ptr<IndexMeta> HnswSearcherTest::_index_meta_ptr;

void HnswSearcherTest::SetUp(void) {
  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, ailego::Params());
}

void HnswSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(HnswSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswSearcherTest, TestRnnSearchInnerProduct) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("InnerProduct", 0, ailego::Params());

  ASSERT_EQ(0, builder->init(index_meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestRnnSearchInnerProduct";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = -results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(-radius, results[topk - 1].score());
}

TEST_F(HnswSearcherTest, TestRnnSearchCosine) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dist(gen);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  converter->transform(holder);

  auto converted_holder = converter->result();

  ASSERT_EQ(0, builder->init(index_meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(converted_holder));
  ASSERT_EQ(0, builder->build(converted_holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestRnnSearchCosine";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  std::string new_query;
  IndexQueryMeta new_meta;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = 0.5f;
  ctx->set_threshold(radius);
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswSearcherTest, TestRnnSearchMipsSquaredEuclidean) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("MipsSquaredEuclidean", 0, ailego::Params());

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestStreamerDump.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t doc_cnt = 1000UL;
  auto streamer_ctx = streamer->create_context();
  ASSERT_TRUE(!!streamer_ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    streamer->add_impl(i, vec.data(), qmeta, streamer_ctx);
  }

  {
    // Test Reset Threshold
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = 1.0;
    }

    size_t topk = 50;
    streamer_ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, streamer_ctx));
    auto &results = streamer_ctx->result();
    ASSERT_EQ(topk, results.size());

    float radius = -results[topk / 2].score();
    streamer_ctx->set_threshold(radius);
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, streamer_ctx));
    ASSERT_GT(topk, results.size());
    for (size_t k = 0; k < results.size(); ++k) {
      ASSERT_GE(radius, results[k].score());
    }

    streamer_ctx->reset_threshold();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, streamer_ctx));
    ASSERT_EQ(topk, results.size());
    ASSERT_LT(-radius, results[topk - 1].score());
  }

  auto path = _dir + "/TestStreamerDump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));

  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto searcher_ctx = searcher->create_context();
  ASSERT_TRUE(!!searcher_ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }

  {
    size_t topk = 50;
    searcher_ctx->set_topk(topk);
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, searcher_ctx));
    auto &results = searcher_ctx->result();
    ASSERT_EQ(topk, results.size());

    float radius = -results[topk / 2].score();
    searcher_ctx->set_threshold(radius);
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, searcher_ctx));
    ASSERT_GT(topk, results.size());
    for (size_t k = 0; k < results.size(); ++k) {
      ASSERT_GE(radius, results[k].score());
    }

    // Test Reset Threshold
    searcher_ctx->reset_threshold();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, searcher_ctx));
    ASSERT_EQ(topk, results.size());
    ASSERT_LT(-radius, results[topk - 1].score());
  }
}

TEST_F(HnswSearcherTest, TestGeneral) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  ailego::Params params;
  // params.set("proxima.hnsw.builder.max_neighbor_count", 16);
  params.set("proxima.hnsw.builder.scaling_factor", 16);
  params.set("proxima.hnsw.builder.ef_construction", 10);
  params.set("proxima.hnsw.builder.thread_count", 2);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 1);
  ASSERT_EQ(0, searcher->init(searcherParams));


  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 200;
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  // do linear search test
  {
    float query[dim];
    for (size_t i = 0; i < dim; ++i) {
      query[i] = 3.1f;
    }
    ASSERT_EQ(0, searcher->search_bf_impl(query, qmeta, linearCtx));
    auto &linearResult = linearCtx->result();
    ASSERT_EQ(3UL, linearResult[0].key());
    ASSERT_EQ(4UL, linearResult[1].key());
    ASSERT_EQ(2UL, linearResult[2].key());
    ASSERT_EQ(5UL, linearResult[3].key());
    ASSERT_EQ(1UL, linearResult[4].key());
    ASSERT_EQ(6UL, linearResult[5].key());
    ASSERT_EQ(0UL, linearResult[6].key());
    ASSERT_EQ(7UL, linearResult[7].key());
    for (size_t i = 8; i < topk; ++i) {
      ASSERT_EQ(i, linearResult[i].key());
    }
  }

  // do linear search by p_keys test
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {8, 9, 10, 11, 3, 2, 1, 0};
  {
    float query[dim];
    for (size_t i = 0; i < dim; ++i) {
      query[i] = 3.1f;
    }
    ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(query, p_keys, qmeta,
                                                    linearByPKeysCtx));
    auto &linearByPKeysResult = linearByPKeysCtx->result();
    ASSERT_EQ(8, linearByPKeysResult.size());
    ASSERT_EQ(3UL, linearByPKeysResult[0].key());
    ASSERT_EQ(2UL, linearByPKeysResult[1].key());
    ASSERT_EQ(1UL, linearByPKeysResult[2].key());
    ASSERT_EQ(0UL, linearByPKeysResult[3].key());
    ASSERT_EQ(8UL, linearByPKeysResult[4].key());
    ASSERT_EQ(9UL, linearByPKeysResult[5].key());
    ASSERT_EQ(10UL, linearByPKeysResult[6].key());
    ASSERT_EQ(11UL, linearByPKeysResult[7].key());
  }

  size_t step = 50;
  for (size_t i = 0; i < doc_cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // TODO: check
    // ASSERT_EQ(topk, knnResult.size());
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }
  float recall = totalHits * step * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / doc_cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.90f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSearcherTest, TestClearAndReload) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 1000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  ailego::Params params;
  params.set("proxima.hnsw.builder.thread_count", 3);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.check_crc_enable", true);
  searcherParams.set("proxima.hnsw.searcher.max_scan_ratio",
                     1.1f);  // including upper layer
  ASSERT_EQ(0, searcher->init(searcherParams));


  auto storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!knnCtx);
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 100;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
  auto &knnResult = knnCtx->result();
  ASSERT_EQ(topk, knnResult.size());
  auto &linearResult = linearCtx->result();
  ASSERT_EQ(topk, linearResult.size());
  auto &stats = searcher->stats();
  ASSERT_EQ(doc_cnt, stats.loaded_count());
  // ASSERT_GT(stats.loaded_costtime(), 0UL);

  //! cleanup
  ASSERT_EQ(0, searcher->cleanup());
  ASSERT_EQ(nullptr, searcher->create_context());
  ASSERT_EQ(IndexError_Runtime,
            searcher->load(storage, IndexMetric::Pointer()));
  ASSERT_EQ(0UL, stats.loaded_count());

  ASSERT_EQ(0, searcher->init(searcherParams));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  linearCtx = searcher->create_context();
  knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!knnCtx);
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
  auto &knnResult1 = knnCtx->result();
  ASSERT_EQ(topk, knnResult1.size());
  auto &linearResult1 = linearCtx->result();
  ASSERT_EQ(topk, linearResult1.size());
  ASSERT_EQ(doc_cnt, stats.loaded_count());

  //! unload
  ASSERT_EQ(0, searcher->unload());
  ASSERT_EQ(nullptr, searcher->create_context());
  ASSERT_EQ(0UL, stats.loaded_count());
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  linearCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  linearCtx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
  auto &linearResult2 = linearCtx->result();
  ASSERT_EQ(topk, linearResult2.size());
  ASSERT_EQ(doc_cnt, stats.loaded_count());
}

TEST_F(HnswSearcherTest, TestFilter) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 100UL;
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
    p_keys[0].push_back(i);
  }
  ailego::Params params;
  params.set("proxima.hnsw.builder.thread_count", 3);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.check_crc_enable", true);
  searcherParams.set("proxima.hnsw.searcher.max_scan_ratio", 1.0f);
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 10;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linearByPKeysCtx));

  auto filterFunc = [](uint64_t key) {
    if (key == 10UL || key == 11UL) {
      return true;
    }
    return false;
  };
  auto &knnResult = knnCtx->result();
  ASSERT_EQ(topk, knnResult.size());
  ASSERT_EQ(10UL, knnResult[0].key());
  ASSERT_EQ(11UL, knnResult[1].key());
  ASSERT_EQ(9UL, knnResult[2].key());

  auto &linearResult = linearCtx->result();
  ASSERT_EQ(topk, linearResult.size());
  ASSERT_EQ(10UL, linearResult[0].key());
  ASSERT_EQ(11UL, linearResult[1].key());
  ASSERT_EQ(9UL, linearResult[2].key());

  auto &linearByPKeysResult = linearByPKeysCtx->result();
  ASSERT_EQ(topk, linearByPKeysResult.size());
  ASSERT_EQ(10UL, linearByPKeysResult[0].key());
  ASSERT_EQ(11UL, linearByPKeysResult[1].key());
  ASSERT_EQ(9UL, linearByPKeysResult[2].key());

  knnCtx->set_filter(filterFunc);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
  auto &knnResult1 = knnCtx->result();
  ASSERT_EQ(topk, knnResult1.size());
  ASSERT_EQ(9UL, knnResult1[0].key());
  ASSERT_EQ(12UL, knnResult1[1].key());
  ASSERT_EQ(8UL, knnResult1[2].key());

  linearCtx->set_filter(filterFunc);
  ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
  auto &linearResult1 = linearCtx->result();
  ASSERT_EQ(topk, linearResult1.size());
  ASSERT_EQ(9UL, linearResult1[0].key());
  ASSERT_EQ(12UL, linearResult1[1].key());
  ASSERT_EQ(8UL, linearResult1[2].key());

  linearByPKeysCtx->set_filter(filterFunc);
  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linearByPKeysCtx));
  auto &linearByPKeysResult1 = linearByPKeysCtx->result();
  ASSERT_EQ(topk, linearByPKeysResult1.size());
  ASSERT_EQ(9UL, linearByPKeysResult1[0].key());
  ASSERT_EQ(12UL, linearByPKeysResult1[1].key());
  ASSERT_EQ(8UL, linearByPKeysResult1[2].key());
}

TEST_F(HnswSearcherTest, TestStreamerDump) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestStreamerDump.index", true));
  ASSERT_EQ(0, streamer->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 5000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  auto path = _dir + "/TestStreamerDump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // do searcher knn
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  auto read_storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  size_t step = 50;
  for (size_t i = 0; i < cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // ASSERT_EQ(topk, knnResult.size());
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }
  float recall = totalHits * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSearcherTest, TestSharedContext) {
  auto gen_holder = [](int start, size_t doc_cnt) {
    auto holder =
        make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
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
  auto gen_index = [&gen_holder](int start, size_t docs, std::string path) {
    auto holder = gen_holder(start, docs);
    IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
    ailego::Params params;
    builder->init(*_index_meta_ptr, params);
    builder->train(holder);
    builder->build(holder);
    auto dumper = IndexFactory::CreateDumper("FileDumper");
    dumper->create(path);
    builder->dump(dumper);
    dumper->close();

    IndexSearcher::Pointer searcher =
        IndexFactory::CreateSearcher("HnswSearcher");
    auto name = rand() % 2 ? "FileReadStorage" : "MMapFileReadStorage";
    auto storage = IndexFactory::CreateStorage(name);
    storage->open(path, false);
    params.set("proxima.hnsw.searcher.visit_bloomfilter_enable", rand() % 2);
    searcher->init(ailego::Params());
    searcher->load(storage, IndexMetric::Pointer());
    return searcher;
  };

  srand(ailego::Realtime::MilliSeconds());
  size_t docs1 = rand() % 500 + 100;
  size_t docs2 = rand() % 5000 + 100;
  size_t docs3 = rand() % 50000 + 100;
  auto path1 = _dir + "/TestSharedContext.index1";
  auto path2 = _dir + "/TestSharedContext.index2";
  auto path3 = _dir + "/TestSharedContext.index3";
  auto searcher1 = gen_index(0, docs1, path1);
  auto searcher2 = gen_index(1, docs2, path2);
  auto searcher3 = gen_index(2, docs3, path3);

  srand(ailego::Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto do_test = [&]() {
    IndexSearcher::Context::Pointer ctx;
    switch (rand() % 3) {
      case 0:
        ctx = searcher1->create_context();
        break;
      case 1:
        ctx = searcher2->create_context();
        break;
      case 2:
        ctx = searcher3->create_context();
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

      EXPECT_EQ(0, ret);
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

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  storage->init(ailego::Params());
  storage->open(_dir + "/TestSharedContext.index4", true);
  streamer->init(*_index_meta_ptr, ailego::Params());
  streamer->open(storage);
  NumericalVector<float> query(dim);
  auto ctx1 = streamer->create_context();
  EXPECT_EQ(IndexError_Unsupported,
            searcher1->search_impl(query.data(), qmeta, ctx1));

  auto ctx2 = searcher1->create_context();
  EXPECT_EQ(IndexError_Unsupported,
            streamer->search_impl(query.data(), qmeta, ctx2));
}

TEST_F(HnswSearcherTest, TestProvider) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  std::vector<key_t> keys(doc_cnt);
  srand(ailego::Realtime::MilliSeconds());
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<uint16_t> dt(
        0, std::numeric_limits<uint16_t>::max());
    for (size_t i = 0; i < doc_cnt; ++i) {
      keys[i] = dt(mt);
    }
  } else {
    std::iota(keys.begin(), keys.end(), 0U);
    std::transform(keys.begin(), keys.end(), keys.begin(),
                   [&](key_t k) { return step * k; });
    if (rand_order) {
      uint32_t seed = ailego::Realtime::Seconds();
      std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    }
  }
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = keys[i];
    }
    ASSERT_TRUE(holder->emplace(keys[i], vec));
  }
  ailego::Params params;
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestProvider";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 1);
  ASSERT_EQ(0, searcher->init(searcherParams));
  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto provider = searcher->create_provider();
  for (size_t i = 0; i < keys.size(); ++i) {
    const float *d1 =
        reinterpret_cast<const float *>(provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d1[j], keys[i]);
    }
  }

  auto iter = provider->create_iterator();
  size_t cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, doc_cnt);

  ASSERT_EQ(dim, provider->dimension());
  ASSERT_EQ(_index_meta_ptr->element_size(), provider->element_size());
  ASSERT_EQ(_index_meta_ptr->data_type(), provider->data_type());
}

TEST_F(HnswSearcherTest, TestMipsEuclideanMetric) {
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("MipsSquaredEuclidean", 0, ailego::Params());
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  const size_t COUNT = 10000UL;
  for (size_t i = 0; i < COUNT; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 100.0f;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestMipsEuclideanMetric";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ailego::Params params;
  params.set("proxima.hnsw.searcher.ef", 10);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  EXPECT_EQ(results.size(), topk);
  EXPECT_NEAR((uint64_t)(COUNT - 1), results[0].key(), 20);
}

TEST_F(HnswSearcherTest, TestRandomPaddingTopk) {
  std::mt19937 mt{};
  std::uniform_real_distribution<float> gen(0.0f, 1.0f);
  constexpr size_t static dim = 8;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  const size_t COUNT = 10000UL;
  for (size_t i = 0; i < COUNT; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = gen(mt);
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }
  ASSERT_EQ(0, builder->init(meta, ailego::Params()));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestRandomPadding";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ailego::Params params;
  params.set("proxima.hnsw.searcher.force_padding_result_enable", true);
  params.set("proxima.hnsw.searcher.scan_ratio", 0.01f);
  ASSERT_TRUE(searcher != nullptr);
  ASSERT_EQ(0, searcher->init(params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  std::uniform_int_distribution<uint32_t> gen_int(1, COUNT);
  size_t topk = gen_int(mt);
  ctx->set_topk(topk);
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  EXPECT_EQ(results.size(), topk);
  for (size_t i = 0; i < results.size(); ++i) {
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(results[i].key(), results[j].key());
    }
  }

  ctx->set_filter([](uint64_t key) { return true; });
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto &results1 = ctx->result();
  EXPECT_EQ(results1.size(), 0);
}


TEST_F(HnswSearcherTest, TestBruteForceSetupInContext) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;
  // params.set("proxima.hnsw.builder.max_neighbor_count", 16);
  params.set("proxima.hnsw.builder.scaling_factor", 16);
  params.set("proxima.hnsw.builder.ef_construction", 10);
  params.set("proxima.hnsw.builder.thread_count", 2);
  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGeneral";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 1);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t topk = 200;
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  bool set_bf_threshold = false;
  bool use_update = false;

  size_t step = 50;
  for (size_t i = 0; i < doc_cnt; i += step) {
    auto linearCtx = searcher->create_context();
    auto knnCtx = searcher->create_context();

    ASSERT_TRUE(!!linearCtx);
    ASSERT_TRUE(!!linearCtx);

    linearCtx->set_topk(topk);
    knnCtx->set_topk(topk);

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      if (use_update) {
        ailego::Params searcherParamsExtra;

        searcherParamsExtra.set("proxima.hnsw.searcher.brute_force_threshold",
                                doc_cnt);
        knnCtx->update(searcherParamsExtra);
      } else {
        knnCtx->set_bruteforce_threshold(doc_cnt);
      }

      use_update = !use_update;
    }
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    // auto t3 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      linearTotalTime += t2 - t1;
    } else {
      knnTotalTime += t2 - t1;
    }

    set_bf_threshold = !set_bf_threshold;

    auto &knnResult = knnCtx->result();
    // TODO: check
    // ASSERT_EQ(topk, knnResult.size());
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }
  float recall = totalHits * step * step * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * step * 1.0f / doc_cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.90f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSearcherTest, TestCosine) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestCosine.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 5000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  float fixed_value = float(cnt) / 2;
  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_vec;
    IndexQueryMeta new_meta;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  auto path = _dir + "/TestCosine";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();

  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  NumericalVector<float> qvec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        qvec[j] = fixed_value;
      else
        qvec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0,
              reformer->transform(qvec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
    topk1Hits += i == knnResult[0].key();

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    for (size_t k = 0; k < topk; ++k) {
      totalCnts++;
      for (size_t j = 0; j < topk; ++j) {
        if (linearResult[j].key() == knnResult[k].key()) {
          totalHits++;
          break;
        }
      }
    }
  }

  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 1.0f / query_cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;

  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.90f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswSearcherTest, TestFetchVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestFetchVector.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  auto path = _dir + "/TestFetchVector";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    float vector_value = *(float *)(vector);
    ASSERT_EQ(vector_value, i);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  for (size_t i = 0; i < query_cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);
    float vector_value = *((float *)(knnResult[0].vector()));
    ASSERT_EQ(vector_value, i);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswSearcherTest, TestFetchVectorCosine) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineFp32Converter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestFetchVectorCosine.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  IndexQueryMeta new_meta;

  const float epsilon = 1e-2;
  float fixed_value = float(cnt) / 2;
  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  auto path = _dir + "/TestFetchVectorCosine";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  NumericalVector<float> qvec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        qvec[j] = fixed_value;
      else
        qvec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0,
              reformer->transform(qvec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(knnResult[0].vector(), new_meta, &denormalized_vec);

    float vector_value = *(((float *)(denormalized_vec.data()) + dim - 1));
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}


TEST_F(HnswSearcherTest, TestFetchVectorCosineHalfFloatConverter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP16, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineHalfFloatConverter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(
      0, storage->open(_dir + "/TestFetchVectorCosineHalfFloatConverter.index",
                       true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP16, dim);
  IndexQueryMeta new_meta;

  const float epsilon = 0.1;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-2.0, 2.0);

  std::vector<NumericalVector<uint16_t>> vecs;
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<uint16_t> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      float value = dist(gen);
      vec[j] = ailego::FloatHelper::ToFP16(value);
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));

    vecs.push_back(vec);
  }

  auto path = _dir + "/TestFetchVectorCosineHalfFloatConverter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    uint16_t expected_vec_value = vecs[i][dim - 1];

    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(uint16_t));
    reformer->revert(vector, new_meta, &denormalized_vec);

    uint16_t vector_value = *((uint16_t *)(denormalized_vec.data()) + dim - 1);
    float vector_value_float = ailego::FloatHelper::ToFP32(vector_value);

    float expected_vec_float = ailego::FloatHelper::ToFP32(expected_vec_value);

    EXPECT_NEAR(expected_vec_float, vector_value_float, epsilon);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  NumericalVector<uint16_t> qvec(dim);

  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(uint16_t));
    reformer->revert(knnResult[0].vector(), new_meta, &denormalized_vec);

    uint16_t expected_vec_value = vec[dim - 1];
    uint16_t vector_value =
        *(((uint16_t *)(denormalized_vec.data()) + dim - 1));

    float vector_value_float = ailego::FloatHelper::ToFP32(vector_value);
    float expected_vec_float = ailego::FloatHelper::ToFP32(expected_vec_value);

    EXPECT_NEAR(expected_vec_float, vector_value_float, epsilon);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswSearcherTest, TestFetchVectorCosineFp16Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineFp16Converter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestFetchVectorCosineFp16Converter.index",
                             true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  IndexQueryMeta new_meta;

  const float epsilon = 0.1;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-2.0, 2.0);

  std::vector<NumericalVector<float>> vecs;
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dist(gen);
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));

    vecs.push_back(vec);
  }

  auto path = _dir + "/TestFetchVectorCosineFp16Converter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    float expected_vec_value = vecs[i][dim - 1];

    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);
    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);

    EXPECT_NEAR(expected_vec_value, vector_value, epsilon);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  NumericalVector<float> qvec(dim);

  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(knnResult[0].vector(), new_meta, &denormalized_vec);

    float expected_vec_value = vec[dim - 1];
    float vector_value = *(((float *)(denormalized_vec.data()) + dim - 1));

    EXPECT_NEAR(expected_vec_value, vector_value, epsilon);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswSearcherTest, TestFetchVectorCosineInt8Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineInt8Converter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestFetchVectorCosineInt8Converter.index",
                             true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  IndexQueryMeta new_meta;

  const float epsilon = 1e-2;
  float fixed_value = float(cnt) / 2;
  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  auto path = _dir + "/TestFetchVectorCosineInt8Converter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  NumericalVector<float> qvec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        qvec[j] = fixed_value;
      else
        qvec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0,
              reformer->transform(qvec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(knnResult[0].vector(), new_meta, &denormalized_vec);

    float vector_value = *(((float *)(denormalized_vec.data()) + dim - 1));
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswSearcherTest, TestFetchVectorCosineInt4Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer, nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 100);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dim);
  index_meta_raw.set_metric("Cosine", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("CosineInt4Converter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestFetchVectorCosineInt4Converter.index",
                             true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 2000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  IndexQueryMeta new_meta;

  const float epsilon = 1e-2;
  float fixed_value = float(cnt) / 2;
  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  auto path = _dir + "/TestFetchVectorCosineInt4Converter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 100);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = searcher->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  size_t query_cnt = 200U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  size_t topk = 100;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  NumericalVector<float> qvec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;

    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        qvec[j] = fixed_value;
      else
        qvec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0,
              reformer->transform(qvec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              searcher->search_bf_impl(new_query.data(), new_meta, linearCtx));
    auto t3 = ailego::Realtime::MicroSeconds();

    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector(), nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(knnResult[0].vector(), new_meta, &denormalized_vec);

    float vector_value = *(((float *)(denormalized_vec.data()) + dim - 1));
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswSearcherTest, TestGroup) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGroup";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_NE(searcher, nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 50);
  searcherParams.set("proxima.hnsw.searcher.max_scan_ratio", 0.8);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t group_topk = 20;
  uint64_t total_time = 0;

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;

    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;

    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_num = 5;

  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();

  total_time += t2 - t1;

  std::cout << "total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    // const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    // std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }
  }

  // do linear search by p_keys test
  auto groupbyFuncLinear = [](uint64_t key) {
    uint32_t group_id = key % 10;

    return std::string("g_") + std::to_string(group_id);
  };

  auto linear_pk_ctx = searcher->create_context();

  linear_pk_ctx->set_group_params(group_num, group_topk);
  linear_pk_ctx->set_group_by(groupbyFuncLinear);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {4, 3, 2, 1, 5, 6, 7, 8, 9, 10};

  ASSERT_EQ(0, searcher->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linear_pk_ctx));
  auto &linear_by_pkeys_group_result = linear_pk_ctx->group_result();
  ASSERT_EQ(linear_by_pkeys_group_result.size(), group_num);

  for (uint32_t i = 0; i < linear_by_pkeys_group_result.size(); ++i) {
    // const std::string &group_id = linear_by_pkeys_group_result[i].group_id();
    auto &result = linear_by_pkeys_group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    // std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }

    ASSERT_EQ(10 - i, result[0].key());
  }
}

TEST_F(HnswSearcherTest, TestGroupNotEnoughNum) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGroupNotEnoughNum";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_NE(searcher, nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 50);
  searcherParams.set("proxima.hnsw.searcher.max_scan_ratio", 0.8);
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t group_topk = 20;
  uint64_t total_time = 0;

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;

    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;

    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_num = 12;
  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();
  total_time += t2 - t1;

  std::cout << "total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();
  ASSERT_EQ(group_result.size(), 10);

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    // const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    // std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }
  }
}

TEST_F(HnswSearcherTest, TestGroupInBruteforceSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("HnswBuilder");
  ASSERT_NE(builder, nullptr);
  auto holder =
      make_shared<OnePassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 5000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  ailego::Params params;

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));
  ASSERT_EQ(0, builder->train(holder));
  ASSERT_EQ(0, builder->build(holder));
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  string path = _dir + "/TestGroupInBruteforceSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_NE(searcher, nullptr);
  ailego::Params searcherParams;
  searcherParams.set("proxima.hnsw.searcher.ef", 50);
  searcherParams.set("proxima.hnsw.searcher.max_scan_ratio", 0.8);
  searcherParams.set("proxima.hnsw.searcher.brute_force_threshold",
                     2 * doc_cnt);

  ASSERT_EQ(0, searcher->init(searcherParams));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t group_topk = 20;
  uint64_t total_time = 0;

  auto groupbyFunc = [](uint64_t key) {
    uint32_t group_id = key / 10 % 10;

    // std::cout << "key: " << key << ", group id: " << group_id << std::endl;

    return std::string("g_") + std::to_string(group_id);
  };

  size_t group_num = 5;
  ctx->set_group_params(group_num, group_topk);
  ctx->set_group_by(groupbyFunc);

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();
  total_time += t2 - t1;

  std::cout << "total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();
  ASSERT_EQ(group_result.size(), 5);

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    // const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    // std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }
  }
}

TEST_F(HnswSearcherTest, TestBinaryConverter) {
  uint32_t dimension = 256;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  // params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  // params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  // params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  // params.set(PARAM_HNSW_STREAMER_EF, 5);
  // params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

  ailego::Params stg_params;

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, dimension);
  index_meta_raw.set_metric("InnerProduct", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("BinaryConverter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta_raw, converter_params);

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(_dir + "/TestBinaryConverter.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-2.0, 2.0);
  std::vector<NumericalVector<float>> vecs;

  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);
    }

    std::string new_vec;
    IndexQueryMeta new_meta;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));

    vecs.push_back(vec);
  }

  auto path = _dir + "/TestBinaryConverter";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, dumper->close());

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  ASSERT_TRUE(searcher != nullptr);

  ailego::Params searcherParams;
  ASSERT_EQ(0, searcher->init(searcherParams));

  auto read_storage = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, read_storage->open(path, false));
  ASSERT_EQ(0, searcher->load(read_storage, IndexMetric::Pointer()));

  size_t query_cnt = 200U;
  auto knnCtx = searcher->create_context();

  float epison = 1e-6;
  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];
    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    size_t topk = 50;
    knnCtx->set_topk(topk);
    ASSERT_EQ(0, searcher->search_impl(new_query.data(), new_meta, knnCtx));
    auto &results = knnCtx->result();
    ASSERT_EQ(topk, results.size());
    ASSERT_EQ(i, results[0].key());
    ASSERT_NEAR(0, results[0].score(), epison);
  }
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif