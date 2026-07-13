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

#include "diskann_searcher.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <ailego/math/distance.h>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include <zvec/core/framework/index_framework.h>
#include "diskann_holder.h"
#include "diskann_params.h"

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t static dim = 64;

class DiskAnnSearcherTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

  static std::string _dir;
  static shared_ptr<IndexMeta> _index_meta_ptr;
};

std::string DiskAnnSearcherTest::_dir("DiskAnnSearcherTest/");
shared_ptr<IndexMeta> DiskAnnSearcherTest::_index_meta_ptr;

void DiskAnnSearcherTest::SetUp(void) {
  LoggerBroker::SetLevel(Logger::LEVEL_INFO);

  _index_meta_ptr.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  _index_meta_ptr->set_metric("SquaredEuclidean", 0, Params());
}

void DiskAnnSearcherTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", _dir.c_str());
  system(cmdBuf);
}

TEST_F(DiskAnnSearcherTest, TestGeneral) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

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
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

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

  size_t step = 500;
  for (size_t i = 0; i < doc_cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // TODO: check
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

  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.80f);
  EXPECT_GT(cost, 2.0f);
}

TEST_F(DiskAnnSearcherTest, TestNodeCache) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestNodeCache";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.cache_node_num", 32);
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

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

  size_t step = 500;
  for (size_t i = 0; i < doc_cnt; i += step) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();

    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    // TODO: check
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

  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.80f);
  EXPECT_GT(cost, 2.0f);
}

TEST_F(DiskAnnSearcherTest, TestFilter) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestFilter";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.cache_node_num", 32);
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));
  auto ctx = searcher->create_context();
  ASSERT_TRUE(!!ctx);

  auto linearCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();

  ASSERT_TRUE(!!linearCtx);
  ASSERT_TRUE(!!linearByPKeysCtx);
  ASSERT_TRUE(!!knnCtx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  size_t topk = 200;
  linearCtx->set_topk(topk);
  linearByPKeysCtx->set_topk(topk);
  knnCtx->set_topk(topk);

  size_t key = 50;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = key + 0.1f;
  }

  // no filter
  {
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
    ASSERT_EQ(50UL, knnResult[0].key());
    ASSERT_EQ(51UL, knnResult[1].key());
    ASSERT_EQ(49UL, knnResult[2].key());

    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(50UL, linearResult[0].key());
    ASSERT_EQ(51UL, linearResult[1].key());
    ASSERT_EQ(49UL, linearResult[2].key());
  }

  // with filter
  {
    auto filterFunc = [](uint64_t key) {
      if (key == 50UL || key == 51UL || key == 49UL) {
        return true;
      }
      return false;
    };


    knnCtx->set_filter(filterFunc);
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
    ASSERT_EQ(52UL, knnResult[0].key());
    ASSERT_EQ(48UL, knnResult[1].key());
    ASSERT_EQ(53UL, knnResult[2].key());

    linearCtx->set_filter(filterFunc);
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(52UL, linearResult[0].key());
    ASSERT_EQ(48UL, linearResult[1].key());
    ASSERT_EQ(53UL, linearResult[2].key());
  }
}

TEST_F(DiskAnnSearcherTest, TestGroup) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestGroup";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

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
    vec[j] = query_value / 10 + 0.1f;
  }

  auto t1 = Realtime::MicroSeconds();
  ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, ctx));
  auto t2 = Realtime::MicroSeconds();

  total_time += t2 - t1;

  auto &group_result = ctx->group_result();

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    const std::string &group_id = group_result[i].group_id();
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }
  }

#if 0
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
    const std::string &group_id = linear_by_pkeys_group_result[i].group_id();
    auto &result = linear_by_pkeys_group_result[i].docs();

    ASSERT_GT(result.size(), 0);
    std::cout << "Group ID: " << group_id << std::endl;

    for (uint32_t j = 0; j < result.size(); ++j) {
      std::cout << "\tKey: " << result[j].key() << std::fixed
                << std::setprecision(3) << ", Score: " << result[j].score()
                << std::endl;
    }

    ASSERT_EQ(10 - i, result[0].key());
  }
#endif
}

TEST_F(DiskAnnSearcherTest, TestFetchVector) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestFetchVector";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

  auto storage = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, storage->open(path, false));
  ASSERT_EQ(0, searcher->load(storage, IndexMetric::Pointer()));

  size_t query_cnt = 20U;
  auto linearCtx = searcher->create_context();
  auto knnCtx = searcher->create_context();
  auto linearByPKeysCtx = searcher->create_context();
  knnCtx->set_fetch_vector(true);

  for (size_t i = 0; i < doc_cnt; i += doc_cnt / 10) {
    std::string vec_value;
    ASSERT_EQ(0, searcher->get_vector(i, linearCtx, vec_value));

    float vector_value = *(const float *)(vec_value.data());
    ASSERT_EQ(vector_value, i);
  }

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  NumericalVector<float> vec(dim);
  for (size_t i = 0; i < query_cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();
    ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
    knnTotalTime += t2 - t1;
    linearTotalTime += t3 - t2;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());

    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());
    ASSERT_EQ(i, linearResult[0].key());

    ASSERT_NE(knnResult[0].vector_string(), "");
    float vector_value = *((float *)(knnResult[0].vector_string().data()));
    ASSERT_EQ(vector_value, i);
  }
}

TEST_F(DiskAnnSearcherTest, TestRnnSearch) {
  IndexBuilder::Pointer builder = IndexFactory::CreateBuilder("DiskAnnBuilder");
  ASSERT_NE(builder, nullptr);

  auto holder =
      make_shared<MultiPassIndexHolder<IndexMeta::DataType::DT_FP32>>(dim);
  size_t doc_cnt = 10000UL;
  for (size_t i = 0; i < doc_cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_TRUE(holder->emplace(i, vec));
  }

  Params params;

  params.set("zvec.diskann.builder.max_degree", 32);
  params.set("zvec.diskann.builder.list_size", 300);
  params.set("zvec.diskann.builder.max_pq_chunk_num", 32);
  params.set("zvec.diskann.builder.threads", 4);

  ASSERT_EQ(0, builder->init(*_index_meta_ptr, params));

  ASSERT_EQ(0, builder->train(holder));

  ASSERT_EQ(0, builder->build(holder));

  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);

  string path = _dir + "/TestRnnSearch";
  ASSERT_EQ(0, dumper->create(path));
  ASSERT_EQ(0, builder->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto &stats = builder->stats();
  ASSERT_EQ(doc_cnt, stats.trained_count());
  ASSERT_EQ(doc_cnt, stats.built_count());
  ASSERT_EQ(doc_cnt, stats.dumped_count());
  ASSERT_EQ(0UL, stats.discarded_count());
  ASSERT_GT(stats.trained_costtime(), 0UL);
  ASSERT_GT(stats.built_costtime(), 0UL);

  // test searcher
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("DiskAnnSearcher");
  ASSERT_TRUE(searcher != nullptr);

  Params search_params;
  search_params.set("zvec.diskann.searcher.list_size", 500);

  ASSERT_EQ(0, searcher->init(search_params));

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
