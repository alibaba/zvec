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
#include "hnsw_streamer.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace std;
using namespace testing;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t static dim = 16;

class HnswStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string HnswStreamerTest::dir_("streamer_test/");
shared_ptr<IndexMeta> HnswStreamerTest::index_meta_ptr_;

void HnswStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, ailego::Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void HnswStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(HnswStreamerTest, TestAddVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set("proxima.hnsw.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.streamer.scaling_factor", 5U);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/AddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < 1000UL; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  streamer->flush(0UL);
  streamer.reset();
}

// TODO: context cannot shared by different searcher
TEST_F(HnswStreamerTest, TestLinearSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set("proxima.hnsw.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.streamer.scaling_factor", 5U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> vec(dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 1) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(1U);
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 10.1f;
  }
  ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &result = ctx->result();
  ASSERT_EQ(100U, result.size());
  ASSERT_EQ(10, result[0].key());
  ASSERT_EQ(11, result[1].key());
  ASSERT_EQ(5, result[10].key());
  ASSERT_EQ(0, result[20].key());
  ASSERT_EQ(30, result[30].key());
  ASSERT_EQ(35, result[35].key());
  ASSERT_EQ(99, result[99].key());
}

// TODO: context cannot shared by different searcher

TEST_F(HnswStreamerTest, TestLinearSearchByKeys) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set("proxima.hnsw.streamer.max_neighbor_count", 16U);
  params.set("proxima.hnsw.streamer.upper_neighbor_count", 8U);
  params.set("proxima.hnsw.streamer.scaling_factor", 5U);
  params.set("proxima.hnsw.streamer.get_vector_enable", true);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestLinearSearchByKeys.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> vec(dim);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0].resize(cnt);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    p_keys[0][i] = i;
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 1) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(1U);
    ASSERT_EQ(
        0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(
        0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = 10.1f;
    }
    ASSERT_EQ(
        0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(100U, result.size());
    ASSERT_EQ(10, result[0].key());
    ASSERT_EQ(11, result[1].key());
    ASSERT_EQ(5, result[10].key());
    ASSERT_EQ(0, result[20].key());
    ASSERT_EQ(30, result[30].key());
    ASSERT_EQ(35, result[35].key());
    ASSERT_EQ(99, result[99].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = 10.1f;
    }
    p_keys[0] = {{cnt + 1, 10, 1, 15, cnt + 2}};
    ASSERT_EQ(
        0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(3U, result.size());
    ASSERT_EQ(10, result[0].key());
    ASSERT_EQ(15, result[1].key());
    ASSERT_EQ(1, result[2].key());
  }

  {
    ctx->set_topk(100U);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = 9.1f;
    }
    p_keys[0].clear();
    for (size_t j = 0; j < cnt; j += 10) {
      p_keys[0].push_back((uint64_t)j);
    }
    ASSERT_EQ(
        0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
    auto &result = ctx->result();
    ASSERT_EQ(100U, result.size());
    ASSERT_EQ(10, result[0].key());
    ASSERT_EQ(0, result[1].key());
    ASSERT_EQ(100, result[10].key());
    ASSERT_EQ(200, result[20].key());
    ASSERT_EQ(300, result[30].key());
    ASSERT_EQ(350, result[35].key());
    ASSERT_EQ(990, result[99].key());
  }
}

TEST_F(HnswStreamerTest, TestKnnSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestKnnSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
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

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
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
  float topk1Recall = topk1Hits * 1.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
  // // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswStreamerTest, TestAddAndSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestAddAndSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  // streamer->print_debug_info();

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  for (size_t i = 0; i < cnt; i += 100) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
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
  float topk1Recall = topk1Hits * 100.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswStreamerTest, TestKnnSearchRandomData) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  constexpr size_t static dim = 128;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 128);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 20);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 200);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_EF, 10);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestKnnSearchRandomData", true));
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t cnt = 1500;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    streamer->add_impl(i + cnt, vec.data(), qmeta, ctx);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  size_t topk = 100;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  cnt = 500;
  for (size_t i = 0; i < cnt; i += 1) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t3 = ailego::Realtime::MicroSeconds();
    knnTotalTime += t3 - t2;
    linearTotalTime += t2 - t1;

    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
    auto &linearResult = linearCtx->result();
    ASSERT_EQ(topk, linearResult.size());

    topk1Hits += linearResult[0].key() == knnResult[0].key();

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

  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;

  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 1.0f / cnt;
  // float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.50f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 5.0f);
}

TEST_F(HnswStreamerTest, TestOpenClose) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  constexpr size_t static dim = 2048;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ASSERT_NE(nullptr, storage2);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TessOpenAndClose1", true));
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TessOpenAndClose2", true));
  ASSERT_EQ(0, streamer->init(meta, params));
  auto checkIter = [](size_t base, size_t total,
                      IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = base;
    size_t cnt = 0;
    while (iter->is_valid()) {
      float *data = (float *)iter->data();
      ASSERT_EQ(cur, iter->key());
      for (size_t d = 0; d < dim; ++d) {
        ASSERT_FLOAT_EQ((float)cur, data[d]);
      }
      iter->next();
      cur += 2;
      cnt++;
    }
    ASSERT_EQ(cnt, total);
  };

  size_t testCnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < testCnt; i += 2) {
    float v1 = (float)i;
    ASSERT_EQ(0, streamer->open(storage1));
    auto ctx = streamer->create_context();
    ASSERT_TRUE(!!ctx);
    float vec1[dim];
    for (size_t d = 0; d < dim; ++d) {
      vec1[d] = v1;
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec1, qmeta, ctx));
    checkIter(0, i / 2 + 1, streamer);
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());

    float v2 = (float)(i + 1);
    float vec2[dim];
    for (size_t d = 0; d < dim; ++d) {
      vec2[d] = v2;
    }
    ASSERT_EQ(0, streamer->open(storage2));
    ctx = streamer->create_context();
    ASSERT_TRUE(!!ctx);
    ASSERT_EQ(0, streamer->add_impl(i + 1, vec2, qmeta, ctx));
    checkIter(1, i / 2 + 1, streamer);
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }

  IndexStreamer::Pointer streamer1 =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer1->init(meta, params));
  ASSERT_EQ(0, streamer1->open(storage1));

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer2->init(meta, params));
  ASSERT_EQ(0, streamer2->open(storage2));

  checkIter(0, testCnt / 2, streamer1);
  checkIter(1, testCnt / 2, streamer2);
}

TEST_F(HnswStreamerTest, TestCreateIterator) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, true);
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestCreateIterator", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto checkIter = [](size_t total, IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = 0;
    while (iter->is_valid()) {
      float *data = (float *)iter->data();
      ASSERT_EQ(cur, iter->key());
      for (size_t d = 0; d < dim; ++d) {
        ASSERT_FLOAT_EQ((float)cur, data[d]);
      }
      iter->next();
      cur++;
    }
    ASSERT_EQ(cur, total);
  };

  NumericalVector<float> vec(dim);
  size_t cnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    checkIter(i + 1, streamer);
  }

  streamer->flush(0UL);
  streamer->close();
  ASSERT_EQ(0, streamer->open(storage));
  checkIter(cnt, streamer);

  // check getVector
  auto provider = streamer->create_provider();
  for (size_t i = 0; i < cnt; i++) {
    const float *data = (const float *)provider->get_vector(i);
    ASSERT_NE(data, nullptr);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(i, data[j]);
    }
  }
}

TEST_F(HnswStreamerTest, TestNoInit) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  streamer->cleanup();
}

TEST_F(HnswStreamerTest, TestForceFlush) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, true);
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  stg_params.set("proxima.mmap_file.storage.copy_on_write", true);
  stg_params.set("proxima.mmap_file.storage.force_flush", true);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestForceFlush", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto checkIter = [](size_t total, IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = 0;
    while (iter->is_valid()) {
      float *data = (float *)iter->data();
      ASSERT_EQ(cur, iter->key());
      for (size_t d = 0; d < dim; ++d) {
        ASSERT_FLOAT_EQ((float)cur, data[d]);
      }
      iter->next();
      cur++;
    }
    ASSERT_EQ(cur, total);
  };

  NumericalVector<float> vec(dim);
  size_t cnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    checkIter(i + 1, streamer);
  }

  streamer->flush(0UL);
  streamer->close();
  storage->close();

  storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestForceFlush", true));
  ASSERT_EQ(0, streamer->open(storage));
  checkIter(cnt, streamer);

  // check getVector
  auto provider = streamer->create_provider();
  for (size_t i = 0; i < cnt; i++) {
    const float *data = (const float *)provider->get_vector(i);
    ASSERT_NE(data, nullptr);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(i, data[j]);
    }
  }
}

TEST_F(HnswStreamerTest, TestKnnMultiThread) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 128);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 10);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 64);
  params.set(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, 30 * 1024 * 1024U);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_EF, 32);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnMultiThread", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    NumericalVector<float> vec(dim);
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    size_t succAdd = 0;
    auto ctx = streamer->create_context();
    for (size_t i = 0; i < addCnt; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = (float)i + baseKey;
      }
      succAdd += !streamer->add_impl(baseKey + i, vec.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    return succAdd;
  };
  auto t2 = std::async(std::launch::async, addVector, 1000, 1000);
  auto t3 = std::async(std::launch::async, addVector, 2000, 1000);
  auto t1 = std::async(std::launch::async, addVector, 0, 1000);
  ASSERT_EQ(1000U, t1.get());
  ASSERT_EQ(1000U, t2.get());
  ASSERT_EQ(1000U, t3.get());
  streamer->close();

  // checking data
  ASSERT_EQ(0, streamer->open(storage));
  auto provider = streamer->create_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  uint64_t min = 1000;
  uint64_t max = 0;
  while (iter->is_valid()) {
    float *data = (float *)iter->data();
    for (size_t d = 0; d < dim; ++d) {
      ASSERT_FLOAT_EQ((float)iter->key(), data[d]);
    }
    total++;
    min = std::min(min, iter->key());
    max = std::max(max, iter->key());
    iter->next();
  }
  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, min);
  ASSERT_EQ(2999, max);

  // ====== multi thread search
  size_t topk = 100;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    NumericalVector<float> vec(dim);
    auto linearCtx = streamer->create_context();
    auto linearByPkeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    linearCtx->set_topk(topk);
    linearByPkeysCtx->set_topk(topk);
    ctx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    for (size_t i = 0; i < cnt; i += 1) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i + 0.1f;
      }
      ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
      ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{0, 1, 2}};
      ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                      linearByPkeysCtx));
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      ASSERT_EQ(i, r2[0].key());
      auto &r3 = linearByPkeysCtx->result();
      ASSERT_EQ(std::min(topk, p_keys[0].size()), r3.size());
#if 0
            printf("linear: %zd => %zd %zd %zd %zd %zd\n", i, r2[0].key,
                   r2[1].key, r2[2].key, r2[3].key, r2[4].key);
            printf("knn: %zd => %zd %zd %zd %zd %zd\n", i, r1[0].key, r1[1].key,
                   r1[2].key, r1[3].key, r1[4].key);
#endif
      for (size_t k = 0; k < topk; ++k) {
        totalCnts++;
        for (size_t j = 0; j < topk; ++j) {
          if (r2[j].key() == r1[k].key()) {
            totalHits++;
            break;
          }
        }
      }
    }
    // printf("%f\n", totalHits * 1.0f / totalCnts);
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

TEST_F(HnswStreamerTest, TestKnnConcurrentAddAndSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 128);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 10);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 64);
  params.set(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, 30 * 1024 * 1024U);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_CHUNK_SIZE, 4096);
  params.set(PARAM_HNSW_STREAMER_EF, 32);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnConcurrentAddAndSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    NumericalVector<float> vec(dim);
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    auto ctx = streamer->create_context();
    size_t succAdd = 0;
    for (size_t i = 0; i < addCnt; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = (float)i + baseKey;
      }
      succAdd += !streamer->add_impl(baseKey + i, vec.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    return succAdd;
  };

  // ====== multi thread search
  auto knnSearch = [&]() {
    size_t topk = 100;
    size_t cnt = 3000;
    NumericalVector<float> vec(dim);
    auto linearCtx = streamer->create_context();
    auto linearByPKeysCtx = streamer->create_context();
    auto ctx = streamer->create_context();
    linearCtx->set_topk(topk);
    linearByPKeysCtx->set_topk(topk);
    ctx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    for (size_t i = 0; i < cnt; i += 1) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i + 0.1f;
      }
      ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
      ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{0, 1, 2}};
      ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                      linearByPKeysCtx));
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      auto &r3 = linearByPKeysCtx->result();
      ASSERT_EQ(std::min(topk, p_keys[0].size()), r3.size());
// ASSERT_EQ(i, r2[0].key);
#if 0
            printf("linear: %zd => %zd %zd %zd %zd %zd\n", i, r2[0].key,
                   r2[1].key, r2[2].key, r2[3].key, r2[4].key);
            printf("knn: %zd => %zd %zd %zd %zd %zd\n", i, r1[0].key, r1[1].key,
                   r1[2].key, r1[3].key, r1[4].key);
#endif
      for (size_t k = 0; k < topk; ++k) {
        totalCnts++;
        for (size_t j = 0; j < topk; ++j) {
          if (r2[j].key() == r1[k].key()) {
            totalHits++;
            break;
          }
        }
      }
    }
    //        printf("%f\n", totalHits * 1.0f / totalCnts);
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };
  auto t0 = std::async(std::launch::async, addVector, 0, 1000);
  ASSERT_EQ(1000, t0.get());
  auto t1 = std::async(std::launch::async, addVector, 1000, 1000);
  auto t2 = std::async(std::launch::async, addVector, 2000, 1000);
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  ASSERT_EQ(1000, t1.get());
  ASSERT_EQ(1000, t2.get());
  s1.wait();
  s2.wait();

  // checking data
  auto provider = streamer->create_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  uint64_t min = 1000;
  uint64_t max = 0;
  while (iter->is_valid()) {
    float *data = (float *)iter->data();
    for (size_t d = 0; d < dim; ++d) {
      ASSERT_FLOAT_EQ((float)iter->key(), data[d]);
    }
    total++;
    min = std::min(min, iter->key());
    max = std::max(max, iter->key());
    iter->next();
  }
  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, min);
  ASSERT_EQ(2999, max);
}

TEST_F(HnswStreamerTest, TestBfThreshold) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_EF, 16);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessBfThreshold", true));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 100000;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(1U);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  streamer->flush(0UL);
  streamer->close();

  IndexStreamer::Pointer streamer1 =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer1, nullptr);
  auto params1 = params;
  params1.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, cnt - 1);
  ASSERT_EQ(0, streamer1->init(*index_meta_ptr_, params1));
  ASSERT_EQ(0, streamer1->open(storage));
  auto ctx1 = streamer1->create_context();

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_NE(streamer2, nullptr);
  auto params2 = params;
  params2.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, cnt);
  ASSERT_EQ(0, streamer2->init(*index_meta_ptr_, params2));
  ASSERT_EQ(0, streamer2->open(storage));
  auto ctx2 = streamer2->create_context();

  // do searcher
  size_t cost1 = 0;
  size_t cost2 = 0;
  for (size_t i = 0; i < 100; ++i) {
    auto t1 = ailego::Monotime::MicroSeconds();
    ASSERT_EQ(0, streamer1->search_impl(vec.data(), qmeta, ctx1));
    auto t2 = ailego::Monotime::MicroSeconds();
    ASSERT_EQ(0, streamer2->search_impl(vec.data(), qmeta, ctx2));
    auto t3 = ailego::Monotime::MicroSeconds();
    cost1 += t2 - t1;
    cost2 += t3 - t2;
  }

  ASSERT_LT(cost1, cost2);

  ailego::Params update_params;
  update_params.set(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE, true);
  update_params.set(PARAM_HNSW_STREAMER_EF, 50);
  ctx1->set_debug_mode(true);
  ctx1->update(update_params);
  ASSERT_EQ(0, streamer1->search_impl(vec.data(), qmeta, ctx1));
  LOG_DEBUG("%s", ctx1->debug_string().c_str());
}

TEST_F(HnswStreamerTest, TestFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 1000);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessFilter", true));
  ASSERT_EQ(0, streamer->open(storage));


  NumericalVector<float> vec(dim);
  size_t cnt = 2000;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(10U);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    p_keys[0].push_back(i);
  }

  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 100.1;
  }
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(10, results.size());
  ASSERT_EQ(100, results[0].key());
  ASSERT_EQ(101, results[1].key());
  ASSERT_EQ(99, results[2].key());

  auto filterFunc = [](uint64_t key) {
    if (key == 100UL || key == 101UL) {
      return true;
    }
    return false;
  };
  ctx->set_filter(filterFunc);

  // after set filter
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto &results1 = ctx->result();
  ASSERT_EQ(10, results1.size());
  ASSERT_EQ(99, results1[0].key());
  ASSERT_EQ(102, results1[1].key());
  ASSERT_EQ(98, results1[2].key());

  // linear
  ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
  auto &results2 = ctx->result();
  ASSERT_EQ(10, results2.size());
  ASSERT_EQ(99, results2[0].key());
  ASSERT_EQ(102, results2[1].key());
  ASSERT_EQ(98, results2[2].key());

  // linear by p_keys
  ASSERT_EQ(0,
            streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta, ctx));
  auto &results3 = ctx->result();
  ASSERT_EQ(10, results3.size());
  ASSERT_EQ(99, results3[0].key());
  ASSERT_EQ(102, results3[1].key());
  ASSERT_EQ(98, results3[2].key());
}

TEST_F(HnswStreamerTest, TestMaxIndexSize) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  constexpr size_t static dim = 128;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessMaxIndexSize", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t vsz0 = 0;
  size_t rss0 = 0;
  if (!ailego::MemoryHelper::SelfUsage(&vsz0, &rss0)) {
    // do not check if get mem usage failed
    return;
  }
  if (vsz0 > 1024 * 1024 * 1024 * 1024UL) {
    // asan mode
    return;
  }

  NumericalVector<float> vec(dim);
  size_t writeCnt1 = 10000;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto ctx = streamer->create_context();
  for (size_t i = 0; i < writeCnt1; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  size_t vsz1 = 0;
  size_t rss1 = 0;
  ailego::MemoryHelper::SelfUsage(&vsz1, &rss1);
  size_t increment1 = rss1 - rss0;
  ASSERT_GT(writeCnt1 * 128 * 4 + writeCnt1 * 100 * 4, increment1 * 0.8f);
  ASSERT_LT(writeCnt1 * 128 * 4 + writeCnt1 * 100 * 4, increment1 * 1.2f);

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(HnswStreamerTest, TestKnnCleanUp) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TessKnnCluenUp1", true));
  ailego::Params params;
  constexpr size_t static dim1 = 32;
  IndexMeta meta1(IndexMeta::DataType::DT_FP32, dim1);
  meta1.set_metric("SquaredEuclidean", 0, ailego::Params());
  NumericalVector<float> vec1(dim1);
  ASSERT_EQ(0, streamer->init(meta1, params));
  ASSERT_EQ(0, streamer->open(storage1));
  IndexQueryMeta qmeta1(IndexMeta::DataType::DT_FP32, dim1);
  auto ctx1 = streamer->create_context();
  ASSERT_EQ(0, streamer->add_impl(1, vec1.data(), qmeta1, ctx1));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());

  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage2);
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TessKnnCluenUp2", true));
  constexpr size_t static dim2 = 64;
  IndexMeta meta2(IndexMeta::DataType::DT_FP32, dim2);
  meta2.set_metric("SquaredEuclidean", 0, ailego::Params());
  NumericalVector<float> vec2(dim2);
  ASSERT_EQ(0, streamer->init(meta2, params));
  ASSERT_EQ(0, streamer->open(storage2));
  IndexQueryMeta qmeta2(IndexMeta::DataType::DT_FP32, dim2);
  auto ctx2 = streamer->create_context();
  ASSERT_EQ(0, streamer->add_impl(2, vec2.data(), qmeta2, ctx2));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());
}

TEST_F(HnswStreamerTest, TestIndexSizeQuota) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestIndexSizeQuota", true));
  ailego::Params params;
  constexpr size_t static dim = 512;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());
  params.set(PARAM_HNSW_STREAMER_MAX_INDEX_SIZE, 2 * 1024 * 1024U);
  params.set(PARAM_HNSW_STREAMER_CHUNK_SIZE, 100 * 1024U);
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));
  NumericalVector<float> vec(dim);
  size_t writeCnt1 = 850;
  int ret = 0;
  auto ctx = streamer->create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < writeCnt1; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    int iRet = streamer->add_impl(i, vec.data(), qmeta, ctx);
    if (iRet != 0) {
      ret = iRet;
    }
  }
  ASSERT_EQ(IndexError_IndexFull, ret);
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());
}

TEST_F(HnswStreamerTest, TestBloomFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestBloomFilter", true));
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 100);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  ctx->set_topk(10U);
  size_t cnt = 5000;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    if ((i + 1) % 10 == 0) {
      ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
      auto &results = ctx->result();
      ASSERT_EQ(10, results.size());
    }
  }
}

TEST_F(HnswStreamerTest, TestStreamerParams) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestStreamerParams", true));
  ailego::Params params;
  params.set("proxima.hnsw.streamer.docs_hard_limit", 5);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto ctx = streamer->create_context();
  ASSERT_EQ(0, streamer->add_impl(1, vec.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(2, vec.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(3, vec.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(4, vec.data(), qmeta, ctx));
  ASSERT_EQ(0, streamer->add_impl(5, vec.data(), qmeta, ctx));
  ASSERT_EQ(IndexError_IndexFull,
            streamer->add_impl(6, vec.data(), qmeta, ctx));
}

#if 0
TEST_F(HnswStreamerTest, TestCheckCrc)
{
    IndexStreamer::Pointer streamer =
        IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_TRUE(streamer != nullptr);

    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ASSERT_NE(nullptr, storage);
    ailego::Params stg_params;
    ASSERT_EQ(0, storage->init(stg_params));
    std::string path = dir_ + "TestCheckCrc";
    ASSERT_EQ(0, storage->open(path, true));
    ailego::Params params;
    params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
    params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
    params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
    params.set(PARAM_HNSW_STREAMER_EF, 100);
    params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
    params.set(PARAM_HNSW_STREAMER_VISIT_BLOOMFILTER_ENABLE, true);
    ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
    ASSERT_EQ(0, streamer->open(storage));

    NumericalVector<float> vec(dim);
    auto ctx = streamer->create_context();
    ASSERT_NE(nullptr, ctx);
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    size_t cnt = 100;
    for (size_t i = 0; i < cnt; i++) {
        for (size_t j = 0; j < dim; ++j) {
            vec[j] = i;
        }
        streamer->add_impl(i, vec.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    streamer->close();
    storage->flush();
    storage->close();

    int fd = open(path.c_str(), O_RDWR);
    ASSERT_GT(fd, 0);
    struct stat fs;
    ASSERT_EQ(0, fstat(fd, &fs));
    char buf[1024];
    pwrite(fd, buf, sizeof(buf), fs.st_size/2);

    ASSERT_EQ(0, storage->open(path, true));
    IndexStreamer::Pointer streamer2 =
        IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_NE(streamer2, nullptr);

    ailego::Params params2;
    params2.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
    params2.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
    params2.set("proxima.hnsw.streamer.check_crc_enable", true);
    ASSERT_EQ(0, streamer2->init(*index_meta_ptr_, params2));
    ASSERT_EQ(0, streamer2->open(storage));
}
#endif

TEST_F(HnswStreamerTest, TestCheckStats) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  std::string path = dir_ + "/TestCheckStats.index";
  ASSERT_EQ(0, storage->open(path, true));
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 100);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, true);
  params.set(PARAM_HNSW_STREAMER_CHUNK_SIZE, 512 * 1024U);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto &stats = streamer->stats();
  ASSERT_EQ(0U, stats.revision_id());
  ASSERT_EQ(0U, stats.loaded_count());
  ASSERT_EQ(0U, stats.added_count());
  ASSERT_EQ(0U, stats.discarded_count());
  ASSERT_EQ(0u, stats.index_size() % ailego::MemoryHelper::PageSize());
  ASSERT_EQ(0U, stats.dumped_size());
  ASSERT_EQ(0U, stats.check_point());
  auto createTime = stats.create_time();
  auto updateTime = stats.update_time();
  ASSERT_GT(createTime, 0UL);
  ASSERT_EQ(createTime, updateTime);

  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  size_t cnt = 3000;
  size_t size1 = stats.index_size();
  size_t size2 = 0;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
    ASSERT_EQ(i + 1, stats.added_count());
    if (i == 0UL) {
      size2 = stats.index_size();
    }
  }
  size_t size3 = stats.index_size();
  ASSERT_GT(size2, size1);
  ASSERT_GT(size3, size2);
  LOG_INFO("size1=%zu size2=%zu size3=%zu", size1, size2, size3);

  uint64_t checkPoint = 23423UL;
  streamer->flush(checkPoint);
  size_t size4 = stats.index_size();
  ASSERT_EQ(size3, size4);
  auto stats1 = streamer->stats();
  ASSERT_EQ(1U, stats1.revision_id());
  ASSERT_EQ(0U, stats1.loaded_count());
  ASSERT_EQ(cnt, stats1.added_count());
  ASSERT_EQ(0U, stats1.discarded_count());
  ASSERT_GT(stats1.index_size(), 0U);
  ASSERT_EQ(0U, stats1.dumped_size());
  ASSERT_EQ(checkPoint, stats1.check_point());
  auto createTime1 = stats1.create_time();
  auto updateTime1 = stats1.update_time();
  ASSERT_GE(updateTime1, createTime1);
  ASSERT_EQ(createTime, createTime1);
  streamer->close();

  ASSERT_EQ(0, streamer->open(storage));
  auto &stats2 = streamer->stats();
  ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  ASSERT_EQ(0, streamer->add_impl(10000UL, vec.data(), qmeta, ctx));
  ASSERT_EQ(2U, stats2.revision_id());
  ASSERT_EQ(cnt, stats2.loaded_count());
  ASSERT_EQ(1U, stats2.added_count());
  ASSERT_EQ(0U, stats2.discarded_count());
  ASSERT_GT(stats1.index_size(), 0);
  ASSERT_EQ(0U, stats2.dumped_size());
  ASSERT_EQ(checkPoint, stats2.check_point());
  auto createTime2 = stats2.create_time();
  auto updateTime2 = stats2.update_time();
  ASSERT_EQ(createTime2, createTime1);
  ASSERT_GE(updateTime2, updateTime1);

  sleep(1);
  streamer->flush(checkPoint + 1);
  ASSERT_NE(0, streamer->add_impl(0U, vec.data(), qmeta, ctx));
  auto &stats3 = streamer->stats();
  ASSERT_EQ(2U, stats3.revision_id());
  ASSERT_EQ(cnt, stats3.loaded_count());
  ASSERT_EQ(1U, stats3.added_count());
  ASSERT_EQ(1U, stats3.discarded_count());
  ASSERT_EQ(stats2.index_size(), stats3.index_size());
  ASSERT_EQ(0U, stats3.dumped_size());
  ASSERT_EQ(checkPoint + 1, stats3.check_point());
  auto createTime3 = stats3.create_time();
  auto updateTime3 = stats3.update_time();
  ASSERT_EQ(createTime3, createTime1);
  ASSERT_GT(updateTime3, updateTime2);

  auto dpath = dir_ + "/dumpIndex";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(0, dumper->create(dpath));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, dumper->close());
  size_t doc_cnt = stats3.loaded_count() + stats3.added_count();
  struct stat st;
  ASSERT_EQ(3001UL, doc_cnt);
  ASSERT_EQ(0, stat(dpath.c_str(), &st));
  ASSERT_LT(st.st_size - stats3.dumped_size(), 8192);

  streamer->close();
}

TEST_F(HnswStreamerTest, TestCheckDuplicateAndGetVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestCheckDuplicateAndGetVec", true));
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_STREAMER_FILTER_SAME_KEY, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < 1000; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }
  for (size_t i = 0; i < 1000; i += 10) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ASSERT_EQ(IndexError_Duplicate,
              streamer->add_impl(i, vec.data(), qmeta, ctx));
  }
  auto provider = streamer->create_provider();
  for (size_t i = 0; i < 1000; i++) {
    const float *data = (const float *)provider->get_vector(i);
    ASSERT_NE(data, nullptr);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(i, data[j]);
    }
  }

  streamer->flush(0UL);
  streamer.reset();
}

class TestDumper : public IndexDumper {
  virtual int init(const ailego::Params &) {
    return 0;
  }
  virtual int cleanup(void) {
    return 0;
  }
  virtual int create(const std::string &path) {
    return 0;
  }
  virtual uint32_t magic(void) const {
    return 0;
  }
  virtual int close(void) {
    return 0;
  }
  virtual int append(const std::string &id, size_t data_size,
                     size_t padding_size, uint32_t crc) {
    usleep(100000);
    return 0;
  }
  virtual size_t write(const void *data, size_t len) {
    return len;
  }
};

TEST_F(HnswStreamerTest, TestDumpIndexAndAdd) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestDumpIndexAndAdd", true));
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  ASSERT_NE(nullptr, ctx);
  int code = 0;
  std::mutex mutex;
  auto addVector = [&](int a, int b) {
    int success = 0;
    mutex.unlock();
    for (int i = a; i < b; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i;
      }
      int ret = streamer->add_impl(i, vec.data(), qmeta, ctx);
      if (ret != 0) {
        code = ret;
        ASSERT_EQ(IndexError_Unsupported, code);
        i = i - 1;  // retry
        usleep(10000);
      } else {
        success++;
      }
    }
    std::cout << "addVector: " << success << " success" << std::endl;
  };
  mutex.lock();
  addVector(0, 2000);
  mutex.lock();
  auto t2 = std::async(std::launch::async, addVector, 2000, 3000);
  auto path1 = dir_ + "/dumpIndex1";
  auto dumper1 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper1, nullptr);
  ASSERT_EQ(0, dumper1->create(path1));
  mutex.lock();  // sync: wait addVector start and release lock
  auto test_dumper = std::make_shared<TestDumper>();
  ASSERT_EQ(0, streamer->dump(test_dumper));
  mutex.unlock();
  ASSERT_EQ(0, streamer->dump(dumper1));
  ASSERT_EQ(0, dumper1->close());
  t2.get();
  streamer->close();
  ASSERT_EQ(IndexError_Unsupported, code);

  // check dump index
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->init(ailego::Params()));
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto iter = searcher->create_provider()->create_iterator();
  size_t docs = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    docs++;
    iter->next();
  }
  ASSERT_GE(docs, 2000U);

  // check streamer
  ASSERT_EQ(0, streamer->open(storage));
  iter = streamer->create_provider()->create_iterator();
  docs = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    docs++;
    iter->next();
  }
  ASSERT_EQ(docs, 3000U);
}


TEST_F(HnswStreamerTest, TestProvider) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestGetVector", true));
  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 5);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);

  //! prepare data
  size_t docs = 10000UL;
  srand(ailego::Realtime::MilliSeconds());
  std::vector<key_t> keys(docs);
  bool rand_key = rand() % 2;
  bool rand_order = rand() % 2;
  size_t step = rand() % 2 + 1;
  LOG_DEBUG("randKey=%u randOrder=%u step=%zu", rand_key, rand_order, step);
  if (rand_key) {
    std::mt19937 mt;
    std::uniform_int_distribution<size_t> dt(
        0, std::numeric_limits<size_t>::max());
    for (size_t i = 0; i < docs; ++i) {
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
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < keys.size(); i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = keys[i];
    }
    streamer->add_impl(keys[i], vec.data(), qmeta, ctx);
  }

  auto path1 = dir_ + "/TestGetVector1";
  auto dumper1 = IndexFactory::CreateDumper("FileDumper");
  ASSERT_NE(dumper1, nullptr);
  ASSERT_EQ(0, dumper1->create(path1));
  ASSERT_EQ(0, streamer->dump(dumper1));
  ASSERT_EQ(0, dumper1->close());
  streamer->close();

  // check dump index
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("HnswSearcher");
  auto container = IndexFactory::CreateStorage("FileReadStorage");
  ASSERT_EQ(0, container->init(ailego::Params()));
  ASSERT_EQ(0, container->open(path1, false));
  ASSERT_NE(searcher, nullptr);
  ASSERT_EQ(0, searcher->init(ailego::Params()));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));
  auto iter = searcher->create_provider()->create_iterator();
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
  ASSERT_EQ(cnt, docs);

  // check streamer
  ASSERT_EQ(0, streamer->open(storage));
  iter = streamer->create_provider()->create_iterator();
  cnt = 0;
  while (iter->is_valid()) {
    auto key = iter->key();
    const float *d = reinterpret_cast<const float *>(iter->data());
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d[j], key);
    }
    cnt++;
    iter->next();
  }
  ASSERT_EQ(cnt, docs);


  auto searcher_provider = searcher->create_provider();
  auto streamer_provider = streamer->create_provider();
  for (size_t i = 0; i < keys.size(); ++i) {
    const float *d1 =
        reinterpret_cast<const float *>(searcher_provider->get_vector(keys[i]));
    ASSERT_TRUE(d1);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d1[j], keys[i]);
    }

    const float *d2 =
        reinterpret_cast<const float *>(streamer_provider->get_vector(keys[i]));
    ASSERT_TRUE(d2);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_FLOAT_EQ(d2[j], keys[i]);
    }
  }

  ASSERT_EQ(dim, streamer_provider->dimension());
  ASSERT_EQ(index_meta_ptr_->element_size(), streamer_provider->element_size());
  ASSERT_EQ(index_meta_ptr_->data_type(), streamer_provider->data_type());
}

TEST_F(HnswStreamerTest, TestSharedContext) {
  auto create_streamer = [](std::string path) {
    IndexStreamer::Pointer streamer =
        IndexFactory::CreateStreamer("HnswStreamer");
    auto storage = IndexFactory::CreateStorage("MMapFileStorage");
    ailego::Params stg_params;
    storage->init(stg_params);
    storage->open(path, true);
    ailego::Params params;
    streamer->init(*index_meta_ptr_, params);
    streamer->open(storage);
    return streamer;
  };
  auto streamer1 = create_streamer(dir_ + "TestSharedContext.index1");
  auto streamer2 = create_streamer(dir_ + "TestSharedContext.index2");
  auto streamer3 = create_streamer(dir_ + "TestSharedContext.index3");

  srand(ailego::Realtime::MilliSeconds());
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  auto do_test = [&](int start) {
    auto code = rand() % 3;
    IndexStreamer::Context::Pointer ctx;
    switch (code) {
      case 0:
        ctx = streamer1->create_context();
        break;
      case 1:
        ctx = streamer2->create_context();
        break;
      case 2:
        ctx = streamer3->create_context();
        break;
    };
    ctx->set_topk(1);
    uint64_t key1 = start + 0;
    uint64_t key2 = start + 1;
    uint64_t key3 = start + 2;
    NumericalVector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = 0.1f;
    }
    for (int i = 0; i < 1000; ++i) {
      NumericalVector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = rand();
      }
      int ret = 0;
      auto code = rand() % 3;
      switch (code) {
        case 0:
          streamer1->add_impl(key1, vec.data(), qmeta, ctx);
          key1 += 3;
          ret = streamer1->search_impl(query.data(), qmeta, ctx);
          break;
        case 1:
          streamer2->add_impl(key2, vec.data(), qmeta, ctx);
          key2 += 3;
          streamer2->add_impl(key2, vec.data(), qmeta, ctx);
          key2 += 3;
          ret = streamer2->search_impl(query.data(), qmeta, ctx);
          break;
        case 2:
          streamer3->add_impl(key3, vec.data(), qmeta, ctx);
          key3 += 3;
          streamer3->add_impl(key3, vec.data(), qmeta, ctx);
          key3 += 3;
          streamer3->add_impl(key3, vec.data(), qmeta, ctx);
          key3 += 3;
          ret = streamer3->search_impl(query.data(), qmeta, ctx);
          break;
      }
      EXPECT_EQ(0, ret);
      auto &results = ctx->result();
      EXPECT_EQ(1, results.size());
      EXPECT_EQ(code, results[0].key() % 3);
    }
  };

  auto t1 = std::async(std::launch::async, do_test, 0);
  auto t2 = std::async(std::launch::async, do_test, 30000000);
  t1.wait();
  t2.wait();
}

TEST_F(HnswStreamerTest, TestMipsEuclideanMetric) {
  constexpr size_t static dim = 32;
  std::srand(ailego::Realtime::MilliSeconds());
  // int injection_type = rand() % 2;
  int injection_type = 0;

  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  ailego::Params params;
  params.set("proxima.mips_euclidean.metric.injection_type", injection_type);
  meta.set_metric("MipsSquaredEuclidean", 0, params);
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestMipsSquaredEuclidean", true));
  const size_t COUNT = 10000;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  {
    IndexStreamer::Pointer streamer =
        IndexFactory::CreateStreamer("HnswStreamer");
    ASSERT_TRUE(streamer != nullptr);
    ASSERT_EQ(0, streamer->init(meta, params));
    ASSERT_EQ(0, streamer->open(storage));
    const auto &metric_params = streamer->meta().metric_params();
    EXPECT_FLOAT_EQ(0.0, metric_params.get_as_float(
                             "proxima.mips_euclidean.metric.max_l2_norm"));
    auto ctx = streamer->create_context();
    for (size_t i = COUNT; i < 2 * COUNT; i++) {
      float vec[dim];
      for (size_t d = 0; d < dim; ++d) {
        vec[d] = i;
      }
      ASSERT_EQ(0, streamer->add_impl(i, vec, qmeta, ctx));
    }
    ASSERT_EQ(0, streamer->flush(0UL));
    ASSERT_EQ(0, streamer->close());
  }
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));
  const auto &metric_params = streamer->meta().metric_params();
  // NoTrain for LocalizedSpherical (type == 1), so max_l2_norm equals to 0
  EXPECT_FLOAT_EQ(
      injection_type == 0 ? 0.0f : 113131.0f,
      metric_params.get_as_float("proxima.mips_euclidean.metric.max_l2_norm"));
  auto ctx = streamer->create_context();
  for (size_t i = 0; i < COUNT; i++) {
    float vec[dim];
    for (size_t d = 0; d < dim; ++d) {
      vec[d] = i;
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec, qmeta, ctx));
  }
  float vec[dim];
  for (size_t d = 0; d < dim; ++d) {
    vec[d] = 1.0;
  }

  ctx->set_topk(10);
  ASSERT_EQ(0, streamer->search_impl(vec, qmeta, ctx));
  const auto &results = ctx->result();
  EXPECT_EQ(results.size(), 10);
  EXPECT_NEAR((uint64_t)(2 * COUNT - 1), results[0].key(), 10);
}

TEST_F(HnswStreamerTest, TestBruteForceSetupInContext) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0,
            storage->open(dir_ + "/TestBruteForceSetupInContext.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
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

  size_t topk = 200;
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;

  bool set_bf_threshold = false;
  bool use_update = false;

  for (size_t i = 0; i < cnt; i++) {
    auto linearCtx = streamer->create_context();
    auto knnCtx = streamer->create_context();

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
        ailego::Params streamerParamsExtra;

        streamerParamsExtra.set("proxima.hnsw.streamer.brute_force_threshold",
                                cnt);
        knnCtx->update(streamerParamsExtra);
      } else {
        knnCtx->set_bruteforce_threshold(cnt);
      }

      use_update = !use_update;
    }
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));

    auto t2 = ailego::Realtime::MicroSeconds();

    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));

    // auto t3 = ailego::Realtime::MicroSeconds();

    if (set_bf_threshold) {
      linearTotalTime += t2 - t1;
    } else {
      knnTotalTime += t2 - t1;
    }

    set_bf_threshold = !set_bf_threshold;

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
  float topk1Recall = topk1Hits * 1.0f / cnt;
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

TEST_F(HnswStreamerTest, TestKnnSearchCosine) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestKnnSearchCosine.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 4000U;
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

  size_t query_cnt = 200U;
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;


  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestFetchVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestFetchVector.index", true));
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

  for (size_t i = 0; i < cnt; i++) {
    const void *vector = streamer->get_vector(i);
    ASSERT_NE(vector, nullptr);

    float vector_value = *(float *)(vector);
    ASSERT_FLOAT_EQ(vector_value, i);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 200U;
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
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
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
    ASSERT_FLOAT_EQ(vector_value, i);
  }
  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
}

TEST_F(HnswStreamerTest, TestFetchVectorCosine) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestFetchVectorCosine.index", true));
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

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = streamer->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 200U;
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestFetchVectorCosineHalfFloatConverter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
      0, storage->open(dir_ + "/TestFetchVectorCosineHalfFloatConverter.index",
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
      vec[j] = ailego::FloatHelper::ToFP16(dist(gen));
    }

    std::string new_vec;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));

    vecs.push_back(vec);
  }

  for (size_t i = 0; i < cnt; i++) {
    uint16_t expected_vec_value = vecs[i][dim - 1];

    const void *vector = streamer->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(uint16_t));
    reformer->revert(vector, new_meta, &denormalized_vec);

    uint16_t vector_value = *((uint16_t *)(denormalized_vec.data()) + dim - 1);
    float vector_value_float = ailego::FloatHelper::ToFP32(vector_value);

    float expected_vec_float = ailego::FloatHelper::ToFP32(expected_vec_value);

    EXPECT_NEAR(expected_vec_float, vector_value_float, epsilon);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 200U;
  size_t topk = 30;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestFetchVectorCosineFp16Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestFetchVectorCosineFp16Converter.index",
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

  for (size_t i = 0; i < cnt; i++) {
    float expected_vec_value = vecs[i][dim - 1];

    const void *vector = streamer->get_vector(i);


    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);
    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);

    EXPECT_NEAR(expected_vec_value, vector_value, epsilon);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 200U;
  size_t topk = 30;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;

  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestFetchVectorCosineInt8Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestFetchVectorCosineInt8Converter.index",
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

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = streamer->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 200U;
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestFetchVectorCosineInt4Converter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestFetchVectorCosineInt4Converter.index",
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

  for (size_t i = 0; i < cnt; i++) {
    float add_on = i * 10;

    const void *vector = streamer->get_vector(i);
    ASSERT_NE(vector, nullptr);

    std::string denormalized_vec;
    denormalized_vec.resize(dim * sizeof(float));
    reformer->revert(vector, new_meta, &denormalized_vec);

    float vector_value = *((float *)(denormalized_vec.data()) + dim - 1);
    EXPECT_NEAR(vector_value, fixed_value + add_on, epsilon);
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  knnCtx->set_fetch_vector(true);

  size_t query_cnt = 100U;
  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  for (size_t i = 0; i < query_cnt; i++) {
    float add_on = i * 10;
    for (size_t j = 0; j < dim; ++j) {
      if (j < dim / 4)
        vec[j] = fixed_value;
      else
        vec[j] = fixed_value + add_on;
    }

    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0,
              streamer->search_bf_impl(new_query.data(), new_meta, linearCtx));
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

TEST_F(HnswStreamerTest, TestRnnSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  // params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestRnnSearchInnerProduct.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 1000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }

  size_t topk = 50;
  ctx->set_topk(topk);
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswStreamerTest, TestRnnSearchInnerProduct) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 50);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  // params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;

  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dim);
  index_meta.set_metric("InnerProduct", 0, ailego::Params());

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestRnnSearchInnerProduct.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 1000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }

    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }

  size_t topk = 50;
  ctx->set_topk(topk);

  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = -results[topk / 2].score();
  ctx->set_threshold(radius);
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(-radius, results[topk - 1].score());
}

TEST_F(HnswStreamerTest, TestRnnSearchCosine) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  // params.set(PARAM_HNSW_STREAMER_EF, 5);
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
  ASSERT_EQ(0, storage->open(dir_ + "/TestRnnSearchCosine.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 1000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dist(gen);
    }

    std::string new_vec;
    IndexQueryMeta new_meta;

    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  size_t topk = 50;
  ctx->set_topk(topk);

  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 1.0;
  }

  std::string new_query;
  IndexQueryMeta new_meta;
  ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

  ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, ctx));
  auto &results = ctx->result();
  ASSERT_EQ(topk, results.size());

  float radius = 0.5f;
  ctx->set_threshold(radius);
  ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, ctx));
  ASSERT_GT(topk, results.size());
  for (size_t k = 0; k < results.size(); ++k) {
    ASSERT_GE(radius, results[k].score());
  }

  // Test Reset Threshold
  ctx->reset_threshold();
  ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, ctx));
  ASSERT_EQ(topk, results.size());
  ASSERT_LT(radius, results[topk - 1].score());
}

TEST_F(HnswStreamerTest, TestGroup) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_HNSW_STREAMER_GET_VECTOR_ENABLE, true);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 5000U;
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

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

  size_t query_value = cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();

  total_time += t2 - t1;
  std::cout << "Total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();
  ASSERT_EQ(group_result.size(), group_num);

  for (uint32_t i = 0; i < group_result.size(); ++i) {
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);

    // const std::string &group_id = group_result[i].group_id();
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

  auto linear_pk_ctx = streamer->create_context();

  linear_pk_ctx->set_group_params(group_num, group_topk);
  linear_pk_ctx->set_group_by(groupbyFuncLinear);

  std::vector<std::vector<uint64_t>> p_keys;
  p_keys.resize(1);
  p_keys[0] = {4, 3, 2, 1, 5, 6, 7, 8, 9, 10};

  ASSERT_EQ(0, streamer->search_bf_by_p_keys_impl(vec.data(), p_keys, qmeta,
                                                  linear_pk_ctx));
  auto &linear_by_pkeys_group_result = linear_pk_ctx->group_result();
  ASSERT_EQ(linear_by_pkeys_group_result.size(), group_num);

  for (uint32_t i = 0; i < linear_by_pkeys_group_result.size(); ++i) {
    auto &result = linear_by_pkeys_group_result[i].docs();

    ASSERT_GT(result.size(), 0);

    // const std::string &group_id = linear_by_pkeys_group_result[i].group_id();
    //  std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }

    ASSERT_EQ(10 - i, result[0].key());
  }
}

TEST_F(HnswStreamerTest, TestGroupNotEnoughNum) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGroupNotEnoughNum.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 5000U;
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

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

  size_t query_value = cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();
  total_time += t2 - t1;

  std::cout << "Total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();

  ASSERT_EQ(group_result.size(), 10);
  for (uint32_t i = 0; i < group_result.size(); ++i) {
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);

    // const std::string &group_id = group_result[i].group_id();
    // std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }
  }
}

TEST_F(HnswStreamerTest, TestGroupInBruteforceSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  size_t cnt = 5000U;

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, cnt * 2);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0,
            storage->open(dir_ + "/TestGroupInBruteforceSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);

  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i / 10.0;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

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

  size_t query_value = cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = float(query_value) / 10 + 0.1f;
  }

  auto t1 = ailego::Realtime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
  auto t2 = ailego::Realtime::MicroSeconds();
  total_time += t2 - t1;

  std::cout << "Total time: " << total_time << std::endl;

  auto &group_result = ctx->group_result();

  ASSERT_EQ(group_result.size(), 5);
  for (uint32_t i = 0; i < group_result.size(); ++i) {
    auto &result = group_result[i].docs();

    ASSERT_GT(result.size(), 0);

    // const std::string &group_id = group_result[i].group_id();
    //  std::cout << "Group ID: " << group_id << std::endl;

    // for (uint32_t j = 0; j < result.size(); ++j) {
    //   std::cout << "\tKey: " << result[j].key() << std::fixed
    //             << std::setprecision(3) << ", Score: " << result[j].score()
    //             << std::endl;
    // }
  }
}

TEST_F(HnswStreamerTest, TestBinaryConverter) {
  uint32_t dimension = 2560;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  // params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  // params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  // params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  // params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);

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
  ASSERT_EQ(0, storage->open(dir_ + "/TestBinaryConverter.index", true));
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

  size_t query_cnt = 200U;
  auto knnCtx = streamer->create_context();

  float epison = 1e-6;
  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];
    std::string new_query;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->transform(vec.data(), qmeta, &new_query, &new_meta));

    size_t topk = 50;
    knnCtx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_impl(new_query.data(), new_meta, knnCtx));
    auto &results = knnCtx->result();
    ASSERT_EQ(topk, results.size());
    ASSERT_EQ(i, results[0].key());
    ASSERT_NEAR(0, results[0].score(), epison);
  }
}

TEST_F(HnswStreamerTest, TestAddAndSearchWithID) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(streamer != nullptr);

  ailego::Params params;
  params.set(PARAM_HNSW_STREAMER_MAX_NEIGHBOR_COUNT, 10);
  params.set(PARAM_HNSW_STREAMER_SCALING_FACTOR, 16);
  params.set(PARAM_HNSW_STREAMER_EFCONSTRUCTION, 10);
  params.set(PARAM_HNSW_STREAMER_EF, 5);
  params.set(PARAM_HNSW_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestAddAndSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  size_t cnt = 20000U;
  auto ctx = streamer->create_context();
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i += 4) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_with_id_impl(i, vec.data(), qmeta, ctx);
  }

  for (size_t i = 2; i < cnt; i += 4) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_with_id_impl(i, vec.data(), qmeta, ctx);
  }

  // streamer->print_debug_info();

  size_t topk = 200;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  uint64_t knnTotalTime = 0;
  uint64_t linearTotalTime = 0;
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  for (size_t i = 0; i < cnt / 10; i += 2) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = ailego::Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
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

    for (size_t j = 0; j < topk; ++j) {
      ASSERT_NE(linearResult[j].key(), kInvalidKey);
      ASSERT_NE(linearResult[j].index(), kInvalidKey);
      auto linearVec = static_cast<const float *>(
          streamer->get_vector_by_id(linearResult[j].index()));

      for (size_t z = 0; z < dim; ++z) {
        ASSERT_FLOAT_EQ(linearVec[z], linearResult[j].index());
      }
    }
    for (size_t j = 0; j < topk; ++j) {
      ASSERT_NE(knnResult[j].key(), kInvalidKey);
      ASSERT_NE(knnResult[j].index(), kInvalidKey);
      auto knnVec = static_cast<const float *>(
          streamer->get_vector_by_id(knnResult[j].index()));
      for (size_t z = 0; z < dim; ++z) {
        ASSERT_FLOAT_EQ(knnVec[z], knnResult[j].index());
      }
    }
  }
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 100.0f / cnt;
  float cost = linearTotalTime * 1.0f / knnTotalTime;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
  // EXPECT_GT(cost, 2.0f);
}

TEST_F(HnswStreamerTest, TestBasicRefiner) {
  uint32_t dimension = 1120;

  IndexStreamer::Pointer base_streamer =
      IndexFactory::CreateStreamer("HnswStreamer");
  ASSERT_TRUE(base_streamer != nullptr);

  IndexStreamer::Pointer refine_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(refine_streamer != nullptr);

  IndexRefiner::Pointer refiner = IndexFactory::CreateRefiner("BasicRefiner");
  ASSERT_TRUE(refiner != nullptr);

  ailego::Params params;
  IndexMeta index_meta(IndexMeta::DataType::DT_FP32, dimension);
  index_meta.set_metric("InnerProduct", 0, ailego::Params());

  ailego::Params converter_params;
  auto converter = IndexFactory::CreateConverter("BinaryConverter");
  ASSERT_TRUE(converter != nullptr);

  converter->init(index_meta, converter_params);

  IndexMeta index_meta_binary = converter->meta();

  auto reformer =
      IndexFactory::CreateReformer(index_meta_binary.reformer_name());
  ASSERT_TRUE(reformer != nullptr);

  ASSERT_EQ(0, reformer->init(index_meta_binary.reformer_params()));

  // base streamer
  ailego::Params base_stg_params;
  auto base_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, base_storage->init(base_stg_params));
  ASSERT_EQ(0, base_storage->open(dir_ + "/TestBasicRefinerBase.index", true));
  ASSERT_EQ(0, base_streamer->init(index_meta_binary, params));
  ASSERT_EQ(0, base_streamer->open(base_storage));

  auto base_ctx = base_streamer->create_context();
  ASSERT_TRUE(!!base_ctx);

  // refine streamer
  ailego::Params refine_stg_params;
  auto refine_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, refine_storage->init(refine_stg_params));
  ASSERT_EQ(0,
            refine_storage->open(dir_ + "/TestBasicRefinerRefine.index", true));
  ASSERT_EQ(0, refine_streamer->init(index_meta, params));
  ASSERT_EQ(0, refine_streamer->open(refine_storage));
  auto refine_ctx = refine_streamer->create_context();
  ASSERT_TRUE(!!refine_ctx);

  ailego::Params refiner_params;
  ASSERT_EQ(0, refiner->init(base_streamer, refine_streamer, refiner_params));

  auto ctx = refiner->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dimension);

  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<float> dist(-2.0, 2.0);
  std::vector<NumericalVector<float>> vecs;

  size_t cnt = 5000U;
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dimension);
    for (size_t j = 0; j < dimension; ++j) {
      vec[j] = dist(gen);
    }

    std::string binary_vec;
    IndexQueryMeta binary_qmeta;

    ASSERT_EQ(0,
              reformer->convert(vec.data(), qmeta, &binary_vec, &binary_qmeta));
    ASSERT_EQ(0, refiner->add_impl(i, binary_vec.data(), binary_qmeta,
                                   vec.data(), qmeta, ctx));

    vecs.push_back(vec);
  }

  size_t query_cnt = 200U;
  // size_t query_cnt = 1U;

  auto searcherCtx = refiner->create_context();

  for (size_t i = 0; i < query_cnt; i++) {
    auto &vec = vecs[i];

    // float abs_value{0};
    // for (size_t j = 0; j < dimension; ++j) {
    //   std::cout << "dim: " << j << ", value: " << vec[j] << std::endl;

    //   abs_value += std::abs(vec[j]);
    // }
    // std::cout << "abs value: " << abs_value << std::endl;

    std::string new_query;
    IndexQueryMeta binary_qmeta;
    ASSERT_EQ(
        0, reformer->transform(vec.data(), qmeta, &new_query, &binary_qmeta));

    size_t topk = 50;
    searcherCtx->set_topk(topk);
    ASSERT_EQ(0, refiner->search_impl(new_query.data(), binary_qmeta,
                                      vec.data(), qmeta, searcherCtx));
    auto &results = searcherCtx->result();
    ASSERT_EQ(topk, results.size());
    ASSERT_EQ(i, results[0].key());

    // for (size_t i = 0; i < results.size(); ++i) {
    //   std::cout << i << ", id: " << results[i].index()
    //             << ", score: " << results[i].score() << std::endl;
    // }
  }
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif