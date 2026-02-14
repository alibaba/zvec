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

#include <cstddef>
#include <future>
#include <string>
#include <vector>
#include <ailego/utility/math_helper.h>
#include <ailego/utility/memory_helper.h>
#include <gtest/gtest.h>
#include <zvec/ailego/buffer/buffer_manager.h>
#include <zvec/ailego/encoding/json/mod_json.h>
#include <zvec/core/framework/index_framework.h>
#include <zvec/core/framework/index_streamer.h>
#include "algorithm/flat/flat_utility.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace zvec::core;
using namespace zvec::ailego;
using namespace std;

constexpr size_t static dim = 16;

class FlatStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);
  void hybrid_scale(std::vector<float> &dense_value,
                    std::vector<float> &sparse_value, float alpha_scale);

  static std::string dir_;
  static std::shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string FlatStreamerTest::dir_("streamer_test/");
std::shared_ptr<IndexMeta> FlatStreamerTest::index_meta_ptr_;

void FlatStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (std::nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, dim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, Params());

  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

void FlatStreamerTest::TearDown(void) {
  char cmdBuf[100];
  snprintf(cmdBuf, 100, "rm -rf %s", dir_.c_str());
  system(cmdBuf);
}

TEST_F(FlatStreamerTest, TestAddVector) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/AddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  auto provider = streamer->create_provider();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < 1000UL; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    const float *data = (float *)provider->get_vector(i);
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_EQ(data[j], i);
    }
  }

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(FlatStreamerTest, TestLinearSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/Test/AddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  auto provider = streamer->create_provider();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 1000UL;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 1) {
    NumericalVector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(topk, result1.size());
    for (size_t j = 0; j < dim; ++j) {
      const float *data = (float *)provider->get_vector(result1[0].key());
      ASSERT_EQ(data[j], i);
    }
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }

  ctx->set_topk(100U);
  NumericalVector<float> vec(dim);
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

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(FlatStreamerTest, TestAddAndSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestAddAndSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  const size_t topk = 200U, cnt = 2000U;
  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ctx->set_topk(topk);
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
    auto &knnResult = ctx->result();
    ASSERT_EQ(std::min(i + 1, topk), knnResult.size());
  }
}

TEST_F(FlatStreamerTest, TestAddAndSearcherSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestAddAndSearcherSearch.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  const size_t topk = 200U, cnt = 2000U;
  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ctx->set_topk(topk);
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }

  std::string path1 = dir_ + "/TestAddAndSearcherSearchDump";
  auto dumper = IndexFactory::CreateDumper("FileDumper");
  ASSERT_EQ(0, dumper->init(Params()));
  ASSERT_EQ(0, dumper->create(path1));
  ASSERT_EQ(0, streamer->dump(dumper));
  ASSERT_EQ(0, dumper->close());

  auto container = IndexFactory::CreateStorage("MMapFileReadStorage");
  ASSERT_EQ(0, container->init(Params()));
  ASSERT_EQ(0, container->open(path1, false));
  IndexSearcher::Pointer searcher =
      IndexFactory::CreateSearcher("FlatSearcher");
  ASSERT_EQ(0, searcher->init(Params()));
  ASSERT_EQ(0, searcher->load(container, IndexMetric::Pointer()));

  auto linearCtx = searcher->create_context();
  linearCtx->set_topk(topk);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, linearCtx));
    auto &knnResult = linearCtx->result();
    ASSERT_EQ(topk, knnResult.size());
  }
}

TEST_F(FlatStreamerTest, TestLinearSearchRandomData) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  constexpr size_t static dim = 128;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestKnnSearchRandomData", true));
  ASSERT_EQ(0, streamer->init(meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t2 = Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t3 = Realtime::MicroSeconds();
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
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 1.0f / cnt;
#if 1
  printf(
      "knnTotalTime=%zu linearTotalTime=%zu totalHits=%d totalCnts=%d "
      "R@%zd=%f R@1=%f\n",
      (size_t)knnTotalTime, (size_t)linearTotalTime, totalHits, totalCnts, topk,
      recall, topk1Recall);
#endif
  EXPECT_GT(recall, 0.50f);
  EXPECT_GT(topk1Recall, 0.80f);
}

TEST_F(FlatStreamerTest, TestOpenClose) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  constexpr size_t static dim = 2048;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  Params params;
  // params.set(PARAM_FLAT_COLUMN_MAJOR_ORDER, false);
  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  ASSERT_NE(nullptr, storage2);
  Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TestOpenAndClose1", true));
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TestOpenAndClose2", true));
  ASSERT_EQ(0, streamer->init(meta, params));
  auto checkIter = [](size_t base, size_t total,
                      IndexStreamer::Pointer &streamer) {
    auto provider = streamer->create_provider();
    auto iter = provider->create_iterator();
    ASSERT_TRUE(!!iter);
    size_t cur = base;
    size_t cnt = 0;
    while (iter->is_valid()) {
      float *data = (float *)provider->get_vector(cur);
      for (size_t d = 0; d < dim; ++d) {
        ASSERT_EQ((float)cur, data[d]);
      }
      iter->next();
      cur += 2;
      cnt++;
    }
    ASSERT_EQ(cnt, total);
  };

  size_t testCnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer1->init(meta, params));
  ASSERT_EQ(0, streamer1->open(storage1));

  IndexStreamer::Pointer streamer2 =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);
  ASSERT_EQ(0, streamer2->init(meta, params));
  ASSERT_EQ(0, streamer2->open(storage2));

  checkIter(0, testCnt / 2, streamer1);
  checkIter(1, testCnt / 2, streamer2);
}

TEST_F(FlatStreamerTest, TestNoInit) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  streamer->cleanup();
}

TEST_F(FlatStreamerTest, TestForceFlush) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
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
      float *data = (float *)provider->get_vector(cur);
      for (size_t d = 0; d < dim; ++d) {
        ASSERT_EQ((float)cur, data[d]);
      }
      iter->next();
      cur++;
    }
    ASSERT_EQ(cur, total);
  };

  NumericalVector<float> vec(dim);
  size_t cnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
      ASSERT_EQ(i, data[j]);
    }
  }
}

TEST_F(FlatStreamerTest, TestMultiThread) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnMultiThread", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    NumericalVector<float> vec(dim);
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
      ASSERT_EQ((float)iter->key(), data[d]);
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
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
      ASSERT_EQ(i, r2[0].key());
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

TEST_F(FlatStreamerTest, TestConcurrentAddAndSearch) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  constexpr size_t static dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessKnnConcurrentAddAndSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
    NumericalVector<float> vec(dim);
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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
    IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
    for (size_t i = 0; i < cnt; i += 1) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = i + 0.1f;
      }
      ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, ctx));
      ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
      std::vector<std::vector<uint64_t>> p_keys = {{0, 1, 2}};
      auto &r1 = ctx->result();
      ASSERT_EQ(topk, r1.size());
      auto &r2 = linearCtx->result();
      ASSERT_EQ(topk, r2.size());
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
      ASSERT_EQ((float)iter->key(), data[d]);
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

TEST_F(FlatStreamerTest, TestFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessFilter", true));
  ASSERT_EQ(0, streamer->open(storage));


  NumericalVector<float> vec(dim);
  size_t cnt = 2000;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  ctx->set_topk(10U);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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

  auto &results3 = ctx->result();
  ASSERT_EQ(10, results3.size());
  ASSERT_EQ(99, results3[0].key());
  ASSERT_EQ(102, results3[1].key());
  ASSERT_EQ(98, results3[2].key());
}

TEST_F(FlatStreamerTest, TestMaxIndexSize) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  Params params;
  constexpr size_t static dim = 128;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, Params());
  ASSERT_EQ(0, streamer->init(meta, params));
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TessMaxIndexSize", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t vsz0 = 0;
  size_t rss0 = 0;
  if (!MemoryHelper::SelfUsage(&vsz0, &rss0)) {
    // do not check if get mem usage failed
    return;
  }
  if (vsz0 > 1024 * 1024 * 1024 * 1024UL) {
    // asan mode
    return;
  }

  NumericalVector<float> vec(dim);
  size_t writeCnt1 = 10000;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
  auto ctx = streamer->create_context();
  for (size_t i = 0; i < writeCnt1; i++) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_impl(i, vec.data(), qmeta, ctx);
  }
  size_t vsz1 = 0;
  size_t rss1 = 0;
  MemoryHelper::SelfUsage(&vsz1, &rss1);
  size_t increment1 = rss1 - rss0;
  // data + key + block_header
  size_t expect_size =
      writeCnt1 * 128 * 4 + writeCnt1 * 8 + writeCnt1 * 28 / 32;
  LOG_INFO("increment1: %lu, expect_size: %lu", increment1, expect_size);

  ASSERT_GT(expect_size, increment1 * 0.75f);
  ASSERT_LT(expect_size, increment1 * 1.25f);

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(FlatStreamerTest, TestCleanUp) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage1 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage1);
  Params stg_params;
  ASSERT_EQ(0, storage1->init(stg_params));
  ASSERT_EQ(0, storage1->open(dir_ + "TessKnnCluenUp1", true));
  Params params;
  constexpr size_t static dim1 = 32;
  IndexMeta meta1(IndexMeta::DataType::DT_FP32, dim1);
  meta1.set_metric("SquaredEuclidean", 0, Params());
  NumericalVector<float> vec1(dim1);
  ASSERT_EQ(0, streamer->init(meta1, params));
  ASSERT_EQ(0, streamer->open(storage1));
  IndexQueryMeta qmeta1(IndexMeta::DT_FP32, dim1);
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
  meta2.set_metric("SquaredEuclidean", 0, Params());
  NumericalVector<float> vec2(dim2);
  ASSERT_EQ(0, streamer->init(meta2, params));
  ASSERT_EQ(0, streamer->open(storage2));
  IndexQueryMeta qmeta2(IndexMeta::DT_FP32, dim2);
  auto ctx2 = streamer->create_context();
  ASSERT_EQ(0, streamer->add_impl(2, vec2.data(), qmeta2, ctx2));
  ASSERT_EQ(0, streamer->close());
  ASSERT_EQ(0, streamer->cleanup());
}

TEST_F(FlatStreamerTest, TestBloomFilter) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_TRUE(streamer != nullptr);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestBloomFilter", true));
  Params params;
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(dim);
  auto ctx = streamer->create_context();
  ASSERT_NE(nullptr, ctx);
  ctx->set_topk(10U);
  size_t cnt = 5000;
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);
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

TEST_F(FlatStreamerTest, TestGroup) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(streamer, nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t doc_cnt = 5000U;
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DT_FP32, dim);

  for (size_t i = 0; i < doc_cnt; i++) {
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

  size_t query_value = doc_cnt / 2;
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = query_value * 1.0 / 10 + 0.1f;
  }

  auto t1 = Realtime::MicroSeconds();
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, 1, ctx));
  auto t2 = Realtime::MicroSeconds();

  total_time += t2 - t1;
  std::cout << "Total time: " << total_time << std::endl;

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
}

TEST_F(FlatStreamerTest, TestAddAndSearchWithID) {
  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(streamer, nullptr);

  Params params;
  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, streamer->init(*index_meta_ptr_, params));
  ASSERT_EQ(0, streamer->open(storage));
  auto ctx = streamer->create_context();
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 20000U;
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i += 2) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    streamer->add_with_id_impl(i, vec.data(), qmeta, ctx);
  }
  for (size_t i = 1; i < cnt; i += 2) {
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
  for (size_t i = 0; i < cnt; i += 100) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i + 0.1f;
    }
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
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
#if 1
  printf(
      "knnTotalTime=%zu linearTotalTime=%zu totalHits=%d totalCnts=%d "
      "R@%zd=%f R@1=%f\n",
      (size_t)knnTotalTime, (size_t)linearTotalTime, totalHits, totalCnts, topk,
      recall, topk1Recall);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
}

TEST_F(FlatStreamerTest, TestAddAndSearchWithID2) {
  IndexStreamer::Pointer write_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  ASSERT_NE(write_streamer, nullptr);

  Params write_params;
  Params write_stg_params;
  auto write_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, write_storage->init(write_stg_params));
  ASSERT_EQ(0, write_storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, write_streamer->init(*index_meta_ptr_, write_params));
  ASSERT_EQ(0, write_streamer->open(write_storage));
  auto ctx = write_streamer->create_context();
  ASSERT_TRUE(!!ctx);

  size_t cnt = 20000U;
  NumericalVector<float> vec(dim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  for (size_t i = 0; i < cnt; i += 2) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_with_id_impl(i, vec.data(), qmeta, ctx);
  }
  for (size_t i = 1; i < cnt; i += 2) {
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = i;
    }
    write_streamer->add_with_id_impl(i, vec.data(), qmeta, ctx);
  }
  write_streamer->flush(0UL);
  write_streamer->close();
  write_streamer.reset();  //

  IndexStreamer::Pointer read_streamer =
      IndexFactory::CreateStreamer("FlatStreamer");
  Params read_params;
  read_params.set(PARAM_FLAT_USE_ID_MAP, false);
  Params read_stg_params;
  auto read_storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, read_storage->init(read_stg_params));
  ASSERT_EQ(0, read_storage->open(dir_ + "/TestGroup.index", true));
  ASSERT_EQ(0, read_streamer->init(*index_meta_ptr_, read_params));
  ASSERT_EQ(0, read_streamer->open(read_storage));
  auto linearCtx = read_streamer->create_context();
  auto knnCtx = read_streamer->create_context();
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
    auto t1 = Realtime::MicroSeconds();
    ASSERT_EQ(0, read_streamer->search_impl(vec.data(), qmeta, knnCtx));
    auto t2 = Realtime::MicroSeconds();
    ASSERT_EQ(0, read_streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto t3 = Realtime::MicroSeconds();
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
  std::cout << "knnTotalTime: " << knnTotalTime << std::endl;
  std::cout << "linearTotalTime: " << linearTotalTime << std::endl;
  float recall = totalHits * 1.0f / totalCnts;
  float topk1Recall = topk1Hits * 100.0f / cnt;
#if 0
    printf("knnTotalTime=%zd linearTotalTime=%zd totalHits=%d totalCnts=%d "
           "R@%zd=%f R@1=%f cost=%f\n",
           knnTotalTime, linearTotalTime, totalHits, totalCnts, topk, recall,
           topk1Recall, cost);
#endif
  EXPECT_GT(recall, 0.80f);
  EXPECT_GT(topk1Recall, 0.80f);
}

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif