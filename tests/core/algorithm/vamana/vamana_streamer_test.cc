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
#include "vamana_streamer.h"
#include <sys/stat.h>
#include <sys/types.h>
#ifndef _MSC_VER
#include <fcntl.h>
#include <unistd.h>
#endif
#include <future>
#include <iostream>
#include <memory>
#include <gtest/gtest.h>
#include <zvec/ailego/container/vector.h>
#include "tests/test_util.h"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace std;
using namespace testing;
using namespace zvec::ailego;

namespace zvec {
namespace core {

constexpr size_t kDim = 16;

class VamanaStreamerTest : public testing::Test {
 protected:
  void SetUp(void);
  void TearDown(void);

  IndexStreamer::Pointer CreateVamanaStreamer(
      const ailego::Params &extra_params = ailego::Params());

  static std::string dir_;
  static shared_ptr<IndexMeta> index_meta_ptr_;
};

std::string VamanaStreamerTest::dir_("vamana_streamer_test_dir/");
shared_ptr<IndexMeta> VamanaStreamerTest::index_meta_ptr_;

void VamanaStreamerTest::SetUp(void) {
  index_meta_ptr_.reset(new (nothrow)
                            IndexMeta(IndexMeta::DataType::DT_FP32, kDim));
  index_meta_ptr_->set_metric("SquaredEuclidean", 0, ailego::Params());

  zvec::test_util::RemoveTestPath(dir_);
}

void VamanaStreamerTest::TearDown(void) {
  zvec::test_util::RemoveTestPath(dir_);
}

IndexStreamer::Pointer VamanaStreamerTest::CreateVamanaStreamer(
    const ailego::Params &extra_params) {
  auto streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  if (!streamer) return nullptr;

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 100U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 64U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 500U);
  params.merge(extra_params);

  if (streamer->init(*index_meta_ptr_, params) != 0) {
    return nullptr;
  }
  return streamer;
}

TEST_F(VamanaStreamerTest, TestAddVector) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestAddVector", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  for (size_t i = 0; i < 1000UL; i++) {
    NumericalVector<float> vec(kDim);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  streamer->flush(0UL);
  streamer.reset();
}

TEST_F(VamanaStreamerTest, TestLinearSearch) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestLinearSearch.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 5000UL;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  NumericalVector<float> vec(kDim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  size_t topk = 3;
  for (size_t i = 0; i < cnt; i += 1) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ctx->set_topk(1U);
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
    auto &result1 = ctx->result();
    ASSERT_EQ(1UL, result1.size());
    ASSERT_EQ(i, result1[0].key());

    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i) + 0.1f;
    }
    ctx->set_topk(topk);
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, ctx));
    auto &result2 = ctx->result();
    ASSERT_EQ(topk, result2.size());
    ASSERT_EQ(i, result2[0].key());
    ASSERT_EQ(i == cnt - 1 ? i - 1 : i + 1, result2[1].key());
    ASSERT_EQ(i == 0 ? 2 : (i == cnt - 1 ? i - 2 : i - 1), result2[2].key());
  }
}

TEST_F(VamanaStreamerTest, TestKnnSearch) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  ailego::Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestKnnSearch.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(kDim);
  size_t cnt = 5000U;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  size_t topk = 100;
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  int totalHits = 0;
  int totalCnts = 0;
  int topk1Hits = 0;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i) + 0.1f;
    }
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));

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
  EXPECT_GT(recall, 0.90f);
  EXPECT_GT(topk1Recall, 0.95f);
}

TEST_F(VamanaStreamerTest, TestOpenClose) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  constexpr size_t dim_large = 128;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim_large);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 100U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);

  streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestOpenClose.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t testCnt = 200;
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim_large);
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  for (size_t i = 0; i < testCnt; i++) {
    std::vector<float> vec(dim_large);
    for (size_t d = 0; d < dim_large; ++d) {
      vec[d] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }

  ASSERT_EQ(0, streamer->flush(0UL));
  ASSERT_EQ(0, streamer->close());

  // Re-open and verify data
  ASSERT_EQ(0, streamer->open(storage));
  auto provider = streamer->create_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  while (iter->is_valid()) {
    float *data = (float *)iter->data();
    for (size_t d = 0; d < dim_large; ++d) {
      ASSERT_FLOAT_EQ(static_cast<float>(iter->key()), data[d]);
    }
    total++;
    iter->next();
  }
  ASSERT_EQ(testCnt, total);
}

TEST_F(VamanaStreamerTest, TestKnnMultiThread) {
  constexpr size_t dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 64U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 500U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 200U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 1000U);
  params.set(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, 30U * 1024U * 1024U);
  params.set(PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, true);

  auto streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestKnnMultiThread", true));
  ASSERT_EQ(0, streamer->open(storage));

  auto addVector = [&streamer, dim](int baseKey, size_t addCnt) {
    NumericalVector<float> vec(dim);
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    size_t succAdd = 0;
    auto ctx = streamer->create_context();
    for (size_t i = 0; i < addCnt; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i + baseKey);
      }
      succAdd += !streamer->add_impl(baseKey + i, vec.data(), qmeta, ctx);
    }
    streamer->flush(0UL);
    return succAdd;
  };
  auto t1 = std::async(std::launch::async, addVector, 0, 1000);
  auto t2 = std::async(std::launch::async, addVector, 1000, 1000);
  auto t3 = std::async(std::launch::async, addVector, 2000, 1000);
  ASSERT_EQ(1000U, t1.get());
  ASSERT_EQ(1000U, t2.get());
  ASSERT_EQ(1000U, t3.get());
  streamer->close();

  // Verify data
  ASSERT_EQ(0, streamer->open(storage));
  auto provider = streamer->create_provider();
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  uint64_t minKey = 10000;
  uint64_t maxKey = 0;
  while (iter->is_valid()) {
    float *data = (float *)iter->data();
    for (size_t d = 0; d < dim; ++d) {
      ASSERT_FLOAT_EQ(static_cast<float>(iter->key()), data[d]);
    }
    total++;
    minKey = std::min(minKey, iter->key());
    maxKey = std::max(maxKey, iter->key());
    iter->next();
  }
  ASSERT_EQ(3000, total);
  ASSERT_EQ(0, minKey);
  ASSERT_EQ(2999, maxKey);

  // Multi-thread search
  size_t topk = 100;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    NumericalVector<float> vec(dim);
    auto linearCtx = streamer->create_context();
    auto knnCtx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    linearCtx->set_topk(topk);
    knnCtx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    for (size_t i = 0; i < cnt; i += 1) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i) + 0.1f;
      }
      ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
      ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
      auto &knnResult = knnCtx->result();
      ASSERT_EQ(topk, knnResult.size());
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
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

TEST_F(VamanaStreamerTest, TestContiguousMemory) {
  ailego::Params extra;
  extra.set(PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY, true);
  extra.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 2000U);
  auto streamer = CreateVamanaStreamer(extra);
  ASSERT_NE(nullptr, streamer);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestContiguous.index", true));

  // First build with default mmap mode
  {
    auto builder_streamer = CreateVamanaStreamer();
    ASSERT_NE(nullptr, builder_streamer);
    ASSERT_EQ(0, builder_streamer->open(storage));
    auto ctx = builder_streamer->create_context();
    ASSERT_TRUE(!!ctx);

    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
    NumericalVector<float> vec(kDim);
    size_t cnt = 3000UL;
    for (size_t i = 0; i < cnt; i++) {
      for (size_t j = 0; j < kDim; ++j) {
        vec[j] = static_cast<float>(i);
      }
      ASSERT_EQ(0, builder_streamer->add_impl(i, vec.data(), qmeta, ctx));
    }
    ASSERT_EQ(0, builder_streamer->flush(0UL));
    ASSERT_EQ(0, builder_streamer->close());
  }

  // Re-open with contiguous memory mode for search
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 3000UL;
  size_t topk = 50;
  NumericalVector<float> vec(kDim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  auto linearCtx = streamer->create_context();
  auto knnCtx = streamer->create_context();
  linearCtx->set_topk(topk);
  knnCtx->set_topk(topk);
  int totalHits = 0;
  int totalCnts = 0;
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i) + 0.1f;
    }
    ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, knnCtx));
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, linearCtx));
    auto &knnResult = knnCtx->result();
    ASSERT_EQ(topk, knnResult.size());
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
  EXPECT_GT(recall, 0.90f);
}

TEST_F(VamanaStreamerTest, TestContiguousMultiThreadSearch) {
  constexpr size_t dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  // Build with mmap mode
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestContiguousMT", true));

  {
    ailego::Params build_params;
    build_params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 64U);
    build_params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 128U);
    build_params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
    build_params.set(PARAM_VAMANA_STREAMER_EF, 64U);
    build_params.set(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, 30U * 1024U * 1024U);
    build_params.set(PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, true);

    auto builder = IndexFactory::CreateStreamer("VamanaStreamer");
    ASSERT_NE(nullptr, builder);
    ASSERT_EQ(0, builder->init(meta, build_params));
    ASSERT_EQ(0, builder->open(storage));

    auto ctx = builder->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    NumericalVector<float> vec(dim);
    for (size_t i = 0; i < 3000; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i);
      }
      ASSERT_EQ(0, builder->add_impl(i, vec.data(), qmeta, ctx));
    }
    ASSERT_EQ(0, builder->flush(0UL));
    ASSERT_EQ(0, builder->close());
  }

  // Re-open with contiguous memory
  ailego::Params search_params;
  search_params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 64U);
  search_params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 128U);
  search_params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  search_params.set(PARAM_VAMANA_STREAMER_EF, 64U);
  search_params.set(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, 30U * 1024U * 1024U);
  search_params.set(PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, true);
  search_params.set(PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY, true);

  auto searcher = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, searcher);
  ASSERT_EQ(0, searcher->init(meta, search_params));
  ASSERT_EQ(0, searcher->open(storage));

  size_t topk = 50;
  size_t cnt = 3000;
  auto knnSearch = [&]() {
    NumericalVector<float> vec(dim);
    auto linearCtx = searcher->create_context();
    auto knnCtx = searcher->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    linearCtx->set_topk(topk);
    knnCtx->set_topk(topk);
    size_t totalCnts = 0;
    size_t totalHits = 0;
    for (size_t i = 0; i < cnt; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i) + 0.1f;
      }
      ASSERT_EQ(0, searcher->search_impl(vec.data(), qmeta, knnCtx));
      ASSERT_EQ(0, searcher->search_bf_impl(vec.data(), qmeta, linearCtx));
      auto &knnResult = knnCtx->result();
      ASSERT_EQ(topk, knnResult.size());
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
    ASSERT_TRUE((totalHits * 1.0f / totalCnts) > 0.80f);
  };
  auto s1 = std::async(std::launch::async, knnSearch);
  auto s2 = std::async(std::launch::async, knnSearch);
  auto s3 = std::async(std::launch::async, knnSearch);
  s1.wait();
  s2.wait();
  s3.wait();
}

TEST_F(VamanaStreamerTest, TestProvider) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestProvider", true));
  ASSERT_EQ(0, streamer->open(storage));

  size_t cnt = 500;
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  NumericalVector<float> vec(kDim);
  for (size_t i = 0; i < cnt; i++) {
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(i);
    }
    ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
  }
  ASSERT_EQ(0, streamer->flush(0UL));

  auto provider = streamer->create_provider();
  ASSERT_NE(nullptr, provider);
  auto iter = provider->create_iterator();
  ASSERT_TRUE(!!iter);
  size_t total = 0;
  while (iter->is_valid()) {
    ASSERT_NE(nullptr, iter->data());
    float *data = (float *)iter->data();
    for (size_t d = 0; d < kDim; ++d) {
      ASSERT_FLOAT_EQ(static_cast<float>(iter->key()), data[d]);
    }
    total++;
    iter->next();
  }
  ASSERT_EQ(cnt, total);
}

TEST_F(VamanaStreamerTest, TestAddAndSearch) {
  auto streamer = CreateVamanaStreamer();
  ASSERT_NE(nullptr, streamer);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestAddAndSearch.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  NumericalVector<float> vec(kDim);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kDim);
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);

  // Add and search interleaved
  for (size_t batch = 0; batch < 5; batch++) {
    size_t base = batch * 200;
    for (size_t i = 0; i < 200; i++) {
      for (size_t j = 0; j < kDim; ++j) {
        vec[j] = static_cast<float>(base + i);
      }
      ASSERT_EQ(0, streamer->add_impl(base + i, vec.data(), qmeta, ctx));
    }

    // Search for recently added vectors
    size_t current_cnt = (batch + 1) * 200;
    size_t topk = std::min(current_cnt, (size_t)10);
    auto searchCtx = streamer->create_context();
    searchCtx->set_topk(topk);
    for (size_t j = 0; j < kDim; ++j) {
      vec[j] = static_cast<float>(base);
    }
    ASSERT_EQ(0, streamer->search_bf_impl(vec.data(), qmeta, searchCtx));
    auto &result = searchCtx->result();
    ASSERT_EQ(topk, result.size());
    ASSERT_EQ(base, result[0].key());
  }
}

TEST_F(VamanaStreamerTest, TestKnnConcurrentAddAndSearch) {
  constexpr size_t dim = 32;
  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 64U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 128U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 64U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 500U);
  params.set(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, 30U * 1024U * 1024U);
  params.set(PARAM_VAMANA_STREAMER_GET_VECTOR_ENABLE, true);

  auto streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestConcurrentAddSearch", true));
  ASSERT_EQ(0, streamer->open(storage));

  // First add some base data
  {
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    NumericalVector<float> vec(dim);
    for (size_t i = 0; i < 2000; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i);
      }
      ASSERT_EQ(0, streamer->add_impl(i, vec.data(), qmeta, ctx));
    }
  }

  std::atomic<bool> stop_search{false};

  // Concurrent add
  auto addFuture = std::async(std::launch::async, [&]() {
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    NumericalVector<float> vec(dim);
    for (size_t i = 2000; i < 3000; i++) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = static_cast<float>(i);
      }
      streamer->add_impl(i, vec.data(), qmeta, ctx);
    }
    stop_search.store(true);
  });

  // Concurrent search
  auto searchFuture = std::async(std::launch::async, [&]() {
    auto ctx = streamer->create_context();
    IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
    NumericalVector<float> vec(dim);
    ctx->set_topk(10);
    while (!stop_search.load()) {
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = 100.1f;
      }
      int ret = streamer->search_impl(vec.data(), qmeta, ctx);
      ASSERT_EQ(0, ret);
      auto &result = ctx->result();
      ASSERT_GT(result.size(), 0UL);
    }
  });

  addFuture.wait();
  searchFuture.wait();
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
