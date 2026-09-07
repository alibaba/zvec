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
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
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

std::string EncodeUniformUint8Record(size_t dimension, uint32_t seed) {
  std::string record(dimension + sizeof(uint32_t), '\0');
  uint32_t sum_squared = 0;
  for (size_t d = 0; d < dimension; ++d) {
    const uint8_t code =
        static_cast<uint8_t>((seed * 73U + d * 29U + d * seed * 3U) & 0xffU);
    record[d] = static_cast<char>(static_cast<int>(code) - 128);
    sum_squared += static_cast<uint32_t>(code) * code;
  }
  std::memcpy(record.data() + dimension, &sum_squared, sizeof(sum_squared));
  return record;
}

class VamanaStreamerTest : public testing::Test {
 protected:
  void SetUp(void) override;
  void TearDown(void) override;

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
  // static: gives dim static storage duration so the addVector lambda below
  // needs no capture for it (MSVC otherwise demands one, C3493, while Clang
  // warns the capture is unused).
  constexpr size_t static dim = 32;
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

  auto addVector = [&streamer](int baseKey, size_t addCnt) {
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

TEST_F(VamanaStreamerTest, TestContiguousPackedGraphAndExtraValuesLayout) {
  constexpr size_t kOriginalDimension = 128;
  constexpr size_t kEncodedDimension = kOriginalDimension + sizeof(uint32_t);
  constexpr size_t kCount = 192;
  constexpr uint64_t kKeyBase = 10000;

  ailego::Params metric_params;
  metric_params.set("proxima.uniform_uint8.metric.origin_metric_name",
                    std::string("SquaredEuclidean"));
  IndexMeta meta(IndexMeta::DataType::DT_INT8, kEncodedDimension);
  meta.set_metric("UniformUint8", 0, metric_params);

  const auto create_streamer = [&](bool contiguous) {
    ailego::Params params;
    params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
    params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE,
               static_cast<uint32_t>(kCount));
    params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
    params.set(PARAM_VAMANA_STREAMER_EF, static_cast<uint32_t>(kCount));
    params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
    params.set(PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY, contiguous);
    auto result = IndexFactory::CreateStreamer("VamanaStreamer");
    if (result == nullptr || result->init(meta, params) != 0) {
      return IndexStreamer::Pointer{};
    }
    return result;
  };

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(dir_ + "TestContiguousPackedLayout.index", true));

  {
    auto builder = create_streamer(false);
    ASSERT_TRUE(builder);
    ASSERT_EQ(0, builder->open(storage));
    auto context = builder->create_context();
    ASSERT_TRUE(context);
    IndexQueryMeta query_meta(IndexMeta::DataType::DT_INT8, kEncodedDimension);
    for (size_t i = 0; i < kCount; ++i) {
      const auto record = EncodeUniformUint8Record(kOriginalDimension,
                                                   static_cast<uint32_t>(i));
      ASSERT_EQ(0, builder->add_impl(kKeyBase + i, record.data(), query_meta,
                                     context));
    }
    ASSERT_EQ(0, builder->flush(0));
    ASSERT_EQ(0, builder->close());
  }

  auto searcher = create_streamer(true);
  ASSERT_TRUE(searcher);
  ASSERT_EQ(0, searcher->open(storage));
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_INT8, kEncodedDimension);
  for (const size_t probe : {size_t{0}, size_t{17}, size_t{91}, kCount - 1}) {
    const uint64_t expected_key = kKeyBase + probe;
    const auto query = EncodeUniformUint8Record(kOriginalDimension,
                                                static_cast<uint32_t>(probe));

    auto graph_context = searcher->create_context();
    ASSERT_TRUE(graph_context);
    graph_context->set_topk(1);
    ASSERT_EQ(0,
              searcher->search_impl(query.data(), query_meta, graph_context));
    ASSERT_EQ(1U, graph_context->result().size());
    EXPECT_EQ(expected_key, graph_context->result()[0].key());
    EXPECT_FLOAT_EQ(0.0f, graph_context->result()[0].score());

    // A valid filter selects the dual-heap graph path. Keep only the exact
    // probe in the result while still allowing traversal through every node.
    auto filtered_context = searcher->create_context();
    ASSERT_TRUE(filtered_context);
    filtered_context->set_topk(1);
    filtered_context->set_filter(
        [expected_key](uint64_t key) { return key != expected_key; });
    ASSERT_EQ(
        0, searcher->search_impl(query.data(), query_meta, filtered_context));
    ASSERT_EQ(1U, filtered_context->result().size());
    EXPECT_EQ(expected_key, filtered_context->result()[0].key());
    EXPECT_FLOAT_EQ(0.0f, filtered_context->result()[0].score());

    auto brute_force_context = searcher->create_context();
    ASSERT_TRUE(brute_force_context);
    brute_force_context->set_topk(1);
    ASSERT_EQ(0, searcher->search_bf_impl(query.data(), query_meta,
                                          brute_force_context));
    ASSERT_EQ(1U, brute_force_context->result().size());
    EXPECT_EQ(expected_key, brute_force_context->result()[0].key());
    EXPECT_FLOAT_EQ(0.0f, brute_force_context->result()[0].score());
  }

  ASSERT_EQ(0, searcher->close());
}

TEST_F(VamanaStreamerTest, TestContiguousKeepsInt8RecordTailInline) {
  constexpr size_t kOriginalDimension = 128;
  constexpr size_t kCount = 96;
  constexpr uint64_t kKeyBase = 20000;

  IndexMeta raw_meta(IndexMeta::DataType::DT_FP32, kOriginalDimension);
  raw_meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_TRUE(converter);
  ASSERT_EQ(0, converter->init(raw_meta, ailego::Params()));
  const IndexMeta record_meta = converter->meta();
  ASSERT_GT(record_meta.element_size(), kOriginalDimension);

  auto metric = IndexFactory::CreateMetric(record_meta.metric_name());
  ASSERT_TRUE(metric);
  ASSERT_EQ(0, metric->init(record_meta, record_meta.metric_params()));
  EXPECT_EQ(0U, metric->extra_values_size_per_vector());

  auto reformer = IndexFactory::CreateReformer(record_meta.reformer_name());
  ASSERT_TRUE(reformer);
  ASSERT_EQ(0, reformer->init(record_meta.reformer_params()));

  const auto create_streamer = [&](bool contiguous) {
    ailego::Params params;
    params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
    params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE,
               static_cast<uint32_t>(kCount));
    params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
    params.set(PARAM_VAMANA_STREAMER_EF, static_cast<uint32_t>(kCount));
    params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
    params.set(PARAM_VAMANA_STREAMER_USE_CONTIGUOUS_MEMORY, contiguous);
    auto result = IndexFactory::CreateStreamer("VamanaStreamer");
    if (result == nullptr || result->init(record_meta, params) != 0) {
      return IndexStreamer::Pointer{};
    }
    return result;
  };

  std::vector<std::vector<float>> vectors(
      kCount, std::vector<float>(kOriginalDimension));
  for (size_t i = 0; i < kCount; ++i) {
    uint32_t state = static_cast<uint32_t>(i + 1);
    for (size_t d = 0; d < kOriginalDimension; ++d) {
      state = state * 1664525U + 1013904223U;
      vectors[i][d] =
          static_cast<float>(static_cast<int32_t>(state >> 8U)) / 8388608.0f;
    }
  }

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(dir_ + "TestContiguousInt8Record.index", true));

  IndexQueryMeta raw_query_meta(IndexMeta::DataType::DT_FP32,
                                kOriginalDimension);
  {
    auto builder = create_streamer(false);
    ASSERT_TRUE(builder);
    ASSERT_EQ(0, builder->open(storage));
    auto context = builder->create_context();
    ASSERT_TRUE(context);
    for (size_t i = 0; i < kCount; ++i) {
      std::string record;
      IndexQueryMeta encoded_meta;
      ASSERT_EQ(0, reformer->convert(vectors[i].data(), raw_query_meta, &record,
                                     &encoded_meta));
      ASSERT_EQ(record_meta.element_size(), record.size());
      ASSERT_EQ(0, builder->add_impl(kKeyBase + i, record.data(), encoded_meta,
                                     context));
    }
    ASSERT_EQ(0, builder->flush(0));
    ASSERT_EQ(0, builder->close());
  }

  auto searcher = create_streamer(true);
  ASSERT_TRUE(searcher);
  ASSERT_EQ(0, searcher->open(storage));
  const std::array<size_t, 4> probes{{0, 17, 53, kCount - 1}};
  for (size_t probe : probes) {
    std::string query;
    IndexQueryMeta query_meta;
    ASSERT_EQ(0, reformer->transform(vectors[probe].data(), raw_query_meta,
                                     &query, &query_meta));
    ASSERT_EQ(record_meta.element_size(), query.size());

    auto graph_context = searcher->create_context();
    ASSERT_TRUE(graph_context);
    graph_context->set_topk(1);
    ASSERT_EQ(0,
              searcher->search_impl(query.data(), query_meta, graph_context));
    ASSERT_EQ(1U, graph_context->result().size());
    EXPECT_EQ(kKeyBase + probe, graph_context->result()[0].key());
    EXPECT_NEAR(0.0f, graph_context->result()[0].score(), 1e-4f);

    auto brute_force_context = searcher->create_context();
    ASSERT_TRUE(brute_force_context);
    brute_force_context->set_topk(1);
    ASSERT_EQ(0, searcher->search_bf_impl(query.data(), query_meta,
                                          brute_force_context));
    ASSERT_EQ(1U, brute_force_context->result().size());
    EXPECT_EQ(kKeyBase + probe, brute_force_context->result()[0].key());
    EXPECT_NEAR(0.0f, brute_force_context->result()[0].score(), 1e-4f);
  }

  ASSERT_EQ(0, searcher->close());
  ASSERT_EQ(0, storage->close());
}

TEST_F(VamanaStreamerTest, TestContiguousPackedGraphTracksNeighborUpdates) {
  constexpr uint32_t kMaxDegree = 8;
  constexpr uint32_t kCount = 12;

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, kMaxDegree);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, kCount);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);
  auto builder = CreateVamanaStreamer(params);
  ASSERT_TRUE(builder);

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(
      0, storage->open(dir_ + "TestContiguousPackedGraphUpdates.index", true));
  ASSERT_EQ(0, builder->open(storage));

  auto context = builder->create_context();
  ASSERT_TRUE(context);
  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kDim);
  for (uint32_t i = 0; i < kCount; ++i) {
    std::array<float, kDim> vector{};
    for (size_t d = 0; d < vector.size(); ++d) {
      vector[d] = static_cast<float>(i * 17U + d);
    }
    ASSERT_EQ(0, builder->add_impl(i, vector.data(), query_meta, context));
  }
  ASSERT_EQ(0, builder->flush(0));
  ASSERT_EQ(0, builder->close());
  builder.reset();

  IndexStreamer::Stats stats;
  VamanaContiguousStreamerEntity entity(stats);
  entity.set_use_key_info_map(true);
  entity.set_vector_size(index_meta_ptr_->element_size());
  entity.set_max_degree(kMaxDegree);
  entity.set_search_list_size(kCount);
  entity.set_max_occlusion_size(VamanaEntity::kDefaultMaxOcclusionSize);
  ASSERT_EQ(0, entity.init(kCount));
  ASSERT_EQ(0, entity.open(storage, 0, false));
  ASSERT_EQ(0, entity.build_contiguous_memory());

  const std::vector<std::pair<node_id_t, dist_t>> replacement = {{1U, 1.0F},
                                                                 {2U, 2.0F}};
  ASSERT_EQ(0, entity.update_neighbors(0U, replacement));
  auto neighbors = entity.get_neighbors(0U);
  ASSERT_EQ(2U, neighbors.size());
  EXPECT_EQ(1U, neighbors[0]);
  EXPECT_EQ(2U, neighbors[1]);

  entity.add_neighbor(0U, 2U, 3U);
  neighbors = entity.get_neighbors(0U);
  ASSERT_EQ(3U, neighbors.size());
  EXPECT_EQ(3U, neighbors[2]);

  entity.degrade_to_mmap();
  neighbors = entity.get_neighbors(0U);
  ASSERT_EQ(3U, neighbors.size());
  EXPECT_EQ(1U, neighbors[0]);
  EXPECT_EQ(2U, neighbors[1]);
  EXPECT_EQ(3U, neighbors[2]);
  ASSERT_EQ(0, entity.close());
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

// Test concurrent build (parallel add_impl) which was crashing due to
// unprotected node_chunks_ / node_chunk_bases_ access during chunk allocation.
TEST_F(VamanaStreamerTest, TestConcurrentBuild) {
  constexpr size_t dim = kDim;
  constexpr size_t total_vectors = 5000;
  constexpr size_t thread_count = 4;

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 100U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 64U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 500U);
  params.set(PARAM_VAMANA_STREAMER_MAX_INDEX_SIZE, 50U * 1024U * 1024U);

  IndexMeta meta(IndexMeta::DataType::DT_FP32, dim);
  meta.set_metric("SquaredEuclidean", 0, ailego::Params());

  auto streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer);
  ASSERT_EQ(0, streamer->init(meta, params));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ailego::Params stg_params;
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestConcurrentBuild", true));
  ASSERT_EQ(0, streamer->open(storage));

  // Parallel insertion from multiple threads (mimics local_builder behavior)
  std::atomic<int> error_count{0};
  std::vector<std::future<void>> futures;

  for (size_t t = 0; t < thread_count; ++t) {
    futures.push_back(std::async(std::launch::async, [&, t]() {
      auto ctx = streamer->create_context();
      ASSERT_TRUE(!!ctx);
      IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
      NumericalVector<float> vec(dim);

      for (size_t i = t; i < total_vectors; i += thread_count) {
        for (size_t j = 0; j < dim; ++j) {
          vec[j] = static_cast<float>(i) + static_cast<float>(j) * 0.01f;
        }
        int ret = streamer->add_impl(i, vec.data(), qmeta, ctx);
        if (ret != 0) {
          error_count.fetch_add(1);
          return;
        }
      }
    }));
  }

  for (auto &f : futures) {
    f.wait();
  }
  ASSERT_EQ(0, error_count.load());

  // Verify search still works correctly after concurrent build
  auto search_ctx = streamer->create_context();
  ASSERT_TRUE(!!search_ctx);
  search_ctx->set_topk(1);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, dim);
  NumericalVector<float> vec(dim);
  for (size_t j = 0; j < dim; ++j) {
    vec[j] = 0.0f;
  }
  ASSERT_EQ(0, streamer->search_impl(vec.data(), qmeta, search_ctx));
  auto &result = search_ctx->result();
  ASSERT_GT(result.size(), 0UL);
}

TEST_F(VamanaStreamerTest, TestAsymmetricQueryMetric) {
  constexpr size_t kTestDimension = 2;

  ailego::Params metric_params;
  metric_params.set("proxima.mips_euclidean.metric.injection_type", 0);
  IndexMeta meta(IndexMeta::DataType::DT_FP32, kTestDimension);
  meta.set_metric("MipsSquaredEuclidean", 0, metric_params);

  ailego::Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 8U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 16U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 16U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 0U);

  auto streamer = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_TRUE(streamer);
  ASSERT_EQ(0, streamer->init(meta, params));

  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_TRUE(storage);
  ASSERT_EQ(0, storage->init(ailego::Params()));
  ASSERT_EQ(0, storage->open(dir_ + "TestAsymmetricQueryMetric.index", true));
  ASSERT_EQ(0, streamer->open(storage));

  IndexQueryMeta query_meta(IndexMeta::DataType::DT_FP32, kTestDimension);
  NumericalVector<float> unit_record(kTestDimension);
  unit_record[0] = 1.0f;
  unit_record[1] = 0.0f;
  NumericalVector<float> scaled_record(kTestDimension);
  scaled_record[0] = 2.0f;
  scaled_record[1] = 0.0f;

  auto context = streamer->create_context();
  ASSERT_TRUE(context);
  ASSERT_EQ(0, streamer->add_impl(10, unit_record.data(), query_meta, context));
  ASSERT_EQ(0,
            streamer->add_impl(20, scaled_record.data(), query_meta, context));

  auto *vamana_context = dynamic_cast<VamanaContext *>(context.get());
  ASSERT_TRUE(vamana_context);
  EXPECT_FLOAT_EQ(1.0f, vamana_context->dist_calculator().dist(
                            unit_record.data(), scaled_record.data()));

  context->set_topk(2);
  ASSERT_EQ(0,
            streamer->search_bf_impl(unit_record.data(), query_meta, context));
  ASSERT_EQ(2UL, context->result().size());
  EXPECT_EQ(20UL, context->result()[0].key());
  EXPECT_FLOAT_EQ(-2.0f, context->result()[0].score());
  EXPECT_EQ(10UL, context->result()[1].key());
  EXPECT_FLOAT_EQ(-1.0f, context->result()[1].score());
  EXPECT_FLOAT_EQ(-2.0f, vamana_context->dist_calculator().dist(
                             unit_record.data(), scaled_record.data()));

  auto graph_context = streamer->create_context();
  ASSERT_TRUE(graph_context);
  graph_context->set_topk(1);
  ASSERT_EQ(
      0, streamer->search_impl(unit_record.data(), query_meta, graph_context));
  ASSERT_EQ(1UL, graph_context->result().size());
  EXPECT_EQ(20UL, graph_context->result()[0].key());
  EXPECT_FLOAT_EQ(-2.0f, graph_context->result()[0].score());

  auto primary_key_context = streamer->create_context();
  ASSERT_TRUE(primary_key_context);
  primary_key_context->set_topk(2);
  const std::vector<std::vector<uint64_t>> primary_keys{{10, 20}};
  ASSERT_EQ(
      0, streamer->search_bf_by_p_keys_impl(unit_record.data(), primary_keys,
                                            query_meta, primary_key_context));
  ASSERT_EQ(2UL, primary_key_context->result().size());
  EXPECT_EQ(20UL, primary_key_context->result()[0].key());
  EXPECT_FLOAT_EQ(-2.0f, primary_key_context->result()[0].score());
}

// Test Vamana + INT8 quantization + rotation end-to-end
TEST_F(VamanaStreamerTest, TestInt8WithRotate) {
  constexpr size_t kTestDim = 128;
  constexpr size_t kCnt = 2000U;
  constexpr size_t kTopk = 10;

  IndexStreamer::Pointer streamer =
      IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer);

  Params params;
  params.set(PARAM_VAMANA_STREAMER_MAX_DEGREE, 32U);
  params.set(PARAM_VAMANA_STREAMER_SEARCH_LIST_SIZE, 100U);
  params.set(PARAM_VAMANA_STREAMER_ALPHA, 1.2f);
  params.set(PARAM_VAMANA_STREAMER_EF, 64U);
  params.set(PARAM_VAMANA_STREAMER_BRUTE_FORCE_THRESHOLD, 500U);

  IndexMeta index_meta_raw(IndexMeta::DataType::DT_FP32, kTestDim);
  index_meta_raw.set_metric("SquaredEuclidean", 0, Params());

  // Create INT8 converter with rotation enabled
  Params converter_params;
  converter_params.set("integer_streaming.converter.enable_rotate", true);
  auto converter = IndexFactory::CreateConverter("Int8StreamingConverter");
  ASSERT_NE(nullptr, converter);
  ASSERT_EQ(0, converter->init(index_meta_raw, converter_params));

  IndexMeta index_meta = converter->meta();

  auto reformer = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_NE(nullptr, reformer);
  ASSERT_EQ(0, reformer->init(index_meta.reformer_params()));

  Params stg_params;
  auto storage = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage);
  ASSERT_EQ(0, storage->init(stg_params));
  ASSERT_EQ(0, storage->open(dir_ + "TestInt8WithRotate.index", true));
  ASSERT_EQ(0, streamer->init(index_meta, params));
  ASSERT_EQ(0, streamer->open(storage));

  // Add 2000 vectors
  auto ctx = streamer->create_context();
  ASSERT_TRUE(!!ctx);
  IndexQueryMeta qmeta(IndexMeta::DataType::DT_FP32, kTestDim);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < kCnt; i++) {
    NumericalVector<float> vec(kTestDim);
    for (size_t j = 0; j < kTestDim; ++j) vec[j] = dist(gen);

    std::string new_vec;
    IndexQueryMeta new_meta;
    ASSERT_EQ(0, reformer->convert(vec.data(), qmeta, &new_vec, &new_meta));
    ASSERT_EQ(0, streamer->add_impl(i, new_vec.data(), new_meta, ctx));
  }

  streamer->flush(0UL);
  streamer.reset();
  storage.reset();

  // Reopen: reformer should auto-detect rotator from storage
  auto storage2 = IndexFactory::CreateStorage("MMapFileStorage");
  ASSERT_NE(nullptr, storage2);
  ASSERT_EQ(0, storage2->init(stg_params));
  ASSERT_EQ(0, storage2->open(dir_ + "TestInt8WithRotate.index", false));

  auto streamer2 = IndexFactory::CreateStreamer("VamanaStreamer");
  ASSERT_NE(nullptr, streamer2);
  ASSERT_EQ(0, streamer2->init(index_meta, params));
  ASSERT_EQ(0, streamer2->open(storage2));

  auto reformer2 = IndexFactory::CreateReformer(index_meta.reformer_name());
  ASSERT_NE(nullptr, reformer2);
  ASSERT_EQ(0, reformer2->init(index_meta.reformer_params()));
  ASSERT_EQ(0, reformer2->load(storage2));

  // Search: verify knn results are non-empty
  auto knnCtx = streamer2->create_context();
  knnCtx->set_topk(kTopk);
  auto linearCtx = streamer2->create_context();
  linearCtx->set_topk(kTopk);

  NumericalVector<float> query(kTestDim);
  for (size_t j = 0; j < kTestDim; ++j) query[j] = dist(gen);

  std::string new_query;
  IndexQueryMeta new_qmeta;
  ASSERT_EQ(0,
            reformer2->transform(query.data(), qmeta, &new_query, &new_qmeta));
  ASSERT_EQ(0, streamer2->search_impl(new_query.data(), new_qmeta, knnCtx));
  ASSERT_EQ(0,
            streamer2->search_bf_impl(new_query.data(), new_qmeta, linearCtx));

  EXPECT_EQ(kTopk, knnCtx->result().size());
  EXPECT_EQ(kTopk, linearCtx->result().size());
}

}  // namespace core
}  // namespace zvec

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
